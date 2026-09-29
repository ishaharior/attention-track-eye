"""Browser interface for live webcam gaze tracking and document heatmaps.

Pipeline (reuses gaze_core / phase3 directly):

    browser webcam frame (JPEG dataURL)
        -> EyeTracker.process          (MediaPipe face/eye features + head pose)
        -> ScreenCalibrator.update     (5/9/13-point calibration, optional)
        -> ScreenMapper.map            (affine | polynomial | MLP -> display px)
        -> AttentionHeatmap.add        (confidence-weighted, document pixels)
        -> /api/finish                 (time decay -> Gaussian blur -> TURBO
                                        blue->red colormap over the document)

Run:  python web_app.py   then open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import base64
import binascii
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from gaze_core import (
    AttentionHeatmap,
    EyeTracker,
    GazeConfig,
    MappingSample,
    ScreenCalibrator,
    ScreenMapper,
    robust_median,
)
from phase3_attention_heatmap import EventDetector

APP_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(APP_DIR, "web_interface.html")

# hard caps so a huge upload cannot exhaust memory or the heatmap grid
MAX_DOC_DIM = 2400

# grid-mapping mode: one coloured digit every GRID_INTERVAL_S seconds; the
# frames shown during each digit become (features -> digit position) samples
GRID_INTERVAL_S = 5.0
GRID_MAX_SAMPLES_PER_TARGET = 60
GRID_MIN_SAMPLES_PER_TARGET = 3
GRID_MIN_TARGETS = 3

LOGGER = logging.getLogger("web_app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

STATE_LOCK = threading.Lock()
_tracker: Optional[EyeTracker] = None
_session: Optional["TrackingSession"] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def decode_data_url(data_url: str) -> Optional[np.ndarray]:
    """Decode a base64 data URL into a BGR image, or None on failure."""
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        buffer = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image if image is not None and image.size else None
    except (binascii.Error, ValueError, cv2.error):
        return None


def encode_png_data_url(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def get_tracker() -> EyeTracker:
    global _tracker
    if _tracker is None:
        LOGGER.info("loading MediaPipe FaceMesh (first request)...")
        started = time.perf_counter()
        _tracker = EyeTracker()
        LOGGER.info("tracker ready in %.2fs", time.perf_counter() - started)
    return _tracker


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
class TrackingSession:
    """State for one upload -> calibrate -> track -> report cycle.

    The mapper/calibrator operate in *display* pixels (the CSS rectangle the
    document occupies in the browser viewport).  The heatmap accumulates in
    *document natural* pixels so the final report matches the uploaded file's
    own scale.
    """

    def __init__(
        self,
        doc_width: int,
        doc_height: int,
        rect: Dict[str, float],
        calibration: bool,
        points: int,
        image_bgr: np.ndarray,
        config: GazeConfig,
    ) -> None:
        self.doc_width = int(doc_width)
        self.doc_height = int(doc_height)
        self.rect = {k: float(rect[k]) for k in ("x", "y", "w", "h")}
        self.config = config
        self.doc_image = image_bgr
        self.tmpdir = tempfile.mkdtemp(prefix="gaze_web_")

        disp_w = max(int(round(self.rect["w"])), 1)
        disp_h = max(int(round(self.rect["h"])), 1)

        self.calibrator: Optional[ScreenCalibrator] = None
        if calibration:
            self.calibrator = ScreenCalibrator(
                width=disp_w,
                height=disp_h,
                config=config,
                points=points,
            )
            self.calibrator.start()
            self.mode = "calibrating"
        else:
            self.mode = "tracking"

        self.mapper = ScreenMapper(
            width=disp_w,
            height=disp_h,
            config=config,
            path=os.path.join(self.tmpdir, "session_calibration.npz"),
            smoothing=config.ema_alpha,
        )

        # sigma is defined in screen pixels; scale it to document width so the
        # blob covers the same fraction of whatever document is uploaded
        sigma_doc = clamp(
            config.gaussian_sigma_px * self.doc_width / max(config.screen_width, 1),
            10.0,
            150.0,
        )
        self.heatmap = AttentionHeatmap(
            width=self.doc_width,
            height=self.doc_height,
            cell=config.heatmap_cell_size,
            sigma=sigma_doc,
            decay_tau=config.heatmap_decay_tau,
            colormap=cv2.COLORMAP_TURBO,
            alpha=config.heatmap_alpha,
            gamma=config.heatmap_gamma,
            min_confidence=config.heatmap_min_confidence,
        )
        self.detector = EventDetector(
            method=config.fixation_method,
            dispersion_px=config.fixation_dispersion_px,
            min_duration_ms=config.fixation_min_duration_ms,
            velocity_threshold=config.saccade_velocity_threshold,
        )

        # (t_ms, x, y, confidence) in display pixels
        self.samples: List[Tuple[float, float, float, float]] = []
        # eye-to-camera distance: live smoothed value plus min/max history;
        # the first value (or the distance at calibration time) becomes the
        # mapper's reference and drift past config.distance_drift_warn_pct
        # triggers a recalibration warning in the UI and report
        self.distance_cm: Optional[float] = None
        self.distance_sum = 0.0
        self.distance_count = 0
        self.distance_min_cm: Optional[float] = None
        self.distance_max_cm: Optional[float] = None
        self.distance_warning = False
        # grid mapping: digit id -> [(features, confidence)] and id -> target
        self.grid_samples: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        self.grid_targets: Dict[int, Tuple[float, float]] = {}
        self.grid_stats: Optional[Dict[str, object]] = None
        self.frames = 0
        self.face_frames = 0
        self.inference_ms: List[float] = []
        self.started_at = time.time()
        self.calib_residual: Optional[Dict[str, float]] = None
        self.calibration_error: Optional[str] = None
        self.finished = False

    def dispose(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @property
    def calibration_summary(self) -> Optional[Dict[str, object]]:
        if self.calibrator is None:
            return None
        summary: Dict[str, object] = {
            "points": self.calibrator.n_points,
            "index": int(self.calibrator.index),
            "total": len(self.calibrator.targets),
            "progress": float(self.calibrator.progress),
            "accepted": int(self.calibrator.total_accepted),
            "rejected": int(self.calibrator.rejected),
            "done": bool(self.calibrator.done),
            "fitted": bool(self.mapper.calibrated),
        }
        if self.calib_residual is not None:
            summary["residual"] = self.calib_residual
        if self.calibration_error is not None:
            summary["error"] = self.calibration_error
        return summary

    def current_target(self) -> Optional[Dict[str, float]]:
        if self.calibrator is None:
            return None
        target = self.calibrator.target_px
        if target is None:
            return None
        return {"x": float(target[0]), "y": float(target[1])}

    @property
    def reference_distance_cm(self) -> Optional[float]:
        return self.mapper.reference_distance_cm

    def note_distance(self, distance_cm: Optional[float]) -> None:
        """Update distance history and flag drift from the reference distance."""
        if distance_cm is None or not math.isfinite(float(distance_cm)):
            return
        value = float(distance_cm)
        self.distance_cm = value
        self.distance_sum += value
        self.distance_count += 1
        self.distance_min_cm = value if self.distance_min_cm is None else min(self.distance_min_cm, value)
        self.distance_max_cm = value if self.distance_max_cm is None else max(self.distance_max_cm, value)
        if self.mapper.reference_distance_cm is None:
            # first observation (uncalibrated) becomes the reference
            self.mapper.set_reference_distance(value)
        reference = self.mapper.reference_distance_cm
        if reference:
            drift = 100.0 * abs(value - reference) / reference
            self.distance_warning = drift > self.config.distance_drift_warn_pct

    @property
    def distance_drift_pct(self) -> Optional[float]:
        reference = self.mapper.reference_distance_cm
        if reference is None or self.distance_cm is None:
            return None
        return 100.0 * abs(self.distance_cm - reference) / reference

    @property
    def mean_distance_cm(self) -> Optional[float]:
        if self.distance_count == 0:
            return None
        return self.distance_sum / self.distance_count


def complete_calibration(session: TrackingSession) -> None:
    """Fit the mapping model on collected samples and measure in-sample error."""
    assert session.calibrator is not None
    samples = session.calibrator.finalize()
    # lock the mapper's distance reference to where the user sat while
    # calibrating; later leaning shows up as drift + gaze compensation
    if not session.mapper.fit(samples, reference_distance_cm=session.distance_cm):
        session.calibration_error = (
            "fit failed: not enough feature spread - sit closer, improve lighting, "
            "or redo calibration"
        )
        session.mode = "calibration_failed"
        return
    errors = []
    for sample in session.mapper.samples:
        predicted = session.mapper.model.predict(
            np.asarray(sample.features, dtype=np.float64).reshape(1, -1)
        )[0]
        errors.append(
            math.hypot(predicted[0] - sample.target[0], predicted[1] - sample.target[1])
        )
    if errors:
        ordered = sorted(errors)
        session.calib_residual = {
            "mean_px": float(sum(errors) / len(errors)),
            "median_px": float(ordered[len(ordered) // 2]),
            "max_px": float(ordered[-1]),
            "sample_count": len(errors),
        }
    session.mode = "tracking"
    LOGGER.info(
        "calibration fitted: %s, residual mean %.1f px",
        session.mapper.mapping_name,
        (session.calib_residual or {}).get("mean_px", float("nan")),
    )


def handle_frame(
    session: TrackingSession,
    tracker: EyeTracker,
    frame: np.ndarray,
    t_ms: float,
    grid: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run one frame through Phase 1 (+ Phase 2 while calibrating/tracking).

    `grid` is the currently displayed grid-mapping digit (client-driven):
    ``{"active", "id", "x", "y", "restart"}`` in document-natural pixels.
    While a digit is on screen its feature vectors are collected next to the
    digit position so /api/grid/finish can refit the mapping.
    """
    result = tracker.process(frame)
    session.frames += 1
    session.inference_ms.append(result.inference_ms)
    session.note_distance(result.distance_cm)

    face_found = result.features is not None or bool(result.eyes)
    if face_found:
        session.face_frames += 1

    response: Dict[str, object] = {
        "mode": session.mode,
        "face": bool(face_found),
        "confidence": float(result.confidence),
        "inference_ms": float(result.inference_ms),
        "frame": session.frames,
        "distance_warning": bool(session.distance_warning),
    }
    if result.distance_cm is not None:
        response["distance_cm"] = round(float(result.distance_cm), 1)
    if session.reference_distance_cm is not None:
        response["distance_reference_cm"] = round(float(session.reference_distance_cm), 1)
    drift = session.distance_drift_pct
    if drift is not None:
        response["distance_drift_pct"] = round(drift, 1)
    if result.confidences is not None:
        response["confidence_breakdown"] = result.confidences.as_dict()
    if result.head_pose is not None:
        response["head_pose"] = result.head_pose.as_dict()

    if session.mode == "calibrating" and session.calibrator is not None:
        status = session.calibrator.update(result.features, result.confidence, result.gaze)
        response["calibration"] = session.calibration_summary
        response["calibration"]["status"] = status
        response["calibration"]["target"] = session.current_target()
        if session.calibrator.done:
            complete_calibration(session)
            response["mode"] = session.mode
            response["calibration"] = session.calibration_summary
    elif session.mode == "tracking":
        position = session.mapper.map(
            result.gaze, result.features, result.confidence,
            distance_cm=result.distance_cm,
        )
        if position is not None:
            x, y = position
            scale_x = session.doc_width / max(session.rect["w"], 1e-6)
            scale_y = session.doc_height / max(session.rect["h"], 1e-6)
            natural_x = clamp(x * scale_x, 0.0, session.doc_width - 1)
            natural_y = clamp(y * scale_y, 0.0, session.doc_height - 1)
            session.heatmap.add(
                natural_x,
                natural_y,
                t_ms / 1000.0,
                weight=1.0,
                confidence=result.confidence,
            )
            session.samples.append((t_ms, x, y, result.confidence))
            session.detector.update(
                t_ms, x, y, result.confidence,
                min_confidence=0.0,
            )
            response["gaze"] = {"x": float(x), "y": float(y)}
            response["mapper_stats"] = dict(session.mapper.stats)

    if grid and grid.get("active"):
        try:
            grid_id = int(grid["id"])
            grid_x = float(grid["x"])
            grid_y = float(grid["y"])
            restart = bool(grid.get("restart", False))
        except (KeyError, TypeError, ValueError):
            grid_id = -1
        else:
            if restart:
                session.grid_samples.clear()
                session.grid_targets.clear()
            if (
                grid_id >= 0
                and result.features is not None
                and result.confidence >= session.config.calibration_min_confidence
            ):
                session.grid_targets[grid_id] = (grid_x, grid_y)
                bucket = session.grid_samples.setdefault(grid_id, [])
                if len(bucket) < GRID_MAX_SAMPLES_PER_TARGET:
                    bucket.append(
                        (np.asarray(result.features, dtype=np.float64), float(result.confidence))
                    )
                    response["grid"] = {"id": grid_id, "collected": len(bucket)}
    return response


def build_report(
    session: TrackingSession,
    apply_decay: bool,
) -> Dict[str, object]:
    """Render the session heatmap over the uploaded document and summarise it.

    Decay (when requested) is applied end-anchored: each stored point is
    weighted by exp(-(t_end - t) / tau), i.e. the same time-based rule the
    live pipeline uses, evaluated once for the finished session instead of
    per frame, so old gaze does not literally vanish from a long report.
    """
    heatmap = session.heatmap
    decay_tau = 0.0
    points = heatmap.points_matrix()
    if len(points):
        # the report grid is always re-derived from stored points so repeated
        # finishes (with or without decay) produce identical results
        heatmap.grid[:] = 0.0
        t_end = float(points[-1, 0])
        if apply_decay and heatmap.decay_enabled:
            decay_tau = heatmap.decay_tau
        for t, x, y, weight in points:
            col = int(x / heatmap.cell)
            row = int(y / heatmap.cell)
            if 0 <= row < heatmap.grid_h and 0 <= col < heatmap.grid_w:
                factor = (
                    math.exp(-max(t_end - float(t), 0.0) / decay_tau)
                    if decay_tau > 0.0
                    else 1.0
                )
                heatmap.grid[row, col] += float(weight) * factor

    composited = heatmap.render(background=session.doc_image)
    session.detector.finalize()
    event_summary = session.detector.summary(attempted_samples=len(session.samples))

    confidences = [s[3] for s in session.samples]
    mapper_stats = dict(session.mapper.stats)
    total_mapped = max(sum(mapper_stats.values()), 1)

    stats: Dict[str, object] = {
        "duration_s": round(time.time() - session.started_at, 1),
        "frames": session.frames,
        "face_rate": round(session.face_frames / max(session.frames, 1), 3),
        "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else 0.0,
        "gaze_samples": len(session.samples),
        "heatmap_points": int(heatmap.total_hits),
        "rejected_low_confidence": int(heatmap.rejected_low_confidence),
        "fixations": int(event_summary.get("fixation_count", 0)),
        "saccades": int(event_summary.get("saccade_count", 0)),
        "mean_fixation_ms": round(
            float(event_summary.get("average_fixation_duration_ms", 0.0)), 0
        ),
        "mean_inference_ms": round(float(np.mean(session.inference_ms)), 1)
        if session.inference_ms
        else 0.0,
        "calibration": session.calib_residual,
        "grid": session.grid_stats,
        "distance": {
            "reference_cm": round(float(session.reference_distance_cm), 1)
            if session.reference_distance_cm is not None
            else None,
            "mean_cm": round(float(session.mean_distance_cm), 1)
            if session.mean_distance_cm is not None
            else None,
            "min_cm": round(float(session.distance_min_cm), 1)
            if session.distance_min_cm is not None
            else None,
            "max_cm": round(float(session.distance_max_cm), 1)
            if session.distance_max_cm is not None
            else None,
            "drift_pct": round(float(session.distance_drift_pct), 1)
            if session.distance_drift_pct is not None
            else None,
            "drift_warning": bool(session.distance_warning),
            "warn_threshold_pct": session.config.distance_drift_warn_pct,
        },
        "mapping": {
            "model": session.mapper.mapping_name,
            "valid_sample_rate": round(session.mapper.valid_sample_rate, 3),
            "stats": mapper_stats,
            "mapped_share": round(session.mapper.stats.get("accepted", 0) / total_mapped, 3),
        },
        "heatmap": {
            "cell_px": heatmap.cell,
            "sigma_px": round(heatmap.sigma, 1),
            "gamma": heatmap.gamma,
            "alpha": heatmap.alpha,
            "decay_tau_s": decay_tau,
            "colormap": "TURBO (blue -> red)",
        },
    }
    session.finished = True
    return {
        "heatmap_png": encode_png_data_url(composited),
        "doc": {"width": session.doc_width, "height": session.doc_height},
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.route("/")
def index() -> Response:
    with open(HTML_PATH, encoding="utf-8") as handle:
        return Response(handle.read(), mimetype="text/html")


@app.route("/api/session", methods=["POST"])
def create_session() -> Response:
    global _session
    payload = request.get_json(silent=True) or {}
    image_bgr = decode_data_url(str(payload.get("image", "")))
    if image_bgr is None:
        return jsonify({"error": "document image missing or undecodable"}), 400

    rect = payload.get("rect") or {}
    try:
        doc_height, doc_width = image_bgr.shape[:2]
        rect_w = float(rect["w"])
        rect_h = float(rect["h"])
        if rect_w <= 0 or rect_h <= 0:
            raise ValueError("rect must be positive")
        points = int(payload.get("points", 5))
        calibration = bool(payload.get("calibration", True))
        if points == 0:
            # "None — rough centre mapping": no calibrator, points are irrelevant
            calibration = False
            points = 5
        if points not in (5, 9, 13):
            raise ValueError("points must be 5, 9 or 13")
        model = str(payload.get("model", "affine"))
        if model not in ("affine", "polynomial", "mlp"):
            raise ValueError("model must be affine, polynomial or mlp")
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"invalid session payload: {exc}"}), 400

    try:
        config = GazeConfig(
            calibration_points=points,
            mapping_model=model,
            # adaptive smoothing by default: strong while the gaze is still,
            # light while it moves (the fixed EMA lags or jitters at one end)
            smoothing_filter=str(payload.get("smoothing", "oneeuro")),
        )
        config.validate()
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"invalid session payload: {exc}"}), 400

    with STATE_LOCK:
        if _session is not None:
            _session.dispose()
        # downscale very large uploads so base64 + heatmap stay responsive
        longest = max(doc_width, doc_height)
        if longest > MAX_DOC_DIM:
            scale = MAX_DOC_DIM / longest
            image_bgr = cv2.resize(
                image_bgr,
                (max(int(doc_width * scale), 1), max(int(doc_height * scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
            doc_height, doc_width = image_bgr.shape[:2]
        _session = TrackingSession(
            doc_width=doc_width,
            doc_height=doc_height,
            rect={"x": rect.get("x", 0), "y": rect.get("y", 0), "w": rect_w, "h": rect_h},
            calibration=calibration,
            points=points,
            image_bgr=image_bgr,
            config=config,
        )
        get_tracker()  # load the model up-front so the first frame is fast
        session = _session
    return jsonify(
        {
            "mode": session.mode,
            "doc": {"width": session.doc_width, "height": session.doc_height},
            "calibration": session.calibration_summary,
            "model": model,
            "colormap": "TURBO (blue -> red)",
        }
    )


@app.route("/api/frame", methods=["POST"])
def frame() -> Response:
    payload = request.get_json(silent=True) or {}
    image_bgr = decode_data_url(str(payload.get("image", "")))
    if image_bgr is None:
        return jsonify({"error": "frame missing or undecodable"}), 400
    t_ms = float(payload.get("t_ms", time.monotonic() * 1000.0))

    with STATE_LOCK:
        if _session is None or _session.finished:
            return jsonify({"error": "no active session"}), 409
        try:
            tracker = get_tracker()
            response = handle_frame(_session, tracker, image_bgr, t_ms,
                                    grid=payload.get("grid"))
        except Exception as exc:  # keep the loop alive on a single bad frame
            LOGGER.exception("frame processing failed")
            return jsonify({"error": f"frame failed: {exc}"}), 500
    return jsonify(response)


@app.route("/api/grid/finish", methods=["POST"])
def grid_finish() -> Response:
    """Refit the mapping model from the collected grid-digit samples.

    Each digit contributes one representative (median) feature vector paired
    with the digit's display position.  The error before and after the refit
    is reported so the improvement is measured, not claimed.
    """
    with STATE_LOCK:
        if _session is None:
            return jsonify({"error": "no active session"}), 409
        if _session.mode != "tracking":
            return jsonify({"error": "grid mapping runs while tracking"}), 409
        session = _session
        config = session.config

        new_samples: List[MappingSample] = []
        pre_errors: List[float] = []
        total_samples = 0
        for grid_id, (natural_x, natural_y) in session.grid_targets.items():
            rows = session.grid_samples.get(grid_id, [])
            total_samples += len(rows)
            if len(rows) < GRID_MIN_SAMPLES_PER_TARGET:
                continue
            representative = robust_median(
                [row[0] for row in rows], config.calibration_outlier_z
            )
            if representative is None:
                continue
            display_x = natural_x * session.rect["w"] / max(session.doc_width, 1)
            display_y = natural_y * session.rect["h"] / max(session.doc_height, 1)
            target = (float(display_x), float(display_y))
            if session.mapper.calibrated:
                predicted = session.mapper.model.predict(
                    representative.reshape(1, -1)
                )[0]
                pre_errors.append(
                    math.hypot(predicted[0] - target[0], predicted[1] - target[1])
                )
            new_samples.append(MappingSample(representative, target))

        session.grid_samples.clear()
        session.grid_targets.clear()

        if len(new_samples) < GRID_MIN_TARGETS:
            return jsonify({
                "error": (
                    f"only {len(new_samples)} usable digit(s) with enough samples - "
                    f"need at least {GRID_MIN_TARGETS} (face visible, looking at the digits)"
                ),
                "fitted": False,
            }), 400

        merged = list(session.mapper.samples) + new_samples
        fitted = session.mapper.fit(merged)

        post_errors: List[float] = []
        if fitted:
            for sample in new_samples:
                predicted = session.mapper.model.predict(
                    np.asarray(sample.features, dtype=np.float64).reshape(1, -1)
                )[0]
                post_errors.append(
                    math.hypot(predicted[0] - sample.target[0],
                               predicted[1] - sample.target[1])
                )

        session.grid_stats = {
            "targets": len(new_samples),
            "used_targets": len(new_samples),
            "samples": total_samples,
            "pre_error_px": round(float(np.mean(pre_errors)), 1) if pre_errors else None,
            "post_error_px": round(float(np.mean(post_errors)), 1) if post_errors else None,
            "fitted": bool(fitted),
            "interval_s": GRID_INTERVAL_S,
        }
        if not fitted:
            session.grid_stats["error"] = "mapping fit failed (not enough feature spread)"
        LOGGER.info("grid mapping: %s", session.grid_stats)
    return jsonify(session.grid_stats)


@app.route("/api/finish", methods=["POST"])
def finish() -> Response:
    payload = request.get_json(silent=True) or {}
    apply_decay = bool(payload.get("decay", False))
    with STATE_LOCK:
        if _session is None:
            return jsonify({"error": "no active session"}), 409
        if _session.mode == "calibrating":
            return jsonify({"error": "calibration is not finished yet"}), 409
        if _session.frames == 0:
            return jsonify({"error": "no frames were processed"}), 409
        try:
            report = build_report(_session, apply_decay)
        except Exception as exc:
            LOGGER.exception("report generation failed")
            return jsonify({"error": f"report failed: {exc}"}), 500
    return jsonify(report)


@app.route("/api/reset", methods=["POST"])
def reset() -> Response:
    global _session
    with STATE_LOCK:
        if _session is not None:
            _session.dispose()
            _session = None
    return jsonify({"mode": "idle"})


@app.route("/api/health", methods=["GET"])
def health() -> Response:
    with STATE_LOCK:
        state = {
            "tracker_loaded": _tracker is not None,
            "session": None if _session is None else _session.mode,
        }
    return jsonify(state)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live webcam gaze heatmap web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"Open http://{args.host}:{args.port} in your browser "
          f"(webcam permission required; localhost is a secure context)")
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except OSError as exc:
        print(f"Could not start the server on {args.host}:{args.port} ({exc}).\n"
              f"Another program is probably using that port — try:\n"
              f"    python web_app.py --port 5001")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
