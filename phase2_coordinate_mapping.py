"""Phase 2: calibration and gaze mapping (feature vector -> screen pixels).

Responsibilities:
- 5 / 9 / 13 point calibration with stabilization, multi-sample collection,
  low-confidence rejection and MAD-based outlier rejection
- mapping models: affine (baseline), polynomial, optional small MLP
- outlier rejection + configurable EMA smoothing
- screen-space validation mode with quantitative error metrics
- evaluation report export (JSON + CSV)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gaze_core import (
    TARGET_LAYOUTS,
    ConfidenceBreakdown,
    EyeTracker,
    GazeConfig,
    GazeResult,
    HeadPose,
    MappingSample,
    ScreenCalibrator,
    ScreenMapper,
    build_mapping_model,
)


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: gaze -> screen coordinate mapping")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--screen", default="1920x1080", help="virtual screen size, e.g. 1920x1080")
    parser.add_argument("--gain", type=float, default=3.0, help="raw mapping gain around screen centre")
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor in (0, 1]; 1 disables smoothing",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="deprecated alias of --ema-alpha",
    )
    parser.add_argument("--preview", type=int, default=960, help="preview width of the virtual screen")
    parser.add_argument("--pupil", choices=("hybrid", "iris", "image"), default="hybrid")
    parser.add_argument("--calibration", default="calibration.npz", help="calibration file")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--trail", type=int, default=120, help="gaze trail length in frames")

    # research-oriented calibration controls
    parser.add_argument(
        "--points",
        type=int,
        default=9,
        choices=sorted(TARGET_LAYOUTS),
        help="calibration point layout: 5 (lightweight), 9 (default) or 13",
    )
    parser.add_argument(
        "--samples-per-point",
        type=int,
        default=20,
        help="accepted samples collected per calibration target",
    )
    parser.add_argument(
        "--stabilize-frames",
        type=int,
        default=12,
        help="dwell frames required before a target starts collecting samples",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="samples below this combined confidence are rejected",
    )
    parser.add_argument(
        "--mapping",
        choices=("affine", "polynomial", "mlp"),
        default="affine",
        help="gaze mapping model",
    )

    # validation / evaluation
    parser.add_argument(
        "--validate",
        action="store_true",
        help="start in validation mode: look at the displayed targets and collect error metrics",
    )
    parser.add_argument(
        "--validation-points",
        type=int,
        default=None,
        choices=sorted(TARGET_LAYOUTS),
        help="target layout used for validation (defaults to --points)",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=15,
        help="predictions collected per validation target",
    )
    parser.add_argument(
        "--report-prefix",
        default="evaluation_report",
        help="prefix for the JSON/CSV evaluation report",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="after validation, refit affine/polynomial/mlp on the calibration data and compare",
    )
    parser.add_argument("--screen-width-cm", type=float, default=None, help="physical display width in cm")
    parser.add_argument("--camera-distance-cm", type=float, default=None, help="camera to screen distance in cm")
    return parser.parse_args()


def parse_screen(value: str) -> Tuple[int, int]:
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError("screen must look like 1920x1080") from None


# ---------------------------------------------------------------------------
# validation + evaluation metrics
# ---------------------------------------------------------------------------
def error_statistics(errors: Sequence[float]) -> Dict[str, float]:
    """Mean / median / std / 95th percentile / maximum error in pixels."""
    data = np.asarray(errors, dtype=np.float64).ravel()
    if data.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
        "count": int(data.size),
    }


def angular_error_degrees(
    error_px: float,
    screen_width_px: int,
    screen_width_cm: Optional[float],
    camera_distance_cm: Optional[float],
) -> Optional[float]:
    """Angular error in degrees when the physical display geometry is known.

    Returns None when the geometry is unavailable - angular error is never
    reported from guessed numbers.
    """
    if not screen_width_cm or not camera_distance_cm or screen_width_px <= 0:
        return None
    cm_per_px = screen_width_cm / float(screen_width_px)
    angle = math.degrees(math.atan(error_px * cm_per_px / camera_distance_cm))
    return float(angle)


def head_pose_condition(yaw: Optional[float], pitch: Optional[float]) -> str:
    if yaw is None or pitch is None:
        return "unknown"
    ay, ap = abs(yaw), abs(pitch)
    if ay <= 8.0 and ap <= 8.0:
        return "neutral"
    if ay >= 20.0 or ap >= 20.0:
        return "moderate"
    return "slight"


@dataclass
class ValidationRecord:
    target_index: int
    target_x: float
    target_y: float
    predicted_x: float
    predicted_y: float
    confidence: float
    features: Optional[np.ndarray] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    @property
    def error_px(self) -> float:
        return math.hypot(self.predicted_x - self.target_x, self.predicted_y - self.target_y)


@dataclass
class ValidationRun:
    """Guided screen-space validation: known targets, measured prediction error."""

    width: int
    height: int
    targets: Sequence[Tuple[float, float]]
    samples_per_target: int = 15
    stabilize_frames: int = 15
    min_confidence: float = 0.35
    active: bool = False
    index: int = 0
    hits: int = 0
    stabilized: bool = False
    collected: int = 0
    attempts: int = 0
    records: List[ValidationRecord] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.index >= len(self.targets)

    @property
    def target_px(self) -> Optional[Tuple[float, float]]:
        if not self.active or self.done:
            return None
        tx, ty = self.targets[self.index]
        return tx * self.width, ty * self.height

    @property
    def progress(self) -> float:
        if not self.stabilized:
            return 0.3 * min(self.hits / max(self.stabilize_frames, 1), 1.0)
        return 0.3 + 0.7 * min(self.collected / max(self.samples_per_target, 1), 1.0)

    def start(self) -> None:
        self.active = True
        self.index = 0
        self.hits = 0
        self.stabilized = False
        self.collected = 0
        self.attempts = 0
        self.records = []

    def cancel(self) -> None:
        self.active = False

    def step(
        self,
        position: Optional[Tuple[float, float]],
        confidence: float,
        head_pose: Optional[HeadPose] = None,
        features: Optional[np.ndarray] = None,
    ) -> str:
        """Advance one frame and record a prediction when the target is stable."""
        if not self.active:
            return "idle"
        self.attempts += 1
        if position is None or confidence < self.min_confidence:
            self.hits = 0
            self.stabilized = False
            return "invalid"
        if not self.stabilized:
            self.hits += 1
            if self.hits < self.stabilize_frames:
                return "stabilizing"
            self.stabilized = True
        target = self.targets[self.index]
        self.records.append(
            ValidationRecord(
                target_index=self.index,
                target_x=target[0] * self.width,
                target_y=target[1] * self.height,
                predicted_x=position[0],
                predicted_y=position[1],
                confidence=float(confidence),
                features=None if features is None else np.asarray(features, dtype=np.float64).copy(),
                yaw=head_pose.yaw if head_pose and head_pose.valid else None,
                pitch=head_pose.pitch if head_pose and head_pose.valid else None,
                roll=head_pose.roll if head_pose and head_pose.valid else None,
            )
        )
        self.collected += 1
        if self.collected >= self.samples_per_target:
            self.index += 1
            self.hits = 0
            self.stabilized = False
            self.collected = 0
            if self.done:
                self.active = False
                return "done"
            return "target_complete"
        return "collecting"

    def feature_target_pairs(
        self,
    ) -> Optional[Tuple[List[np.ndarray], List[Tuple[float, float]]]]:
        """Feature vectors and known coordinates of every recorded sample."""
        if not self.records or any(record.features is None for record in self.records):
            return None
        features = [record.features for record in self.records if record.features is not None]
        targets = [(record.target_x, record.target_y) for record in self.records]
        return features, targets


def summarize_validation(
    records: Sequence[ValidationRecord],
    attempts: int = 0,
) -> Dict[str, object]:
    """Aggregate a validation run into the research evaluation structure."""
    errors = [record.error_px for record in records]
    summary = error_statistics(errors)
    confidences = [record.confidence for record in records]
    per_target: Dict[str, Dict[str, float]] = {}
    grouped: Dict[int, List[float]] = {}
    for record in records:
        grouped.setdefault(record.target_index, []).append(record.error_px)
    for index in sorted(grouped):
        stats = error_statistics(grouped[index])
        per_target[str(index)] = {k: v for k, v in stats.items() if k != "count"}
        per_target[str(index)]["count"] = stats["count"]
    conditions: Dict[str, Dict[str, float]] = {}
    condition_errors: Dict[str, List[float]] = {}
    for record in records:
        condition = head_pose_condition(record.yaw, record.pitch)
        condition_errors.setdefault(condition, []).append(record.error_px)
    for condition, values in sorted(condition_errors.items()):
        conditions[condition] = error_statistics(values)
    return {
        "overall": summary,
        "per_target": per_target,
        "by_head_pose": conditions,
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "valid_sample_rate": float(len(records) / attempts) if attempts else 0.0,
        "recorded_samples": len(records),
        "attempts": attempts,
    }


def evaluate_models_on_features(
    calibration_samples: Sequence[MappingSample],
    validation_features: Sequence[np.ndarray],
    validation_targets: Sequence[Tuple[float, float]],
    model_names: Sequence[str],
    config: GazeConfig,
    screen: Tuple[int, int],
) -> Dict[str, Dict[str, float]]:
    """Fit each model on calibration samples and report validation error statistics."""
    if not calibration_samples or not validation_features:
        return {}
    x = np.vstack([np.asarray(s.features, dtype=np.float64) for s in calibration_samples])
    y = np.array([s.target for s in calibration_samples], dtype=np.float64)
    vx = np.vstack([np.asarray(f, dtype=np.float64) for f in validation_features])
    vy = np.array(validation_targets, dtype=np.float64)
    results: Dict[str, Dict[str, float]] = {}
    for name in model_names:
        try:
            model = build_mapping_model(name, config)
        except ValueError:
            continue
        if hasattr(model, "screen"):
            model.screen = screen  # type: ignore[attr-defined]
        if not model.fit(x, y):
            results[name] = {"fit": 0.0}
            continue
        predicted = model.predict(vx)
        errors = np.linalg.norm(predicted - vy, axis=1)
        stats = error_statistics([float(e) for e in errors])
        stats["fit"] = 1.0
        results[name] = stats
    return results


def build_report(
    *,
    model: str,
    calibration_method: str,
    calibration_points: int,
    samples_per_point: int,
    validation: Dict[str, object],
    valid_sample_rate: Optional[float] = None,
    average_confidence: Optional[float] = None,
    angular: Optional[Dict[str, float]] = None,
    comparisons: Optional[Dict[str, Dict[str, float]]] = None,
    config: Optional[GazeConfig] = None,
) -> Dict[str, object]:
    overall = dict(validation.get("overall", {}))  # type: ignore[arg-type]
    report: Dict[str, object] = {
        "model": model,
        "calibration_method": calibration_method,
        "number_of_calibration_points": calibration_points,
        "samples_per_point": samples_per_point,
        "mean_error_px": overall.get("mean"),
        "median_error_px": overall.get("median"),
        "standard_deviation_px": overall.get("std"),
        "p95_error_px": overall.get("p95"),
        "max_error_px": overall.get("max"),
        "valid_sample_rate": valid_sample_rate
        if valid_sample_rate is not None
        else validation.get("valid_sample_rate"),
        "average_confidence": average_confidence
        if average_confidence is not None
        else validation.get("average_confidence"),
        "recorded_samples": validation.get("recorded_samples"),
        "per_target_error_px": validation.get("per_target"),
        "error_by_head_pose_px": validation.get("by_head_pose"),
        "angular_error_degrees": angular,
        "model_comparison": comparisons,
        "config": as_config_dict(config),
    }
    return report


def as_config_dict(config: Optional[GazeConfig]) -> Dict[str, object]:
    if config is None:
        return {}
    data = dict(config.__dict__)
    data.pop("confidence_weights", None)
    return data


def save_report(report: Dict[str, object], prefix: str) -> Tuple[str, str]:
    json_path = f"{prefix}.json"
    csv_path = f"{prefix}.csv"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)
    columns = [
        "model",
        "calibration_method",
        "number_of_calibration_points",
        "samples_per_point",
        "mean_error_px",
        "median_error_px",
        "standard_deviation_px",
        "p95_error_px",
        "max_error_px",
        "valid_sample_rate",
        "average_confidence",
        "recorded_samples",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({key: report.get(key) for key in columns})
    return json_path, csv_path


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_validation_pairs(run: ValidationRun, path: str) -> str:
    """Store validation targets/predictions so models can be compared offline."""
    np.savez(
        path,
        targets=np.array([[r.target_x, r.target_y] for r in run.records], dtype=np.float32),
        predictions=np.array([[r.predicted_x, r.predicted_y] for r in run.records], dtype=np.float32),
        confidences=np.array([r.confidence for r in run.records], dtype=np.float32),
        target_index=np.array([r.target_index for r in run.records], dtype=np.int32),
        yaw=np.array([r.yaw if r.yaw is not None else np.nan for r in run.records], dtype=np.float32),
        pitch=np.array([r.pitch if r.pitch is not None else np.nan for r in run.records], dtype=np.float32),
    )
    return path


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def draw_grid(canvas: np.ndarray, step: int = 160) -> None:
    for x in range(0, canvas.shape[1], step):
        cv2.line(canvas, (x, 0), (x, canvas.shape[0]), (44, 40, 38), 1)
    for y in range(0, canvas.shape[0], step):
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (44, 40, 38), 1)
    cv2.line(canvas, (canvas.shape[1] // 2, 0), (canvas.shape[1] // 2, canvas.shape[0]), (60, 56, 52), 1)
    cv2.line(canvas, (0, canvas.shape[0] // 2), (canvas.shape[1], canvas.shape[0] // 2), (60, 56, 52), 1)


def draw_trail(canvas: np.ndarray, trail, scale: float) -> None:
    points = list(trail)
    if len(points) < 2:
        return
    for index in range(1, len(points)):
        ratio = index / len(points)
        start = (int(points[index - 1][0] * scale), int(points[index - 1][1] * scale))
        end = (int(points[index][0] * scale), int(points[index][1] * scale))
        color = (int(60 + 180 * ratio), int(220 - 120 * ratio), 40)
        cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)


def draw_cursor(canvas: np.ndarray, x: float, y: float, scale: float, confidence: float) -> None:
    cx, cy = int(x * scale), int(y * scale)
    cv2.line(canvas, (cx, 0), (cx, canvas.shape[0]), (0, 90, 140), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, cy), (canvas.shape[1], cy), (0, 90, 140), 1, cv2.LINE_AA)
    radius = 12 + int((1.0 - min(max(confidence, 0.0), 1.0)) * 14)
    cv2.circle(canvas, (cx, cy), radius + 6, (0, 120, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 3, (0, 0, 0), -1, cv2.LINE_AA)


def draw_target(
    canvas: np.ndarray,
    center: Tuple[int, int],
    progress: float,
    label: str,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> None:
    cv2.circle(canvas, center, 34, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(canvas, center, 12, color, -1, cv2.LINE_AA)
    arc = int(360 * min(max(progress, 0.0), 1.0))
    if arc > 0:
        cv2.ellipse(canvas, center, (34, 34), -90, 0, arc, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.putText(
        canvas, label, (center[0] - 70, center[1] + 62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )


def draw_calibration(canvas: np.ndarray, calibrator: ScreenCalibrator, scale: float) -> None:
    for tx, ty in calibrator.captured:
        center = (int(tx * calibrator.width * scale), int(ty * calibrator.height * scale))
        cv2.circle(canvas, center, 16, (60, 220, 60), 3, cv2.LINE_AA)
    target = calibrator.target_px
    if target is None:
        return
    center = (int(target[0] * scale), int(target[1] * scale))
    draw_target(canvas, center, calibrator.progress, f"look here  {calibrator.index + 1}/{len(calibrator.targets)}")
    if calibrator.stabilized:
        cv2.putText(
            canvas,
            f"sampling {calibrator.collected}/{calibrator.samples_per_point}",
            (center[0] - 70, center[1] + 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 255, 160),
            1,
            cv2.LINE_AA,
        )


def draw_validation(canvas: np.ndarray, run: ValidationRun, scale: float) -> None:
    for record in run.records:
        center = (int(record.target_x * scale), int(record.target_y * scale))
        cv2.circle(canvas, center, 6, (60, 220, 60), -1, cv2.LINE_AA)
    target = run.target_px
    if target is None:
        return
    center = (int(target[0] * scale), int(target[1] * scale))
    draw_target(
        canvas,
        center,
        run.progress,
        f"validation {run.index + 1}/{len(run.targets)}",
        color=(250, 200, 60),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_config(args: argparse.Namespace, screen: Tuple[int, int]) -> GazeConfig:
    config = GazeConfig(
        screen_width=screen[0],
        screen_height=screen[1],
        calibration_points=args.points,
        calibration_samples_per_point=args.samples_per_point,
        calibration_stabilize_frames=args.stabilize_frames,
        mapping_model=args.mapping,
        ema_alpha=args.ema_alpha if args.smoothing is None else args.smoothing,
        confidence_threshold=args.confidence_threshold,
    )
    config.validate()
    return config


def handle_calibration_done(calibrator: ScreenCalibrator, mapper: ScreenMapper) -> str:
    samples = calibrator.finalize()
    if mapper.fit(samples):
        print(
            f"calibration saved to {mapper.path} "
            f"({len(samples)} samples, {calibrator.n_points} points, "
            f"{calibrator.rejected} outliers rejected)"
        )
        return "calibration saved"
    return "calibration failed (not enough feature spread)"


def main() -> int:
    args = parse_args()
    screen_w, screen_h = parse_screen(args.screen)
    config = build_config(args, (screen_w, screen_h))
    try:
        tracker = EyeTracker(pupil_mode=args.pupil, config=config)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    mapper = ScreenMapper(
        screen_w,
        screen_h,
        gain=args.gain,
        smoothing=config.ema_alpha,
        path=args.calibration,
        config=config,
        model_name=args.mapping,
    )
    calibrator = ScreenCalibrator(screen_w, screen_h, config=config)
    validation = ValidationRun(
        screen_w,
        screen_h,
        TARGET_LAYOUTS[args.validation_points or args.points],
        samples_per_target=args.validation_samples,
        stabilize_frames=args.stabilize_frames,
        min_confidence=config.calibration_min_confidence,
    )
    if args.validate:
        validation.start()

    trail = deque(maxlen=max(args.trail, 4))
    scale = args.preview / screen_w
    preview_size = (args.preview, int(round(screen_h * scale)))

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        tracker.close()
        sys.exit(f"Could not open webcam index {args.camera}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(
        "keys: [c] calibrate  [SPACE] capture point  [v] validation  "
        "[m] cycle mapping model  [r] reset  [n] re-centre  [q/ESC] quit"
    )
    print(
        f"mapping {screen_w}x{screen_h} | model={mapper.model_name} | "
        f"loaded calibration: {mapper.calibrated} | ema_alpha={mapper.smoothing}"
    )
    fps = 0.0
    status_text = "waiting for gaze"
    report_paths: Optional[Tuple[str, str]] = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Frame grab failed, stopping.", file=sys.stderr)
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            loop_start = time.perf_counter()
            result = tracker.process(frame)
            gaze = result.gaze
            features = result.features
            confidence = result.confidence

            # -- calibration ------------------------------------------------
            if calibrator.active:
                state = calibrator.update(features, confidence)
                if state == "invalid":
                    status_text = "low confidence - hold still"
                elif state == "stabilizing":
                    status_text = "stabilizing..."
                elif state == "collecting":
                    status_text = (
                        f"point {calibrator.index + 1}/{len(calibrator.targets)} "
                        f"samples {calibrator.collected}/{calibrator.samples_per_point}"
                    )
                elif state == "captured":
                    status_text = f"captured point {calibrator.index}/{len(calibrator.targets)}"
                elif state == "done":
                    status_text = handle_calibration_done(calibrator, mapper)

            # -- validation -------------------------------------------------
            if validation.active:
                position = mapper.map(gaze, features, confidence)
                state = validation.step(position, confidence, result.head_pose, features)
                if state == "done":
                    summary = summarize_validation(validation.records, validation.attempts)
                    comparisons = None
                    if args.compare_models and mapper.samples:
                        pairs = validation.feature_target_pairs()
                        if pairs is not None:
                            comparisons = evaluate_models_on_features(
                                mapper.samples,
                                pairs[0],
                                pairs[1],
                                ("affine", "polynomial", "mlp"),
                                config,
                                (screen_w, screen_h),
                            )
                    angular = None
                    if validation.records:
                        overall = summary["overall"]  # type: ignore[index]
                        sample = overall.get("mean")
                        if sample is not None and not math.isnan(sample):
                            value = angular_error_degrees(
                                float(sample),
                                screen_w,
                                args.screen_width_cm,
                                args.camera_distance_cm,
                            )
                            if value is not None:
                                angular = {"mean": value}
                    report = build_report(
                        model=mapper.model_name,
                        calibration_method=f"{calibrator.n_points}-point multi-sample",
                        calibration_points=calibrator.n_points,
                        samples_per_point=calibrator.samples_per_point,
                        validation=summary,  # type: ignore[arg-type]
                        valid_sample_rate=mapper.valid_sample_rate,
                        average_confidence=float(summary["average_confidence"]),  # type: ignore[arg-type]
                        angular=angular,
                        comparisons=comparisons,
                        config=config,
                    )
                    report_paths = save_report(report, args.report_prefix)
                    pairs_path = save_validation_pairs(
                        validation, f"{args.report_prefix}_pairs.npz"
                    )
                    status_text = f"validation done: mean {summary['overall']['mean']:.1f} px"  # type: ignore[index]
                    print(f"validation report saved: {report_paths[0]}, {report_paths[1]}")
                    print(f"validation pairs saved: {pairs_path}")
                    if comparisons:
                        for name, stats in comparisons.items():
                            if stats.get("fit"):
                                print(
                                    f"  {name:10s} mean={stats['mean']:.1f} px "
                                    f"median={stats['median']:.1f} px p95={stats['p95']:.1f} px"
                                )
                elif state == "stabilizing":
                    status_text = "validation: hold still"
                elif state == "collecting":
                    status_text = (
                        f"validation {validation.index + 1}/{len(validation.targets)} "
                        f"({validation.collected}/{validation.samples_per_target})"
                    )
            else:
                position = mapper.map(gaze, features, confidence)

            if position is not None:
                trail.append(position)

            canvas = np.full((preview_size[1], preview_size[0], 3), (30, 26, 24), np.uint8)
            draw_grid(canvas)
            draw_trail(canvas, trail, scale)
            if position is not None:
                draw_cursor(canvas, position[0], position[1], scale, confidence)
            draw_calibration(canvas, calibrator, scale)
            draw_validation(canvas, validation, scale)

            fps = 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - loop_start, 1e-6))
            if position is None:
                cv2.putText(
                    canvas,
                    "NO VALID GAZE",
                    (preview_size[0] // 2 - 110, preview_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 140, 255),
                    3,
                    cv2.LINE_AA,
                )
                xy_text = "X=---- Y=----"
            else:
                xy_text = f"X={position[0]:7.1f}  Y={position[1]:7.1f}"
            header = f"{mapper.mapping_name}   {xy_text}   conf={confidence:.2f}  fps={fps:4.1f}"
            cv2.putText(canvas, header, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

            second = _confidence_line(result)
            cv2.putText(canvas, second, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)

            footer = (
                f"{status_text}   |   [c] calibrate [space] capture [v] validate "
                "[m] model [r] reset [n] re-centre [q] quit"
            )
            cv2.putText(
                canvas, footer, (14, preview_size[1] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
            )

            camera_view = tracker.draw_overlay(frame, result, show_hud=False)
            camera_scale = 420 / max(camera_view.shape[1], 1)
            camera_view = cv2.resize(camera_view, (420, int(camera_view.shape[0] * camera_scale)))
            cv2.imshow("webcam", camera_view)
            cv2.imshow("attention-track phase 2 - screen mapping", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                if validation.active:
                    validation.cancel()
                calibrator.start()
                status_text = f"calibration started ({calibrator.n_points} points)"
                trail.clear()
            elif key == ord("v"):
                if calibrator.active:
                    calibrator.cancel()
                validation.start()
                status_text = "validation started"
                trail.clear()
            elif key == ord(" ") and calibrator.active:
                state = calibrator.record(features, confidence)
                if state == "done":
                    status_text = handle_calibration_done(calibrator, mapper)
                else:
                    status_text = "captured manually"
            elif key == ord("m"):
                cycle = ("affine", "polynomial", "mlp")
                next_model = cycle[(cycle.index(mapper.model_name) + 1) % len(cycle)]
                if mapper.samples and mapper.fit(mapper.samples, model_name=next_model):
                    status_text = f"refitted existing calibration as {next_model}"
                    print(f"mapping model -> {next_model}")
                else:
                    mapper.model_name = next_model
                    config.mapping_model = next_model
                    status_text = f"mapping model set to {next_model} (recalibrate to fit)"
                    print(f"mapping model -> {next_model}")
            elif key == ord("r"):
                mapper.reset()
                mapper.recenter()
                calibrator.cancel()
                validation.cancel()
                trail.clear()
                status_text = "calibration reset (raw gain mode)"
                print("calibration reset")
            elif key == ord("n"):
                mapper.recenter()
                trail.clear()
                status_text = "gaze re-centred"

    finally:
        if validation.active and validation.records:
            summary = summarize_validation(validation.records, validation.attempts)
            report = build_report(
                model=mapper.model_name,
                calibration_method=f"{calibrator.n_points}-point multi-sample",
                calibration_points=calibrator.n_points,
                samples_per_point=calibrator.samples_per_point,
                validation=summary,  # type: ignore[arg-type]
                valid_sample_rate=mapper.valid_sample_rate,
                average_confidence=float(summary["average_confidence"]),  # type: ignore[arg-type]
                config=config,
            )
            save_report(report, args.report_prefix)
            save_validation_pairs(validation, f"{args.report_prefix}_pairs.npz")
            print(f"partial validation report saved ({len(validation.records)} samples)")
        elif report_paths:
            print(f"report: {report_paths[0]}, {report_paths[1]}")
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


def _confidence_line(result: GazeResult) -> str:
    conf: Optional[ConfidenceBreakdown] = result.confidences
    pose = result.head_pose
    pose_text = "pose=--"
    if pose is not None and pose.valid:
        pose_text = f"pose y{pose.yaw:+.0f} p{pose.pitch:+.0f} r{pose.roll:+.0f}"
    if conf is None:
        return f"{pose_text}  confidence breakdown unavailable"
    return (
        f"{pose_text}  pup={conf.pupil:.2f} lm={conf.landmark:.2f} "
        f"hp={conf.head_pose:.2f} gz={conf.gaze:.2f} tmp={conf.temporal:.2f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
