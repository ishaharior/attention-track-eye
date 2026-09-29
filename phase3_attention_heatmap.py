"""Phase 3: fixation-aware gaze-density heatmap.

Responsibilities:
- fixation / saccade detection (I-DT dispersion threshold, optional I-VT velocity)
- confidence- and fixation-weighted accumulation
- time-based decay (exp(-dt / tau)), frame-rate independent
- Gaussian blur with sigma specified in screen pixels
- rendering and export (points, grid, fixation/saccade events)

The baseline output is a gaze-density heatmap; it becomes a visual-attention
estimation component only because fixation duration and confidence are folded in.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gaze_core import (
    TARGET_LAYOUTS,
    AttentionHeatmap,
    EyeTracker,
    GazeConfig,
    ScreenCalibrator,
    ScreenMapper,
)


# ---------------------------------------------------------------------------
# fixation / saccade detection (I-DT / I-VT)
# ---------------------------------------------------------------------------
@dataclass
class FixationEvent:
    start_ms: float
    end_ms: float
    duration_ms: float
    centroid: Tuple[float, float]
    confidence: float
    dispersion: float
    samples: int


@dataclass
class SaccadeEvent:
    start_ms: float
    end_ms: float
    duration_ms: float
    amplitude_px: float
    peak_velocity_px_s: float
    confidence: float


@dataclass
class _Sample:
    t_ms: float
    x: float
    y: float
    confidence: float


class EventDetector:
    """Classify gaze samples as fixation / saccade / invalid.

    I-DT (Dispersion Threshold Identification) keeps a sliding window of samples;
    while the window dispersion stays below `dispersion_px` the samples belong to
    one fixation.  A fixation is only reported once it lasts `min_duration_ms`.
    Samples whose dispersion breaks early are classified from their velocity with
    I-VT logic (velocity >= `velocity_threshold` -> saccade, below -> invalid).
    """

    def __init__(
        self,
        method: str = "idt",
        dispersion_px: float = 45.0,
        min_duration_ms: float = 100.0,
        velocity_threshold: float = 1.0,
        max_window_ms: float = 2000.0,
    ) -> None:
        if method not in ("idt", "ivt"):
            raise ValueError("method must be 'idt' or 'ivt'")
        self.method = method
        self.dispersion_px = float(dispersion_px)
        self.min_duration_ms = float(min_duration_ms)
        self.velocity_threshold = float(velocity_threshold)
        self.max_window_ms = float(max_window_ms)
        self.fixations: List[FixationEvent] = []
        self.saccades: List[SaccadeEvent] = []
        self._pending: List[_Sample] = []
        self._carry: List[_Sample] = []
        self._run_kind: Optional[str] = None
        self._confirmed = False
        self._previous: Optional[_Sample] = None

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _dispersion(samples: Sequence[_Sample]) -> float:
        if not samples:
            return 0.0
        xs = [s.x for s in samples]
        ys = [s.y for s in samples]
        return float(max(max(xs) - min(xs), max(ys) - min(ys)))

    @staticmethod
    def _duration(samples: Sequence[_Sample]) -> float:
        if len(samples) < 2:
            return 0.0
        return float(samples[-1].t_ms - samples[0].t_ms)

    @property
    def current_duration_ms(self) -> float:
        return self._duration(self._pending)

    @property
    def current_dispersion(self) -> float:
        return self._dispersion(self._pending)

    @property
    def current_confidence(self) -> float:
        if not self._pending:
            return 0.0
        return float(np.mean([s.confidence for s in self._pending]))

    def velocity(self, sample: _Sample) -> float:
        if self._previous is None:
            return 0.0
        dt = max(sample.t_ms - self._previous.t_ms, 1e-3)
        distance = math.hypot(sample.x - self._previous.x, sample.y - self._previous.y)
        return distance / dt  # px per ms

    # -- emission ---------------------------------------------------------
    def _emit_fixation(self, samples: Sequence[_Sample]) -> None:
        if len(samples) < 2:
            return
        centroid = (
            float(np.mean([s.x for s in samples])),
            float(np.mean([s.y for s in samples])),
        )
        self.fixations.append(
            FixationEvent(
                start_ms=samples[0].t_ms,
                end_ms=samples[-1].t_ms,
                duration_ms=self._duration(samples),
                centroid=centroid,
                confidence=float(np.mean([s.confidence for s in samples])),
                dispersion=self._dispersion(samples),
                samples=len(samples),
            )
        )

    def _emit_saccade(self, samples: Sequence[_Sample]) -> None:
        if len(samples) < 2:
            return
        peak = 0.0
        for previous, current in zip(samples, samples[1:]):
            dt = max(current.t_ms - previous.t_ms, 1e-3)
            distance = math.hypot(current.x - previous.x, current.y - previous.y)
            peak = max(peak, distance / dt)
        self.saccades.append(
            SaccadeEvent(
                start_ms=samples[0].t_ms,
                end_ms=samples[-1].t_ms,
                duration_ms=self._duration(samples),
                amplitude_px=float(
                    math.hypot(samples[-1].x - samples[0].x, samples[-1].y - samples[0].y)
                ),
                peak_velocity_px_s=float(peak * 1000.0),
                confidence=float(np.mean([s.confidence for s in samples])),
            )
        )

    def _max_velocity(self, samples: Sequence[_Sample]) -> float:
        peak = 0.0
        for previous, current in zip(samples, samples[1:]):
            dt = max(current.t_ms - previous.t_ms, 1e-3)
            peak = max(peak, math.hypot(current.x - previous.x, current.y - previous.y) / dt)
        return peak

    def _close_window(self, breaking_sample: Optional[_Sample] = None) -> None:
        """Close the current sample window.

        A confirmed window becomes a fixation event (merged with any trimmed
        head of the same fixation).  Otherwise the window is movement: it becomes
        a saccade when its velocity is high enough to deserve that label.
        """
        samples = list(self._pending) if breaking_sample is None else list(self._pending[:-1])
        if self._confirmed:
            merged = list(self._carry) + samples
            if len(merged) >= 2 and self._duration(merged) >= self.min_duration_ms:
                self._emit_fixation(merged)
            # the movement that ends a fixation is the saccade between two fixations
            if breaking_sample is not None and samples:
                previous = samples[-1]
                dt = breaking_sample.t_ms - previous.t_ms
                distance = math.hypot(
                    breaking_sample.x - previous.x, breaking_sample.y - previous.y
                )
                if dt > 0.0 and distance / dt >= self.velocity_threshold:
                    self._emit_saccade([previous, breaking_sample])
        elif len(samples) >= 2 and self._max_velocity(samples) >= self.velocity_threshold:
            self._emit_saccade(samples)
        self._carry = []
        self._pending = [] if breaking_sample is None else [breaking_sample]
        self._confirmed = False
        self._run_kind = None

    # -- public API -------------------------------------------------------
    def update(
        self,
        t_ms: float,
        x: float,
        y: float,
        confidence: float,
        min_confidence: float = 0.0,
        valid: bool = True,
    ) -> str:
        """Classify one gaze sample ('fixation' | 'saccade' | 'invalid')."""
        if not valid or confidence < min_confidence or not math.isfinite(x) or not math.isfinite(y):
            self._close_window()
            self._previous = None
            return "invalid"

        sample = _Sample(float(t_ms), float(x), float(y), float(confidence))

        if self.method == "ivt":
            velocity = self.velocity(sample)
            kind = "stable" if velocity < self.velocity_threshold else "move"
            if self._run_kind is not None and self._run_kind != kind:
                self._close_window()
                self._pending = []
            self._run_kind = kind
            self._pending.append(sample)
            self._trim_window()
            if kind == "stable" and self._duration(self._pending) >= self.min_duration_ms:
                self._confirmed = True
            self._previous = sample
            return "fixation" if kind == "stable" else "saccade"

        # --- I-DT (default): dispersion threshold identification ----------
        self._pending.append(sample)
        self._trim_window()
        dispersion = self._dispersion(self._pending)
        duration = self._duration(self._pending)

        if dispersion <= self.dispersion_px:
            if duration >= self.min_duration_ms:
                self._confirmed = True
            self._previous = sample
            return "fixation"

        # dispersion exceeded: close the window and restart from this sample
        velocity = self.velocity(sample)
        self._close_window(breaking_sample=sample)
        self._previous = sample
        return "saccade" if velocity >= self.velocity_threshold else "invalid"

    def _trim_window(self) -> None:
        """Keep the live window bounded without losing confirmed fixation samples."""
        while len(self._pending) > 1 and self._duration(self._pending) > self.max_window_ms:
            popped = self._pending.pop(0)
            if self._confirmed:
                self._carry.append(popped)

    def finalize(self) -> None:
        """Flush any fixation still open at the end of a session."""
        self._close_window()

    def summary(self, attempted_samples: int = 0) -> Dict[str, float]:
        durations = [f.duration_ms for f in self.fixations]
        return {
            "fixation_count": len(self.fixations),
            "saccade_count": len(self.saccades),
            "average_fixation_duration_ms": float(np.mean(durations)) if durations else 0.0,
            "median_fixation_duration_ms": float(np.median(durations)) if durations else 0.0,
            "fixation_detection_rate": (
                float(
                    sum(f.samples for f in self.fixations) / attempted_samples
                )
                if attempted_samples
                else 0.0
            ),
            "attempted_samples": attempted_samples,
        }

    def reset(self) -> None:
        self.fixations = []
        self.saccades = []
        self._pending = []
        self._carry = []
        self._run_kind = None
        self._confirmed = False
        self._previous = None

    def events_payload(self) -> Dict[str, List[Dict[str, object]]]:
        return {
            "fixations": [asdict(f) for f in self.fixations],
            "saccades": [asdict(s) for s in self.saccades],
        }


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: fixation-aware gaze-density heatmap")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--screen", default="1920x1080", help="virtual screen size")
    parser.add_argument("--gain", type=float, default=3.0, help="raw mapping gain")
    parser.add_argument("--ema-alpha", type=float, default=0.3, help="EMA smoothing factor")
    parser.add_argument("--smoothing", type=float, default=None, help="deprecated alias of --ema-alpha")
    parser.add_argument("--calibration", default="calibration.npz", help="calibration file")
    parser.add_argument("--mapping", choices=("affine", "polynomial", "mlp"), default="affine")
    parser.add_argument("--points", type=int, default=9, choices=sorted(TARGET_LAYOUTS))
    parser.add_argument("--samples-per-point", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)

    # heatmap parameters
    parser.add_argument("--cell", type=int, default=4, help="grid cell size in screen pixels")
    parser.add_argument("--sigma", type=float, default=45.0, help="Gaussian sigma in screen pixels")
    parser.add_argument(
        "--decay-tau",
        type=float,
        default=5.0,
        help="time decay constant in seconds (0 disables decay)",
    )
    parser.add_argument("--gamma", type=float, default=0.75, help="gamma correction exponent")
    parser.add_argument("--alpha", type=float, default=0.7, help="heatmap opacity")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="samples below this confidence are not accumulated",
    )
    parser.add_argument("--colormap", type=int, default=cv2.COLORMAP_TURBO, help="cv2 colormap id")
    parser.add_argument("--background", choices=("camera", "canvas"), default="camera")
    parser.add_argument("--preview", type=int, default=960, help="preview width")

    # fixation detection
    parser.add_argument("--fixation-method", choices=("idt", "ivt"), default="idt")
    parser.add_argument("--fixation-dispersion", type=float, default=45.0, help="I-DT dispersion in px")
    parser.add_argument("--fixation-min-duration", type=float, default=100.0, help="minimum fixation ms")
    parser.add_argument("--saccade-velocity", type=float, default=1.0, help="velocity threshold px/ms")

    parser.add_argument("--pupil", choices=("hybrid", "iris", "image"), default="hybrid")
    parser.add_argument("--points-out", default="attention_points.npy", help="gaze point log")
    parser.add_argument("--grid-out", default="attention_grid.npy", help="accumulation matrix dump")
    parser.add_argument("--events-out", default="fixation_events.json", help="fixation/saccade export")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def parse_screen(value: str) -> Tuple[int, int]:
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError("screen must look like 1920x1080") from None


def build_config(args: argparse.Namespace, screen: Tuple[int, int]) -> GazeConfig:
    config = GazeConfig(
        screen_width=screen[0],
        screen_height=screen[1],
        calibration_points=args.points,
        calibration_samples_per_point=args.samples_per_point,
        mapping_model=args.mapping,
        ema_alpha=args.ema_alpha if args.smoothing is None else args.smoothing,
        confidence_threshold=args.confidence_threshold,
        heatmap_cell_size=args.cell,
        heatmap_decay_tau=args.decay_tau,
        gaussian_sigma_px=args.sigma,
        heatmap_gamma=args.gamma,
        heatmap_alpha=args.alpha,
        heatmap_min_confidence=args.min_confidence,
        fixation_method=args.fixation_method,
        fixation_dispersion_px=args.fixation_dispersion,
        fixation_min_duration_ms=args.fixation_min_duration,
        saccade_velocity_threshold=args.saccade_velocity,
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def make_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (28, 24, 22), np.uint8)
    for x in range(0, width, 160):
        cv2.line(canvas, (x, 0), (x, height), (44, 40, 38), 1)
    for y in range(0, height, 160):
        cv2.line(canvas, (0, y), (width, y), (44, 40, 38), 1)
    cv2.rectangle(canvas, (0, 0), (width, 46), (62, 56, 52), -1)
    cv2.putText(
        canvas, "virtual desktop", (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
    )
    return canvas


def draw_events(canvas: np.ndarray, detector: EventDetector, scale: float) -> None:
    """Overlay recent fixation centroids (radius grows with fixation duration)."""
    for fixation in detector.fixations[-40:]:
        cx = int(fixation.centroid[0] * scale)
        cy = int(fixation.centroid[1] * scale)
        radius = max(6, int(min(fixation.duration_ms, 2000.0) / 60.0))
        cv2.circle(canvas, (cx, cy), radius, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, (80, 255, 120), -1, cv2.LINE_AA)


def draw_hud(
    canvas: np.ndarray,
    mapper: ScreenMapper,
    heat: AttentionHeatmap,
    detector: EventDetector,
    fps: float,
    position,
    status: str,
    show_blur: bool,
    confidence: float = 0.0,
    decay_on: bool = True,
    distance_cm: Optional[float] = None,
) -> None:
    x_scale = canvas.shape[1] / mapper.width
    y_scale = canvas.shape[0] / mapper.height
    if position is not None:
        cx, cy = int(position[0] * x_scale), int(position[1] * y_scale)
        cv2.circle(canvas, (cx, cy), 16, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (cx - 26, cy), (cx + 26, cy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - 26), (cx, cy + 26), (0, 255, 255), 1, cv2.LINE_AA)
        xy = f"X={position[0]:7.1f} Y={position[1]:7.1f}"
    else:
        xy = "X=----- Y=-----"
    distance = f"  d={distance_cm:.0f}cm" if distance_cm is not None else ""
    header = (
        f"fps={fps:4.1f}  {xy}  conf={confidence:.2f}{distance}  "
        f"points={heat.total_hits}  peak={heat.peak():6.1f}  {mapper.mapping_name}"
    )
    if not mapper.calibrated:
        header += "  (press c to calibrate)"
    cv2.putText(canvas, header, (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    summary = detector.summary()
    line2 = (
        f"fixations={summary['fixation_count']}  "
        f"avg_dur={summary['average_fixation_duration_ms']:.0f}ms  "
        f"saccades={summary['saccade_count']}  "
        f"disp_th={detector.dispersion_px:.0f}px  {detector.method.upper()}"
    )
    cv2.putText(canvas, line2, (14, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 255, 160), 1, cv2.LINE_AA)

    mode = "PAUSED" if heat.paused else ("blur=on" if show_blur else "blur=off")
    decay_text = f"tau={heat.decay_tau:.1f}s" if decay_on and heat.decay_enabled else "decay=off"
    footer = (
        f"{mode}  {decay_text}  sigma={heat.sigma:.0f}px({heat.sigma_cells:.1f} cells)  "
        f"gamma={heat.gamma:.2f}  grid={heat.grid.shape[1]}x{heat.grid.shape[0]}  {status}  |  "
        "[p]ause [d]ecay [r]eset [s]ave [c]alibrate [n]ew centre [q]uit"
    )
    cv2.putText(
        canvas, footer, (14, canvas.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA,
    )


MATRIX_LEFT = 70  # panel margins around the upscaled grid (pixels)
MATRIX_TOP = 84
MATRIX_RIGHT = 18
MATRIX_BOTTOM = 30

_COLORMAP_NAMES = {
    cv2.COLORMAP_TURBO: "TURBO",
    cv2.COLORMAP_JET: "JET",
    cv2.COLORMAP_BONE: "BONE",
    cv2.COLORMAP_VIRIDIS: "VIRIDIS",
    cv2.COLORMAP_INFERNO: "INFERNO",
    cv2.COLORMAP_MAGMA: "MAGMA",
    cv2.COLORMAP_PLASMA: "PLASMA",
    cv2.COLORMAP_HOT: "HOT",
    cv2.COLORMAP_COOL: "COOL",
}

def _put_text_fitted(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """putText that shrinks the font until the line fits inside the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    while scale > 0.3:
        (width, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if origin[0] + width <= image.shape[1] - 6:
            break
        scale -= 0.05
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_matrix(
    heat: AttentionHeatmap,
    show_blur: bool,
    position: Optional[Tuple[float, float]] = None,
    reference: float = 0.0,
) -> np.ndarray:
    """Draw the accumulation matrix in screen coordinates with a stable scale.

    The view is a diagnostic, so it must answer two questions the old
    per-frame-normalised version could not:

    - *how much is actually accumulated?*  Values are scaled with `log1p`
      against `reference` -- the running maximum of the raw grid for this
      session, maintained by `main` -- then gamma-corrected and colourised with
      the same colormap as `AttentionHeatmap.render`.  Nothing rescales to each
      frame's peak, so weak structure stays visible, growth and decay stay
      readable, and the raw/blur toggle compares both views against one
      denominator (the blurred peak can never exceed the raw peak, so the blur
      view never clips).  The header prints the absolute peak and reference.
    - *where on the screen is this cell?*  Screen-pixel axes, quarter
      gridlines and a live gaze crosshair + cell readout put the matrix in the
      same coordinate frame as the main window.
    """
    grid_h, grid_w = heat.grid.shape
    data = heat.blurred() if show_blur else heat.grid
    peak = float(data.max()) if data.size else 0.0
    ref = max(float(reference), peak, 1e-6)
    if peak <= 1e-6:
        norm = np.zeros_like(data, dtype=np.float32)
    else:
        norm = np.log1p(data)
        norm *= 1.0 / math.log1p(ref)
        np.power(norm, heat.gamma, out=norm)
        np.clip(norm, 0.0, 1.0, out=norm)
    colored = cv2.applyColorMap((norm * 255.0).astype(np.uint8), heat.colormap)

    scale = max(1, min(3, 1000 // max(grid_w, 1)))
    view = cv2.resize(colored, (grid_w * scale, grid_h * scale), interpolation=cv2.INTER_NEAREST)

    # scalar fill: broadcasting a per-channel tuple into a 2 MB buffer is ~5x slower
    panel = np.full(
        (view.shape[0] + MATRIX_TOP + MATRIX_BOTTOM, view.shape[1] + MATRIX_LEFT + MATRIX_RIGHT, 3),
        11,
        dtype=np.uint8,
    )
    x0, y0 = MATRIX_LEFT, MATRIX_TOP
    panel[y0:y0 + view.shape[0], x0:x0 + view.shape[1]] = view

    # quarter gridlines + screen-pixel tick labels (same frame as the main window)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        vx = x0 + int(round(fraction * (view.shape[1] - 1)))
        vy = y0 + int(round(fraction * (view.shape[0] - 1)))
        if 0.0 < fraction < 1.0:
            cv2.line(panel, (vx, y0), (vx, y0 + view.shape[0]), (70, 76, 84), 1, cv2.LINE_AA)
            cv2.line(panel, (x0, vy), (x0 + view.shape[1], vy), (70, 76, 84), 1, cv2.LINE_AA)
        cv2.line(panel, (vx, y0 - 6), (vx, y0), (150, 160, 170), 1, cv2.LINE_AA)
        cv2.line(panel, (vx, y0 + view.shape[0]), (vx, y0 + view.shape[0] + 6), (150, 160, 170), 1, cv2.LINE_AA)
        x_text = f"{int(round(fraction * heat.width))}"
        (tw, th), _ = cv2.getTextSize(x_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            panel, x_text, (vx - tw // 2, y0 + view.shape[0] + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 180, 190), 1, cv2.LINE_AA,
        )
        y_text = f"{int(round(fraction * heat.height))}"
        (tw, th), _ = cv2.getTextSize(y_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.line(panel, (x0 - 6, vy), (x0, vy), (150, 160, 170), 1, cv2.LINE_AA)
        cv2.putText(
            panel, y_text, (x0 - 8 - tw, vy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 180, 190), 1, cv2.LINE_AA,
        )
    cv2.rectangle(panel, (x0 - 1, y0 - 1), (x0 + view.shape[1], y0 + view.shape[0]), (130, 140, 150), 1, cv2.LINE_AA)

    # live gaze crosshair and the value of the cell it lands in
    if position is None:
        gaze_text = "gaze: no valid sample (nothing maps to a cell)"
        gaze_color = (120, 130, 140)
    else:
        col = min(max(int(position[0] / heat.cell), 0), grid_w - 1)
        row = min(max(int(position[1] / heat.cell), 0), grid_h - 1)
        vx = x0 + col * scale + scale // 2
        vy = y0 + row * scale + scale // 2
        cv2.line(panel, (x0, vy), (x0 + view.shape[1], vy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(panel, (vx, y0), (vx, y0 + view.shape[0]), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(
            panel,
            (x0 + col * scale, y0 + row * scale),
            (x0 + (col + 1) * scale - 1, y0 + (row + 1) * scale - 1),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        gaze_text = (
            f"gaze X={position[0]:7.1f} Y={position[1]:7.1f}  "
            f"cell=(col {col}, row {row})  value={float(data[row, col]):.3f}"
        )
        gaze_color = (230, 230, 230)

    mode = "GaussianBlur(grid)" if show_blur else "raw accumulation matrix"
    colormap = _COLORMAP_NAMES.get(heat.colormap, f"id={heat.colormap}")
    _put_text_fitted(
        panel,
        f"{mode}  peak={peak:.3f}  ref={ref:.3f} (session max of raw grid)  "
        f"log1p+gamma={heat.gamma:.2f}  {colormap}",
        (x0, 22),
        0.55,
        (0, 229, 255),
    )
    _put_text_fitted(
        panel,
        f"cell={heat.cell}px  grid={grid_w}x{grid_h} (cols x rows)  screen={heat.width}x{heat.height}  "
        f"origin=top-left  sigma={heat.sigma:.0f}px({heat.sigma_cells:.1f} cells)  "
        f"hits={heat.total_hits}  rejected={heat.rejected_low_confidence}  weight={heat.total_weight:.1f}",
        (x0, 46),
        0.5,
        (170, 210, 255),
    )
    _put_text_fitted(panel, gaze_text, (x0, 70), 0.5, gaze_color)
    return panel


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
FIXATION_RATE_PER_SECOND = 10.0  # weight added per second of fixation
SACCADE_FACTOR = 0.1  # saccades contribute an order of magnitude less


def save_events(detector: EventDetector, path: str, summary: Dict[str, float]) -> str:
    payload = {"summary": summary, **detector.events_payload()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


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
    heat = AttentionHeatmap(screen_w, screen_h, config=config, colormap=args.colormap)
    detector = EventDetector(
        method=config.fixation_method,
        dispersion_px=config.fixation_dispersion_px,
        min_duration_ms=config.fixation_min_duration_ms,
        velocity_threshold=config.saccade_velocity_threshold,
    )
    canvas_background = make_canvas(screen_w, screen_h)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        tracker.close()
        sys.exit(f"Could not open webcam index {args.camera}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(
        "keys: [p] pause  [d] decay  [r] reset  [s] save  [c] calibrate  "
        "[n] re-centre  [v] matrix view  [q/ESC] quit"
    )
    print(
        f"mapping {screen_w}x{screen_h} | model={mapper.model_name} | "
        f"loaded calibration: {mapper.calibrated} | fixation={config.fixation_method} | "
        f"tau={config.heatmap_decay_tau}s"
    )
    fps = 0.0
    status = "accumulating gaze density"
    show_blur = True
    position = None
    decay_on = heat.decay_enabled
    matrix_scale = 0.0  # running session reference for the matrix view
    previous_time: Optional[float] = None
    samples_attempted = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Frame grab failed, stopping.", file=sys.stderr)
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            loop_start = time.perf_counter()
            now = time.monotonic()
            dt_seconds = 0.0 if previous_time is None else max(now - previous_time, 0.0)
            previous_time = now

            result = tracker.process(frame)
            gaze = result.gaze
            features = result.features
            confidence = result.confidence

            if calibrator.active:
                state = calibrator.update(features, confidence)
                if state == "captured":
                    status = f"calibrating {calibrator.index}/{len(calibrator.targets)}"
                elif state == "done":
                    samples = calibrator.finalize()
                    if mapper.fit(samples, reference_distance_cm=result.distance_cm):
                        status = "calibration saved"
                        print(f"calibration saved to {mapper.path}")
                    else:
                        status = "calibration failed"
                elif state in ("stabilizing", "collecting"):
                    status = "hold still on the target"

            position = mapper.map(
                gaze, features, confidence, distance_cm=result.distance_cm
            )

            # -- fixation / saccade classification -------------------------
            label = "invalid"
            if position is not None:
                samples_attempted += 1
                label = detector.update(
                    now * 1000.0,
                    position[0],
                    position[1],
                    confidence,
                    min_confidence=config.heatmap_min_confidence,
                    valid=True,
                )

            # -- accumulation ----------------------------------------------
            if position is not None and not heat.paused and not calibrator.active:
                if label == "fixation":
                    fixation_weight = dt_seconds * FIXATION_RATE_PER_SECOND
                elif label == "saccade":
                    fixation_weight = dt_seconds * FIXATION_RATE_PER_SECOND * SACCADE_FACTOR
                else:
                    fixation_weight = 0.0
                if fixation_weight > 0.0:
                    heat.add(
                        position[0],
                        position[1],
                        now,
                        confidence=confidence,
                        fixation_weight=fixation_weight,
                    )

            if not heat.paused and decay_on:
                heat.decay_step(dt_seconds)

            # running reference: the matrix view scales against the largest raw
            # value seen this session instead of re-normalising to each frame's peak
            matrix_scale = max(matrix_scale, float(heat.grid.max()))

            background = canvas_background if args.background == "canvas" else frame
            composite = heat.render(background)
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - loop_start, 1e-6))
            draw_hud(
                composite, mapper, heat, detector, fps, position, status,
                show_blur, confidence, decay_on,
                distance_cm=result.distance_cm,
            )
            draw_events(composite, detector, composite.shape[1] / max(mapper.width, 1))

            inset = tracker.draw_overlay(frame, result, show_hud=False)
            inset_scale = 360 / max(inset.shape[1], 1)
            inset = cv2.resize(inset, (360, max(int(inset.shape[0] * inset_scale), 1)))
            x0 = composite.shape[1] - inset.shape[1] - 14
            y0 = 120
            if y0 + inset.shape[0] <= composite.shape[0] and x0 >= 0:
                roi = composite[y0:y0 + inset.shape[0], x0:x0 + inset.shape[1]]
                composite[y0:y0 + inset.shape[0], x0:x0 + inset.shape[1]] = cv2.addWeighted(
                    roi, 0.25, inset, 0.9, 0
                )
                cv2.rectangle(
                    composite, (x0, y0), (x0 + inset.shape[1], y0 + inset.shape[0]),
                    (255, 255, 255), 1,
                )

            if args.background == "camera":
                # screen-space density over the video is a coordinate mismatch unless
                # it is called out; drawn after the inset so it is never occluded
                _put_text_fitted(
                    composite,
                    f"density is in SCREEN px ({screen_w}x{screen_h}) - video is context only; "
                    "use --background canvas to match the matrix window",
                    (14, 128),
                    0.5,
                    (120, 210, 255),
                )

            preview_scale = args.preview / max(composite.shape[1], 1)
            preview = cv2.resize(
                composite,
                (args.preview, max(int(composite.shape[0] * preview_scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("attention-track phase 3 - gaze-density heatmap", preview)
            cv2.imshow("accumulation matrix", draw_matrix(heat, show_blur, position, matrix_scale))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                heat.paused = not heat.paused
                status = "paused" if heat.paused else "accumulating gaze density"
            elif key == ord("d"):
                decay_on = not decay_on
                status = f"decay tau={heat.decay_tau:.1f}s" if decay_on else "decay off"
            elif key == ord("r"):
                heat.reset()
                detector.reset()
                matrix_scale = 0.0
                status = "matrix cleared"
            elif key == ord("s"):
                summary = detector.summary(samples_attempted)
                print(
                    f"saved {heat.save_points(args.points_out)} ({heat.total_hits} points, "
                    f"weight sum {heat.total_weight:.1f}), {heat.save_grid(args.grid_out)} "
                    f"and {save_events(detector, args.events_out, summary)}"
                )
            elif key == ord("v"):
                show_blur = not show_blur
            elif key == ord("n"):
                mapper.recenter()
                status = "gaze re-centred"
            elif key == ord("c"):
                calibrator.start()
                status = "calibration started"
            elif key == ord(" ") and calibrator.active:
                state = calibrator.record(features, confidence)
                if state == "done":
                    samples = calibrator.finalize()
                    if mapper.fit(samples, reference_distance_cm=result.distance_cm):
                        status = "calibration saved"
                        print(f"calibration saved to {mapper.path}")
    finally:
        detector.finalize()
        points_path = heat.save_points(args.points_out)
        grid_path = heat.save_grid(args.grid_out)
        summary = detector.summary(samples_attempted)
        events_path = save_events(detector, args.events_out, summary)
        print(
            f"saved {points_path} shape={heat.points_matrix().shape}, {grid_path} "
            f"shape={heat.grid.shape} and {events_path}"
        )
        print(
            f"fixations={summary['fixation_count']} "
            f"saccades={summary['saccade_count']} "
            f"avg_fixation_ms={summary['average_fixation_duration_ms']:.0f} "
            f"fixation_rate={summary['fixation_detection_rate']:.2f}"
        )
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
