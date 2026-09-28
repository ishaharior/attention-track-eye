"""Core library for attention-track.

Phase 1 helpers: EyeTracker (eye ROI cropping + pupil tracking on MediaPipe Face Mesh)
Phase 2 helpers: ScreenMapper / ScreenCalibrator (gaze -> 1920x1080 pixels)
Phase 3 helpers: AttentionHeatmap (accumulation matrix + cv2.GaussianBlur heatmap)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

MEDIAPIPE_ERROR: Optional[str] = None
mp = None
try:
    import mediapipe as mp

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        raise AttributeError("mediapipe.solutions.face_mesh is unavailable")
except Exception as exc:
    mp = None
    MEDIAPIPE_ERROR = str(exc)

RECT_W = 200
RECT_H = 100
EYE_PAD = 1.25
MIN_CALIBRATION_SPAN = 0.25

EYE_TEMPLATES = (
    (33, 159, 133, 145, 468,
     (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)),
    (362, 386, 263, 374, 473,
     (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)),
)
IRIS_FALLBACK = {468: 473, 473: 468}


def _shoelace(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _expand(pts: np.ndarray, pad: float) -> np.ndarray:
    center = pts.mean(axis=0)
    return (center + (pts - center) * pad).astype(np.float32)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    poly = np.ascontiguousarray(polygon.reshape(-1, 1, 2), dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0.0


def _otsu_threshold(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
    if mask is not None and int(mask.max()) == 0:
        mask = None
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0.0:
        return 127
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    best_t = 127
    for t in range(256):
        w_b += float(hist[t])
        if w_b <= 0.0:
            continue
        w_f = total - w_b
        if w_f <= 0.0:
            break
        sum_b += t * float(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return best_t


def detect_pupil_in_rect(
    rect_bgr: np.ndarray,
    aperture_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float, float, float]]:
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = aperture_mask
    if mask is None or int(mask.max()) == 0:
        mask = np.full(gray.shape[:2], 255, np.uint8)
    thr = _otsu_threshold(gray, mask)
    binary = np.zeros(gray.shape[:2], np.uint8)
    binary[(gray <= thr) & (mask > 0)] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    aperture_area = float((mask > 0).sum())
    if aperture_area <= 0.0:
        return None
    best: Optional[Tuple[float, float, float, float]] = None
    height, width = gray.shape[:2]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(6.0, 0.0008 * aperture_area) or area > 0.45 * aperture_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        area_ratio = area / aperture_area
        confidence = 0.55 * min(circularity / 0.85, 1.0) + 0.45 * min(area_ratio / 0.06, 1.0)
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            confidence *= 0.6
        radius = float(np.sqrt(area / np.pi))
        if best is None or confidence > best[3]:
            best = (float(cx), float(cy), radius, float(confidence))
    return best


@dataclass
class EyeObservation:
    side: str
    rect: np.ndarray
    aperture_mask: np.ndarray
    contour_rect: np.ndarray
    contour_frame: np.ndarray
    quad_frame: np.ndarray
    pupil_rect: Optional[Tuple[float, float]]
    pupil_frame: Optional[Tuple[float, float]]
    gaze: Optional[Tuple[float, float]]
    source: str
    confidence: float


@dataclass
class GazeResult:
    gaze: Optional[Tuple[float, float]]
    confidence: float
    eyes: List[EyeObservation] = field(default_factory=list)
    inference_ms: float = 0.0


class EyeTracker:
    """Crops both eye regions out of a frame and tracks the pupil inside them."""

    def __init__(
        self,
        pupil_mode: str = "hybrid",
        refine_landmarks: bool = True,
        max_num_faces: int = 1,
        rect_size: Tuple[int, int] = (RECT_W, RECT_H),
        pad: float = EYE_PAD,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None:
            raise RuntimeError(
                "MediaPipe Face Mesh unavailable: "
                f"{MEDIAPIPE_ERROR or 'unknown error'}. Install with: pip install mediapipe"
            )
        if pupil_mode not in ("hybrid", "iris", "image"):
            raise ValueError("pupil_mode must be one of: hybrid, iris, image")
        self.pupil_mode = pupil_mode
        self.refine_landmarks = refine_landmarks
        self.rect_w, self.rect_h = int(rect_size[0]), int(rect_size[1])
        self.pad = float(pad)
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.last_raw_result = None

    def close(self) -> None:
        try:
            self.mesh.close()
        except Exception:
            pass

    def __enter__(self) -> "EyeTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _select_pupil(
        self,
        rect: np.ndarray,
        aperture: np.ndarray,
        iris_rect: Optional[np.ndarray],
    ) -> Tuple[Optional[Tuple[float, float]], str, float]:
        detection = detect_pupil_in_rect(rect, aperture)
        if self.pupil_mode == "iris":
            if iris_rect is not None:
                return (float(iris_rect[0]), float(iris_rect[1])), "iris", 1.0
            if detection is not None:
                return (detection[0], detection[1]), "image", detection[3]
            return None, "none", 0.0
        if self.pupil_mode == "image":
            if detection is not None:
                return (detection[0], detection[1]), "image", detection[3]
            if iris_rect is not None:
                return (float(iris_rect[0]), float(iris_rect[1])), "iris", 0.7
            return None, "none", 0.0
        if detection is not None and detection[3] >= 0.6:
            return (detection[0], detection[1]), "image", detection[3]
        if iris_rect is not None:
            return (float(iris_rect[0]), float(iris_rect[1])), "iris", 0.95
        if detection is not None:
            return (detection[0], detection[1]), "image", detection[3]
        return None, "none", 0.0

    def process(self, frame: np.ndarray) -> GazeResult:
        started = time.perf_counter()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        raw = self.mesh.process(rgb)
        self.last_raw_result = raw
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not raw.multi_face_landmarks:
            return GazeResult(None, 0.0, [], inference_ms)
        landmarks = raw.multi_face_landmarks[0].landmark
        points = np.array([[p.x * width, p.y * height] for p in landmarks], dtype=np.float32)

        eyes: List[EyeObservation] = []
        for outer_i, top_i, inner_i, bottom_i, iris_i, contour_ids in EYE_TEMPLATES:
            if max(iris_i, *contour_ids) >= len(points):
                continue
            contour = points[list(contour_ids)]
            quad = _expand(points[[outer_i, top_i, inner_i, bottom_i]].copy(), self.pad)
            target = np.array(
                [
                    [0.0, self.rect_h / 2.0],
                    [self.rect_w / 2.0, 0.0],
                    [float(self.rect_w), self.rect_h / 2.0],
                    [self.rect_w / 2.0, float(self.rect_h)],
                ],
                dtype=np.float32,
            )
            if _shoelace(quad) * _shoelace(target) < 0.0:
                quad = quad[[0, 3, 2, 1]]
                target = target[[0, 3, 2, 1]]
            matrix = cv2.getPerspectiveTransform(quad, target)
            inverse = cv2.getPerspectiveTransform(target, quad)
            rect = cv2.warpPerspective(frame, matrix, (self.rect_w, self.rect_h))
            contour_rect = (
                cv2.perspectiveTransform(contour.reshape(-1, 1, 2), matrix)
                .reshape(-1, 2)
                .astype(np.float32)
            )
            aperture = np.zeros((self.rect_h, self.rect_w), np.uint8)
            poly = np.round(contour_rect).astype(np.int32)
            if len(poly) >= 3:
                cv2.fillPoly(aperture, [poly], 255)

            iris_point = None
            for candidate in (iris_i, IRIS_FALLBACK[iris_i]):
                if candidate < len(points) and _point_in_polygon(points[candidate], contour):
                    iris_point = points[candidate]
                    break
            iris_rect = None
            if iris_point is not None:
                iris_rect = cv2.perspectiveTransform(
                    np.array([[iris_point]], dtype=np.float32), matrix
                )[0][0]

            pupil_rect, source, confidence = self._select_pupil(rect, aperture, iris_rect)
            pupil_frame = None
            gaze = None
            if pupil_rect is not None:
                mapped = cv2.perspectiveTransform(
                    np.array([[pupil_rect]], dtype=np.float32), inverse
                )[0][0]
                pupil_frame = (float(mapped[0]), float(mapped[1]))
                gaze = (pupil_rect[0] / self.rect_w, pupil_rect[1] / self.rect_h)
            eyes.append(
                EyeObservation(
                    side="",
                    rect=rect,
                    aperture_mask=aperture,
                    contour_rect=contour_rect,
                    contour_frame=contour.astype(np.float32),
                    quad_frame=quad.astype(np.float32),
                    pupil_rect=pupil_rect,
                    pupil_frame=pupil_frame,
                    gaze=gaze,
                    source=source,
                    confidence=confidence,
                )
            )

        if not eyes:
            return GazeResult(None, 0.0, [], inference_ms)
        eyes.sort(key=lambda eye: float(eye.contour_frame[:, 0].mean()))
        eyes[0].side = "image-left"
        if len(eyes) > 1:
            eyes[1].side = "image-right"

        valid = [eye for eye in eyes if eye.gaze is not None]
        if not valid:
            return GazeResult(None, 0.0, eyes, inference_ms)
        gaze = (
            float(np.mean([eye.gaze[0] for eye in valid])),
            float(np.mean([eye.gaze[1] for eye in valid])),
        )
        confidence = float(np.mean([eye.confidence for eye in valid]))
        return GazeResult(gaze, confidence, eyes, inference_ms)

    def draw_overlay(
        self,
        frame: np.ndarray,
        result: GazeResult,
        show_contours: bool = True,
        show_hud: bool = True,
    ) -> np.ndarray:
        view = frame.copy()
        for eye in result.eyes:
            if show_contours:
                poly = np.round(eye.contour_frame).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [poly], True, (0, 255, 0), 1, cv2.LINE_AA)
                quad = np.round(eye.quad_frame).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [quad], True, (255, 255, 0), 1, cv2.LINE_AA)
            if eye.pupil_frame is not None:
                px, py = int(round(eye.pupil_frame[0])), int(round(eye.pupil_frame[1]))
                cv2.circle(view, (px, py), 7, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(view, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
            if eye.gaze is not None:
                center = eye.quad_frame.mean(axis=0)
                start = (int(round(center[0])), int(round(center[1])))
                end = (
                    int(round(center[0] + (eye.gaze[0] - 0.5) * 240.0)),
                    int(round(center[1] + (eye.gaze[1] - 0.5) * 240.0)),
                )
                cv2.arrowedLine(view, start, end, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
            if eye.pupil_frame is not None:
                label = f"{eye.side}:{eye.source}:{eye.confidence:.2f}"
                origin = (
                    int(eye.quad_frame[:, 0].min()),
                    max(14, int(eye.quad_frame[:, 1].min()) - 6),
                )
                cv2.putText(view, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        if show_hud:
            self._draw_gaze_field(view, result)
        return view

    def _draw_gaze_field(self, view: np.ndarray, result: GazeResult) -> None:
        height = view.shape[0]
        box_w, box_h = 240, 140
        x0, y0 = 12, height - box_h - 12
        cv2.rectangle(view, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), -1)
        cv2.rectangle(view, (x0, y0), (x0 + box_w, y0 + box_h), (200, 200, 200), 1)
        cv2.putText(
            view, "GAZE FIELD (1920x1080)", (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA,
        )
        ix, iy = x0 + 8, y0 + 26
        iw, ih = box_w - 16, box_h - 34
        cv2.rectangle(view, (ix, iy), (ix + iw, iy + ih), (70, 70, 70), 1)
        cv2.line(view, (ix + iw // 2, iy), (ix + iw // 2, iy + ih), (70, 70, 70), 1)
        cv2.line(view, (ix, iy + ih // 2), (ix + iw, iy + ih // 2), (70, 70, 70), 1)
        if result.gaze is not None:
            gx, gy = result.gaze
            cx = int(round(ix + min(max(gx, 0.0), 1.0) * iw))
            cy = int(round(iy + min(max(gy, 0.0), 1.0) * ih))
            cv2.circle(view, (cx, cy), 7, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(view, (cx, cy), 11, (0, 255, 255), 1, cv2.LINE_AA)

    def draw_crops(self, result: GazeResult, scale: float = 2.0) -> np.ndarray:
        if not result.eyes:
            placeholder = np.zeros((self.rect_h, self.rect_w, 3), np.uint8)
            cv2.putText(placeholder, "NO FACE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
            tiles = [placeholder]
        else:
            tiles = []
            for eye in result.eyes:
                tile = eye.rect.copy()
                poly = np.round(eye.contour_rect).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(tile, [poly], True, (0, 255, 0), 1, cv2.LINE_AA)
                if eye.pupil_rect is not None:
                    px, py = int(round(eye.pupil_rect[0])), int(round(eye.pupil_rect[1]))
                    cv2.circle(tile, (px, py), 10, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.drawMarker(tile, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
                cv2.putText(
                    tile, f"{eye.side} [{eye.source}]", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
                )
                tiles.append(tile)
        width = max(tile.shape[1] for tile in tiles)
        rows = []
        for tile in tiles:
            if tile.shape[1] != width:
                pad = np.zeros((tile.shape[0], width - tile.shape[1], 3), np.uint8)
                tile = np.hstack([tile, pad])
            rows.append(tile)
            rows.append(np.full((2, width, 3), 60, np.uint8))
        stack = np.vstack(rows[:-1])
        if scale != 1.0:
            stack = cv2.resize(stack, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return stack

    def draw_mediapipe_mesh(self, frame: np.ndarray) -> np.ndarray:
        if self.last_raw_result is None or not self.last_raw_result.multi_face_landmarks:
            return frame
        view = frame.copy()
        drawing = mp.solutions.drawing_utils
        style = mp.solutions.drawing_styles
        for face in self.last_raw_result.multi_face_landmarks:
            drawing.draw_landmarks(
                view,
                face,
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=style.get_default_face_mesh_tesselation_style(),
            )
        return view


class ScreenMapper:
    """Translates normalized eye movement into pixel coordinates on a screen."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        gain: float = 3.0,
        smoothing: float = 0.3,
        path: str = "calibration.npz",
        auto_center: bool = True,
        warmup_frames: int = 45,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.gain = float(gain)
        self.smoothing = float(smoothing)
        self.path = path
        self.auto_center = bool(auto_center)
        self.warmup_frames = int(warmup_frames)
        self.affine: Optional[np.ndarray] = None
        self._smoothed: Optional[Tuple[float, float]] = None
        self._baseline: Optional[Tuple[float, float]] = None
        self._warm_sum = np.zeros(2, dtype=np.float64)
        self._warm_n = 0
        self.load()

    @property
    def calibrated(self) -> bool:
        return self.affine is not None

    @property
    def mapping_name(self) -> str:
        if self.calibrated:
            return "affine calibration"
        if self.auto_center and self._baseline is not None:
            return f"raw gain={self.gain:.1f} (auto-centred)"
        return f"raw gain={self.gain:.1f}"

    def recenter(self) -> None:
        self._baseline = None
        self._warm_sum[:] = 0.0
        self._warm_n = 0
        self._smoothed = None

    def _update_baseline(self, gx: float, gy: float) -> Tuple[float, float]:
        if not self.auto_center or self.calibrated:
            return (0.5, 0.5)
        if self._baseline is None:
            self._warm_sum += (gx, gy)
            self._warm_n += 1
            running = self._warm_sum / self._warm_n
            if self._warm_n >= self.warmup_frames:
                self._baseline = (
                    float(np.clip(running[0], 0.15, 0.85)),
                    float(np.clip(running[1], 0.15, 0.85)),
                )
                return self._baseline
            return (float(running[0]), float(running[1]))
        return self._baseline

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            data = np.load(self.path)
            affine = np.asarray(data["affine"], dtype=np.float32)
            width = int(data["width"])
            height = int(data["height"])
            if affine.shape == (2, 3) and width == self.width and height == self.height:
                self.affine = affine
        except Exception:
            self.affine = None

    def fit(self, samples: Sequence[Tuple[float, float, float, float]]) -> bool:
        if len(samples) < 4:
            return False
        gaze = np.array([[gx, gy] for gx, gy, _, _ in samples], dtype=np.float64)
        span = gaze.max(axis=0) - gaze.min(axis=0)
        if float(span.min()) < MIN_CALIBRATION_SPAN:
            return False
        design = np.array([[gx, gy, 1.0] for gx, gy, _, _ in samples], dtype=np.float64)
        targets = np.array([[x, y] for _, _, x, y in samples], dtype=np.float64)
        solution, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
        if rank < 3:
            return False
        self.affine = solution.T.astype(np.float32)
        self._smoothed = None
        np.savez(
            self.path,
            affine=self.affine,
            width=self.width,
            height=self.height,
            samples=np.array(samples, dtype=np.float32),
        )
        return True

    def reset(self) -> None:
        self.affine = None
        self._smoothed = None
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass

    def map(self, gaze: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if gaze is None:
            return None
        gx, gy = gaze
        if self.affine is not None:
            vector = np.array([gx, gy, 1.0], dtype=np.float32)
            x = float(np.dot(self.affine[0], vector))
            y = float(np.dot(self.affine[1], vector))
        else:
            base_x, base_y = self._update_baseline(gx, gy)
            x = (0.5 + (gx - base_x) * self.gain) * self.width
            y = (0.5 + (gy - base_y) * self.gain) * self.height
        x = float(min(max(x, 0.0), self.width - 1))
        y = float(min(max(y, 0.0), self.height - 1))
        if self._smoothed is None:
            self._smoothed = (x, y)
        else:
            a = self.smoothing
            self._smoothed = (
                a * x + (1.0 - a) * self._smoothed[0],
                a * y + (1.0 - a) * self._smoothed[1],
            )
        return self._smoothed


class ScreenCalibrator:
    """Five point calibration run: corners plus screen centre."""

    TARGETS = ((0.08, 0.08), (0.92, 0.08), (0.92, 0.92), (0.08, 0.92), (0.5, 0.5))

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        tolerance: float = 0.1,
        dwell_frames: int = 10,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.tolerance = float(tolerance)
        self.dwell_frames = int(dwell_frames)
        self.active = False
        self.index = 0
        self.hits = 0
        self.samples: List[Tuple[float, float, float, float]] = []
        self.captured: List[Tuple[float, float]] = []

    @property
    def done(self) -> bool:
        return self.index >= len(self.TARGETS)

    @property
    def target_px(self) -> Optional[Tuple[float, float]]:
        if not self.active or self.done:
            return None
        tx, ty = self.TARGETS[self.index]
        return tx * self.width, ty * self.height

    @property
    def progress(self) -> float:
        return min(self.hits / max(self.dwell_frames, 1), 1.0)

    def start(self) -> None:
        self.active = True
        self.index = 0
        self.hits = 0
        self.samples = []
        self.captured = []

    def cancel(self) -> None:
        self.active = False

    def update(self, gaze: Optional[Tuple[float, float]]) -> str:
        if not self.active:
            return "idle"
        if gaze is None:
            self.hits = 0
            return "aiming"
        tx, ty = self.TARGETS[self.index]
        if abs(gaze[0] - tx) <= self.tolerance and abs(gaze[1] - ty) <= self.tolerance:
            self.hits += 1
        else:
            self.hits = 0
        if self.hits >= self.dwell_frames:
            return self.record(gaze)
        return "aiming"

    def record(self, gaze: Optional[Tuple[float, float]]) -> str:
        if not self.active or gaze is None or self.done:
            return "idle"
        tx, ty = self.TARGETS[self.index]
        self.samples.append((float(gaze[0]), float(gaze[1]), tx * self.width, ty * self.height))
        self.captured.append((tx, ty))
        self.index += 1
        self.hits = 0
        if self.done:
            self.active = False
            return "done"
        return "captured"


class AttentionHeatmap:
    """Accumulates screen points in a NumPy matrix and blurs them into a heatmap."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        cell: int = 4,
        sigma: float = 45.0,
        decay: float = 0.997,
        colormap: int = cv2.COLORMAP_TURBO,
        alpha: float = 0.7,
        initial_capacity: int = 8192,
    ) -> None:
        self.cell = max(int(cell), 1)
        self.width = int(width)
        self.height = int(height)
        self.grid_w = max(self.width // self.cell, 1)
        self.grid_h = max(self.height // self.cell, 1)
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.sigma = float(sigma)
        self.decay = float(decay)
        self.colormap = colormap
        self.alpha = float(alpha)
        self.total_hits = 0
        self.paused = False
        self._points = np.zeros((max(initial_capacity, 64), 3), dtype=np.float32)
        self._count = 0

    @property
    def kernel_size(self) -> int:
        sigma_cells = self.sigma / self.cell
        return int(2 * round(3.0 * sigma_cells) + 1)

    def points_matrix(self) -> np.ndarray:
        return self._points[: self._count].copy()

    def save_points(self, path: str = "attention_points.npy") -> str:
        np.save(path, self.points_matrix())
        return path

    def save_grid(self, path: str = "attention_grid.npy") -> str:
        np.save(path, self.grid.copy())
        return path

    def reset(self) -> None:
        self.grid[:] = 0.0
        self._points[: self._count] = 0.0
        self._count = 0
        self.total_hits = 0

    def _append_point(self, t: float, x: float, y: float) -> None:
        if self._count >= self._points.shape[0]:
            grown = np.zeros((self._points.shape[0] * 2, 3), dtype=np.float32)
            grown[: self._count] = self._points[: self._count]
            self._points = grown
        self._points[self._count] = (t, x, y)
        self._count += 1

    def add(self, x: float, y: float, t: float, weight: float = 1.0) -> bool:
        col = int(x / self.cell)
        row = int(y / self.cell)
        if col < 0 or row < 0 or col >= self.grid_w or row >= self.grid_h:
            return False
        self.grid[row, col] += weight
        self.total_hits += 1
        self._append_point(t, x, y)
        return True

    def decay_step(self, scale: float = 1.0) -> None:
        if self.decay >= 1.0:
            return
        self.grid *= float(self.decay) ** scale

    def blurred(self) -> np.ndarray:
        kernel = self.kernel_size
        return cv2.GaussianBlur(self.grid, (kernel, kernel), 0)

    def peak(self) -> float:
        return float(self.blurred().max())

    def render(self, background: Optional[np.ndarray] = None) -> np.ndarray:
        blurred = self.blurred()
        peak = float(blurred.max())
        if peak <= 1e-6:
            norm = np.zeros_like(blurred)
        else:
            norm = np.clip(blurred / peak, 0.0, 1.0)
        shaped = np.power(norm, 0.75)
        heat = cv2.applyColorMap((shaped * 255.0).astype(np.uint8), self.colormap)
        if background is None:
            return heat
        bg_h, bg_w = background.shape[:2]
        if (bg_h, bg_w) != (self.grid_h, self.grid_w):
            heat = cv2.resize(heat, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
            shaped = cv2.resize(shaped, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
        amount = (self.alpha * shaped).astype(np.float32)[..., None]
        mixed = background.astype(np.float32) * (1.0 - amount) + heat.astype(np.float32) * amount
        return mixed.astype(np.uint8)
