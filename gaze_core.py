"""Core library for attention-track.

Shared definitions used by all three phases:

- configuration .......... GazeConfig
- data structures ........ HeadPose, ConfidenceBreakdown, FeatureVector helpers
- feature definitions .... FEATURE_NAMES / build_feature_vector
- confidence calculations  confidence_breakdown / fused-eye quality weighting
- Phase 1 helpers ........ EyeTracker (eye ROI cropping, pupil tracking, head pose,
                           feature vector, confidence-weighted eye fusion)
- Phase 2 helpers ........ ScreenMapper / ScreenCalibrator (feature -> 1920x1080 pixels)
- Phase 3 helpers ........ AttentionHeatmap (fixation-weighted, time-decayed grid)

Terminology: pupil/iris and facial features are inputs to a calibrated gaze-estimation
model. Pupil position alone is never treated as the direct gaze direction.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("attention_track")

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

# --------------------------------------------------------------------
# Feature vector definition
# ---------------------------------------------------------------------------
# left/right_eye_*  : pupil position relative to the eye-aperture centre,
#                     normalized by the aperture size (eye-centred reference frame)
# left/right_iris_* : pupil/iris position inside the normalized 200x100 eye crop
#                     (image-centred reference frame, 0..1)
# head_*            : head pose in degrees, divided by HEAD_POSE_SCALE_DEG so that
#                     the three pose channels stay on a comparable numeric scale.
FEATURE_NAMES: Tuple[str, ...] = (
    "left_eye_x",
    "left_eye_y",
    "right_eye_x",
    "right_eye_y",
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
    "head_yaw",
    "head_pitch",
    "head_roll",
)
FEATURE_DIM = len(FEATURE_NAMES)
HEAD_POSE_SCALE_DEG = 30.0
NEUTRAL_IRIS = 0.5  # value used when an eye is missing (feature is imputed)


@dataclass
class GazeConfig:
    """Centralized configuration for the whole gaze-estimation pipeline."""

    # display / calibration
    screen_width: int = 1920
    screen_height: int = 1080
    calibration_points: int = 9
    calibration_samples_per_point: int = 20
    calibration_stabilize_frames: int = 12
    calibration_outlier_z: float = 3.0
    calibration_min_confidence: float = 0.35
    min_calibration_span: float = 0.25

    # mapping
    mapping_model: str = "affine"  # affine | polynomial | mlp
    polynomial_ridge: float = 1e-2
    mlp_hidden: int = 16
    mlp_epochs: int = 400
    mlp_learning_rate: float = 5e-3
    random_seed: int = 7

    # temporal smoothing / outliers
    ema_alpha: float = 0.3
    confidence_threshold: float = 0.5
    # 0.0 = auto (1.5 x screen diagonal).  Deliberately beyond any reachable
    # on-screen jump so legitimate saccades are never rejected for being fast;
    # lower it only for constrained setups.
    outlier_max_jump_px: float = 0.0
    outlier_max_speed_px_s: float = 0.0  # 0.0 = disabled

    # eye fusion
    min_total_eye_quality: float = 0.12
    temporal_confidence_tau_s: float = 0.35

    # heatmap (Phase 3)
    heatmap_cell_size: int = 4
    heatmap_decay_tau: float = 5.0  # seconds
    gaussian_sigma_px: float = 45.0  # screen pixels
    heatmap_gamma: float = 0.75
    heatmap_alpha: float = 0.7
    heatmap_min_confidence: float = 0.35

    # fixation / saccade detection (Phase 3)
    fixation_method: str = "idt"  # idt | ivt
    fixation_dispersion_px: float = 45.0
    fixation_min_duration_ms: float = 100.0
    saccade_velocity_threshold: float = 1.0  # px per ms (1000 px/s)

    # confidence weights (kept explicit so the combination stays interpretable)
    confidence_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "pupil": 0.3,
            "landmark": 0.2,
            "head_pose": 0.15,
            "gaze": 0.25,
            "temporal": 0.1,
        }
    )

    def validate(self) -> None:
        if self.mapping_model not in ("affine", "polynomial", "mlp"):
            raise ValueError(f"unknown mapping_model: {self.mapping_model!r}")
        if self.fixation_method not in ("idt", "ivt"):
            raise ValueError(f"unknown fixation_method: {self.fixation_method!r}")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.calibration_points not in TARGET_LAYOUTS:
            raise ValueError(f"calibration_points must be one of {sorted(TARGET_LAYOUTS)}")
        if self.calibration_samples_per_point < 1:
            raise ValueError("calibration_samples_per_point must be >= 1")


@dataclass
class HeadPose:
    """Camera-relative head orientation in degrees (solvePnP based)."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    valid: bool = False
    confidence: float = 0.0
    reprojection_error: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "yaw": round(self.yaw, 3),
            "pitch": round(self.pitch, 3),
            "roll": round(self.roll, 3),
            "confidence": round(self.confidence, 3),
            "reprojection_error": round(self.reprojection_error, 2),
        }


@dataclass
class ConfidenceBreakdown:
    """Separate, interpretable confidence components (never a single opaque number).

    pupil ...... reliability of pupil/iris localization
    landmark ... stability/reliability of the facial landmarks
    head_pose . stability and plausibility of the head-pose estimate
    gaze ....... reliability of the combined gaze estimate
    temporal ... consistency with the recent gaze trajectory
    """

    pupil: float = 0.0
    landmark: float = 0.0
    head_pose: float = 0.0
    gaze: float = 0.0
    temporal: float = 0.0

    def combined(self, weights: Optional[Dict[str, float]] = None) -> float:
        weights = weights or {
            "pupil": 0.3,
            "landmark": 0.2,
            "head_pose": 0.15,
            "gaze": 0.25,
            "temporal": 0.1,
        }
        total = 0.0
        weight_sum = 0.0
        for name, weight in weights.items():
            total += float(getattr(self, name)) * float(weight)
            weight_sum += float(weight)
        return float(total / weight_sum) if weight_sum > 0 else 0.0

    def as_dict(self) -> Dict[str, float]:
        data = {name: round(float(getattr(self, name)), 3) for name in self.__dataclass_fields__}
        data["combined"] = round(self.combined(), 3)
        return data


# ---------------------------------------------------------------------------
# Landmark / head-pose model
# ---------------------------------------------------------------------------
# 3D head model points (approximate millimetres) matched with 2D MediaPipe
# landmarks.  x points to image right, y points up, z points toward the camera,
# i.e. the model frame matches the camera frame for a frontal face.
HEAD_POSE_LANDMARK_IDS: Tuple[int, ...] = (1, 152, 33, 263, 61, 291)
HEAD_POSE_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # eye corner on image left
        (225.0, 170.0, -135.0),  # eye corner on image right
        (-150.0, -150.0, -125.0),  # mouth corner image left
        (150.0, -150.0, -125.0),  # mouth corner image right
    ],
    dtype=np.float64,
)
# expected yaw/pitch/roll plausibility limits (degrees)
HEAD_POSE_LIMITS = (60.0, 50.0, 45.0)
LANDMARK_LEFT_RIGHT_EYE = (33, 263)


def _camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole model with a ~60 deg horizontal field of view (typical webcam)."""
    focal = float(max(width, height)) * 1.1
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def estimate_head_pose(
    points: np.ndarray,
    width: int,
    height: int,
    previous: Optional[HeadPose] = None,
) -> HeadPose:
    """Estimate yaw/pitch/roll from stable facial landmarks with solvePnP.

    The result is reported together with a confidence value derived from the
    reprojection error, angular plausibility and temporal consistency.  Head-pose
    compensation reduces (but does not eliminate) error caused by head movement.
    """
    if points is None or len(points) <= max(HEAD_POSE_LANDMARK_IDS):
        return HeadPose()
    if width <= 0 or height <= 0:
        return HeadPose()

    image_points = np.array(
        [[points[i][0], points[i][1]] for i in HEAD_POSE_LANDMARK_IDS], dtype=np.float64
    )
    if not np.all(np.isfinite(image_points)):
        return HeadPose()
    # degenerate geometry (collapsed or all-zero landmarks) cannot define a pose
    extent = image_points.max(axis=0) - image_points.min(axis=0)
    if float(extent[0]) < 5.0 or float(extent[1]) < 5.0:
        return HeadPose()

    camera = _camera_matrix(width, height)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = None, None, None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            HEAD_POSE_MODEL,
            image_points,
            camera,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                HEAD_POSE_MODEL, image_points, camera, dist, flags=cv2.SOLVEPNP_EPNP
            )
    except cv2.error as exc:  # pragma: no cover - depends on OpenCV build
        LOGGER.debug("solvePnP failed: %s", exc)
        ok = False
    if not ok or rvec is None or tvec is None:
        return HeadPose()

    rotation, _ = cv2.Rodrigues(rvec)
    # Euler angles (ZYX convention) from the object->camera rotation matrix.
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(float(rotation[2, 1]), float(rotation[2, 2])))
        yaw = math.degrees(math.atan2(-float(rotation[2, 0]), sy))
        roll = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    else:  # gimbal lock
        pitch = math.degrees(math.atan2(-float(rotation[1, 2]), float(rotation[1, 1])))
        yaw = math.degrees(math.atan2(-float(rotation[2, 0]), sy))
        roll = 0.0

    projected, _ = cv2.projectPoints(
        HEAD_POSE_MODEL, rvec, tvec, camera, dist
    )
    error = float(
        np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1))
    )

    pose = HeadPose(
        yaw=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
        valid=True,
        reprojection_error=error,
    )

    # reprojection quality: ~2 px is excellent, >= 25 px is unusable
    error_term = float(np.clip(1.0 - error / 25.0, 0.0, 1.0))
    # plausibility: how far the angles sit inside their expected limits
    plausibility = float(
        np.clip(
            min(
                1.0 - abs(pose.yaw) / HEAD_POSE_LIMITS[0],
                1.0 - abs(pose.pitch) / HEAD_POSE_LIMITS[1],
                1.0 - abs(pose.roll) / HEAD_POSE_LIMITS[2],
            ),
            0.0,
            1.0,
        )
    )
    stability = 1.0
    if previous is not None and previous.valid:
        delta = max(abs(pose.yaw - previous.yaw), abs(pose.pitch - previous.pitch))
        stability = float(np.clip(1.0 - max(0.0, delta - 6.0) / 30.0, 0.3, 1.0))
    pose.confidence = float(np.clip(0.5 * error_term + 0.3 * plausibility + 0.2 * stability, 0.0, 1.0))
    return pose


def landmark_confidence(
    points: np.ndarray,
    width: int,
    height: int,
    previous_points: Optional[np.ndarray] = None,
) -> float:
    """Reliability of the facial landmarks: presence, geometry and temporal stability."""
    if points is None or len(points) < max(i for i in LANDMARK_LEFT_RIGHT_EYE) + 1:
        return 0.0
    left = points[LANDMARK_LEFT_RIGHT_EYE[0]]
    right = points[LANDMARK_LEFT_RIGHT_EYE[1]]
    iod = float(np.linalg.norm(left - right))  # inter-ocular distance in px
    if iod <= 1.0:
        return 0.0
    # face must occupy a reasonable part of the frame
    size_term = float(np.clip((iod / max(width, 1)) / 0.12, 0.0, 1.0))
    # temporal stability: displacement relative to face size (blink/landmark noise)
    stability = 1.0
    if previous_points is not None and len(previous_points) == len(points):
        shift = float(np.mean(np.linalg.norm(points - previous_points, axis=1)))
        stability = float(np.clip(1.0 - (shift / iod) / 0.12, 0.0, 1.0))
    return float(np.clip(0.45 * size_term + 0.55 * stability, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature vector + confidence-weighted eye fusion
# ---------------------------------------------------------------------------
def eye_aperture_stats(eye: "EyeObservation") -> Tuple[float, float]:
    """(width, height) of the eye aperture in frame pixels."""
    contour = eye.contour_frame
    if contour is None or len(contour) < 3:
        return (0.0, 0.0)
    x0, y0 = contour.min(axis=0)
    x1, y1 = contour.max(axis=0)
    return (float(x1 - x0), float(y1 - y0))


def eye_feature_offsets(eye: "EyeObservation") -> Tuple[float, float, float]:
    """Eye-centred pupil offset (normalized by aperture size) + raw quality flags.

    Returns (offset_x, offset_y, aperture_area) with offsets in approx. [-1, 1].
    """
    if eye.pupil_frame is None or eye.contour_frame is None or len(eye.contour_frame) < 3:
        return (0.0, 0.0, 0.0)
    center = eye.contour_frame.mean(axis=0)
    width, height = eye_aperture_stats(eye)
    if width <= 1.0 or height <= 1.0:
        return (0.0, 0.0, 0.0)
    dx = float((eye.pupil_frame[0] - center[0]) / (width / 2.0))
    dy = float((eye.pupil_frame[1] - center[1]) / (height / 2.0))
    aperture_area = float(
        abs(cv2.contourArea(np.ascontiguousarray(eye.contour_frame, dtype=np.float32)))
    )
    return (
        float(np.clip(dx, -1.5, 1.5)),
        float(np.clip(dy, -1.5, 1.5)),
        aperture_area,
    )


def build_feature_vector(
    eyes: Sequence["EyeObservation"],
    head_pose: Optional[HeadPose],
) -> np.ndarray:
    """Build the normalized gaze feature vector.

    Missing eyes are imputed with neutral values; the per-eye quality scores keep
    the fusion stage from trusting the imputed channels.
    """
    features = np.full(FEATURE_DIM, 0.0, dtype=np.float64)
    sides = {"image-left": 0, "image-right": 2}
    for eye in eyes:
        base = sides.get(eye.side)
        if base is None or eye.gaze is None:
            continue
        offset_x, offset_y, _ = eye_feature_offsets(eye)
        features[base + 0] = offset_x
        features[base + 1] = offset_y
        features[base + 4] = float(np.clip(eye.gaze[0], 0.0, 1.0))
        features[base + 5] = float(np.clip(eye.gaze[1], 0.0, 1.0))
    # irises default to the centre when the eye is missing
    if features[4] == 0.0 and features[5] == 0.0:
        features[4] = NEUTRAL_IRIS
        features[5] = NEUTRAL_IRIS
    if features[6] == 0.0 and features[7] == 0.0:
        features[6] = NEUTRAL_IRIS
        features[7] = NEUTRAL_IRIS
    if head_pose is not None and head_pose.valid:
        features[8] = head_pose.yaw / HEAD_POSE_SCALE_DEG
        features[9] = head_pose.pitch / HEAD_POSE_SCALE_DEG
        features[10] = head_pose.roll / HEAD_POSE_SCALE_DEG
    return features


def eye_quality(
    eye: "EyeObservation",
    previous_gaze: Optional[Tuple[float, float]] = None,
    pose_confidence: Optional[float] = None,
) -> float:
    """Per-eye quality in [0, 1] used as the eye-fusion weight.

    Combines pupil-detection confidence, pupil contour quality, landmark/geometry
    stability, image quality (local contrast) and temporal consistency.
    """
    if eye.gaze is None or eye.pupil_rect is None:
        return 0.0

    pupil_conf = float(np.clip(eye.confidence, 0.0, 1.0))

    # pupil contour quality: contour must sit inside the aperture and not on its border
    contour_ok = 1.0
    if eye.contour_rect is not None and len(eye.contour_rect) >= 3:
        x0, y0 = eye.contour_rect.min(axis=0)
        x1, y1 = eye.contour_rect.max(axis=0)
        margin = 2.0
        if x0 < margin or y0 < margin or x1 > eye.rect.shape[1] - margin or y1 > eye.rect.shape[0] - margin:
            contour_ok = 0.45
    else:
        contour_ok = 0.3

    # landmark geometry: aperture should be a plausible eye shape
    width, height = eye_aperture_stats(eye)
    geometry = 1.0
    if width < 4.0 or height < 3.0:
        geometry = 0.2
    else:
        aspect = width / max(height, 1.0)
        if not 2.0 <= aspect <= 9.0:
            geometry = 0.5

    # image quality: contrast inside the eye crop (low contrast => poor detection)
    gray = cv2.cvtColor(eye.rect, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    image_quality = float(np.clip(contrast / 35.0, 0.0, 1.0))

    # temporal consistency of this eye's gaze estimate
    temporal = 1.0
    if previous_gaze is not None:
        delta = math.hypot(eye.gaze[0] - previous_gaze[0], eye.gaze[1] - previous_gaze[1])
        temporal = float(np.clip(1.0 - max(0.0, delta - 0.25) / 0.5, 0.0, 1.0))

    # visibility: how much of the aperture actually contains usable pixels
    visible = 1.0
    if eye.aperture_mask is not None and eye.aperture_mask.size > 0:
        coverage = float((eye.aperture_mask > 0).mean())
        visible = float(np.clip(coverage / 0.35, 0.0, 1.0))

    score = (
        0.35 * pupil_conf
        + 0.15 * contour_ok
        + 0.15 * geometry
        + 0.15 * image_quality
        + 0.10 * temporal
        + 0.10 * visible
    )
    if pose_confidence is not None:
        score *= float(np.clip(pose_confidence, 0.4, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def fuse_eye_gazes(
    eyes: Sequence["EyeObservation"],
    qualities: Dict[str, float],
    config: Optional[GazeConfig] = None,
) -> Optional[Tuple[float, float]]:
    """Confidence-weighted fusion of the two eyes (never a plain average).

        fused = (left*q_left + right*q_right) / (q_left + q_right)

    Returns None when both quality values are extremely low, so the sample can be
    treated as invalid instead of silently producing a plausible-looking point.
    """
    config = config or GazeConfig()
    valid: List[Tuple[Tuple[float, float], float]] = []
    for eye in eyes:
        if eye.gaze is None:
            continue
        quality = float(qualities.get(eye.side, 0.0))
        if quality <= 0.0:
            continue
        valid.append((eye.gaze, quality))
    if not valid:
        return None
    total = sum(q for _, q in valid)
    if total < config.min_total_eye_quality:
        LOGGER.debug("eye fusion rejected: total quality %.3f", total)
        return None
    gx = sum(g[0] * q for g, q in valid) / total
    gy = sum(g[1] * q for g, q in valid) / total
    return (float(gx), float(gy))


def temporal_confidence(
    current: Optional[Tuple[float, float]],
    previous: Optional[Tuple[float, float]],
    elapsed_s: float,
    config: Optional[GazeConfig] = None,
) -> float:
    """Consistency with recent gaze estimates (1.0 = unchanged, -> 0 = implausible jump)."""
    config = config or GazeConfig()
    if current is None:
        return 0.0
    if previous is None or elapsed_s <= 0.0:
        return 1.0
    delta = math.hypot(current[0] - previous[0], current[1] - previous[1])
    # a real saccade covers ~0.3 of the normalized range in well under 100 ms
    allowed = 0.3 + elapsed_s * 4.0
    if delta <= allowed:
        return 1.0
    ratio = allowed / max(delta, 1e-6)
    return float(np.clip(ratio, 0.0, 1.0))



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
    quality: float = 0.0


@dataclass
class GazeResult:
    """One processed frame.

    `gaze` is the confidence-weighted fused pupil/iris position in normalized eye
    coordinates.  `features` is the formal feature vector consumed by the mapping
    models.  `confidences` separates pupil / landmark / head-pose / gaze / temporal
    reliability instead of collapsing everything into one number.
    """

    gaze: Optional[Tuple[float, float]]
    confidence: float
    eyes: List[EyeObservation] = field(default_factory=list)
    inference_ms: float = 0.0
    features: Optional[np.ndarray] = None
    head_pose: Optional[HeadPose] = None
    confidences: Optional[ConfidenceBreakdown] = None

    @property
    def valid(self) -> bool:
        return self.gaze is not None and self.features is not None


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
        config: Optional[GazeConfig] = None,
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
        self.config = config or GazeConfig()
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
        # temporal state used by the confidence model
        self._previous_points: Optional[np.ndarray] = None
        self._previous_gaze: Optional[Tuple[float, float]] = None
        self._previous_pose: Optional[HeadPose] = None
        self._previous_eye_gaze: Dict[str, Tuple[float, float]] = {}
        self._previous_time: Optional[float] = None

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
            head_pose = estimate_head_pose(points, width, height, self._previous_pose)
            landmark_conf = landmark_confidence(points, width, height, self._previous_points)
            self._remember(points, None, head_pose)
            breakdown = ConfidenceBreakdown(
                landmark=landmark_conf,
                head_pose=head_pose.confidence if head_pose.valid else 0.0,
            )
            return GazeResult(None, 0.0, [], inference_ms, None, head_pose, breakdown)

        eyes.sort(key=lambda eye: float(eye.contour_frame[:, 0].mean()))
        eyes[0].side = "image-left"
        if len(eyes) > 1:
            eyes[1].side = "image-right"

        head_pose = estimate_head_pose(points, width, height, self._previous_pose)
        landmark_conf = landmark_confidence(points, width, height, self._previous_points)

        # --- per-eye quality (drives confidence-weighted fusion) -------------
        qualities: Dict[str, float] = {}
        for eye in eyes:
            eye.quality = eye_quality(
                eye,
                self._previous_eye_gaze.get(eye.side),
                head_pose.confidence if head_pose.valid else None,
            )
            qualities[eye.side] = eye.quality

        fused = fuse_eye_gazes(eyes, qualities, self.config)
        features = build_feature_vector(eyes, head_pose)

        now = time.monotonic()
        elapsed = 0.0 if self._previous_time is None else max(now - self._previous_time, 0.0)
        temporal_conf = temporal_confidence(fused, self._previous_gaze, elapsed, self.config)

        with_gaze = [eye for eye in eyes if eye.gaze is not None]
        if with_gaze:
            weight_total = sum(qualities.get(eye.side, 0.0) for eye in with_gaze)
            if weight_total > 0.0:
                pupil_conf = sum(
                    qualities.get(eye.side, 0.0) ** 2 for eye in with_gaze
                ) / weight_total
            else:
                pupil_conf = 0.0
        else:
            pupil_conf = 0.0

        head_conf = head_pose.confidence if head_pose.valid else 0.0
        completeness = 1.0 if len(with_gaze) >= 2 else 0.6
        gaze_conf = float(
            np.clip(
                0.45 * pupil_conf
                + 0.25 * landmark_conf
                + 0.20 * head_conf
                + 0.10 * completeness,
                0.0,
                1.0,
            )
        )
        breakdown = ConfidenceBreakdown(
            pupil=float(pupil_conf),
            landmark=float(landmark_conf),
            head_pose=float(head_conf),
            gaze=gaze_conf,
            temporal=temporal_conf,
        )
        self._remember(points, fused, head_pose, eyes)

        if fused is None:
            return GazeResult(None, 0.0, eyes, inference_ms, features, head_pose, breakdown)
        confidence = breakdown.combined(self.config.confidence_weights)
        return GazeResult(fused, confidence, eyes, inference_ms, features, head_pose, breakdown)

    def _remember(
        self,
        points: np.ndarray,
        gaze: Optional[Tuple[float, float]],
        head_pose: HeadPose,
        eyes: Sequence[EyeObservation] = (),
    ) -> None:
        self._previous_points = points
        if gaze is not None:
            self._previous_gaze = gaze
        self._previous_eye_gaze = {
            eye.side: eye.gaze for eye in eyes if eye.side and eye.gaze is not None
        }
        self._previous_pose = head_pose if head_pose.valid else self._previous_pose
        self._previous_time = time.monotonic()

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


# ---------------------------------------------------------------------------
# Calibration target layouts (normalized screen coordinates)
# ---------------------------------------------------------------------------
def _ordered_grid(points: Tuple[Tuple[float, float], ...]) -> Tuple[Tuple[float, float], ...]:
    """Start at the screen centre, then sweep the outer ring (easier to follow)."""
    ordered = [p for p in points if p == (0.5, 0.5)]
    ordered += [p for p in points if p != (0.5, 0.5)]
    return tuple(ordered)


_GRID_9 = _ordered_grid(
    tuple(
        (x, y)
        for y in (0.1, 0.5, 0.9)
        for x in (0.1, 0.5, 0.9)
    )
)
TARGET_LAYOUTS: Dict[int, Tuple[Tuple[float, float], ...]] = {
    # lightweight legacy layout: four corners + centre
    5: ((0.5, 0.5), (0.08, 0.08), (0.92, 0.08), (0.92, 0.92), (0.08, 0.92)),
    9: _GRID_9,
    # 9-point grid plus four inner-diamond points for interior coverage
    13: _GRID_9
    + ((0.5, 0.25), (0.25, 0.5), (0.75, 0.5), (0.5, 0.75)),
}


def robust_median(rows: Sequence[np.ndarray], z_threshold: float = 3.0) -> Optional[np.ndarray]:
    """Median of a set of feature vectors after dropping extreme outliers.

    Uses a MAD-based z score so a handful of contaminated frames cannot drag the
    representative sample away from the target.
    """
    if not rows:
        return None
    data = np.vstack([np.asarray(r, dtype=np.float64).ravel() for r in rows])
    if data.shape[0] == 1:
        return data[0]
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    scale = np.where(mad > 1e-6, 1.4826 * mad, np.std(data, axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    distance = np.mean(np.abs(data - median) / scale, axis=1)
    keep = distance <= z_threshold
    if not np.any(keep):
        return median
    return np.median(data[keep], axis=0)


# ---------------------------------------------------------------------------
# Mapping models (Phase 2): affine / polynomial / small MLP
# ---------------------------------------------------------------------------
# Affine stays the transparent baseline.  Polynomial and MLP are optional and are
# only "better" if measured validation data says so.
@dataclass
class MappingSample:
    """Features -> known screen coordinate (one accepted calibration frame)."""

    features: np.ndarray
    target: Tuple[float, float]
    confidence: float = 1.0


def polynomial_features(x: np.ndarray) -> np.ndarray:
    """Second-order expansion: linear terms, squares and pairwise interactions."""
    x = np.asarray(x, dtype=np.float64)
    squares = x ** 2
    interactions = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            interactions.append(x[:, i] * x[:, j])
    if interactions:
        return np.hstack([x, squares, np.column_stack(interactions)])
    return np.hstack([x, squares])


class MappingModel:
    """Base class for feature -> screen-coordinate regression models."""

    name = "base"

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        self.feature_dim = int(feature_dim)

    def fit(self, x: np.ndarray, y: np.ndarray) -> bool:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> Dict[str, np.ndarray]:
        return {}

    def load_params(self, data: Dict[str, np.ndarray]) -> None:
        return None

    @property
    def fitted(self) -> bool:
        return False


class AffineModel(MappingModel):
    """X = a1*f1 + ... + b1, Y = c1*f1 + ... + b2 (least squares)."""

    name = "affine"

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__(feature_dim)
        self.coef: Optional[np.ndarray] = None  # (2, feature_dim + 1)

    @property
    def fitted(self) -> bool:
        return self.coef is not None

    def fit(self, x: np.ndarray, y: np.ndarray) -> bool:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 3:
            return False
        design = np.hstack([x, np.ones((x.shape[0], 1))])
        if x.shape[0] < design.shape[1]:
            LOGGER.warning(
                "affine fit is under-determined (%d samples, %d parameters)",
                x.shape[0],
                design.shape[1],
            )
        solution, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 3 or not np.all(np.isfinite(solution)):
            return False
        self.coef = solution.T
        self.feature_dim = int(x.shape[1])
        return True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("affine model is not fitted")
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        design = np.hstack([x, np.ones((x.shape[0], 1))])
        return design @ self.coef.T

    def params(self) -> Dict[str, np.ndarray]:
        return {"coef": np.asarray(self.coef, dtype=np.float64)}

    def load_params(self, data: Dict[str, np.ndarray]) -> None:
        self.coef = np.asarray(data["coef"], dtype=np.float64)
        self.feature_dim = self.coef.shape[1] - 1


class PolynomialModel(MappingModel):
    """Second-order polynomial regression with light ridge regularization."""

    name = "polynomial"

    def __init__(self, feature_dim: int = FEATURE_DIM, ridge: float = 1e-2) -> None:
        super().__init__(feature_dim)
        self.ridge = float(ridge)
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.coef: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return self.coef is not None

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.scale is not None
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def fit(self, x: np.ndarray, y: np.ndarray) -> bool:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 4:
            return False
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        design = np.hstack([polynomial_features(self._standardize(x)), np.ones((x.shape[0], 1))])
        reg = np.eye(design.shape[1]) * self.ridge
        reg[-1, -1] = 0.0  # never regularize the intercept
        try:
            solution = np.linalg.solve(design.T @ design + reg, design.T @ y)
        except np.linalg.LinAlgError:
            return False
        if not np.all(np.isfinite(solution)):
            return False
        self.coef = solution
        self.feature_dim = int(x.shape[1])
        return True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("polynomial model is not fitted")
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        design = np.hstack([polynomial_features(self._standardize(x)), np.ones((x.shape[0], 1))])
        return design @ self.coef

    def params(self) -> Dict[str, np.ndarray]:
        return {
            "coef": np.asarray(self.coef, dtype=np.float64),
            "mean": np.asarray(self.mean, dtype=np.float64),
            "scale": np.asarray(self.scale, dtype=np.float64),
        }

    def load_params(self, data: Dict[str, np.ndarray]) -> None:
        self.coef = np.asarray(data["coef"], dtype=np.float64)
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)
        # infer feature dimension from the stored expansion size
        d = 1
        while True:
            expected = d + d + d * (d - 1) // 2 + 1
            if expected == self.coef.shape[0]:
                self.feature_dim = d
                break
            d += 1
            if d > 64:
                break


class MLPModel(MappingModel):
    """Small fully connected network: Dense -> tanh -> Dense -> (X, Y).

    Optional (Priority 3): implemented in plain NumPy so no extra dependency is
    introduced, and it is never required for the basic system to run.
    """

    name = "mlp"

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden: int = 16,
        epochs: int = 400,
        learning_rate: float = 5e-3,
        seed: int = 7,
        screen: Tuple[int, int] = (1920, 1080),
    ) -> None:
        super().__init__(feature_dim)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.screen = (int(screen[0]), int(screen[1]))
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.w0: Optional[np.ndarray] = None
        self.b0: Optional[np.ndarray] = None
        self.w1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.train_loss = float("nan")

    @property
    def fitted(self) -> bool:
        return self.w0 is not None and self.w1 is not None

    def fit(self, x: np.ndarray, y: np.ndarray) -> bool:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 8:
            return False
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        xs = (x - self.mean) / self.scale
        ys = y / np.array([self.screen[0], self.screen[1]], dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        limit0 = math.sqrt(6.0 / (xs.shape[1] + self.hidden))
        limit1 = math.sqrt(6.0 / (self.hidden + 2))
        w0 = rng.uniform(-limit0, limit0, (xs.shape[1], self.hidden))
        b0 = np.zeros(self.hidden)
        w1 = rng.uniform(-limit1, limit1, (self.hidden, 2))
        b1 = np.zeros(2)

        # Adam state
        moments = [np.zeros_like(p) for p in (w0, b0, w1, b1)]
        velocities = [np.zeros_like(p) for p in (w0, b0, w1, b1)]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = self.learning_rate
        reg = 1e-4
        loss = float("nan")
        for step in range(1, self.epochs + 1):
            hidden = np.tanh(xs @ w0 + b0)
            out = hidden @ w1 + b1
            error = out - ys
            loss = float(np.mean(np.sum(error ** 2, axis=1)) + reg * (np.sum(w0 ** 2) + np.sum(w1 ** 2)))

            grad_out = 2.0 * error / xs.shape[0]
            gw1 = hidden.T @ grad_out + 2.0 * reg * w1
            gb1 = grad_out.sum(axis=0)
            grad_hidden = grad_out @ w1.T * (1.0 - hidden ** 2)
            gw0 = xs.T @ grad_hidden + 2.0 * reg * w0
            gb0 = grad_hidden.sum(axis=0)

            grads = [gw0, gb0, gw1, gb1]
            params = [w0, b0, w1, b1]
            for i in range(4):
                moments[i] = beta1 * moments[i] + (1.0 - beta1) * grads[i]
                velocities[i] = beta2 * velocities[i] + (1.0 - beta2) * grads[i] ** 2
                m_hat = moments[i] / (1.0 - beta1 ** step)
                v_hat = velocities[i] / (1.0 - beta2 ** step)
                params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            w0, b0, w1, b1 = params
            if not np.isfinite(loss):
                return False

        self.w0, self.b0, self.w1, self.b1 = w0, b0, w1, b1
        self.train_loss = loss
        self.feature_dim = int(x.shape[1])
        return True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("mlp model is not fitted")
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        xs = (x - self.mean) / self.scale
        hidden = np.tanh(xs @ self.w0 + self.b0)
        out = hidden @ self.w1 + self.b1
        return out * np.array([self.screen[0], self.screen[1]], dtype=np.float64)

    def params(self) -> Dict[str, np.ndarray]:
        return {
            "w0": np.asarray(self.w0, dtype=np.float64),
            "b0": np.asarray(self.b0, dtype=np.float64),
            "w1": np.asarray(self.w1, dtype=np.float64),
            "b1": np.asarray(self.b1, dtype=np.float64),
            "mean": np.asarray(self.mean, dtype=np.float64),
            "scale": np.asarray(self.scale, dtype=np.float64),
        }

    def load_params(self, data: Dict[str, np.ndarray]) -> None:
        self.w0 = np.asarray(data["w0"], dtype=np.float64)
        self.b0 = np.asarray(data["b0"], dtype=np.float64)
        self.w1 = np.asarray(data["w1"], dtype=np.float64)
        self.b1 = np.asarray(data["b1"], dtype=np.float64)
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)
        self.feature_dim = self.w0.shape[0]


def build_mapping_model(name: str, config: Optional[GazeConfig] = None) -> MappingModel:
    config = config or GazeConfig()
    name = (name or "affine").lower()
    if name == "affine":
        return AffineModel()
    if name == "polynomial":
        return PolynomialModel(ridge=config.polynomial_ridge)
    if name == "mlp":
        return MLPModel(
            hidden=config.mlp_hidden,
            epochs=config.mlp_epochs,
            learning_rate=config.mlp_learning_rate,
            seed=config.random_seed,
            screen=(config.screen_width, config.screen_height),
        )
    raise ValueError(f"unknown mapping model: {name!r}")


class ScreenMapper:
    """Maps the gaze feature vector onto pixel coordinates on a screen.

    Responsibilities: model fitting/loading, feature->screen regression, outlier
    rejection and configurable EMA smoothing.  When no calibration exists the
    legacy raw-gain mapping around the screen centre is used.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        gain: float = 3.0,
        smoothing: float = 0.3,
        path: str = "calibration.npz",
        auto_center: bool = True,
        warmup_frames: int = 45,
        config: Optional[GazeConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.gain = float(gain)
        self.config = config or GazeConfig()
        self.smoothing = float(smoothing) if smoothing is not None else self.config.ema_alpha
        self.path = path
        self.auto_center = bool(auto_center)
        self.warmup_frames = int(warmup_frames)
        self.model_name = model_name or self.config.mapping_model
        self.model: Optional[MappingModel] = None
        self.samples: List[MappingSample] = []
        self.metadata: Dict[str, object] = {}
        self._smoothed: Optional[Tuple[float, float]] = None
        self._last_raw: Optional[Tuple[float, float]] = None
        self._last_time: Optional[float] = None
        self._baseline: Optional[Tuple[float, float]] = None
        self._warm_sum = np.zeros(2, dtype=np.float64)
        self._warm_n = 0
        self.stats: Dict[str, int] = {
            "accepted": 0,
            "rejected_low_confidence": 0,
            "rejected_off_screen": 0,
            "rejected_jump": 0,
            "rejected_missing_features": 0,
        }
        self.load()

    # -- state ------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.model is not None and self.model.fitted

    @property
    def mapping_name(self) -> str:
        if self.calibrated:
            return f"{self.model_name} calibration"
        if self.auto_center and self._baseline is not None:
            return f"raw gain={self.gain:.1f} (auto-centred)"
        return f"raw gain={self.gain:.1f}"

    @property
    def valid_sample_rate(self) -> float:
        total = sum(self.stats.values())
        return float(self.stats["accepted"] / total) if total else 0.0

    def recenter(self) -> None:
        self._baseline = None
        self._warm_sum[:] = 0.0
        self._warm_n = 0
        self._smoothed = None

    def _update_baseline(self, gx: float, gy: float) -> Tuple[float, float]:
        if not auto_center_safe(self.auto_center, self.calibrated):
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

    # -- persistence ------------------------------------------------------
    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            data = np.load(self.path, allow_pickle=False)
            keys = set(data.files)
            width = int(data["width"]) if "width" in keys else -1
            height = int(data["height"]) if "height" in keys else -1
            if width != self.width or height != self.height:
                LOGGER.warning("calibration file screen size %dx%d does not match %dx%d",
                               width, height, self.width, self.height)
                return
            if "model" in keys:
                name = str(np.asarray(data["model"]).item())
                model = build_mapping_model(name, self.config)
                model.load_params({k: data[k] for k in model.params() if k in keys})
                if model.fitted:
                    self.model = model
                    self.model_name = name
                    if "samples" in keys and "targets" in keys:
                        feats = np.asarray(data["samples"], dtype=np.float64)
                        tgts = np.asarray(data["targets"], dtype=np.float64)
                        self.samples = [
                            MappingSample(f, (float(t[0]), float(t[1])))
                            for f, t in zip(feats, tgts)
                        ]
                return
            if "affine" in keys:  # legacy file: 2-feature affine on (gx, gy)
                affine = np.asarray(data["affine"], dtype=np.float64)
                if affine.shape == (2, 3):
                    model = AffineModel(feature_dim=2)
                    model.coef = affine
                    self.model = model
                    self.model_name = "affine"
        except Exception as exc:
            LOGGER.warning("could not load calibration %s: %s", self.path, exc)
            self.model = None

    def save(self) -> str:
        if not self.calibrated:
            raise RuntimeError("nothing to save")
        payload: Dict[str, np.ndarray] = {
            "model": np.array(self.model_name),
            "width": np.array(self.width),
            "height": np.array(self.height),
            "feature_dim": np.array(self.model.feature_dim),
        }
        if self.samples:
            payload["samples"] = np.array([s.features for s in self.samples], dtype=np.float32)
            payload["targets"] = np.array([s.target for s in self.samples], dtype=np.float32)
        payload.update(self.model.params())
        np.savez(self.path, **payload)
        return self.path

    # -- fitting ----------------------------------------------------------
    @staticmethod
    def _coerce_samples(
        samples: Sequence,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Accept MappingSample rows (new) or legacy (gx, gy, x, y) tuples."""
        features: List[np.ndarray] = []
        targets: List[Tuple[float, float]] = []
        for sample in samples:
            if isinstance(sample, MappingSample):
                features.append(np.asarray(sample.features, dtype=np.float64).ravel())
                targets.append((float(sample.target[0]), float(sample.target[1])))
            elif isinstance(sample, (tuple, list)) and len(sample) == 4:
                # legacy: (gx, gy, screen_x, screen_y)
                features.append(np.array([sample[0], sample[1]], dtype=np.float64))
                targets.append((float(sample[2]), float(sample[3])))
            elif isinstance(sample, (tuple, list)) and len(sample) == 2:
                feature, target = sample
                features.append(np.asarray(feature, dtype=np.float64).ravel())
                targets.append((float(target[0]), float(target[1])))
            else:
                return None
        if not features:
            return None
        lengths = {len(f) for f in features}
        if len(lengths) != 1:
            return None
        return np.vstack(features), np.array(targets, dtype=np.float64)

    def fit(self, samples: Sequence, model_name: Optional[str] = None) -> bool:
        """Fit the selected mapping model to Features -> Screen Coordinate samples."""
        coerced = self._coerce_samples(samples)
        if coerced is None:
            return False
        x, y = coerced
        if len(x) < 4:
            return False
        span = x.max(axis=0) - x.min(axis=0)
        if int((span >= self.config.min_calibration_span).sum()) < 2:
            LOGGER.warning("calibration rejected: feature spread too small")
            return False
        name = model_name or self.model_name
        model = build_mapping_model(name, self.config)
        if isinstance(model, MLPModel):
            model.screen = (self.width, self.height)
        if not model.fit(x, y):
            LOGGER.warning("fit failed for model %s", name)
            return False
        self.model = model
        self.model_name = name
        self.samples = [
            MappingSample(f, (float(t[0]), float(t[1]))) for f, t in zip(x, y)
        ]
        self._smoothed = None
        self._last_raw = None
        self._last_time = None
        try:
            self.save()
        except Exception as exc:
            LOGGER.warning("could not save calibration: %s", exc)
        return True

    def reset(self) -> None:
        self.model = None
        self.samples = []
        self.metadata = {}
        self._smoothed = None
        self._last_raw = None
        self._last_time = None
        self.stats = {key: 0 for key in self.stats}
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass

    # -- prediction -------------------------------------------------------
    def _reject(self, reason: str) -> None:
        self.stats[reason] = self.stats.get(reason, 0) + 1

    def map(
        self,
        gaze: Optional[Tuple[float, float]],
        features: Optional[np.ndarray] = None,
        confidence: float = 1.0,
    ) -> Optional[Tuple[float, float]]:
        """Map one gaze sample to screen pixels, or None when it is rejected.

        Rejection reasons (outlier rejection, item 17): missing features, low
        confidence, off-screen prediction, implausible jump.
        """
        if gaze is None:
            return None
        if confidence < self.config.confidence_threshold:
            self._reject("rejected_low_confidence")
            return None

        gx, gy = gaze
        x: Optional[float] = None
        y: Optional[float] = None

        if self.calibrated:
            if features is None or len(features) != self.model.feature_dim:
                if self.model.feature_dim == 2:
                    vector = np.array([gx, gy], dtype=np.float64)
                else:
                    self._reject("rejected_missing_features")
                    return None
            else:
                vector = np.asarray(features, dtype=np.float64).ravel()
            predicted = self.model.predict(vector)[0]
            x, y = float(predicted[0]), float(predicted[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                self._reject("rejected_off_screen")
                return None
            # physically implausible: far outside the screen
            margin_x = 0.10 * self.width
            margin_y = 0.10 * self.height
            if x < -margin_x or x > self.width + margin_x or y < -margin_y or y > self.height + margin_y:
                self._reject("rejected_off_screen")
                return None
        else:
            base_x, base_y = self._update_baseline(gx, gy)
            x = (0.5 + (gx - base_x) * self.gain) * self.width
            y = (0.5 + (gy - base_y) * self.gain) * self.height

        x = float(min(max(x, 0.0), self.width - 1))
        y = float(min(max(y, 0.0), self.height - 1))

        now = time.monotonic()
        if self._last_raw is not None and self._last_time is not None:
            jump = math.hypot(x - self._last_raw[0], y - self._last_raw[1])
            max_jump = self.config.outlier_max_jump_px or (
                1.5 * math.hypot(self.width, self.height)
            )
            if jump > max_jump:
                # physically implausible displacement: real saccades stay inside
                # the screen, so they can never reach this threshold
                self._reject("rejected_jump")
                return None
            if self.config.outlier_max_speed_px_s > 0.0:
                dt = max(now - self._last_time, 1e-3)
                if jump / dt > self.config.outlier_max_speed_px_s and jump > max_jump * 0.5:
                    self._reject("rejected_jump")
                    return None
        self._last_raw = (x, y)
        self._last_time = now
        self.stats["accepted"] += 1

        # configurable EMA smoothing (does not hardcode the factor)
        alpha = self.smoothing
        if self._smoothed is None or alpha >= 1.0:
            self._smoothed = (x, y)
        else:
            self._smoothed = (
                alpha * x + (1.0 - alpha) * self._smoothed[0],
                alpha * y + (1.0 - alpha) * self._smoothed[1],
            )
        return self._smoothed


def auto_center_safe(auto_center: bool, calibrated: bool) -> bool:
    return bool(auto_center) and not calibrated


class ScreenCalibrator:
    """Multi-point, multi-sample calibration run.

    Layouts: 5 (lightweight legacy), 9 (research default) and 13 points.
    Every target is stabilized first, then several frames are collected; low
    confidence samples and extreme outliers are rejected, and a representative
    (median) feature vector is stored per target next to the raw accepted samples.
    """

    TARGETS = TARGET_LAYOUTS[5]

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        tolerance: float = 0.1,
        dwell_frames: int = 10,
        config: Optional[GazeConfig] = None,
        points: Optional[int] = None,
        samples_per_point: Optional[int] = None,
    ) -> None:
        self.config = config or GazeConfig()
        self.width = int(width)
        self.height = int(height)
        self.tolerance = float(tolerance)
        self.dwell_frames = int(dwell_frames)
        self.n_points = int(points or self.config.calibration_points)
        if self.n_points not in TARGET_LAYOUTS:
            raise ValueError(f"points must be one of {sorted(TARGET_LAYOUTS)}")
        self.targets = TARGET_LAYOUTS[self.n_points]
        self.samples_per_point = int(
            samples_per_point or self.config.calibration_samples_per_point
        )
        self.stabilize_frames = int(self.config.calibration_stabilize_frames)
        self.active = False
        self.index = 0
        self.hits = 0  # stabilization counter (kept name for backward compatibility)
        self.stabilized = False
        self.collected = 0
        self.rejected = 0
        self.samples: List[MappingSample] = []
        self.captured: List[Tuple[float, float]] = []
        self._current: List[np.ndarray] = []
        self._current_confidences: List[float] = []
        self._point_samples: List[List[np.ndarray]] = [[] for _ in self.targets]
        self.representative: List[Optional[np.ndarray]] = [None] * len(self.targets)

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
        """0..1 arc: stabilization followed by sample collection for this target."""
        if not self.stabilized:
            return min(self.hits / max(self.stabilize_frames, 1), 1.0) * 0.3
        return 0.3 + 0.7 * min(self.collected / max(self.samples_per_point, 1), 1.0)

    @property
    def total_accepted(self) -> int:
        return len(self.samples)

    def start(self) -> None:
        self.active = True
        self.index = 0
        self.hits = 0
        self.stabilized = False
        self.collected = 0
        self.rejected = 0
        self.samples = []
        self.captured = []
        self._current = []
        self._current_confidences = []
        self._point_samples = [[] for _ in self.targets]
        self.representative = [None] * len(self.targets)

    def cancel(self) -> None:
        self.active = False

    def _reset_target_state(self) -> None:
        self.hits = 0
        self.stabilized = False
        self.collected = 0
        self._current = []
        self._current_confidences = []

    def _advance(self) -> str:
        self.captured.append(self.targets[self.index])
        rows = list(self._current)
        self._point_samples[self.index] = rows
        self.representative[self.index] = robust_median(
            rows, self.config.calibration_outlier_z
        )
        self.index += 1
        self._reset_target_state()
        if self.done:
            self.active = False
            return "done"
        return "captured"

    def update(
        self,
        features: Optional[np.ndarray],
        confidence: float = 1.0,
        gaze: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Feed one frame. `gaze` is only used for the legacy tolerance preview."""
        if not self.active:
            return "idle"
        if features is None or confidence < self.config.calibration_min_confidence:
            self.hits = 0
            self.stabilized = False
            self.rejected += 1
            return "invalid"
        if not self.stabilized:
            self.hits += 1
            if self.hits < self.stabilize_frames:
                return "stabilizing"
            self.stabilized = True
        self._current.append(np.asarray(features, dtype=np.float64).ravel())
        self._current_confidences.append(float(confidence))
        self.collected += 1
        if self.collected >= self.samples_per_point:
            return self._advance()
        return "collecting"

    def record(
        self,
        features: Optional[np.ndarray],
        confidence: float = 1.0,
        gaze: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Manual capture: accept what has been collected for the current target."""
        if not self.active or self.done:
            return "idle"
        if features is None:
            return "invalid"
        self.stabilized = True
        if not self._current:
            self._current.append(np.asarray(features, dtype=np.float64).ravel())
            self._current_confidences.append(float(confidence))
            self.collected += 1
        return self._advance()

    def finalize(self) -> List[MappingSample]:
        """Reject extreme outliers per target and return every accepted sample.

        Returns the mapping dataset `Features -> Known Screen Coordinate`; the
        per-target median feature vectors stay available in `self.representative`.
        """
        z_threshold = self.config.calibration_outlier_z
        accepted: List[MappingSample] = []
        for index, target in enumerate(self.targets):
            rows = self._point_samples[index] if index < len(self._point_samples) else []
            if not rows:
                continue
            data = np.vstack([np.asarray(row, dtype=np.float64).ravel() for row in rows])
            median = robust_median(rows, z_threshold)
            if median is None:
                continue
            self.representative[index] = median
            if data.shape[0] > 1:
                mad = np.median(np.abs(data - median), axis=0)
                scale = np.where(mad > 1e-6, 1.4826 * mad, np.std(data, axis=0))
                scale = np.where(scale > 1e-6, scale, 1.0)
                distance = np.mean(np.abs(data - median) / scale, axis=1)
                keep = distance <= z_threshold
                if not np.any(keep):
                    keep = np.ones(data.shape[0], dtype=bool)
            else:
                keep = np.ones(1, dtype=bool)
            self.rejected += int((~keep).sum())
            target_px = (target[0] * self.width, target[1] * self.height)
            for row, kept in zip(data, keep):
                if kept:
                    accepted.append(MappingSample(row, target_px))
        self.samples = accepted
        return accepted



class AttentionHeatmap:
    """Fixation- and confidence-weighted gaze-density heatmap.

    Accumulation -> time-based decay -> Gaussian blur -> normalization ->
    gamma correction -> colormap rendering.

    Units: `sigma` is expressed in **screen pixels** and converted to heatmap-grid
    cells through `sigma_cells = sigma / cell_size`.  Decay is time-based
    (`exp(-dt / tau)`) so the result does not depend on the camera frame rate.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        cell: int = 4,
        sigma: float = 45.0,
        decay_tau: float = 5.0,
        colormap: int = cv2.COLORMAP_TURBO,
        alpha: float = 0.7,
        gamma: float = 0.75,
        min_confidence: float = 0.35,
        config: Optional[GazeConfig] = None,
        initial_capacity: int = 8192,
    ) -> None:
        if config is not None:
            cell = config.heatmap_cell_size
            sigma = config.gaussian_sigma_px
            decay_tau = config.heatmap_decay_tau
            alpha = config.heatmap_alpha
            gamma = config.heatmap_gamma
            min_confidence = config.heatmap_min_confidence
        self.cell = max(int(cell), 1)
        self.width = int(width)
        self.height = int(height)
        self.grid_w = max(self.width // self.cell, 1)
        self.grid_h = max(self.height // self.cell, 1)
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.sigma = float(sigma)  # screen pixels
        self.decay_tau = float(decay_tau)  # seconds
        self.colormap = colormap
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.min_confidence = float(min_confidence)
        self.total_hits = 0
        self.total_weight = 0.0
        self.rejected_low_confidence = 0
        self.paused = False
        self._points = np.zeros((max(initial_capacity, 64), 4), dtype=np.float32)
        self._count = 0

    @property
    def sigma_cells(self) -> float:
        """Gaussian sigma converted from screen pixels into grid cells."""
        return self.sigma / self.cell

    @property
    def kernel_size(self) -> int:
        return int(2 * round(3.0 * self.sigma_cells) + 1)

    @property
    def decay_enabled(self) -> bool:
        return self.decay_tau > 0.0

    def points_matrix(self) -> np.ndarray:
        """Columns: time, x, y, accumulated weight."""
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
        self.total_weight = 0.0
        self.rejected_low_confidence = 0

    def _append_point(self, t: float, x: float, y: float, weight: float) -> None:
        if self._count >= self._points.shape[0]:
            grown = np.zeros((self._points.shape[0] * 2, 4), dtype=np.float32)
            grown[: self._count] = self._points[: self._count]
            self._points = grown
        self._points[self._count] = (t, x, y, weight)
        self._count += 1

    def add(
        self,
        x: float,
        y: float,
        t: float,
        weight: float = 1.0,
        confidence: float = 1.0,
        fixation_weight: float = 1.0,
    ) -> bool:
        """Accumulate one gaze sample.

            heatmap += weight * confidence * fixation_weight

        Samples whose confidence is below `min_confidence` are discarded instead
        of contributing equally with reliable ones.
        """
        if confidence < self.min_confidence:
            self.rejected_low_confidence += 1
            return False
        contribution = float(weight) * float(confidence) * float(fixation_weight)
        if contribution <= 0.0:
            return False
        col = int(x / self.cell)
        row = int(y / self.cell)
        if col < 0 or row < 0 or col >= self.grid_w or row >= self.grid_h:
            return False
        self.grid[row, col] += contribution
        self.total_hits += 1
        self.total_weight += contribution
        self._append_point(t, x, y, contribution)
        return True

    def decay_step(self, dt_seconds: float = 0.0) -> None:
        """Time-based decay: grid *= exp(-dt / tau) (frame-rate independent)."""
        if not self.decay_enabled or dt_seconds <= 0.0:
            return
        self.grid *= float(np.exp(-dt_seconds / self.decay_tau))

    def blurred(self) -> np.ndarray:
        return cv2.GaussianBlur(self.grid, (0, 0), sigmaX=self.sigma_cells)

    def peak(self) -> float:
        return float(self.blurred().max())

    def render(self, background: Optional[np.ndarray] = None) -> np.ndarray:
        blurred = self.blurred()
        peak = float(blurred.max())
        if peak <= 1e-6:
            norm = np.zeros_like(blurred)
        else:
            norm = np.clip(blurred / peak, 0.0, 1.0)
        shaped = np.power(norm, self.gamma)
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
