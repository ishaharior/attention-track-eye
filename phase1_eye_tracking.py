"""Phase 1: eye and head-pose feature extraction.

Crops the eye regions from the webcam feed, localizes the pupil/iris, estimates
the head pose (yaw/pitch/roll) and builds the normalized gaze feature vector that
Phase 2 consumes.  Pupil/iris and facial features are inputs to a calibrated
gaze-estimation model - the pupil position alone is never treated as the gaze.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np

from gaze_core import (
    FEATURE_NAMES,
    GazeConfig,
    GazeResult,
    HeadPose,
    EyeTracker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: eye ROI, pupil tracking, head pose")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument(
        "--pupil",
        choices=("hybrid", "iris", "image"),
        default="hybrid",
        help="pupil detector: MediaPipe iris landmarks, image thresholding, or both",
    )
    parser.add_argument("--no-mirror", action="store_true", help="disable horizontal flip")
    parser.add_argument("--mesh", action="store_true", help="draw the full face mesh tessellation")
    parser.add_argument("--zoom", type=float, default=2.0, help="eye crop zoom factor")
    parser.add_argument("--snapshot", default="phase1_snapshot.jpg", help="snapshot file name")
    parser.add_argument("--show-features", action="store_true", help="print the feature vector")
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit(f"Could not open webcam index {args.camera}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return capture


def format_pose(pose: Optional[HeadPose]) -> str:
    if pose is None or not pose.valid:
        return "yaw=--- pitch=--- roll=---"
    return f"yaw={pose.yaw:+6.1f} pitch={pose.pitch:+6.1f} roll={pose.roll:+6.1f}"


def format_confidences(result: GazeResult) -> str:
    if result.confidences is None:
        return "pup=-.- lm=-.- hp=-.- gz=-.- tmp=-.-"
    c = result.confidences
    return (
        f"pup={c.pupil:.2f} lm={c.landmark:.2f} hp={c.head_pose:.2f} "
        f"gz={c.gaze:.2f} tmp={c.temporal:.2f}"
    )


def draw_feature_panel(canvas: np.ndarray, result: GazeResult) -> None:
    if result.features is None:
        return
    lines = ["feature vector:"]
    values = np.round(result.features, 3)
    row = ", ".join(f"{name}={value:+.2f}" for name, value in zip(FEATURE_NAMES[:4], values[:4]))
    lines.append("  " + row)
    row = ", ".join(f"{name}={value:+.2f}" for name, value in zip(FEATURE_NAMES[4:8], values[4:8]))
    lines.append("  " + row)
    row = ", ".join(f"{name}={value:+.2f}" for name, value in zip(FEATURE_NAMES[8:], values[8:]))
    lines.append("  " + row)
    y = 96
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 120), 1, cv2.LINE_AA)
        y += 20


def main() -> int:
    args = parse_args()
    config = GazeConfig()
    try:
        tracker = EyeTracker(pupil_mode=args.pupil, config=config)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    capture = open_camera(args)
    window = "attention-track phase 1 - eye + head-pose features"
    fps = 0.0
    show_mesh = args.mesh
    show_hud = True
    show_features = args.show_features
    print("keys: [i] cycle pupil mode  [m] face mesh  [h] hud  [f] features  [s] snapshot  [q/ESC] quit")

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

            view = tracker.draw_overlay(frame, result, show_hud=show_hud)
            if show_mesh:
                view = tracker.draw_mediapipe_mesh(view)
            crops = tracker.draw_crops(result, scale=args.zoom)
            scale = view.shape[0] / max(crops.shape[0], 1)
            crops = cv2.resize(crops, (max(int(crops.shape[1] * scale), 1), view.shape[0]))
            separator = np.full((view.shape[0], 4, 3), 30, np.uint8)
            canvas = np.hstack([view, separator, crops])

            fps = 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - loop_start, 1e-6))
            gaze_text = "no valid gaze"
            if result.gaze is not None:
                gaze_text = (
                    f"gaze=({result.gaze[0]:.3f}, {result.gaze[1]:.3f}) "
                    f"conf={result.confidence:.2f}"
                )
            distance_text = ""
            if result.distance_cm is not None:
                distance_text = f"  dist={result.distance_cm:.0f}cm"
            header = (
                f"fps={fps:4.1f}  track={result.inference_ms:5.1f}ms  "
                f"pupil={tracker.pupil_mode}  {gaze_text}{distance_text}"
            )
            cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            if show_hud:
                cv2.putText(
                    canvas,
                    "head pose  " + format_pose(result.head_pose),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 180),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "confidence  " + format_confidences(result),
                    (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
                qualities = "  ".join(
                    f"{eye.side.split('-')[1]}={eye.quality:.2f}" for eye in result.eyes
                )
                cv2.putText(
                    canvas,
                    f"eye quality  {qualities or 'n/a'}",
                    (10, 94),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 220, 180),
                    1,
                    cv2.LINE_AA,
                )
                if show_features:
                    draw_feature_panel(canvas, result)
            cv2.putText(
                canvas,
                "[i] pupil mode   [m] mesh   [h] hud   [f] features   [s] snapshot   [q] quit",
                (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("i"):
                modes = ("hybrid", "iris", "image")
                tracker.pupil_mode = modes[(modes.index(tracker.pupil_mode) + 1) % len(modes)]
            elif key == ord("m"):
                show_mesh = not show_mesh
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("f"):
                show_features = not show_features
            elif key == ord("s"):
                cv2.imwrite(args.snapshot, canvas)
                print(f"saved {args.snapshot}")
    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
