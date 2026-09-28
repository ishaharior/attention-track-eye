"""Phase 1: crop the eye regions from the webcam feed and track pupil movement."""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from gaze_core import EyeTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: eye ROI cropping + pupil tracking")
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
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit(f"Could not open webcam index {args.camera}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return capture


def main() -> int:
    args = parse_args()
    try:
        tracker = EyeTracker(pupil_mode=args.pupil)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    capture = open_camera(args)
    window = "attention-track phase 1 - eye tracking"
    fps = 0.0
    show_mesh = args.mesh
    show_hud = True
    print("keys: [i] cycle pupil mode  [m] face mesh  [h] hud  [s] snapshot  [q/ESC] quit")

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
            gaze_text = "no face"
            if result.gaze is not None:
                gaze_text = f"gaze=({result.gaze[0]:.3f}, {result.gaze[1]:.3f}) conf={result.confidence:.2f}"
            header = (
                f"fps={fps:4.1f}  track={result.inference_ms:5.1f}ms  "
                f"pupil={tracker.pupil_mode}  {gaze_text}"
            )
            cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "[i] pupil mode   [m] mesh   [h] hud   [s] snapshot   [q] quit",
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
