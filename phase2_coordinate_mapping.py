"""Phase 2: map normalized eye movement onto 1920x1080 screen coordinates."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

from gaze_core import EyeTracker, ScreenCalibrator, ScreenMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: gaze -> 1920x1080 coordinate mapping")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--screen", default="1920x1080", help="virtual screen size, e.g. 1920x1080")
    parser.add_argument("--gain", type=float, default=3.0, help="raw mapping gain around screen centre")
    parser.add_argument("--smoothing", type=float, default=0.3, help="exponential smoothing factor, 1 = off")
    parser.add_argument("--preview", type=int, default=960, help="preview width of the virtual screen")
    parser.add_argument("--pupil", choices=("hybrid", "iris", "image"), default="hybrid")
    parser.add_argument("--calibration", default="calibration.npz", help="calibration file")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--trail", type=int, default=120, help="gaze trail length in frames")
    return parser.parse_args()


def parse_screen(value: str) -> tuple:
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError("screen must look like 1920x1080") from None


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


def draw_calibration(canvas: np.ndarray, calibrator: ScreenCalibrator, scale: float) -> None:
    for tx, ty in calibrator.captured:
        center = (int(tx * calibrator.width * scale), int(ty * calibrator.height * scale))
        cv2.circle(canvas, center, 16, (60, 220, 60), 3, cv2.LINE_AA)
    target = calibrator.target_px
    if target is None:
        return
    center = (int(target[0] * scale), int(target[1] * scale))
    cv2.circle(canvas, center, 34, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(canvas, center, 12, (0, 255, 255), -1, cv2.LINE_AA)
    arc = int(360 * calibrator.progress)
    if arc > 0:
        cv2.ellipse(canvas, center, (34, 34), -90, 0, arc, (0, 255, 0), 4, cv2.LINE_AA)
    label = f"look here  {calibrator.index + 1}/{len(calibrator.TARGETS)}"
    cv2.putText(canvas, label, (center[0] - 70, center[1] + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    screen_w, screen_h = parse_screen(args.screen)
    try:
        tracker = EyeTracker(pupil_mode=args.pupil)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    mapper = ScreenMapper(screen_w, screen_h, args.gain, args.smoothing, args.calibration)
    calibrator = ScreenCalibrator(screen_w, screen_h)
    trail = deque(maxlen=max(args.trail, 4))
    scale = args.preview / screen_w
    preview_size = (args.preview, int(round(screen_h * scale)))

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        tracker.close()
        sys.exit(f"Could not open webcam index {args.camera}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("keys: [c] start calibration  [SPACE] capture point  [r] reset calibration  [n] re-centre  [q/ESC] quit")
    print(f"mapping {screen_w}x{screen_h} | loaded calibration: {mapper.calibrated}")
    fps = 0.0
    status_text = "waiting for gaze"

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

            if calibrator.active:
                state = calibrator.update(gaze)
                if state == "captured":
                    status_text = f"captured point {calibrator.index}/{len(calibrator.TARGETS)}"
                elif state == "done":
                    if mapper.fit(calibrator.samples):
                        status_text = "calibration saved"
                        print(f"calibration saved to {mapper.path}")
                    else:
                        status_text = "calibration failed (not enough spread)"
                elif state == "aiming":
                    status_text = "hold still on the target"

            position = mapper.map(gaze)
            if position is not None:
                trail.append(position)

            canvas = np.full((preview_size[1], preview_size[0], 3), (30, 26, 24), np.uint8)
            draw_grid(canvas)
            draw_trail(canvas, trail, scale)
            if position is not None:
                draw_cursor(canvas, position[0], position[1], scale, result.confidence)
            draw_calibration(canvas, calibrator, scale)

            fps = 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - loop_start, 1e-6))
            if position is None:
                cv2.putText(canvas, "NO FACE", (preview_size[0] // 2 - 80, preview_size[1] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 140, 255), 3, cv2.LINE_AA)
                xy_text = "X=---- Y=----"
            else:
                xy_text = f"X={position[0]:7.1f}  Y={position[1]:7.1f}"
            header = f"{mapper.mapping_name}   {xy_text}   conf={result.confidence:.2f}  fps={fps:4.1f}"
            cv2.putText(canvas, header, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            footer = f"{status_text}   |   [c] calibrate  [space] capture  [n] re-centre  [r] reset  [q] quit"
            cv2.putText(canvas, footer, (14, preview_size[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1, cv2.LINE_AA)

            camera_view = tracker.draw_overlay(frame, result, show_hud=False)
            camera_scale = 420 / max(camera_view.shape[1], 1)
            camera_view = cv2.resize(camera_view, (420, int(camera_view.shape[0] * camera_scale)))
            cv2.imshow("webcam", camera_view)
            cv2.imshow("attention-track phase 2 - 1920x1080 screen", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                calibrator.start()
                status_text = "calibration started"
                trail.clear()
            elif key == ord(" ") and calibrator.active:
                state = calibrator.record(gaze)
                if state == "done":
                    if mapper.fit(calibrator.samples):
                        status_text = "calibration saved"
                        print(f"calibration saved to {mapper.path}")
                else:
                    status_text = "captured manually"
            elif key == ord("r"):
                mapper.reset()
                mapper.recenter()
                calibrator.cancel()
                trail.clear()
                status_text = "calibration reset"
                print("calibration reset")
            elif key == ord("n"):
                mapper.recenter()
                trail.clear()
                status_text = "gaze re-centred"

    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
