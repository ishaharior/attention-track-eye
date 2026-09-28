"""Phase 3: accumulate gaze points into a NumPy matrix and render a live attention heatmap."""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from gaze_core import AttentionHeatmap, EyeTracker, ScreenCalibrator, ScreenMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: live attention heatmap")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--screen", default="1920x1080", help="virtual screen size")
    parser.add_argument("--gain", type=float, default=3.0, help="raw mapping gain")
    parser.add_argument("--smoothing", type=float, default=0.3, help="exponential smoothing, 1 = off")
    parser.add_argument("--calibration", default="calibration.npz", help="calibration file")
    parser.add_argument("--cell", type=int, default=4, help="grid cell size in screen pixels")
    parser.add_argument("--sigma", type=float, default=45.0, help="Gaussian blur sigma in screen pixels")
    parser.add_argument("--decay", type=float, default=0.997, help="per frame matrix decay, 1.0 = off")
    parser.add_argument("--alpha", type=float, default=0.7, help="heatmap opacity")
    parser.add_argument("--colormap", type=int, default=cv2.COLORMAP_TURBO, help="cv2 colormap id")
    parser.add_argument("--background", choices=("camera", "canvas"), default="camera")
    parser.add_argument("--preview", type=int, default=960, help="preview width")
    parser.add_argument("--pupil", choices=("hybrid", "iris", "image"), default="hybrid")
    parser.add_argument("--points-out", default="attention_points.npy", help="gaze point log")
    parser.add_argument("--grid-out", default="attention_grid.npy", help="accumulation matrix dump")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def parse_screen(value: str) -> tuple:
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError("screen must look like 1920x1080") from None


def make_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (28, 24, 22), np.uint8)
    for x in range(0, width, 160):
        cv2.line(canvas, (x, 0), (x, height), (44, 40, 38), 1)
    for y in range(0, height, 160):
        cv2.line(canvas, (0, y), (width, y), (44, 40, 38), 1)
    cv2.rectangle(canvas, (0, 0), (width, 46), (62, 56, 52), -1)
    cv2.putText(canvas, "virtual desktop 1920x1080", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 2, cv2.LINE_AA)
    return canvas


def draw_hud(
    canvas: np.ndarray,
    mapper: ScreenMapper,
    heat: AttentionHeatmap,
    fps: float,
    position,
    status: str,
    show_blur: bool,
    confidence: float = 0.0,
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
    header = (
        f"fps={fps:4.1f}  {xy}  conf={confidence:.2f}  points={heat.total_hits}  "
        f"peak={heat.peak():6.1f}  {mapper.mapping_name}"
    )
    if not mapper.calibrated:
        header += "  (press c to calibrate)"
    cv2.putText(canvas, header, (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    mode = "PAUSED" if heat.paused else ("blur=on" if show_blur else "blur=off")
    decay_text = f"decay={heat.decay:.3f}" if heat.decay < 1.0 else "decay=off"
    footer = f"{mode}  {decay_text}  grid={heat.grid.shape[1]}x{heat.grid.shape[0]}  {status}  |  [p]ause [d]ecay [r]eset [s]ave [c]alibrate [n]ew centre [q]uit"
    cv2.putText(canvas, footer, (14, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (210, 210, 210), 1, cv2.LINE_AA)


def draw_matrix(heat: AttentionHeatmap, show_blur: bool) -> np.ndarray:
    data = heat.blurred() if show_blur else heat.grid
    peak = float(data.max())
    if peak <= 1e-6:
        normalized = np.zeros_like(data, dtype=np.uint8)
    else:
        normalized = np.clip(data / peak, 0.0, 1.0)
        normalized = (normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_BONE)
    label = "GaussianBlur(grid)" if show_blur else "raw NumPy accumulation matrix"
    cv2.putText(colored, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return colored


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
    heat = AttentionHeatmap(
        screen_w,
        screen_h,
        cell=args.cell,
        sigma=args.sigma,
        decay=args.decay,
        colormap=args.colormap,
        alpha=args.alpha,
    )
    canvas_background = make_canvas(screen_w, screen_h)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        tracker.close()
        sys.exit(f"Could not open webcam index {args.camera}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("keys: [p] pause  [d] decay  [r] reset  [s] save  [c] calibrate  [n] re-centre  [v] matrix view  [q/ESC] quit")
    fps = 0.0
    status = "accumulating"
    show_blur = True
    position = None

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
                    status = f"calibrating {calibrator.index}/{len(calibrator.TARGETS)}"
                elif state == "done":
                    if mapper.fit(calibrator.samples):
                        status = "calibration saved"
                        print(f"calibration saved to {mapper.path}")
                elif state == "aiming":
                    status = "hold still on the target"

            position = mapper.map(gaze)
            if position is not None and not heat.paused and not calibrator.active:
                heat.add(position[0], position[1], time.time())
            if not heat.paused:
                heat.decay_step()

            background = canvas_background if args.background == "canvas" else frame
            composite = heat.render(background)
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - loop_start, 1e-6))
            draw_hud(composite, mapper, heat, fps, position, status, show_blur, result.confidence)

            inset = tracker.draw_overlay(frame, result, show_hud=False)
            inset_scale = 360 / max(inset.shape[1], 1)
            inset = cv2.resize(inset, (360, max(int(inset.shape[0] * inset_scale), 1)))
            x0 = composite.shape[1] - inset.shape[1] - 14
            y0 = 96
            if y0 + inset.shape[0] <= composite.shape[0] and x0 >= 0:
                roi = composite[y0:y0 + inset.shape[0], x0:x0 + inset.shape[1]]
                composite[y0:y0 + inset.shape[0], x0:x0 + inset.shape[1]] = cv2.addWeighted(roi, 0.25, inset, 0.9, 0)
                cv2.rectangle(composite, (x0, y0), (x0 + inset.shape[1], y0 + inset.shape[0]), (255, 255, 255), 1)

            preview_scale = args.preview / max(composite.shape[1], 1)
            preview = cv2.resize(
                composite,
                (args.preview, max(int(composite.shape[0] * preview_scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("attention-track phase 3 - heatmap", preview)
            cv2.imshow("accumulation matrix", draw_matrix(heat, show_blur))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                heat.paused = not heat.paused
                status = "paused" if heat.paused else "accumulating"
            elif key == ord("d"):
                heat.decay = 1.0 if heat.decay < 1.0 else args.decay
                status = "decay off" if heat.decay >= 1.0 else f"decay {heat.decay:.3f}"
            elif key == ord("r"):
                heat.reset()
                status = "matrix cleared"
            elif key == ord("s"):
                print(f"saved {heat.save_points(args.points_out)} ({heat.total_hits} points) "
                      f"and {heat.save_grid(args.grid_out)}")
            elif key == ord("v"):
                show_blur = not show_blur
            elif key == ord("n"):
                mapper.recenter()
                status = "gaze re-centred"
            elif key == ord("c"):
                calibrator.start()
                status = "calibration started"
            elif key == ord(" ") and calibrator.active:
                state = calibrator.record(gaze)
                if state == "done" and mapper.fit(calibrator.samples):
                    status = "calibration saved"
                    print(f"calibration saved to {mapper.path}")
    finally:
        points_path = heat.save_points(args.points_out)
        grid_path = heat.save_grid(args.grid_out)
        print(f"saved {points_path} shape={heat.points_matrix().shape} and {grid_path} shape={heat.grid.shape}")
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
