"""Headless tests for the browser interface (web_app.py).

These cover everything that does not require a webcam: session lifecycle,
the real MediaPipe frame path on face-less frames, calibration fitting,
heatmap report rendering and the blue->red colormap endpoints.  Live camera
behaviour must still be checked manually in a browser.
"""

import base64
import unittest

import cv2
import numpy as np

import web_app
from gaze_core import FEATURE_DIM, AttentionHeatmap


def png_data_url(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def blank_frame(width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width, 3), 240, dtype=np.uint8)


def synthetic_features(rng, target) -> np.ndarray:
    features = rng.normal(0.0, 0.15, FEATURE_DIM)
    features[0] += (target[0] - 0.5) * 2.0
    features[1] += (target[1] - 0.5) * 2.0
    return features


def drive_calibration(session, rng) -> None:
    """Feed the calibrator enough synthetic samples to finish every target."""
    calibrator = session.calibrator
    per_target = calibrator.stabilize_frames + calibrator.samples_per_point
    for target in calibrator.targets:
        for _ in range(per_target + 2):
            if calibrator.done:
                return
            features = synthetic_features(rng, target)
            calibrator.update(features, 0.9)
    assert calibrator.done


def inject_gaze(session) -> None:
    """Synthetic session history: one hot cluster, one weak cluster."""
    rng = np.random.default_rng(11)
    for i in range(40):
        t_ms = i * 250.0
        x = float(100 + rng.normal(0, 4))
        y = float(100 + rng.normal(0, 4))
        session.heatmap.add(x, y, t_ms / 1000.0, confidence=0.9)
        session.samples.append((t_ms, x * 0.5, y * 0.5, 0.9))
        session.detector.update(t_ms, x * 0.5, y * 0.5, 0.9)
    for i in range(8):
        t_ms = 10000.0 + i * 250.0
        x = float(250 + rng.normal(0, 3))
        y = float(180 + rng.normal(0, 3))
        session.heatmap.add(x, y, t_ms / 1000.0, confidence=0.5)
        session.samples.append((t_ms, x * 0.5, y * 0.5, 0.5))
        session.detector.update(t_ms, x * 0.5, y * 0.5, 0.5)


class WebAppTest(unittest.TestCase):
    def setUp(self):
        web_app.app.config["TESTING"] = True
        self.client = web_app.app.test_client()
        self._clear_session()

    def tearDown(self):
        self._clear_session()

    @staticmethod
    def _clear_session():
        with web_app.STATE_LOCK:
            if web_app._session is not None:
                web_app._session.dispose()
                web_app._session = None

    @staticmethod
    def _session_payload(calibration=True, points=5):
        return {
            "image": png_data_url(blank_frame()),
            "rect": {"x": 0, "y": 0, "w": 640, "h": 480},
            "points": points,
            "model": "affine",
            "calibration": calibration,
        }

    # -- routing / validation --------------------------------------------
    def test_index_served(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        body = res.get_data(as_text=True)
        self.assertIn("gaze-density heatmap", body)
        self.assertIn("Finish — build heatmap report", body)
        self.assertIn("blue = low gaze density", body.lower())

    def test_session_rejects_missing_image(self):
        res = self.client.post("/api/session", json={})
        self.assertEqual(res.status_code, 400)

    def test_session_rejects_bad_points(self):
        payload = self._session_payload(points=7)
        res = self.client.post("/api/session", json=payload)
        self.assertEqual(res.status_code, 400)

    def test_frame_and_finish_need_a_session(self):
        frame = blank_frame()
        res = self.client.post("/api/frame", json={"image": png_data_url(frame)})
        self.assertEqual(res.status_code, 409)
        res = self.client.post("/api/finish", json={})
        self.assertEqual(res.status_code, 409)

    def test_frame_rejects_bad_image(self):
        self.client.post("/api/session", json=self._session_payload())
        res = self.client.post("/api/frame", json={"image": "not-an-image"})
        self.assertEqual(res.status_code, 400)

    # -- pipeline paths ----------------------------------------------------
    def test_session_and_faceless_frame(self):
        res = self.client.post("/api/session", json=self._session_payload())
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["mode"], "calibrating")
        self.assertEqual(data["doc"], {"width": 320, "height": 240})

        res = self.client.post(
            "/api/frame", json={"image": png_data_url(blank_frame()), "t_ms": 1000.0}
        )
        self.assertEqual(res.status_code, 200)
        frame = res.get_json()
        self.assertFalse(frame["face"])
        self.assertEqual(frame["mode"], "calibrating")
        self.assertIn("status", frame["calibration"])
        self.assertIn("target", frame["calibration"])

        # finish is refused while calibration is still running
        res = self.client.post("/api/finish", json={})
        self.assertEqual(res.status_code, 409)

    def test_calibration_fits_and_reports_residual(self):
        self.client.post("/api/session", json=self._session_payload())
        with web_app.STATE_LOCK:
            session = web_app._session
            drive_calibration(session, np.random.default_rng(3))
            self.assertTrue(session.calibrator.done)
            web_app.complete_calibration(session)
            self.assertEqual(session.mode, "tracking")
            self.assertTrue(session.mapper.calibrated)
            residual = session.calib_residual
        self.assertIsNotNone(residual)
        self.assertLess(residual["mean_px"], 200.0)
        self.assertEqual(residual["sample_count"], len(session.mapper.samples))

    def test_tracking_report_and_colours(self):
        self.client.post("/api/session", json=self._session_payload(calibration=False))
        res = self.client.post(
            "/api/frame", json={"image": png_data_url(blank_frame()), "t_ms": 500.0}
        )
        self.assertEqual(res.get_json()["mode"], "tracking")

        with web_app.STATE_LOCK:
            inject_gaze(web_app._session)

        res = self.client.post("/api/finish", json={"decay": False})
        self.assertEqual(res.status_code, 200)
        report = res.get_json()
        self.assertTrue(report["heatmap_png"].startswith("data:image/png;base64,"))
        self.assertEqual(report["doc"], {"width": 320, "height": 240})
        stats = report["stats"]
        for key in (
            "duration_s", "frames", "face_rate", "mean_confidence", "gaze_samples",
            "heatmap_points", "fixations", "saccades", "calibration", "mapping",
            "heatmap",
        ):
            self.assertIn(key, stats)
        self.assertEqual(stats["gaze_samples"], 48)
        self.assertGreaterEqual(stats["fixations"], 1)
        self.assertIsNone(stats["calibration"])
        self.assertEqual(stats["heatmap"]["colormap"], "TURBO (blue -> red)")

        encoded = report["heatmap_png"].split(",", 1)[1]
        image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (240, 320, 3))

        # hot cluster: red-dominant blend over the grey document
        hot = image[100, 100]
        self.assertGreater(int(hot[2]), 130)              # red channel
        self.assertGreater(int(hot[2]), int(hot[0]) + 50)  # red > blue
        # weak cluster: clearly less heat than the hot one
        bg = 240.0
        hot_energy = float(np.abs(image[95:106, 95:106].astype(np.float32) - bg).sum())
        weak_energy = float(np.abs(image[175:186, 245:256].astype(np.float32) - bg).sum())
        self.assertGreater(hot_energy, weak_energy)
        # far corner: background untouched (blur must not reach it)
        corner = image[5, 5].astype(np.int16)
        self.assertLess(int(np.abs(corner - bg).max()), 8)

        # second finish with end-anchored time decay must also succeed
        res = self.client.post("/api/finish", json={"decay": True})
        self.assertEqual(res.status_code, 200)
        decayed = res.get_json()
        self.assertEqual(decayed["stats"]["heatmap"]["decay_tau_s"], 5.0)

        res = self.client.post("/api/reset", json={})
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(self.client.get("/api/health").get_json()["session"])


class ColormapTest(unittest.TestCase):
    def test_blue_to_red_endpoints(self):
        heat = AttentionHeatmap(
            width=320, height=240, cell=4, sigma=10.0, decay_tau=5.0,
            colormap=cv2.COLORMAP_TURBO, alpha=0.7, gamma=0.75, min_confidence=0.35,
        )
        self.assertTrue(heat.add(100.0, 100.0, 0.0, confidence=1.0))
        rendered = heat.render(None)
        peak = rendered[25, 25].astype(int)     # grid cell of the hot spot
        floor = rendered[2, 2].astype(int)      # far away, near-zero density
        self.assertGreater(int(peak[2]), int(peak[0]) + 50)   # hot: red
        self.assertGreaterEqual(int(floor[0]), int(floor[2]))  # cold: blue-ish
        self.assertFalse(heat.add(100.0, 100.0, 1.0, confidence=0.2))  # gate


if __name__ == "__main__":
    unittest.main()
