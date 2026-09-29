"""Offline test suite for the attention-track-eye upgrade.

Everything runs without a webcam: synthetic landmarks, synthetic calibration
data, synthetic gaze trajectories.  Run with:

    python -m unittest -v test_gaze_pipeline
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

import gaze_core as gc
from gaze_core import (
    ConfidenceBreakdown,
    EyeObservation,
    GazeConfig,
    HeadPose,
    MappingSample,
    OneEuroFilter,
    ScreenCalibrator,
    ScreenMapper,
    build_feature_vector,
    build_mapping_model,
    eye_quality,
    estimate_eye_distance_cm,
    estimate_head_pose,
    fuse_eye_gazes,
    landmark_confidence,
    robust_median,
    temporal_confidence,
)
from phase2_coordinate_mapping import (
    ValidationRecord,
    ValidationRun,
    angular_error_degrees,
    build_report,
    distance_condition,
    error_statistics,
    evaluate_models_on_features,
    save_report,
    summarize_validation,
)
from phase3_attention_heatmap import (
    MATRIX_BOTTOM,
    MATRIX_LEFT,
    MATRIX_RIGHT,
    MATRIX_TOP,
    EventDetector,
    draw_matrix,
)


SCREEN = (1920, 1080)


def make_eye(side: str, gaze, quality: float = 0.8) -> EyeObservation:
    rect = np.full((100, 200, 3), 60, np.uint8)
    mask = np.full((100, 200), 255, np.uint8)
    contour = np.array([[40.0, 30.0], [160.0, 30.0], [160.0, 70.0], [40.0, 70.0]], np.float32)
    if gaze is None:
        return EyeObservation(side, rect, mask, contour.copy(), contour.copy(), contour.copy(),
                              None, None, None, "none", 0.0)
    pupil_rect = (gaze[0] * 200.0, gaze[1] * 100.0)
    pupil_frame = (40.0 + gaze[0] * 120.0, 30.0 + gaze[0] * 0 + gaze[1] * 40.0)
    return EyeObservation(
        side=side, rect=rect, aperture_mask=mask, contour_rect=contour.copy(),
        contour_frame=contour.copy(), quad_frame=contour.copy(),
        pupil_rect=pupil_rect, pupil_frame=pupil_frame, gaze=gaze,
        source="image", confidence=quality, quality=quality,
    )


def projected_landmarks(yaw=0.0, pitch=0.0, roll=0.0, distance=700.0,
                        width=1280, height=720) -> np.ndarray:
    ry, rx, rz = map(math.radians, (yaw, pitch, roll))
    ry_mat = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    rx_mat = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    rz_mat = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    rotation = rz_mat @ rx_mat @ ry_mat
    object_points = (rotation @ gc.HEAD_POSE_MODEL.T).T + np.array([0.0, 0.0, distance])
    camera = gc._camera_matrix(width, height)
    projected = (camera @ object_points.T).T
    xy = projected[:, :2] / projected[:, 2:3]
    points = np.zeros((478, 2), dtype=np.float32)
    for index, value in zip(gc.HEAD_POSE_LANDMARK_IDS, xy):
        points[index] = value
    # fill the rest of the mesh with plausible coordinates so helper checks pass
    points[10] = (width / 2, height / 2)
    points[200] = (width / 2, height / 2 - 40)
    return points


class TestConfiguration(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = GazeConfig()
        config.validate()
        self.assertEqual(config.mapping_model, "affine")
        self.assertEqual(config.calibration_points, 9)

    def test_invalid_values_raise(self):
        with self.assertRaises(ValueError):
            GazeConfig(mapping_model="deep-learning").validate()
        with self.assertRaises(ValueError):
            GazeConfig(ema_alpha=0.0).validate()
        with self.assertRaises(ValueError):
            GazeConfig(calibration_points=7).validate()

    def test_smoothing_and_distance_keys_are_validated(self):
        GazeConfig(smoothing_filter="oneeuro").validate()
        GazeConfig(smoothing_filter="ema").validate()
        for kwargs in (
            {"smoothing_filter": "median"},
            {"oneeuro_min_cutoff": 0.0},
            {"oneeuro_d_cutoff": -1.0},
            {"oneeuro_beta": -0.01},
            {"distance_drift_warn_pct": 0.0},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                GazeConfig(**kwargs).validate()
        self.assertEqual(GazeConfig().smoothing_filter, "ema")   # documented default
        self.assertEqual(GazeConfig().distance_drift_warn_pct, 15.0)

    def test_target_layouts(self):
        self.assertEqual({k: len(v) for k, v in gc.TARGET_LAYOUTS.items()}, {5: 5, 9: 9, 13: 13})
        for layout in gc.TARGET_LAYOUTS.values():
            self.assertEqual(len(set(layout)), len(layout))


class TestFeatureVector(unittest.TestCase):
    def test_feature_names_count(self):
        self.assertEqual(gc.FEATURE_DIM, 11)
        self.assertEqual(len(gc.FEATURE_NAMES), 11)

    def test_neutral_features_without_eyes(self):
        pose = HeadPose(yaw=30.0, pitch=0.0, roll=0.0, valid=True, confidence=0.9)
        features = build_feature_vector([], pose)
        self.assertEqual(features.shape, (11,))
        self.assertAlmostEqual(features[4], 0.5)  # imputed iris centres
        self.assertAlmostEqual(features[5], 0.5)
        self.assertAlmostEqual(features[6], 0.5)
        self.assertAlmostEqual(features[7], 0.5)
        self.assertAlmostEqual(features[8], 30.0 / gc.HEAD_POSE_SCALE_DEG)

    def test_features_from_both_eyes(self):
        left = make_eye("image-left", (0.6, 0.4))
        right = make_eye("image-right", (0.4, 0.5))
        pose = HeadPose(yaw=-15.0, pitch=10.0, roll=5.0, valid=True, confidence=0.8)
        features = build_feature_vector([left, right], pose)
        self.assertAlmostEqual(features[4], 0.6, places=5)
        self.assertAlmostEqual(features[6], 0.4, places=5)
        self.assertAlmostEqual(features[9], 10.0 / gc.HEAD_POSE_SCALE_DEG, places=5)
        self.assertAlmostEqual(features[10], 5.0 / gc.HEAD_POSE_SCALE_DEG, places=5)
        # eye-centred offsets are finite and non-zero
        self.assertTrue(np.isfinite(features[:4]).all())


class TestHeadPose(unittest.TestCase):
    def test_recovers_known_angles(self):
        for yaw in (-35, -15, 0, 20, 35):
            pose = estimate_head_pose(projected_landmarks(yaw=yaw), 1280, 720)
            self.assertTrue(pose.valid)
            self.assertAlmostEqual(pose.yaw, yaw, delta=1.5)
            self.assertLess(pose.reprojection_error, 1.0)

    def test_recovers_pitch_and_roll(self):
        for pitch in (-20, 0, 25):
            pose = estimate_head_pose(projected_landmarks(pitch=pitch), 1280, 720)
            self.assertAlmostEqual(pose.pitch, pitch, delta=1.5)
        for roll in (-15, 0, 12):
            pose = estimate_head_pose(projected_landmarks(roll=roll), 1280, 720)
            self.assertAlmostEqual(pose.roll, roll, delta=1.5)

    def test_confidence_decreases_with_reprojection_noise(self):
        clean = estimate_head_pose(projected_landmarks(), 1280, 720)
        noisy_points = projected_landmarks()
        noisy_points[gc.HEAD_POSE_LANDMARK_IDS[1]] += np.array([40.0, -30.0], dtype=np.float32)
        noisy = estimate_head_pose(noisy_points, 1280, 720)
        self.assertGreater(clean.confidence, noisy.confidence)

    def test_invalid_input(self):
        pose = estimate_head_pose(np.zeros((478, 2), np.float32), 1280, 720)
        self.assertFalse(pose.valid)
        self.assertEqual(pose.confidence, 0.0)

    def test_landmark_confidence(self):
        points = projected_landmarks()
        high = landmark_confidence(points, 1280, 720)
        self.assertGreater(high, 0.5)
        shifted = points + 6.0
        low = landmark_confidence(points, 1280, 720, previous_points=shifted)
        self.assertLessEqual(low, high)


class TestConfidenceAndFusion(unittest.TestCase):
    def test_breakdown_is_interpretable(self):
        breakdown = ConfidenceBreakdown(pupil=1.0, landmark=1.0, head_pose=1.0, gaze=1.0, temporal=1.0)
        self.assertAlmostEqual(breakdown.combined(), 1.0)
        zero = ConfidenceBreakdown()
        self.assertAlmostEqual(zero.combined(), 0.0)
        self.assertIn("pupil", zero.as_dict())

    def test_confidence_weighted_fusion(self):
        config = GazeConfig()
        good = make_eye("image-left", (0.6, 0.5), quality=0.9)
        poor = make_eye("image-right", (0.2, 0.5), quality=0.1)
        fused = fuse_eye_gazes([good, poor], {"image-left": 0.9, "image-right": 0.1}, config)
        self.assertIsNotNone(fused)
        # weighted mean must sit far closer to the reliable eye than a plain average
        naive = (0.6 + 0.2) / 2
        self.assertLess(abs(fused[0] - 0.6), abs(naive - 0.6))

    def test_rejects_when_both_eyes_are_poor(self):
        config = GazeConfig()
        poor_a = make_eye("image-left", (0.6, 0.5), quality=0.05)
        poor_b = make_eye("image-right", (0.4, 0.5), quality=0.05)
        fused = fuse_eye_gazes(
            [poor_a, poor_b], {"image-left": 0.01, "image-right": 0.01}, config
        )
        self.assertIsNone(fused)

    def test_eye_quality_bounds(self):
        eye = make_eye("image-left", (0.5, 0.5), quality=0.9)
        self.assertTrue(0.0 <= eye_quality(eye) <= 1.0)
        missing = make_eye("image-right", None)
        self.assertEqual(eye_quality(missing), 0.0)

    def test_temporal_confidence(self):
        config = GazeConfig()
        steady = temporal_confidence((0.5, 0.5), (0.5, 0.5), 0.03, config)
        self.assertEqual(steady, 1.0)
        jump = temporal_confidence((0.9, 0.9), (0.1, 0.1), 0.03, config)
        self.assertLess(jump, 0.5)
        missing = temporal_confidence(None, (0.5, 0.5), 0.03, config)
        self.assertEqual(missing, 0.0)


class TestCalibration(unittest.TestCase):
    def test_stabilization_and_collection(self):
        config = GazeConfig(calibration_points=5, calibration_samples_per_point=4,
                            calibration_stabilize_frames=3, calibration_min_confidence=0.3)
        calibrator = ScreenCalibrator(*SCREEN, config=config)
        calibrator.start()
        features = np.zeros(gc.FEATURE_DIM)
        states = [calibrator.update(features, 0.9) for _ in range(3)]
        self.assertEqual(states[:2], ["stabilizing", "stabilizing"])
        self.assertEqual(states[2], "collecting")
        rest = [calibrator.update(features, 0.9) for _ in range(4)]
        self.assertIn("captured", rest)
        self.assertEqual(calibrator.index, 1)

    def test_low_confidence_samples_are_rejected(self):
        config = GazeConfig(calibration_points=5, calibration_samples_per_point=2,
                            calibration_stabilize_frames=1, calibration_min_confidence=0.5)
        calibrator = ScreenCalibrator(*SCREEN, config=config)
        calibrator.start()
        features = np.zeros(gc.FEATURE_DIM)
        self.assertEqual(calibrator.update(features, 0.1), "invalid")
        self.assertEqual(calibrator.rejected, 1)
        self.assertEqual(calibrator.update(features, 0.9), "collecting")

    def test_full_run_and_outlier_rejection(self):
        config = GazeConfig(calibration_points=9, calibration_samples_per_point=6,
                            calibration_stabilize_frames=2, calibration_min_confidence=0.3)
        calibrator = ScreenCalibrator(*SCREEN, config=config)
        calibrator.start()
        rng = np.random.default_rng(0)
        for _ in range(9):
            calibrator.update(rng.normal(0.0, 0.2, gc.FEATURE_DIM), 0.9)  # stabilize
            for i in range(6):
                features = rng.normal(0.0, 0.2, gc.FEATURE_DIM)
                if i == 0:
                    features += 50.0  # extreme outlier on the first sample
                calibrator.update(features, 0.9)
        self.assertTrue(calibrator.done)
        self.assertEqual(len(calibrator.captured), 9)
        samples = calibrator.finalize()
        self.assertEqual(len(samples), 9 * 6 - 9)  # one outlier removed per target
        self.assertGreaterEqual(calibrator.rejected, 9)
        self.assertEqual(len(calibrator.representative), 9)
        self.assertIsNotNone(calibrator.representative[0])
        self.assertTrue(all(isinstance(s, MappingSample) for s in samples))

    def test_manual_capture(self):
        config = GazeConfig(calibration_points=5, calibration_samples_per_point=5)
        calibrator = ScreenCalibrator(*SCREEN, config=config)
        calibrator.start()
        features = np.ones(gc.FEATURE_DIM)
        while calibrator.active:
            state = calibrator.record(features, 0.9)
        self.assertIn(state, ("done", "captured"))
        self.assertEqual(len(calibrator.finalize()), 5)

    def test_robust_median(self):
        rows = [np.zeros(3), np.zeros(3), np.zeros(3), np.array([100.0, 100.0, 100.0])]
        median = robust_median(rows, z_threshold=3.0)
        self.assertTrue(np.allclose(median, 0.0))


def synthetic_calibration_samples(n_points: int = 9, per_point: int = 20, seed: int = 3):
    """Features correlated with known screen coordinates + noise."""
    rng = np.random.default_rng(seed)
    samples = []
    for target in gc.TARGET_LAYOUTS[n_points][:n_points]:
        tx, ty = target[0] * SCREEN[0], target[1] * SCREEN[1]
        for _ in range(per_point):
            features = rng.normal(0.0, 0.15, gc.FEATURE_DIM)
            features[0] += target[0] - 0.5
            features[2] += target[1] - 0.5
            features[8] += (target[0] - 0.5) * 0.6
            samples.append(MappingSample(features, (tx, ty)))
    return samples


class TestMappingModels(unittest.TestCase):
    def test_affine_fit_and_predict(self):
        samples = synthetic_calibration_samples()
        model = build_mapping_model("affine")
        x = np.vstack([s.features for s in samples])
        y = np.array([s.target for s in samples])
        self.assertTrue(model.fit(x, y))
        predicted = model.predict(x)
        self.assertEqual(predicted.shape, y.shape)
        self.assertLess(error_statistics(np.linalg.norm(predicted - y, axis=1))["mean"], 400)

    def test_polynomial_and_mlp_fit(self):
        samples = synthetic_calibration_samples()
        x = np.vstack([s.features for s in samples])
        y = np.array([s.target for s in samples])
        poly = build_mapping_model("polynomial")
        self.assertTrue(poly.fit(x, y))
        self.assertTrue(poly.predict(x[:3]).shape == (3, 2))
        mlp = build_mapping_model("mlp")
        mlp.screen = SCREEN
        self.assertTrue(mlp.fit(x, y))
        self.assertTrue(np.all(np.isfinite(mlp.predict(x[:3]))))

    def test_model_comparison_on_validation_data(self):
        calibration = synthetic_calibration_samples()
        records = []
        rng = np.random.default_rng(5)
        for target in gc.TARGET_LAYOUTS[9]:
            features = rng.normal(0.0, 0.15, gc.FEATURE_DIM)
            features[0] += target[0] - 0.5
            features[2] += target[1] - 0.5
            features[8] += (target[0] - 0.5) * 0.6
            records.append(MappingSample(features, (target[0] * SCREEN[0], target[1] * SCREEN[1])))
        results = evaluate_models_on_features(
            calibration,
            [r.features for r in records],
            [r.target for r in records],
            ("affine", "polynomial", "mlp"),
            GazeConfig(),
            SCREEN,
        )
        self.assertEqual(len(results), 3)
        for name, stats in results.items():
            self.assertEqual(stats.get("fit"), 1.0, name)
            self.assertTrue(math.isfinite(stats["mean"]), name)

    def test_legacy_four_tuple_samples(self):
        legacy = [
            (0.1, 0.1, 100.0, 100.0),
            (0.9, 0.1, 1800.0, 100.0),
            (0.9, 0.9, 1800.0, 980.0),
            (0.1, 0.9, 100.0, 980.0),
            (0.5, 0.5, 960.0, 540.0),
        ]
        mapper = ScreenMapper(*SCREEN, path=os.path.join(tempfile.gettempdir(), "legacy.npz"))
        mapper.reset()
        self.assertTrue(mapper.fit(legacy))
        position = mapper.map((0.5, 0.5), confidence=1.0)
        self.assertIsNotNone(position)
        self.assertAlmostEqual(position[0], 960.0, delta=60)


class TestScreenMapper(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.gettempdir(), "test_calibration.npz")
        if os.path.exists(self.path):
            os.remove(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_raw_mode_without_calibration(self):
        mapper = ScreenMapper(*SCREEN, gain=2.0, smoothing=1.0, path=self.path,
                              warmup_frames=2)
        self.assertFalse(mapper.calibrated)
        first = mapper.map((0.5, 0.5), confidence=1.0)
        self.assertIsNotNone(first)
        self.assertAlmostEqual(first[0], SCREEN[0] / 2, delta=5)
        self.assertIn("raw gain", mapper.mapping_name)

    def test_calibration_roundtrip(self):
        samples = synthetic_calibration_samples()
        mapper = ScreenMapper(*SCREEN, path=self.path, config=GazeConfig(), smoothing=1.0)
        mapper.reset()
        self.assertTrue(mapper.fit(samples))
        self.assertTrue(mapper.calibrated)
        reloaded = ScreenMapper(*SCREEN, path=self.path, config=GazeConfig(), smoothing=1.0)
        self.assertTrue(reloaded.calibrated)
        self.assertEqual(reloaded.model_name, "affine")
        features = samples[0].features
        a = mapper.map((0.5, 0.5), features, 1.0)
        b = reloaded.map((0.5, 0.5), features, 1.0)
        self.assertTrue(np.allclose(a, b, atol=1e-3))

    def test_rejects_low_confidence(self):
        mapper = ScreenMapper(*SCREEN, path=self.path, config=GazeConfig(), smoothing=1.0)
        self.assertIsNone(mapper.map((0.5, 0.5), np.zeros(gc.FEATURE_DIM), confidence=0.1))
        self.assertEqual(mapper.stats["rejected_low_confidence"], 1)

    def test_rejects_missing_features_when_calibrated(self):
        samples = synthetic_calibration_samples()
        mapper = ScreenMapper(*SCREEN, path=self.path, config=GazeConfig(), smoothing=1.0)
        mapper.reset()
        self.assertTrue(mapper.fit(samples))
        self.assertIsNone(mapper.map((0.5, 0.5), None, confidence=1.0))
        self.assertEqual(mapper.stats["rejected_missing_features"], 1)

    def test_rejects_implausible_jump(self):
        config = GazeConfig(outlier_max_jump_px=100.0, outlier_max_speed_px_s=500.0)
        mapper = ScreenMapper(*SCREEN, path=self.path, config=config, gain=1.0, smoothing=1.0,
                              auto_center=False)
        first = mapper.map((0.5, 0.5), confidence=1.0)
        self.assertIsNotNone(first)
        far = mapper.map((0.95, 0.95), confidence=1.0)
        self.assertIsNone(far)
        self.assertEqual(mapper.stats["rejected_jump"], 1)
        near = mapper.map((0.52, 0.5), confidence=1.0)
        self.assertIsNotNone(near)

    def test_ema_smoothing(self):
        mapper = ScreenMapper(*SCREEN, path=self.path, gain=1.0, smoothing=0.5,
                              warmup_frames=1)
        mapper.map((0.5, 0.5), confidence=1.0)
        second = mapper.map((0.6, 0.5), confidence=1.0)
        # raw target would be (0.5 + 0.1*1.0)*1920 = 1152 without smoothing
        self.assertLess(second[0], 1152.0)
        self.assertGreater(second[0], 960.0)

    def test_valid_sample_rate(self):
        mapper = ScreenMapper(*SCREEN, path=self.path, config=GazeConfig(), smoothing=1.0)
        mapper.map((0.5, 0.5), confidence=1.0)
        mapper.map((0.5, 0.5), confidence=0.05)
        self.assertAlmostEqual(mapper.valid_sample_rate, 0.5)

    # -- eye-to-camera distance ------------------------------------------
    @staticmethod
    def _raw_mapper(path, **kwargs):
        """Uncalibrated gain mapper with no auto-centring (x = gx offset)."""
        return ScreenMapper(*SCREEN, path=path, gain=1.0, smoothing=1.0,
                            auto_center=False, warmup_frames=1, **kwargs)

    def test_distance_compensation_scales_offset_from_centre(self):
        mapper = self._raw_mapper(self.path)
        mapper.set_reference_distance(50.0)
        baseline = mapper.map((0.6, 0.5), confidence=1.0)          # reference distance
        leaning = mapper.map((0.6, 0.5), confidence=1.0, distance_cm=70.0)
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(leaning)
        offset = baseline[0] - SCREEN[0] / 2.0                     # 192 px right of centre
        compensated = leaning[0] - SCREEN[0] / 2.0
        self.assertAlmostEqual(compensated / offset, 70.0 / 50.0, delta=0.02)
        self.assertAlmostEqual(leaning[1], baseline[1], delta=1e-6)  # centred vertically

    def test_distance_compensation_is_clamped(self):
        mapper = self._raw_mapper(self.path)
        mapper.set_reference_distance(50.0)
        baseline = mapper.map((0.6, 0.5), confidence=1.0)
        offset = baseline[0] - SCREEN[0] / 2.0
        # absurdly far (ratio 10) and absurdly close (ratio 0.1) both clamp
        far = mapper.map((0.6, 0.5), confidence=1.0, distance_cm=500.0)
        near = mapper.map((0.6, 0.5), confidence=1.0, distance_cm=5.0)
        self.assertAlmostEqual((far[0] - SCREEN[0] / 2.0) / offset,
                               gc.DISTANCE_COMPENSATION_MAX, delta=0.01)
        self.assertAlmostEqual((near[0] - SCREEN[0] / 2.0) / offset,
                               gc.DISTANCE_COMPENSATION_MIN, delta=0.01)

    def test_no_compensation_without_reference_or_distance(self):
        no_reference = self._raw_mapper(self.path)
        without = no_reference.map((0.6, 0.5), confidence=1.0, distance_cm=70.0)
        with_distance = no_reference.map((0.6, 0.5), confidence=1.0, distance_cm=None)
        self.assertEqual(without, with_distance)

        with_reference = self._raw_mapper(self.path)
        with_reference.set_reference_distance(70.0)
        first = with_reference.map((0.6, 0.5), confidence=1.0, distance_cm=70.0)
        second = with_reference.map((0.6, 0.5), confidence=1.0, distance_cm=None)
        self.assertEqual(first, second)

    def test_set_reference_distance_rejects_implausible_values(self):
        mapper = self._raw_mapper(self.path)
        for value in (None, 5.0, 500.0, float("nan"), "far"):
            mapper.set_reference_distance(value)
            self.assertIsNone(mapper.reference_distance_cm)
        mapper.set_reference_distance(60.0)
        self.assertEqual(mapper.reference_distance_cm, 60.0)
        mapper.set_reference_distance(75.0)   # plausible updates win
        self.assertEqual(mapper.reference_distance_cm, 75.0)

    def test_reference_distance_survives_the_calibration_file(self):
        samples = synthetic_calibration_samples()
        mapper = ScreenMapper(*SCREEN, path=self.path, config=GazeConfig(), smoothing=1.0)
        mapper.reset()
        self.assertTrue(mapper.fit(samples, reference_distance_cm=52.0))
        reloaded = ScreenMapper(*SCREEN, path=self.path, config=GazeConfig(), smoothing=1.0)
        self.assertEqual(reloaded.reference_distance_cm, 52.0)
        # refitting without a new distance keeps the recorded reference
        self.assertTrue(mapper.fit(samples))
        self.assertEqual(mapper.reference_distance_cm, 52.0)

    def test_distance_never_pollutes_the_valid_sample_rate(self):
        mapper = self._raw_mapper(self.path)
        mapper.set_reference_distance(50.0)
        self.assertIsNotNone(mapper.map((0.5, 0.5), confidence=1.0, distance_cm=70.0))
        self.assertIsNone(mapper.map((0.5, 0.5), confidence=0.05, distance_cm=70.0))
        self.assertAlmostEqual(mapper.valid_sample_rate, 0.5)
        # only rejection counters live in stats (the rate divides by their sum)
        self.assertEqual(
            set(mapper.stats),
            {"accepted", "rejected_low_confidence", "rejected_off_screen",
             "rejected_jump", "rejected_missing_features"},
        )

    def test_oneeuro_filter_engaged_when_configured(self):
        config = GazeConfig(smoothing_filter="oneeuro")
        mapper = self._raw_mapper(self.path, config=config)
        position = mapper.map((0.6, 0.5), confidence=1.0)
        self.assertIsNotNone(position)
        self.assertIsNotNone(mapper._euro_x._value)
        self.assertEqual(mapper._smoothed[0], mapper._euro_x._value)
        # default EMA config never touches the adaptive filter
        ema_mapper = self._raw_mapper(self.path)
        ema_mapper.map((0.6, 0.5), confidence=1.0)
        self.assertIsNone(ema_mapper._euro_x._value)

    def test_recenter_and_reset_clear_the_adaptive_filter(self):
        mapper = self._raw_mapper(self.path, config=GazeConfig(smoothing_filter="oneeuro"))
        mapper.map((0.6, 0.5), confidence=1.0)
        self.assertIsNotNone(mapper._euro_x._value)
        mapper.recenter()
        self.assertIsNone(mapper._euro_x._value)
        mapper.map((0.6, 0.5), confidence=1.0)
        mapper.reset()
        self.assertIsNone(mapper._euro_x._value)


class TestOneEuroFilter(unittest.TestCase):
    def test_constant_signal_passes_unchanged(self):
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        values = [filt.filter(42.0, i * 0.016) for i in range(20)]
        for value in values:
            self.assertAlmostEqual(value, 42.0, places=6)

    def test_step_response_is_fast_then_settles(self):
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        for i in range(10):
            filt.filter(0.0, i * 0.016)
        step = [filt.filter(100.0, 0.16 + i * 0.016) for i in range(150)]
        # speed raises the cutoff: most of the jump happens in the first frames
        self.assertGreater(step[0], 40.0)
        self.assertGreater(step[4], 80.0)
        self.assertGreater(step[-1], 99.0)   # and the low cutoff settles it

    def test_moves_faster_than_a_fixed_ema_while_moving(self):
        one_hz = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        for i in range(10):
            one_hz.filter(0.0, i * 0.016)
        alpha = 0.15
        ema = 0.0
        one_out = ema_out = 0.0
        for i in range(10):
            t = 0.16 + i * 0.016
            one_out = one_hz.filter(100.0, t)
            ema = alpha * 100.0 + (1.0 - alpha) * ema
            ema_out = ema
        self.assertGreater(one_out, ema_out)

    def test_holds_output_on_non_monotonic_time(self):
        filt = OneEuroFilter()
        filt.filter(10.0, 1.0)
        self.assertEqual(filt.filter(999.0, 0.5), 10.0)

    def test_reset_forgets_state(self):
        filt = OneEuroFilter()
        filt.filter(10.0, 1.0)
        filt.reset()
        self.assertEqual(filt.filter(50.0, 2.0), 50.0)


class TestEyeDistance(unittest.TestCase):
    @staticmethod
    def landmarks(iod_px: float) -> np.ndarray:
        points = np.zeros((478, 2), dtype=np.float32)
        points[33] = (320.0 - iod_px / 2.0, 240.0)
        points[263] = (320.0 + iod_px / 2.0, 240.0)
        return points

    def test_pinhole_value_and_inverse_scaling(self):
        near = estimate_eye_distance_cm(self.landmarks(300.0), 640, 480)
        far = estimate_eye_distance_cm(self.landmarks(150.0), 640, 480)
        self.assertIsNotNone(near)
        self.assertIsNotNone(far)
        focal = 640 * gc.CAMERA_FOCAL_SCALE
        self.assertAlmostEqual(near, focal * gc.EYE_REFERENCE_WIDTH_CM / 300.0, delta=0.5)
        self.assertAlmostEqual(far / near, 2.0, delta=0.01)   # half the IOD, twice the distance

    def test_rejects_missing_or_implausible_input(self):
        self.assertIsNone(estimate_eye_distance_cm(None, 640, 480))
        self.assertIsNone(estimate_eye_distance_cm(self.landmarks(300.0), 0, 480))
        self.assertIsNone(estimate_eye_distance_cm(self.landmarks(300.0), 640, 0))
        self.assertIsNone(estimate_eye_distance_cm(np.zeros((10, 2), np.float32), 640, 480))
        # 600 px IOD at 640 wide -> ~11 cm, below the plausible webcam minimum
        self.assertIsNone(estimate_eye_distance_cm(self.landmarks(600.0), 640, 480))
        # 4 px IOD -> ~1.6 km, far beyond the plausible maximum
        self.assertIsNone(estimate_eye_distance_cm(self.landmarks(4.0), 640, 480))
        # degenerate (identical) points
        self.assertIsNone(estimate_eye_distance_cm(self.landmarks(0.0), 640, 480))

    def test_tracker_smooths_and_keeps_distance(self):
        # bypass EyeTracker.__init__ (loads MediaPipe); only the distance state is used
        tracker = gc.EyeTracker.__new__(gc.EyeTracker)
        tracker.distance_cm = None
        near = estimate_eye_distance_cm(self.landmarks(300.0), 640, 480)
        far = estimate_eye_distance_cm(self.landmarks(150.0), 640, 480)
        first = tracker._update_distance(self.landmarks(300.0), 640, 480)
        self.assertAlmostEqual(first, near)
        # the EMA moves toward the farther reading but never past it in one frame
        second = tracker._update_distance(self.landmarks(150.0), 640, 480)
        self.assertGreater(second, near)
        self.assertLess(second, far)
        self.assertAlmostEqual(
            second,
            gc.DISTANCE_SMOOTHING_ALPHA * far + (1.0 - gc.DISTANCE_SMOOTHING_ALPHA) * near,
        )
        # face lost: no fresh reading, but the smoothed value stays as reference
        self.assertIsNone(tracker._update_distance(None, 640, 480))
        self.assertEqual(tracker.distance_cm, second)

    def test_gaze_result_defaults_to_no_distance(self):
        result = gc.GazeResult(None, 0.0)
        self.assertIsNone(result.distance_cm)


class TestFixationDetection(unittest.TestCase):
    def trajectory(self, fps=30):
        samples = []
        t = 0.0
        rng = np.random.default_rng(1)
        # fixation 1: 500 ms around (500, 500)
        for _ in range(int(0.5 * fps)):
            samples.append((t, 500 + rng.normal(0, 4), 500 + rng.normal(0, 4), 0.9))
            t += 1000.0 / fps
        # saccade: two frames of movement towards (1500, 800)
        samples.append((t, 1000 + rng.normal(0, 4), 650 + rng.normal(0, 4), 0.9))
        t += 1000.0 / fps
        samples.append((t, 1500 + rng.normal(0, 4), 800 + rng.normal(0, 4), 0.9))
        t += 1000.0 / fps
        # fixation 2: 400 ms
        for _ in range(int(0.4 * fps)):
            samples.append((t, 1500 + rng.normal(0, 4), 800 + rng.normal(0, 4), 0.9))
            t += 1000.0 / fps
        # low confidence sample
        samples.append((t, 1500, 800, 0.1))
        t += 1000.0 / fps
        samples.append((t, 1500, 800, 0.1))
        return samples

    def test_idt_detects_fixations_and_saccades(self):
        detector = EventDetector(method="idt", dispersion_px=45.0, min_duration_ms=100.0)
        labels = [detector.update(t, x, y, c, min_confidence=0.35) for t, x, y, c in self.trajectory()]
        detector.finalize()
        self.assertGreaterEqual(len(detector.fixations), 2)
        self.assertGreaterEqual(len(detector.saccades), 1)
        self.assertIn("invalid", labels)
        self.assertIn("saccade", labels)
        self.assertIn("fixation", labels)
        durations = sorted(f.duration_ms for f in detector.fixations)
        self.assertGreaterEqual(durations[0], 90.0)
        # centroids must be near the true fixation positions
        centroids = sorted(f.centroid[0] for f in detector.fixations)
        self.assertLess(abs(centroids[0] - 500), 30)
        self.assertLess(abs(centroids[-1] - 1500), 30)

    def test_ivt_mode(self):
        detector = EventDetector(method="ivt", velocity_threshold=1.0, min_duration_ms=100.0)
        for t, x, y, c in self.trajectory():
            detector.update(t, x, y, c, min_confidence=0.35)
        detector.finalize()
        self.assertGreaterEqual(len(detector.fixations), 1)
        self.assertGreaterEqual(len(detector.saccades), 1)

    def test_summary_and_export(self):
        detector = EventDetector()
        for t, x, y, c in self.trajectory():
            detector.update(t, x, y, c, min_confidence=0.35)
        detector.finalize()
        summary = detector.summary(attempted_samples=40)
        self.assertGreater(summary["fixation_count"], 0)
        self.assertGreater(summary["average_fixation_duration_ms"], 100)
        payload = detector.events_payload()
        self.assertIn("fixations", payload)
        self.assertIn("saccades", payload)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            EventDetector(method="magic")


class TestHeatmap(unittest.TestCase):
    def test_confidence_gate(self):
        from gaze_core import AttentionHeatmap

        heat = AttentionHeatmap(*SCREEN, min_confidence=0.5)
        self.assertFalse(heat.add(100, 100, 0.0, confidence=0.2))
        self.assertEqual(heat.rejected_low_confidence, 1)
        self.assertTrue(heat.add(100, 100, 0.0, confidence=0.9))
        self.assertEqual(heat.total_hits, 1)

    def test_fixation_weighting(self):
        from gaze_core import AttentionHeatmap

        heat = AttentionHeatmap(*SCREEN, min_confidence=0.0)
        heat.add(100, 100, 0.0, confidence=1.0, fixation_weight=0.033)
        weak = heat.grid.sum()
        heat.add(100, 100, 0.0, confidence=1.0, fixation_weight=0.7)
        strong = heat.grid.sum() - weak
        self.assertGreater(strong, weak * 5)

    def test_time_based_decay_is_frame_rate_invariant(self):
        from gaze_core import AttentionHeatmap

        tau = 5.0
        fast = AttentionHeatmap(*SCREEN, decay_tau=tau)
        slow = AttentionHeatmap(*SCREEN, decay_tau=tau)
        fast.grid[:, :] = 100.0
        slow.grid[:, :] = 100.0
        for _ in range(60):  # one second at 60 fps
            fast.decay_step(1.0 / 60.0)
        for _ in range(30):  # one second at 30 fps
            slow.decay_step(1.0 / 30.0)
        expected = 100.0 * math.exp(-1.0 / tau)
        self.assertAlmostEqual(float(fast.grid[0, 0]), expected, delta=1e-3)
        self.assertAlmostEqual(float(slow.grid[0, 0]), expected, delta=1e-3)
        self.assertTrue(np.allclose(fast.grid, slow.grid, rtol=1e-5))

    def test_sigma_units_converted_from_screen_pixels(self):
        from gaze_core import AttentionHeatmap

        heat = AttentionHeatmap(*SCREEN, cell=4, sigma=45.0)
        self.assertAlmostEqual(heat.sigma_cells, 45.0 / 4.0)
        self.assertEqual(heat.kernel_size, 2 * round(3.0 * 45.0 / 4.0) + 1)
        blurred = heat.blurred()
        self.assertEqual(blurred.shape, heat.grid.shape)

    def test_configurable_gamma_alpha_and_render(self):
        from gaze_core import AttentionHeatmap

        config = GazeConfig(heatmap_gamma=0.5, heatmap_alpha=0.4, heatmap_cell_size=8,
                            gaussian_sigma_px=40.0, heatmap_decay_tau=2.0)
        heat = AttentionHeatmap(*SCREEN, config=config)
        self.assertEqual(heat.cell, 8)
        self.assertEqual(heat.gamma, 0.5)
        self.assertEqual(heat.alpha, 0.4)
        self.assertEqual(heat.decay_tau, 2.0)
        heat.add(960, 540, 0.0, confidence=1.0, fixation_weight=1.0)
        image = heat.render()
        self.assertEqual(image.shape[:2], (heat.grid_h, heat.grid_w))

    def test_reset(self):
        from gaze_core import AttentionHeatmap

        heat = AttentionHeatmap(*SCREEN)
        heat.add(10, 10, 0.0, confidence=1.0, fixation_weight=1.0)
        heat.reset()
        self.assertEqual(heat.total_hits, 0)
        self.assertEqual(float(heat.grid.sum()), 0.0)


class TestMatrixView(unittest.TestCase):
    """Phase 3's accumulation-matrix window: stable scale + screen coordinates."""

    @staticmethod
    def _scale(heat) -> int:
        return max(1, min(3, 1000 // heat.grid_w))

    @staticmethod
    def _region(panel, heat, scale: int) -> np.ndarray:
        return panel[
            MATRIX_TOP: MATRIX_TOP + heat.grid_h * scale,
            MATRIX_LEFT: MATRIX_LEFT + heat.grid_w * scale,
        ]

    @staticmethod
    def _index(panel, heat, scale: int, row: int, col: int) -> int:
        """Recover the colormap index rendered at the centre of one grid cell."""
        lut = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(1, -1), heat.colormap
        )[0].astype(int)
        px = panel[MATRIX_TOP + row * scale + scale // 2, MATRIX_LEFT + col * scale + scale // 2].astype(int)
        return int(np.argmin(np.abs(lut - px).sum(1)))

    @staticmethod
    def _heat():
        from gaze_core import AttentionHeatmap

        heat = AttentionHeatmap(*SCREEN, min_confidence=0.0)
        # written directly so the fixtures do not depend on the add() weighting rules
        heat.grid[50, 40] = 10.0  # weak structure far below the session peak
        heat.grid[135, 240] = 1000.0  # the peak cell
        return heat

    def test_panel_layout_carries_axes_and_labels(self):
        heat = self._heat()
        panel = draw_matrix(heat, False, None, 0.0)
        scale = self._scale(heat)

        self.assertEqual(panel.dtype, np.uint8)
        self.assertEqual(panel.ndim, 3)
        self.assertEqual(
            panel.shape,
            (
                heat.grid_h * scale + MATRIX_TOP + MATRIX_BOTTOM,
                heat.grid_w * scale + MATRIX_LEFT + MATRIX_RIGHT,
                3,
            ),
        )
        # header text above the grid, y-axis labels at the left, x-axis below
        self.assertGreater(int(panel[:MATRIX_TOP].max()), 100)
        self.assertGreater(
            int(panel[MATRIX_TOP:MATRIX_TOP + heat.grid_h * scale, :MATRIX_LEFT].max()), 100
        )
        self.assertGreater(int(panel[MATRIX_TOP + heat.grid_h * scale:, :].max()), 100)

    def test_session_reference_replaces_per_frame_normalisation(self):
        heat = self._heat()
        scale = self._scale(heat)
        at_peak = draw_matrix(heat, False, None, 1000.0)
        below_peak = draw_matrix(heat, False, None, 1.0e6)

        # a cell two orders of magnitude below the peak stays visible, where the
        # old per-frame linear scaling rendered it at ~2/255 (effectively black)
        self.assertGreaterEqual(self._index(at_peak, heat, scale, 50, 40), 50)

        # raising the reference dims every cell instead of re-stretching to 255
        self.assertLess(self._index(below_peak, heat, scale, 135, 240),
                        self._index(at_peak, heat, scale, 135, 240))
        self.assertLess(self._index(below_peak, heat, scale, 50, 40),
                        self._index(at_peak, heat, scale, 50, 40))

    def test_blur_view_shares_the_reference_and_does_not_clip(self):
        heat = self._heat()
        scale = self._scale(heat)
        raw = draw_matrix(heat, False, None, 1000.0)
        blurred = draw_matrix(heat, True, None, 1000.0)

        self.assertFalse(np.array_equal(raw, blurred))
        # Gaussian smoothing cannot raise the peak, so against one shared
        # reference the blurred view must sit below the raw peak, never clip
        self.assertLess(self._index(blurred, heat, scale, 135, 240),
                        self._index(raw, heat, scale, 135, 240))

    def test_gaze_crosshair_marks_the_current_cell(self):
        heat = self._heat()
        scale = self._scale(heat)
        still = draw_matrix(heat, False, None, 1000.0)
        tracked = draw_matrix(heat, False, (960.0, 540.0), 1000.0)
        self.assertFalse(np.array_equal(still, tracked))

        col = int(960.0 / heat.cell)
        row = int(540.0 / heat.cell)
        probe_row = MATRIX_TOP + row * scale + scale // 2
        probe_col = MATRIX_LEFT + col * scale + scale // 2
        far_x = MATRIX_LEFT + heat.grid_w * scale - 10  # empty cells, right of the peak
        far_y = MATRIX_TOP + heat.grid_h * scale - 10

        # white crosshair across the view where the data is dark
        self.assertGreater(int(tracked[probe_row, far_x].sum()), 600)
        self.assertLess(int(still[probe_row, far_x].sum()), 200)
        self.assertGreater(int(tracked[far_y, probe_col].sum()), 600)
        self.assertLess(int(still[far_y, probe_col].sum()), 200)

    def test_empty_matrix_stays_uniform_but_labelled(self):
        from gaze_core import AttentionHeatmap

        heat = AttentionHeatmap(*SCREEN, min_confidence=0.0)
        panel = draw_matrix(heat, True, None, 0.0)
        scale = self._scale(heat)
        region = self._region(panel, heat, scale)
        # a sample block clear of the axis frame and the quarter gridlines
        block = region[10:130, 10:230]
        self.assertTrue(np.all(block == block.reshape(-1, 3)[0]))
        self.assertGreater(int(panel[:MATRIX_TOP].max()), 100)


class TestValidation(unittest.TestCase):
    def test_error_statistics(self):
        stats = error_statistics([10.0, 20.0, 30.0, 40.0, 100.0])
        self.assertAlmostEqual(stats["mean"], 40.0)
        self.assertAlmostEqual(stats["median"], 30.0)
        self.assertAlmostEqual(stats["max"], 100.0)
        self.assertEqual(stats["count"], 5)
        empty = error_statistics([])
        self.assertEqual(empty["count"], 0)

    def test_angular_error_requires_geometry(self):
        value = angular_error_degrees(30.0, 1920, None, None)
        self.assertIsNone(value)
        known = angular_error_degrees(30.0, 1920, screen_width_cm=53.0, camera_distance_cm=60.0)
        self.assertIsNotNone(known)
        self.assertGreater(known, 0.0)

    def test_validation_run_collects_records(self):
        run = ValidationRun(*SCREEN, gc.TARGET_LAYOUTS[9], samples_per_target=3,
                            stabilize_frames=2, min_confidence=0.3)
        run.start()
        rng = np.random.default_rng(2)
        state = "idle"
        for _ in range(len(gc.TARGET_LAYOUTS[9])):
            target = run.target_px
            # one frame to stabilize, then three recorded predictions
            state = run.step((target[0] + rng.normal(0, 5), target[1] + rng.normal(0, 5)), 0.9)
            for _ in range(3):
                target = run.target_px
                predicted = (target[0] + rng.normal(0, 20), target[1] + rng.normal(0, 20))
                state = run.step(predicted, 0.9)
        self.assertEqual(state, "done")
        self.assertEqual(len(run.records), 27)
        summary = summarize_validation(run.records, attempts=60)
        self.assertLess(summary["overall"]["mean"], 60.0)
        self.assertEqual(summary["recorded_samples"], 27)
        self.assertAlmostEqual(summary["valid_sample_rate"], 27 / 60, places=5)
        self.assertIn("per_target", summary)
        self.assertIn("by_head_pose", summary)
        self.assertIn("by_distance", summary)   # no distances measured -> unknown bucket
        self.assertEqual(summary["by_distance"]["unknown"]["count"], 27)
        self.assertIsNone(summary["distance_cm"])

    def test_validation_buckets_error_by_eye_distance(self):
        self.assertEqual(distance_condition(None), "unknown")
        self.assertEqual(distance_condition(40.0), "near (<45 cm)")
        self.assertEqual(distance_condition(60.0), "typical (45-70 cm)")
        self.assertEqual(distance_condition(90.0), "far (>70 cm)")

        run = ValidationRun(*SCREEN, gc.TARGET_LAYOUTS[5], samples_per_target=2,
                            stabilize_frames=1, min_confidence=0.3)
        run.start()
        target = run.target_px
        run.step((target[0], target[1]), 0.9, distance_cm=40.0)
        run.step((target[0], target[1]), 0.9, distance_cm=80.0)
        self.assertEqual(len(run.records), 2)
        summary = summarize_validation(run.records, attempts=2)
        self.assertEqual(summary["by_distance"]["near (<45 cm)"]["count"], 1)
        self.assertEqual(summary["by_distance"]["far (>70 cm)"]["count"], 1)
        stats = summary["distance_cm"]
        self.assertAlmostEqual(stats["mean_cm"], 60.0)
        self.assertAlmostEqual(stats["min_cm"], 40.0)
        self.assertAlmostEqual(stats["max_cm"], 80.0)
        # samples without a distance fall into their own bucket instead of vanishing
        run2 = ValidationRun(*SCREEN, gc.TARGET_LAYOUTS[5], samples_per_target=1,
                             stabilize_frames=1, min_confidence=0.3)
        run2.start()
        target = run2.target_px
        run2.step((target[0], target[1]), 0.9)
        summary2 = summarize_validation(run2.records, attempts=1)
        self.assertEqual(summary2["by_distance"]["unknown"]["count"], 1)
        self.assertIsNone(summary2["distance_cm"])
        # the report exposes both new structures
        report = build_report(
            model="affine", calibration_method="test", calibration_points=5,
            samples_per_point=1, validation=summary,
        )
        self.assertEqual(report["error_by_distance_px"], summary["by_distance"])
        self.assertEqual(report["eye_distance_cm"], summary["distance_cm"])

    def test_low_confidence_is_invalid(self):
        run = ValidationRun(*SCREEN, gc.TARGET_LAYOUTS[5], samples_per_target=2,
                            stabilize_frames=1, min_confidence=0.5)
        run.start()
        self.assertEqual(run.step((10.0, 10.0), 0.1), "invalid")

    def test_report_and_files(self):
        records = [
            ValidationRecord(0, 100.0, 100.0, 130.0, 140.0, 0.9, yaw=3.0, pitch=1.0),
            ValidationRecord(0, 100.0, 100.0, 110.0, 120.0, 0.8, yaw=25.0, pitch=2.0),
            ValidationRecord(1, 1800.0, 980.0, 1750.0, 940.0, 0.7, yaw=2.0, pitch=-2.0),
        ]
        summary = summarize_validation(records, attempts=10)
        report = build_report(
            model="affine",
            calibration_method="9-point multi-sample",
            calibration_points=9,
            samples_per_point=20,
            validation=summary,
            config=GazeConfig(),
        )
        self.assertEqual(report["model"], "affine")
        self.assertEqual(report["number_of_calibration_points"], 9)
        self.assertTrue(math.isfinite(report["mean_error_px"]))
        self.assertIn("neutral", report["error_by_head_pose_px"])
        with tempfile.TemporaryDirectory() as folder:
            prefix = os.path.join(folder, "report")
            json_path, csv_path = save_report(report, prefix)
            with open(json_path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertEqual(loaded["model"], "affine")
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("mean_error_px", content)

    def test_head_pose_condition_buckets(self):
        from phase2_coordinate_mapping import head_pose_condition

        self.assertEqual(head_pose_condition(0.0, 0.0), "neutral")
        self.assertEqual(head_pose_condition(12.0, 5.0), "slight")
        self.assertEqual(head_pose_condition(30.0, 5.0), "moderate")
        self.assertEqual(head_pose_condition(None, None), "unknown")


class TestEndToEndSynthetic(unittest.TestCase):
    """Calibrate -> map -> validate on synthetic data, without a webcam."""

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "calibration.npz")
            config = GazeConfig(mapping_model="affine")
            mapper = ScreenMapper(*SCREEN, path=path, config=config, smoothing=1.0)
            samples = synthetic_calibration_samples(9, 20)
            self.assertTrue(mapper.fit(samples))

            run = ValidationRun(*SCREEN, gc.TARGET_LAYOUTS[9], samples_per_target=5,
                                stabilize_frames=1, min_confidence=0.3)
            run.start()
            rng = np.random.default_rng(11)
            for _ in range(len(gc.TARGET_LAYOUTS[9])):
                for _ in range(5):
                    target = run.target_px
                    # reconstruct features that produced this target plus noise
                    features = rng.normal(0.0, 0.15, gc.FEATURE_DIM)
                    features[0] += target[0] / SCREEN[0] - 0.5
                    features[2] += target[1] / SCREEN[1] - 0.5
                    features[8] += (target[0] / SCREEN[0] - 0.5) * 0.6
                    position = mapper.map((0.5, 0.5), features, confidence=0.9)
                    run.step(position, 0.9, features=features)
            self.assertTrue(run.done)
            summary = summarize_validation(run.records, run.attempts)
            # affine recovers the synthetic mapping closely
            self.assertLess(summary["overall"]["mean"], 250.0)

            pairs = run.feature_target_pairs()
            self.assertIsNotNone(pairs)
            comparisons = evaluate_models_on_features(
                samples, pairs[0], pairs[1], ("affine", "polynomial", "mlp"), config, SCREEN
            )
            self.assertEqual(len(comparisons), 3)
            for name, stats in comparisons.items():
                self.assertEqual(stats["fit"], 1.0, name)


class TestEyeTrackerSynthetic(unittest.TestCase):
    """Full Phase 1 path (warp -> pupil -> head pose -> features -> fusion).

    Runs the real `EyeTracker.process` on a geometrically consistent synthetic
    face whose MediaPipe landmarks are injected through a stub mesh, so the eye
    extraction pipeline is exercised without a webcam.
    """

    WIDTH, HEIGHT = 1280, 720
    DISTANCE = 1600.0

    @staticmethod
    def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
        ry, rx, rz = map(math.radians, (yaw, pitch, roll))
        ry_mat = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        rx_mat = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        rz_mat = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        return rz_mat @ rx_mat @ ry_mat

    def model_landmarks(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
        """478 model-space landmarks (millimetres) for a synthetic frontal face."""
        model: dict = {}
        # eye ellipses: contour ids follow the angular order of gc.EYE_TEMPLATES
        left_template = gc.EYE_TEMPLATES[0][5]
        right_template = gc.EYE_TEMPLATES[1][5]
        for i, index in enumerate(left_template):  # theta = pi -> pi/2 -> 0 -> -pi/2
            theta = math.pi - i * math.pi / 8.0
            model[index] = (-175.0 + 50.0 * math.cos(theta), 170.0 - 18.0 * math.sin(theta), -140.0)
        for i, index in enumerate(right_template):  # theta = 0 -> pi/2 -> pi -> 3pi/2
            theta = i * math.pi / 8.0
            model[index] = (275.0 + 50.0 * math.cos(theta), 170.0 - 18.0 * math.sin(theta), -140.0)
        model[468] = (-175.0, 170.0, -145.0)  # left iris centre
        model[473] = (275.0, 170.0, -145.0)  # right iris centre
        # head-pose anchors last so they exactly match HEAD_POSE_MODEL
        for index, coords in zip(gc.HEAD_POSE_LANDMARK_IDS, gc.HEAD_POSE_MODEL):
            model[index] = tuple(coords)
        points = np.array([model.get(i, (0.0, 170.0, -140.0)) for i in range(478)], dtype=np.float64)
        rotation = self._rotation(yaw, pitch, roll)
        return (rotation @ points.T).T + np.array([0.0, 0.0, self.DISTANCE])

    def project(self, model_points: np.ndarray) -> np.ndarray:
        camera = gc._camera_matrix(self.WIDTH, self.HEIGHT)
        projected = (camera @ model_points.T).T
        return (projected[:, :2] / projected[:, 2:3]).astype(np.float32)

    def synthetic_frame(self, image_points: np.ndarray) -> np.ndarray:
        """Render skin + sclera + pupil so pupil detection and contrast work."""
        frame = np.full((self.HEIGHT, self.WIDTH, 3), (140, 150, 165), np.uint8)
        for contour_ids in (gc.EYE_TEMPLATES[0][5], gc.EYE_TEMPLATES[1][5]):
            polygon = np.round(image_points[list(contour_ids)]).astype(np.int32)
            cv2.fillPoly(frame, [polygon], (235, 235, 235))
        for iris_id in (468, 473):
            center = tuple(int(round(v)) for v in image_points[iris_id])
            cv2.circle(frame, center, 6, (25, 25, 30), -1, cv2.LINE_AA)
        return frame

    def stub_mesh(self, image_points: np.ndarray):
        class _StubLandmark:
            def __init__(self, x, y):
                self.x, self.y, self.z = float(x), float(y), 0.0

        class _StubMesh:
            def __init__(self, points):
                self._raw = SimpleNamespace(
                    multi_face_landmarks=[
                        SimpleNamespace(landmark=[_StubLandmark(p[0], p[1]) for p in points])
                    ]
                )

            def process(self, _rgb):
                return self._raw

            def close(self):
                return None

        normalized = image_points / np.array([self.WIDTH, self.HEIGHT], dtype=np.float32)
        return _StubMesh(normalized)

    def setUp(self):
        if gc.mp is None:
            self.skipTest(f"mediapipe unavailable: {gc.MEDIAPIPE_ERROR}")
        from gaze_core import EyeTracker

        self.tracker = EyeTracker(config=GazeConfig())
        self.real_mesh = self.tracker.mesh
        self.image_points = self.project(self.model_landmarks(yaw=4.0, pitch=-3.0, roll=2.0))
        self.tracker.mesh = self.stub_mesh(self.image_points)
        self.frame = self.synthetic_frame(self.image_points)

    def tearDown(self):
        if hasattr(self, "tracker"):
            self.tracker.close()

    def test_process_extracts_features_pose_and_confidence(self):
        result = self.tracker.process(self.frame)
        self.assertIsNotNone(result.gaze)
        self.assertIsNotNone(result.features)
        self.assertEqual(result.features.shape, (gc.FEATURE_DIM,))
        self.assertTrue(np.isfinite(result.features).all())
        self.assertEqual(len(result.eyes), 2)
        for eye in result.eyes:
            self.assertIsNotNone(eye.gaze, eye.side)
            self.assertGreater(eye.quality, 0.2, eye.side)
        # head pose recovers the injected yaw/pitch/roll
        self.assertIsNotNone(result.head_pose)
        self.assertTrue(result.head_pose.valid)
        self.assertAlmostEqual(result.head_pose.yaw, 4.0, delta=3.0)
        self.assertAlmostEqual(result.head_pose.pitch, -3.0, delta=3.0)
        self.assertAlmostEqual(result.head_pose.roll, 2.0, delta=3.0)
        # separate confidence components all behave sensibly
        conf = result.confidences
        self.assertIsNotNone(conf)
        self.assertGreater(conf.pupil, 0.5)
        self.assertGreater(conf.landmark, 0.4)
        self.assertGreater(conf.head_pose, 0.6)
        self.assertGreater(conf.gaze, 0.5)
        self.assertGreater(result.confidence, 0.5)

    def test_temporal_state_on_second_frame(self):
        self.tracker.process(self.frame)
        second = self.tracker.process(self.frame)  # identical frame: temporally steady
        self.assertIsNotNone(second.gaze)
        self.assertGreater(second.confidences.temporal, 0.9)
        # gaze must stay in normalized eye coordinates
        self.assertTrue(0.0 <= second.gaze[0] <= 1.0)
        self.assertTrue(0.0 <= second.gaze[1] <= 1.0)

    def test_overlay_rendering_does_not_crash(self):
        result = self.tracker.process(self.frame)
        view = self.tracker.draw_overlay(self.frame, result)
        self.assertEqual(view.shape, self.frame.shape)
        crops = self.tracker.draw_crops(result, scale=1.0)
        self.assertGreater(crops.size, 0)

    def test_distance_estimate_grows_when_the_face_moves_back(self):
        near = self.tracker.process(self.frame)
        self.assertIsNotNone(near.distance_cm)
        self.assertGreater(near.distance_cm, gc.EYE_DISTANCE_MIN_CM)
        self.assertLess(near.distance_cm, gc.EYE_DISTANCE_MAX_CM)
        # same face 1.5x farther: the inter-ocular width shrinks, so the
        # pinhole estimate grows (the EMA only moves part-way in one frame)
        far_points = self.project(
            self.model_landmarks(yaw=4.0, pitch=-3.0, roll=2.0)
            + np.array([0.0, 0.0, self.DISTANCE * 0.5])
        )
        self.tracker.mesh = self.stub_mesh(far_points)
        far = self.tracker.process(self.synthetic_frame(far_points))
        self.assertIsNotNone(far.distance_cm)
        self.assertGreater(far.distance_cm, near.distance_cm)
        self.assertLess(far.distance_cm, 1.5 * near.distance_cm + 1.0)
        self.assertAlmostEqual(self.tracker.distance_cm, far.distance_cm)

    def test_distance_kept_when_the_face_is_lost(self):
        seen = self.tracker.process(self.frame)
        self.assertIsNotNone(seen.distance_cm)
        self.tracker.mesh = self.real_mesh
        lost = self.tracker.process(np.full_like(self.frame, 240, dtype=np.uint8))
        self.assertIsNone(lost.gaze)
        self.assertIsNone(lost.distance_cm)
        self.assertEqual(self.tracker.distance_cm, seen.distance_cm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
