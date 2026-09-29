# attention-track-eye

A webcam-based **gaze-estimation pipeline** that maps face and eye features to screen
coordinates and renders a fixation-aware **gaze-density heatmap**.

Pupil/iris and facial features are used as inputs to a calibrated gaze-estimation model.
Pupil position alone is never treated as the direct gaze direction.

> **Research integrity note.** This project is a working research *toolkit*, not a
> certified research-grade instrument. No accuracy improvement is claimed anywhere in
> this repository: the code ships with the measurement machinery (validation mode,
> per-target and per-head-pose error statistics, model comparison, JSON/CSV reports),
> and any accuracy statement must come from an experiment **you** run on **your**
> hardware. See [Validation](#7-validation-methodology) and [Limitations](#11-limitations).

---

## 1. System overview

```text
                  WEBCAM
                     │
                     ▼
             Face Detection (MediaPipe FaceMesh, 478 landmarks)
                     │
             ┌───────┴────────┐
             ▼                ▼
       Eye Features       Head Pose (solvePnP → yaw / pitch / roll)
             │                │
             └───────┬────────┘
                     ▼
              Feature Vector (11 normalized channels)
                     │
                     ▼
              Gaze Estimator ── affine / polynomial / MLP
                     │
                     ▼
               Screen X/Y  ── outlier rejection
                     │
                     ▼
             Temporal Filtering (EMA or adaptive 1€ filter)
                     │
                     ▼
           Confidence Estimation (5 separate components)
                     │
              ┌──────┴──────┐
              ▼             ▼
          Fixation       Saccade
          Detection      Detection   (I-DT / I-VT)
              │
              ▼
      Fixation- and Confidence-Weighted Gaze Accumulation
              │
              ▼
        Time-Based Decay   grid *= exp(-Δt / τ)
              │
              ▼
         Gaussian Blur     σ given in screen pixels, converted to cells
              │
              ▼
         Normalization → Gamma Correction → Heatmap Rendering
```

## 2. Files and responsibilities

| File | Lines | Responsibility |
|---|---|---|
| `gaze_core.py` | 2272 | Shared core: `GazeConfig`, feature definitions, confidence model, head pose, eye fusion, eye-distance estimate, calibration, mapping models, heatmap |
| `phase1_eye_tracking.py` | 213 | Phase 1 — eye ROI, pupil/iris detection, head pose, feature vector, eye fusion (live webcam UI) |
| `phase2_coordinate_mapping.py` | 948 | Phase 2 — calibration, mapping models, outlier rejection, smoothing, validation mode, evaluation reports |
| `phase3_attention_heatmap.py` | 877 | Phase 3 — fixation/saccade detection, weighted accumulation, time decay, blur, rendering, export |
| `test_gaze_pipeline.py` | 1235 | Offline pipeline test suite (78 tests, no webcam required) |
| `test_webapp.py` | 506 | Headless browser-interface test suite (20 tests, no webcam required) |
| `web_app.py` | 762 | Flask server: live browser sessions, calibration, frame loop, distance tracking, heatmap report |
| `web_interface.html` | 1028 | Browser UI: document upload, webcam capture, calibration, live gaze dot, distance/warning, report |
| `process_explainer.html` | 1381 | Browser documentation + interactive conceptual simulation |

Three-phase organization is preserved: **Phase 1** eye/head-pose feature extraction,
**Phase 2** coordinate mapping, **Phase 3** gaze-density heatmap.

---

## 3. Phase 1 — Eye and head-pose feature extraction

Implemented in `gaze_core.py` (`EyeTracker`, `estimate_head_pose`, `build_feature_vector`,
`fuse_eye_gazes`, `eye_quality`) and driven by `phase1_eye_tracking.py`.

**What is preserved from the original prototype:** face landmark detection, eye-region
extraction, perspective normalization, pupil/iris detection, left/right eye processing
and a normalized gaze representation.

**What is distinguished:**

- **Pupil/iris localization** — where the pupil is inside the normalized eye crop
  (an image-space quantity, 0–1 within the 200×100 crop).
- **Eye gaze estimate** — the per-eye normalized pupil offset used as a model input.
- **Screen gaze estimate** — only produced by Phase 2 after calibration; never inferred
  from pupil position alone.

### Head pose

Yaw/pitch/roll are estimated with `cv2.solvePnP` from six stable landmarks
(nose tip, chin, two eye corners, two mouth corners) against a 3D head model, using a
pinhole camera with ~60° horizontal field of view (`focal = 1.1 · max(W, H)`).
Head pose enters the feature vector and the confidence model so head movement and
camera-relative face orientation are compensated for in the mapping stage.

Head-pose compensation **reduces** the error caused by head movement; it does not
eliminate it, and the effect is meant to be measured (see
[Validation methodology](#7-validation-methodology)), not assumed.

### Eye-to-camera distance

`estimate_eye_distance_cm` measures how far the eyes are from the camera with the
same pinhole model: `distance = focal_px · EYE_REFERENCE_WIDTH_CM / iod_px`, where
`iod_px` is the pixel width between the two outer eye corners (MediaPipe landmarks
33 / 263) and `EYE_REFERENCE_WIDTH_CM` = 9.5 cm is the average adult canthal width.
The reading is plausibility-gated (20–300 cm), EMA-smoothed across frames
(`DISTANCE_SMOOTHING_ALPHA` = 0.3) and reported as `GazeResult.distance_cm`.

Absolute accuracy is limited by the assumed field of view and by per-person face
width (roughly ±10 %), so the value is honest about what it is: a *measured,
displayed* distance whose **frame-to-frame changes** (which drive drift detection)
are considerably more accurate than the absolute number. Phase 1's HUD, the Phase 2
and 3 HUDs and the web sidebar all show it.

### Feature vector (11 channels)

| Channel(s) | Meaning |
|---|---|
| `left_eye_x`, `left_eye_y`, `right_eye_x`, `right_eye_y` | Pupil position relative to the eye-aperture centre, normalized by aperture size (eye-centred, ≈ −1.5…1.5) |
| `left_iris_x/y`, `right_iris_x/y` | Pupil/iris position inside the normalized 200×100 eye crop (0…1); imputed to 0.5 when an eye is missing |
| `head_yaw`, `head_pitch`, `head_roll` | Head pose in degrees divided by `HEAD_POSE_SCALE_DEG = 30` so all channels share a comparable numeric scale |

### Confidence-weighted eye fusion

The two eyes are **not** simply averaged. Each eye gets a quality score in [0, 1]:

```text
quality = 0.35·pupil_detection_confidence
        + 0.15·pupil_contour_quality
        + 0.15·eye_geometry_plausibility
        + 0.15·image_contrast_in_crop
        + 0.10·temporal_consistency
        + 0.10·aperture_visibility
        scaled by clip(head_pose_confidence, 0.4, 1.0)

fused_gaze = (left·q_left + right·q_right) / (q_left + q_right)
```

When the summed quality falls below `min_total_eye_quality` (0.12) the sample is
rejected instead of silently producing a plausible-looking point.

---

## 4. Phase 2 — Calibration and coordinate mapping

Implemented in `gaze_core.py` (`ScreenCalibrator`, `ScreenMapper`, mapping models) and
driven by `phase2_coordinate_mapping.py`.

### Calibration procedure

Layouts (item "upgrade calibration"):

| Points | Layout | Role |
|---|---|---|
| 5 | 4 corners (8 %/92 %) + centre | Lightweight legacy mode |
| 9 | 3×3 grid at 10/50/90 %, centre first | Research default |
| 13 | 9-point grid + 4 inner-diamond points | Denser interior coverage |

Per target:

1. Display the target.
2. **Stabilize** — dwell for `calibration_stabilize_frames` (default 12) valid frames.
3. **Collect** — accept `calibration_samples_per_point` frames (default 20).
4. **Reject** frames whose combined confidence < `calibration_min_confidence` (0.35);
   a rejected frame resets the stabilization counter.
5. **Reject extreme outliers** per target with a MAD-based z-score
   (`calibration_outlier_z` = 3.0).
6. **Reduce** — compute a robust median feature vector per target
   (`robust_median`), stored as the representative sample.
7. Store every accepted `Features → Known Screen Coordinate` pair for model fitting.

A single frame is never used as the calibration sample. Sample count, dwell time and
confidence gate are all configurable. Calibration is saved to / loaded from
`calibration.npz` (model parameters, samples, targets, screen size).

### Mapping models

Configured with `GazeConfig.mapping_model` (or `--mapping`):

| Model | Form | Notes |
|---|---|---|
| `affine` | `X = a₁f₁+…+b₁`, `Y = c₁f₁+…+b₂` | Transparent least-squares baseline (default) |
| `polynomial` | 2nd-order terms: `x, y, x², y², xy, …` on standardized features | Light ridge regularization (`polynomial_ridge` = 1e-2) |
| `mlp` | Dense(16) → tanh → Dense(2) | Optional (Priority 3), plain NumPy, never mandatory |

More complex is **not** assumed to be better — use `--compare-models` during
validation to measure them against each other.

### Outlier rejection (before screen update and heatmap update)

- combined confidence < `confidence_threshold` (0.5) → rejected
- missing feature vector while calibrated → rejected
- prediction outside the screen by more than a 10 % margin → rejected
- displacement > `outlier_max_jump_px` (default: 1.5 × screen diagonal, deliberately
  beyond any reachable on-screen jump) → rejected

The jump threshold is set so that **legitimate rapid saccades are never rejected for
being fast**; an optional speed limit (`outlier_max_speed_px_s`, 0 = disabled) exists
for constrained setups. Every rejection is counted in `ScreenMapper.stats`.

### Temporal smoothing

Configured with `GazeConfig.smoothing_filter`:

- **`ema`** (default) — `new = α·x + (1−α)·old` with `ema_alpha` (0.3, `1` disables).
  One knob, but lag and jitter are two ends of the same trade-off.
- **`oneeuro`** — the adaptive *1 euro filter*: cutoff = `oneeuro_min_cutoff` +
  `oneeuro_beta·|speed|`, so a still signal is filtered hard (kills jitter) while a
  fast one passes with almost no added lag. The browser demo defaults to it.

No value is hardcoded; smoothing lag is the trade-off you tune.

### Distance compensation

For a fixed gaze angle the spot on the screen moves linearly with the eye-to-screen
distance, so when a distance reference exists (set at calibration, persisted in
`calibration.npz` as `reference_distance_cm`), `ScreenMapper.map` scales the offset
from the screen centre by `distance_cm / reference_distance_cm` — clamped to
[0.7, 1.4] so a wrong reading (a different face, a failed estimate) can never
wreck the mapping. Leaning toward or away from the camera is compensated instead
of showing up as a systematic offset, and drift past `distance_drift_warn_pct`
(15 %) raises a recalibration warning.

---

## 5. Phase 3 — Fixation-aware gaze-density heatmap

Implemented in `gaze_core.py` (`AttentionHeatmap`) and `phase3_attention_heatmap.py`
(`EventDetector`).

### Fixation / saccade detection

At least one established method is implemented — in fact both:

- **I-DT** (Dispersion Threshold Identification, default): a sliding window stays a
  fixation while its dispersion ≤ `fixation_dispersion_px` (45 px) and it lasts ≥
  `fixation_min_duration_ms` (100 ms).
- **I-VT** (Velocity Threshold Identification, `--fixation-method ivt`): velocity below
  `saccade_velocity_threshold` (1.0 px/ms = 1000 px/s) is stable, above is movement.

Samples are classified as `fixation`, `saccade`, or `invalid / low confidence`.
Each fixation stores start time, end time, duration, centroid, confidence, dispersion
and sample count; saccades store duration, amplitude, peak velocity and confidence.

### Accumulation, decay and rendering

```text
contribution = gaze_confidence × fixation_weight × base_weight
fixation_weight = Δt × 10 per second   (saccades × 0.1)
grid[cell] += contribution             (samples below heatmap_min_confidence are dropped)

decay:      grid *= exp(-Δt / τ)        τ = heatmap_decay_tau (5 s), frame-rate independent
blur:       σ_screen_px = 45 → σ_cells = σ / cell_size = 45 / 4 = 11.25 → kernel 69
normalize → gamma (0.75) → TURBO colormap → alpha blend (0.7)
```

A stable 700 ms fixation contributes far more than a single noisy sample
(≈ 7.0 vs ≈ 0.1 in the default weighting), and low-confidence samples contribute less
or are discarded (`heatmap_min_confidence` = 0.35).

**Units:** `gaussian_sigma_px` is specified in **screen pixels** and converted to grid
cells (`sigma_cells = sigma / cell_size`); this is documented on the class and shown in
the Phase 3 HUD. Decay is `exp(-Δt/τ)`, never a per-frame constant, so behaviour is
consistent at different frame rates.

**Accumulation-matrix window:** `[v]` switches the companion OpenCV window between the
raw grid and the blurred one. Both are drawn in screen coordinates — 0…1920 / 0…1080
axis labels, quarter gridlines, a white crosshair with `cell=(col, row)` and `value=`
for the live gaze cell — and scaled with `log1p` against the session's running maximum of
the raw grid instead of each frame's peak, followed by the same gamma + TURBO chain as
the rendered heatmap. Weak cells stay visible, growth and decay stay readable, and the
raw/blur toggle shares one denominator (a Gaussian blur cannot raise the peak, so the
blurred view never clips). The header reports the peak, the reference, σ in px and
cells, grid shape, hits, rejects and accumulated weight.

**Terminology:** the baseline output is a **gaze-density heatmap**. It can be described
as a visual-attention estimation component only because fixation duration and
confidence are folded into the weights.

### Export

`[s]` (or on exit): `attention_points.npy` (time, x, y, weight), `attention_grid.npy`,
`fixation_events.json` (per-fixation / per-saccade records + summary including
fixation detection rate, average fixation duration and saccade count).

---

## 6. Confidence model

Five separate, interpretable components — never one opaque number:

| Component | Meaning | Default weight |
|---|---|---|
| `pupil` | Reliability of pupil/iris localization | 0.30 |
| `landmark` | Stability/reliability of facial landmarks (face size + temporal stability) | 0.20 |
| `head_pose` | Stability and plausibility of the head-pose estimate (reprojection error + angular limits + temporal consistency) | 0.15 |
| `gaze` | Reliability of the combined gaze estimate (`0.45·pupil + 0.25·landmark + 0.20·head_pose + 0.10·two-eye completeness`) | 0.25 |
| `temporal` | Consistency with recent gaze estimates (tolerance grows with Δt so saccades are not punished) | 0.10 |

`combined = Σ(component × weight) / Σ(weights)`. All weights live in
`GazeConfig.confidence_weights` and the breakdown is printed in every HUD.

---

## 7. Validation methodology

Phase 2 has a guided screen-space validation mode (`--validate` or the `[v]` key):

1. Targets from a 5/9/13-point layout are displayed one at a time.
2. After a stabilization period, predictions are collected
   (`--validation-samples`, default 15 per target).
3. For each record the error is
   `error_px = sqrt((predicted_x − target_x)² + (predicted_y − target_y)²)`.
4. Statistics are aggregated overall, **per target** and **per head-pose condition**
   (neutral / slight / moderate from yaw and pitch buckets).

Optional comparisons:

- **Raw / baseline vs calibrated** — run validation once without a calibration file and
  once after calibrating, then compare the two reports.
- **Affine vs polynomial vs MLP** — `--compare-models` refits all three on the stored
  calibration samples and evaluates each on the same validation features.

Optional angular error (`angular_error_degrees`) is reported **only** when the physical
geometry is supplied (`--screen-width-cm` and `--camera-distance-cm`); it is never
computed from guessed numbers.

### Evaluation metrics

Reported fields (JSON + CSV, `--report-prefix`, default `evaluation_report`):

```text
Model · Calibration Method · Number of Calibration Points · Samples per Point
Mean Error · Median Error · Standard Deviation · 95th Percentile Error · Maximum Error
Valid Sample Rate · Average Confidence · Recorded Samples
per-target error · error-by-head-pose · error-by-eye-distance · eye-distance stats
· angular error (optional) · model comparison · config
```

Phase 3 adds fixation detection rate, average fixation duration and saccade count to
`fixation_events.json`.

Run example:

```bash
python phase2_coordinate_mapping.py --screen 1920x1080 --points 9 \
    --samples-per-point 20 --mapping affine --validate --compare-models
```

---

## 8. Configuration

All parameters are centralized in the `GazeConfig` dataclass (`gaze_core.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `screen_width`, `screen_height` | 1920, 1080 | Virtual screen size |
| `calibration_points` | 9 | 5 / 9 / 13 target layout |
| `calibration_samples_per_point` | 20 | Accepted frames per target |
| `calibration_stabilize_frames` | 12 | Dwell before sampling starts |
| `calibration_min_confidence` | 0.35 | Confidence gate during calibration |
| `calibration_outlier_z` | 3.0 | MAD z-score for outlier rejection |
| `min_calibration_span` | 0.25 | Minimum feature spread for a fit |
| `mapping_model` | `affine` | `affine` / `polynomial` / `mlp` |
| `polynomial_ridge` | 1e-2 | Ridge term for polynomial fit |
| `mlp_hidden`, `mlp_epochs`, `mlp_learning_rate` | 16, 400, 5e-3 | Optional MLP |
| `ema_alpha` | 0.3 | EMA smoothing factor (1 = off) |
| `smoothing_filter` | `ema` | `ema` / `oneeuro` (adaptive: strong when still, light when moving) |
| `oneeuro_min_cutoff`, `oneeuro_beta`, `oneeuro_d_cutoff` | 1.0, 0.01, 1.0 | 1 € filter Hz / speed weight / derivative smoothing |
| `distance_drift_warn_pct` | 15.0 | Eye-distance drift from the reference before a warning |
| `confidence_threshold` | 0.5 | Sample rejection gate |
| `outlier_max_jump_px` | 0.0 (auto = 1.5 × diagonal) | Implausible-jump limit |
| `outlier_max_speed_px_s` | 0.0 (disabled) | Optional speed limit |
| `min_total_eye_quality` | 0.12 | Eye-fusion rejection floor |
| `heatmap_cell_size` | 4 | Grid cell in screen px (480×270 grid) |
| `heatmap_decay_tau` | 5.0 | Time-decay constant τ in seconds (0 = off) |
| `gaussian_sigma_px` | 45.0 | Blur σ in **screen pixels** |
| `heatmap_gamma`, `heatmap_alpha` | 0.75, 0.7 | Rendering gamma / opacity |
| `heatmap_min_confidence` | 0.35 | Accumulation confidence gate |
| `fixation_method` | `idt` | `idt` / `ivt` |
| `fixation_dispersion_px` | 45.0 | I-DT dispersion threshold |
| `fixation_min_duration_ms` | 100.0 | Minimum fixation duration |
| `saccade_velocity_threshold` | 1.0 | px/ms for I-VT / saccade label |
| `confidence_weights` | pupil .3 / landmark .2 / head .15 / gaze .25 / temporal .1 | Confidence combination |

CLI flags mirror these (`--points`, `--samples-per-point`, `--ema-alpha`, `--mapping`,
`--cell`, `--sigma`, `--decay-tau`, `--gamma`, `--alpha`, `--min-confidence`,
`--fixation-method`, `--confidence-threshold`, …); run any phase with `--help`.

---

## 9. Running

```bash
pip install -r requirements.txt        # mediapipe, opencv-contrib-python, numpy, flask

python phase1_eye_tracking.py                                   # eye + head-pose features
python phase2_coordinate_mapping.py --screen 1920x1080          # calibration + mapping
python phase2_coordinate_mapping.py --validate --compare-models # validation report
python phase3_attention_heatmap.py --background camera          # gaze-density heatmap
python web_app.py                                               # live browser demo (webcam)
python -m unittest discover -p "test_*.py"                      # offline test suite (98 tests)
```

### Live browser demo (webcam required)

```bash
python web_app.py            # then open http://127.0.0.1:5000
```

1. **Upload** a PNG/JPG document (PDF works too — PDF.js is loaded from a CDN, all
   pages are stacked into one image; PNG/JPG need no internet).
2. **Start camera & tracking** and allow webcam access. Choose either a 5- or 9-point
   **calibration** (follow the dot with your eyes only, head still) or skip it for a
   rough centre mapping. Options also cover the mapping model and time decay.
3. **Watch** the live gaze dot move on the document; the sidebar shows confidence,
   fps, face detection, the measured **eye-to-camera distance** (with a warning when
   it drifts >15 % from the distance you calibrated at) and the calibration residual.
   The frame loop runs as fast as the server can answer (~16 ms floor instead of a
   fixed 80 ms interval), and smoothing defaults to the adaptive 1 € filter so the
   dot keeps up without jitter.
4. **Grid view (optional)** — press **Show grid** to overlay a 6×6 grid. Every 5 s a
   coloured digit (0–9, shuffled) pops up in a different cell; look at each digit.
   **Stop grid — fit mapping** refits the screen ↔ camera model from the collected
   points and reports the measured mapping error before and after the refit — the
   document fills the whole stage so the grid covers it edge to edge.
5. **Finish — build heatmap report** overlays the accumulated, confidence-weighted
   gaze density on the uploaded document at its natural scale — TURBO colormap,
   blue = low density, red = high density — plus session statistics (fixations,
   saccades, durations, calibration residual, eye-distance mean/min/max and drift,
   heatmap parameters, grid-mapping stats when used) and PNG/JSON export.

Everything runs locally: video frames travel only from your browser to
`127.0.0.1`. The demo reuses the exact pipeline from the CLIs (`EyeTracker` →
`ScreenCalibrator`/`ScreenMapper` → `AttentionHeatmap` → `EventDetector`) — it is a
real test of the implemented system, not a simulation. `--host` / `--port` change the bind.

Common hotkeys — Phase 1: `i` pupil mode, `m` mesh, `h` HUD, `f` features, `s` snapshot.
Phase 2: `c` calibrate, `SPACE` capture point, `v` validation, `m` mapping model,
`r` reset, `n` re-centre. Phase 3: `p` pause, `d` decay, `r` reset, `s` save,
`v` matrix view, `c` calibrate, `n` re-centre.

---

## 10. Validation performed on this codebase

Run with `python -m unittest discover -p "test_*.py"` (offline, deterministic):

- **98 tests pass** (78 pipeline + 20 web interface), `pyflakes` is clean on all Python files.
- Covered: configuration validation, feature-vector construction, head-pose recovery of
  known yaw/pitch/roll from projected landmarks, landmark/head-pose confidence
  behaviour, confidence breakdown, confidence-weighted fusion and its rejection path,
  eye-quality bounds, temporal confidence, eye-to-camera distance estimation (pinhole
  value, inverse IOD scaling, plausibility limits, EMA smoothing, keep-on-face-loss),
  the 1 € filter (constant signal, step response, EMA comparison, time guards), 
  calibration stabilization/collection/outlier rejection/manual capture, robust median,
  all three mapping models, calibration save/load round-trip (including the persisted
  distance reference), distance compensation (scaling, clamping, absence cases),
  outlier rejection (low confidence, missing features, implausible jumps), EMA
  smoothing, valid-sample rate, I-DT and I-VT fixation/saccade detection, heatmap
  confidence gating, fixation weighting, frame-rate-invariant time decay, σ unit
  conversion, rendering, validation statistics (including per-distance buckets),
  report/CSV export, head-pose condition buckets, model comparison, and a synthetic
  end-to-end Phase 1 run of the real `EyeTracker.process` (stubbed MediaPipe landmarks)
  that checks eye extraction, feature vector, head-pose recovery, all five
  confidence components and the distance estimate scaling when the face moves back.
- Web interface: session lifecycle, distance frame fields (reference, drift %,
  warning), calibration distance reference, report distance statistics, smoothing
  selection, real MediaPipe frame path on face-less frames, calibration fitting,
  heatmap report rendering and the blue→red colormap endpoints.
- Web interface (`test_webapp.py`): session/frame/finish/reset endpoint contracts and
  error paths, the real MediaPipe frame path on face-less frames, calibration
  stabilization → fit → residual reporting, report generation with the real
  `AttentionHeatmap` render (hot cluster red-dominant, weak cluster weaker, blur does
  not leak into untouched regions), both finish modes (cumulative and end-anchored
  time decay), and the blue→red colormap endpoints.
- All three CLIs start and parse arguments (`--help` verified); `web_app.py` was
  started and served `/`, `/api/health` and expected 409s correctly.

**Not run here (no webcam attached in the development environment):** live camera
capture, real-face landmark/pupil behaviour, on-screen calibration sessions, live
validation sessions, live heatmap accumulation, and an end-to-end browser session
(upload → camera → calibration → finish). Those paths must be exercised on a
machine with a camera; the reports they produce are the only valid basis for accuracy
claims.

---

## 11. Limitations

The system's accuracy depends on conditions that are not controlled by the algorithm:

- **Camera position** — a webcam below the screen forces a gaze/camera geometry that
  raises error compared with a camera near the display; the assumed ~60° FOV pinhole
  model is an approximation.
- **Lighting** — low light, backlighting or strong glare degrade pupil detection and
  landmark stability.
- **Glasses and reflections** — specular reflections can be mistaken for the pupil and
  occlude the iris.
- **Eye shape and individual anatomy** — eyelid shape, monolid eyes, heavy makeup,
  eyelashes and contact lenses change the eye aperture and the apparent pupil.
- **Head movement** — head-pose compensation reduces but does not eliminate error from
  head movement and camera-relative face orientation.
- **Face occlusion** — hands, hair or objects covering the face drop landmark
  confidence and can invalidate a frame entirely.
- **Webcam resolution and frame rate** — a small or noisy eye crop (200×100 from a
  720p frame) limits pupil-localization precision; low frame rates reduce temporal
  confidence resolution and fixation granularity.
- **Monitor geometry** — a second monitor, unusual scaling, or a preview window that is
  not the assumed virtual screen breaks the pixel mapping; angular error needs the real
  physical geometry.
- **Calibration quality** — sparse, rushed or low-confidence calibration produces an
  inaccurate mapping regardless of the model; the model comparison only tells you which
  fits *your* data.

Do not claim research-grade accuracy without measured validation.

---

## 12. Terminology

| Avoid | Use |
|---|---|
| "The pupil position directly represents where the user is looking." | "Pupil/iris and facial features are used as inputs to a calibrated gaze-estimation model." |
| "attention heatmap" for the default output | "**gaze-density heatmap**" |
| "gaze estimation" for the raw pupil offset | "pupil/iris localization" (Phase 1) vs "gaze estimation" (calibrated model) |
| "visual attention estimation" | Only when fixation duration + confidence weighting are actually applied (Phase 3) |

The JavaScript demo in `process_explainer.html` is labeled **Interactive Conceptual
Simulation (Phase 2–3 Visualization Demo)**: it uses mouse position as simulated gaze
and simplified math, and it does **not** validate the Python eye-tracking pipeline.

---

## 13. Research methodology notes

- Every accuracy number must come from a recorded validation run; reports include the
  configuration used so a run can be reproduced.
- Model comparisons (affine / polynomial / MLP, raw vs calibrated) are evaluated on the
  same validation features; a more complex model is not presumed better.
- Head-pose effects are reported per condition (neutral / slight / moderate) so the
  benefit of head-pose compensation can be measured rather than assumed.
- Confidence thresholds, outlier gates and rejection counts are logged
  (`ScreenMapper.stats`, `heatmap.rejected_low_confidence`) so the *valid sample rate*
  is visible next to the error statistics — a low error on 5 % of frames is not a
  result.
- The offline test suite validates algorithmic behaviour on synthetic data. It does not
  constitute an accuracy measurement.

See `process_explainer.html` for the illustrated version of this documentation.
