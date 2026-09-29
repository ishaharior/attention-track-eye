# Attention Track Eye — Research-Grade Upgrade Instructions

## Objective

Upgrade the existing webcam-based eye-gaze and attention heatmap system, **Attention Track Eye**, from a prototype into a more robust and experimentally defensible gaze-estimation pipeline.

The target architecture is:

```text
Webcam
  ↓
Face / Landmark Detection
  ↓
Eye / Iris Features + Head Pose
  ↓
Feature Normalization
  ↓
Gaze Estimation
  ↓
Screen Coordinate Mapping
  ↓
Temporal Filtering
  ↓
Confidence Estimation
  ↓
Fixation / Saccade Detection
  ↓
Fixation-Weighted Gaze Density
  ↓
Time-Based Decay
  ↓
Gaussian Blur
  ↓
Heatmap Rendering
```

Do not rewrite the project blindly. Inspect the existing implementation first, identify where each change belongs, implement incrementally, and preserve working functionality wherever practical.

---

# 1. Phase 1 — Improve Gaze Feature Extraction

Preserve the existing:

- Face landmark detection
- Eye-region extraction
- Perspective normalization
- Pupil/iris detection
- Left/right eye processing
- Normalized gaze representation

However, distinguish clearly between:

- pupil/iris localization
- eye gaze estimation
- screen gaze estimation

Do not treat pupil position alone as direct gaze direction.

---

# 2. Add Head-Pose Estimation

Estimate:

- Yaw
- Pitch
- Roll

Use stable facial landmarks and a suitable camera-model / PnP-based approach where practical.

Use head-pose features as inputs to gaze mapping.

Conceptually:

```text
Eye / Iris Features + Head Pose
                ↓
          Gaze Estimation
```

The purpose is to reduce error caused by head movement and camera-relative face orientation.

Do not claim that head-pose compensation completely eliminates these effects.

---

# 3. Create a Formal Feature Vector

Instead of relying only on `(gx, gy)`, construct a feature vector containing useful gaze-related information.

Example:

```text
left_eye_x
left_eye_y
right_eye_x
right_eye_y
left_iris_x
left_iris_y
right_iris_x
right_iris_y
head_yaw
head_pitch
head_roll
```

Add other stable geometric features where justified.

Normalize features where appropriate.

Avoid unnecessary or highly redundant features.

---

# 4. Improve Eye Fusion

Do not simply average both eyes.

Calculate a quality/confidence value for each eye based on factors such as:

- pupil/iris detection confidence
- eye landmark stability
- image quality
- pupil contour quality
- temporal consistency
- visibility / occlusion

Use confidence-weighted fusion:

```text
fused_gaze =
    (left_gaze * left_quality +
     right_gaze * right_quality)
    /
    (left_quality + right_quality)
```

Handle the case where both quality values are extremely low.

---

# 5. Separate Confidence Types

Do not use one generic confidence value.

Create separate concepts:

### Pupil Detection Confidence
Reliability of pupil/iris localization.

### Landmark Confidence
Stability/reliability of facial landmarks.

### Head-Pose Confidence
Stability and plausibility of head-pose estimation.

### Gaze Confidence
Reliability of the final gaze estimate.

### Temporal Confidence
Consistency with recent gaze estimates.

The final confidence may combine these components.

Keep the calculation interpretable.

---

# 6. Upgrade Calibration

Keep the existing 5-point calibration as a lightweight mode.

Add a research-oriented calibration mode using:

- 9-point calibration
- optionally 13-point calibration

For each target:

1. Display the target.
2. Wait for stabilization.
3. Collect multiple valid frames.
4. Reject low-confidence samples.
5. Reject extreme outliers.
6. Calculate a median or trimmed mean.
7. Store the representative feature vector and known screen coordinate.

Do not use a single frame as the calibration sample.

---

# 7. Improve Calibration Sample Collection

Calibration data should represent:

```text
Features → Known Screen Coordinate
```

rather than:

```text
One frame → Screen Coordinate
```

Make the number of samples per calibration point configurable.

Use a short dwell/stabilization period before collecting samples.

---

# 8. Support Multiple Mapping Models

Keep affine mapping as the transparent baseline.

Add optional nonlinear models.

### Model 1 — Affine

```text
X = a1*f1 + a2*f2 + ... + b1
Y = c1*f1 + c2*f2 + ... + b2
```

### Model 2 — Polynomial Regression

Support second-order terms such as:

```text
x
y
x²
y²
xy
```

and head-pose interactions where justified.

### Model 3 — Small MLP

Optionally support a lightweight neural network:

```text
Input Features
      ↓
Dense Layer
      ↓
Activation
      ↓
Dense Layer
      ↓
Screen X / Y
```

Do not make the MLP mandatory.

---

# 9. Make Mapping Model Configurable

Provide a configuration such as:

```python
mapping_model = "affine"
```

with:

```text
affine
polynomial
mlp
```

The system must allow quantitative comparison between mapping models.

Do not assume the more complex model is better.

---

# 10. Improve Temporal Smoothing

Keep EMA smoothing, but make it configurable.

For example:

```python
ema_alpha = 0.3
```

Do not hardcode the value.

Ensure smoothing does not introduce excessive lag.

An optional adaptive filter may be added later, but EMA should remain available.

---

# 11. Add Fixation and Saccade Detection

Do not treat every raw gaze sample equally.

Implement at least one established method:

- I-VT (Velocity Threshold Identification), or
- I-DT (Dispersion Threshold Identification)

Classify samples as:

```text
Fixation
Saccade
Invalid / Low Confidence
```

For fixations, store:

- start time
- end time
- duration
- centroid
- confidence
- dispersion

---

# 12. Make the Heatmap Fixation-Aware

The baseline output should be described as a **gaze-density heatmap**.

Do not automatically call it an "attention heatmap."

If fixation duration is incorporated, the output can be described as a visual-attention estimation component.

For fixation-aware accumulation:

```text
heatmap_weight ∝ fixation_duration × fixation_confidence
```

A stable 700 ms fixation should contribute more than a single noisy sample.

---

# 13. Replace Frame-Based Decay

Do not use frame-dependent decay such as:

```python
grid *= 0.997
```

Replace it with time-based decay:

```python
decay_factor = exp(-delta_time / tau)
```

where:

```text
delta_time = elapsed time since the previous frame
tau = configurable decay constant
```

This should behave consistently at different frame rates.

---

# 14. Verify Gaussian Blur Units

If the heatmap uses:

```text
cell size = 4 screen pixels
Gaussian sigma = 45 screen pixels
```

convert correctly:

```python
sigma_grid = sigma_screen_pixels / cell_size
```

Clearly document whether sigma is specified in screen pixels or heatmap-grid cells.

---

# 15. Make Heatmap Parameters Configurable

Expose at least:

```text
grid_cell_size
decay_tau
gaussian_sigma
gamma
alpha
```

Use sensible defaults based on the existing implementation, but keep them configurable.

---

# 16. Add Confidence-Aware Heatmap Accumulation

Do not give every gaze sample equal weight.

Use a confidence-aware contribution such as:

```text
heatmap += gaze_confidence * fixation_weight
```

Low-confidence samples should either contribute less or be discarded.

Add a configurable minimum confidence threshold.

---

# 17. Add Outlier Rejection

Before screen-coordinate and heatmap updates, reject clearly invalid samples based on:

- extremely low confidence
- impossible coordinates
- physically implausible sudden jumps
- unstable landmark geometry
- severe head-pose instability
- failed eye detection

Do not reject legitimate rapid saccades simply because they are fast.

---

# 18. Add Screen-Space Validation

Implement a validation mode where the user looks at known screen targets.

For each target calculate:

```text
error_px =
sqrt(
    (predicted_x - target_x)^2 +
    (predicted_y - target_y)^2
)
```

Report:

- Mean Error
- Median Error
- Standard Deviation
- 95th Percentile Error
- Maximum Error

Report error separately for each calibration target where practical.

---

# 19. Add Optional Angular Error

If camera/display geometry is known, optionally calculate:

```text
angular_error_degrees
```

Do not report angular error if the necessary geometry is unavailable.

---

# 20. Test Different Head-Pose Conditions

Where practical, evaluate:

- neutral head
- slight left/right yaw
- slight up/down pitch
- moderate head movement

Report errors separately.

This allows the effect of head-pose compensation to be measured rather than assumed.

---

# 21. Compare Calibration and Mapping Methods

Validation should support comparisons such as:

```text
Raw / Baseline
vs
Calibrated
```

and:

```text
Affine
vs
Polynomial
vs
MLP
```

Do not declare a winner without measured validation data.

---

# 22. Add Research Metrics

Create a structured evaluation report containing:

```text
Model
Calibration Method
Number of Calibration Points
Samples per Point
Mean Error
Median Error
Standard Deviation
95th Percentile Error
Maximum Error
Valid Sample Rate
Average Confidence
```

Optional:

```text
Fixation Detection Rate
Average Fixation Duration
Saccade Count
```

Save results as:

- JSON
- CSV

where practical.

---

# 23. Final Architecture

Use this conceptual architecture:

```text
                  WEBCAM
                     │
                     ▼
             Face Detection
                     │
                     ▼
            Facial Landmarks
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Eye Features          Head Pose
          │                     │
          └──────────┬──────────┘
                     ▼
              Feature Vector
                     │
                     ▼
             Gaze Estimator
          Affine / Polynomial / MLP
                     │
                     ▼
              Screen X/Y
                     │
                     ▼
           Temporal Filtering
                     │
                     ▼
          Confidence Estimation
                     │
             ┌───────┴───────┐
             ▼               ▼
          Fixation         Saccade
          Detection        Detection
             │
             ▼
      Fixation-Weighted
       Gaze Accumulation
             │
             ▼
       Time-Based Decay
             │
             ▼
        Gaussian Blur
             │
             ▼
       Normalization
             │
             ▼
       Gamma Correction
             │
             ▼
       Heatmap Rendering
```

---

# 24. Preserve the Three-Phase Organization

Keep the current structure:

## Phase 1 — Eye and Gaze Feature Extraction

Include:

- face landmarks
- eye extraction
- pupil/iris detection
- eye quality
- head pose
- feature vector
- eye fusion

## Phase 2 — Coordinate Mapping

Include:

- calibration
- affine mapping
- polynomial mapping
- optional MLP
- outlier rejection
- smoothing
- validation

## Phase 3 — Gaze-Density / Visual-Attention Heatmap

Include:

- fixation detection
- saccade detection
- confidence-weighted accumulation
- fixation weighting
- time-based decay
- Gaussian blur
- rendering
- export

---

# 25. Update Terminology

Review all documentation and UI labels.

Avoid:

> "The pupil position directly represents where the user is looking."

Use:

> "Pupil/iris and facial features are used as inputs to a calibrated gaze-estimation model."

Use:

> "Gaze estimation"

and:

> "Gaze-density heatmap"

for the baseline system.

Use:

> "Visual attention estimation"

only when the implemented methodology actually supports that interpretation.

---

# 26. Update the JavaScript Demo

The existing JavaScript demo uses mouse position as simulated gaze.

Keep it, but explicitly label it:

> Interactive Conceptual Simulation

or:

> Phase 2–3 Visualization Demo

Do not imply that it validates the real eye-tracking pipeline.

It is acceptable for the JavaScript demo to use simplified calculations, but its documentation must clearly distinguish it from the actual Python gaze-estimation implementation.

---

# 27. Add a Limitations Section

Document limitations including:

- camera position
- lighting
- glasses/reflections
- eye shape
- individual anatomy
- head movement
- face occlusion
- webcam resolution
- monitor geometry
- calibration quality

Do not claim research-grade accuracy without measured validation.

---

# 28. Code Quality

Use:

- type hints where practical
- modular functions
- configuration objects/dataclasses
- descriptive variable names
- docstrings for major algorithms
- comments for mathematical transformations
- centralized configuration
- robust error handling
- logging rather than excessive print statements

Avoid unnecessary rewrites.

Preserve backward compatibility where practical.

---

# 29. Centralized Configuration

Create a configuration structure similar to:

```python
@dataclass
class GazeConfig:
    screen_width: int = 1920
    screen_height: int = 1080

    calibration_points: int = 9
    calibration_samples_per_point: int = 20

    mapping_model: str = "affine"

    ema_alpha: float = 0.3

    confidence_threshold: float = 0.5

    heatmap_cell_size: int = 4
    heatmap_decay_tau: float = 5.0
    gaussian_sigma_px: float = 45.0
    heatmap_gamma: float = 0.75
    heatmap_alpha: float = 0.7
```

These are example defaults, not mandatory values. Use empirically justified values where the existing implementation provides them.

---

# 30. Suggested File Responsibilities

If the project contains:

```text
gaze_core.py
phase1_eye_tracking.py
phase2_coordinate_mapping.py
phase3_attention_heatmap.py
```

preserve the structure where practical.

### gaze_core.py

Shared:

- feature definitions
- confidence calculations
- data structures
- configuration

### phase1_eye_tracking.py

Implement:

- face landmarks
- eye extraction
- iris/pupil detection
- eye-quality estimation
- head pose
- feature vector
- eye fusion

### phase2_coordinate_mapping.py

Implement:

- calibration
- affine mapping
- polynomial mapping
- optional MLP
- outlier rejection
- EMA
- validation

### phase3_attention_heatmap.py

Implement:

- fixation detection
- saccade detection
- confidence-weighted accumulation
- fixation-weighted accumulation
- time-based decay
- Gaussian blur
- heatmap rendering
- export

---

# 31. Do Not Overengineer

Implement incrementally.

## Priority 1 — Required

1. Head-pose estimation
2. Better calibration
3. Confidence-aware eye fusion
4. Separate confidence measures
5. Fixation detection
6. Time-based heatmap decay
7. Quantitative validation
8. Correct terminology

## Priority 2 — Strongly Recommended

9. Polynomial gaze mapping
10. Confidence-weighted heatmap
11. Outlier rejection
12. Research evaluation report

## Priority 3 — Experimental

13. Small MLP gaze mapping
14. Advanced adaptive filtering
15. Angular-error analysis

Do not make Priority 3 features mandatory for the basic system.

---

# 32. Validation Requirements

After implementation, run all available tests.

Verify:

- application starts
- webcam pipeline works
- landmarks are detected
- eye features are extracted
- head pose is calculated
- calibration works
- gaze coordinates remain within screen bounds
- smoothing works
- confidence values behave sensibly
- fixation detection works
- heatmap accumulation works
- time-based decay works
- heatmap rendering works
- calibration data can be saved and loaded
- evaluation metrics can be generated

If hardware/webcam testing cannot be performed, explicitly state which tests could not be run.

---

# 33. Research Integrity

Do not manufacture accuracy improvements.

Do not state:

> "The new model is more accurate"

unless an actual before/after experiment demonstrates it.

When measurements exist, report them explicitly:

```text
Baseline error: X px
New model error: Y px
Relative change: Z%
```

Do not call the system "research-grade" simply because more algorithms have been added.

Describe the system according to measured performance.

---

# 34. Update the HTML Documentation

Update the existing HTML documentation so that it accurately reflects the final architecture.

Include:

1. System overview
2. Phase 1 — Eye and head-pose feature extraction
3. Phase 2 — Calibration and gaze mapping
4. Phase 3 — Fixation-aware gaze-density heatmap
5. Confidence model
6. Calibration procedure
7. Validation methodology
8. Evaluation metrics
9. Configuration parameters
10. Limitations
11. Interactive demonstration
12. Research methodology notes

Keep diagrams technically accurate and visually clear.

---

# 35. Final Deliverables

After implementation, provide:

## A. Modified Source Files

List every modified and created file.

## B. Architecture Summary

Show the final pipeline in a concise technical diagram.

## C. Algorithm Summary

Explain:

- gaze feature extraction
- head pose
- calibration
- mapping
- smoothing
- confidence
- fixation detection
- heatmap generation

## D. Validation Results

Report what was actually tested and the measured results.

## E. Remaining Limitations

Clearly distinguish implemented functionality from future research work.

## Final Instruction

**Inspect the existing implementation before changing it. Map each requested improvement to the appropriate existing component. Implement incrementally. Preserve working behavior wherever possible. Do not introduce unnecessary dependencies or complexity. Do not claim performance improvements without measured evidence.**
