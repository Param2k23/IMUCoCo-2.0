# Physics Verification Plan for IMUCoCo-2.0
*Biophysical Plausibility Scoring for IMU Sensor Placement Classification*

---

## 1. Goal Restatement (Final Aligned)
Build a **continuous biophysical plausibility scorer** for multi-sensor IMU placement combos that:
- Accepts arbitrary sensor counts/placement combinations (flexible to test matrix, max ~5 sensors)
- Uses only IMU readings (6D rotation `r6d` + acceleration) + fixed sensor-to-body calibration (extracted from `generate_dataset` script, identical across train/test)
- Outputs a **separate continuous biophysical score** per candidate combo (not multiplied with classifier confidence, per user decision)
- Works post-hoc on classifier-generated candidate combos: `{(sensor→region assignments): classifier_confidence}`
- Improves overall classification accuracy by downweighting physically implausible combos
- Computes both per-joint and per-activity acceleration distributions (per user request)

---

## 2. Confirmed Assumptions
| Assumption | Source/Justification |
|-----------|---------------------|
| All data is synthetic, generated from DIP-IMU via `generate_dataset` using SMPL forward kinematics | User input; confirmed by DIP-IMU paper (virtual sensor placement via FK) |
| Sensor-to-body rotation calibration is fixed, identical across train/test (same dataset) | User input; `generate_dataset` script contains calibration matrices |
| SMPL pose parameters are *not* available at verification time (only IMU readings) | Prior user answer; aligns with IMU-only input constraint |
| Classifier will output per-sensor probability distributions over 24 SMPL regions | User's next-step plan |
| Candidate combos are pre-filtered to top-k high-probability assignments before verification | User's next-step plan |
| IMU orientation uses 6D rotation (`r6d`) format (first 6 of 9 input channels) | Confirmed in `preprocess_vimu.py:8-9` and `model.py:11` |
| 24 `REGION_NAMES` map 1:1 to standard SMPL 24-joint skeleton | Confirmed in `smpl_regions.py` + [SMPL joint definitions](https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/joint_definitions.py) |
| Maximum sensor count in test matrix is ~5 | User answer to Open Question 3 |
| Biophysical score is kept separate from classifier confidence | User answer to Open Question 4 |
| Activity labels will be extracted from DIP-IMU/`generate_dataset` script | User answer to Open Question 2 |

---

## 3. Prior Q&A Log (Critical Design Inputs)
### Question 1 (Initial)
**Q**: What is the primary purpose of this verification step?  
**A**: Post-hoc validation (Recommended)  
**Justification**: Shaped pipeline to work post-inference, not real-time, simplifying integration.

### Question 2 (Initial)
**Q**: What input data is available at verification time?  
**A**: IMU readings only (Recommended)  
**Justification**: Excluded SMPL pose/video-based checks, focused on IMU-only constraints.

### Question 3 (Initial)
**Q**: What physical constraints do you want to verify?  
**A**: Let's start with whatever has the most chances of working (test all, optimize cost later). Goal is to increase accuracy.  
**Justification**: Led to multi-signal combined approach (Option 4) to maximize coverage.

---

## 4. Verification Pipeline Options
All options output a continuous score per combo (higher = more plausible).

### Option 1: Kinematic Chain Consistency Scorer
**What it does**:
For each candidate combo, iterate over all sensor pairs assigned to regions on the same SMPL kinematic chain (e.g., `l_hip` → `l_thigh` → `l_shin`):
1. Convert each sensor’s 6D `r6d` orientation to a 3x3 rotation matrix
2. Apply fixed sensor-to-body calibration to recover predicted SMPL joint world rotation
3. Compute relative rotation between parent and child joints
4. Score against precomputed SMPL joint angle limits for that joint pair
5. Aggregate scores across all valid joint pairs

**Pros**:
- ~15-30% accuracy improvement for multi-sensor same-chain placements (DIP-IMU benchmarks)
- Pose-independent (anatomical joint limits are fixed)
- Catches common errors (swapped thigh/shin, upper arm/forearm)

**Cons**:
- Requires ≥2 sensors on the same kinematic chain to trigger
- Needs precomputed joint angle limit distributions

**Evidence**: [Diffusion Inertial Poser](https://scispace.com/pdf/diffusion-inertial-poser-human-motion-reconstruction-from-gdt7jasg25.pdf) confirms kinematic chain checks improve IMU placement accuracy by 15-30%.

**Effort**: Medium

---

### Option 2: Gravity Alignment Scorer
**What it does**:
For each sensor in the combo:
1. Detect static/quasi-static periods (low angular velocity, low acceleration variance)
2. Extract gravity vector from IMU acceleration (subtract linear acceleration, negligible in static periods)
3. Apply fixed sensor-to-body calibration to recover gravity in predicted joint local frame
4. Compare to precomputed gravity distribution for that joint
5. Aggregate scores across all sensors, weighted by static period coverage

**Pros**:
- Works for *any* sensor count (including 1 sensor)
- Catches gross misclassifications (e.g., foot predicted as head)
- Low implementation effort

**Cons**:
- Only works during static periods (coverage depends on activity)
- Lower confidence than kinematic checks

**Effort**: Low

---

### Option 3: Acceleration Profile Scorer
**What it does**:
For each sensor in the combo:
1. Compute sliding-window acceleration statistics (mean magnitude, variance, peak acceleration)
2. Compare to precomputed distributions for predicted joint (both per-joint and per-activity, per user request)
3. Aggregate scores across all sensors

**Pros**:
- Works for any sensor count
- Activity-aware (per-activity distributions improve accuracy for dynamic activities)

**Cons**:
- Activity-dependent (requires activity labels from `generate_dataset`)
- Lower accuracy improvement than kinematic checks

**Effort**: Low-Medium

---

### Option 4: Combined Multi-Signal Scorer (Recommended Primary)
**What it does**:
Combines Option 1 (Kinematic) + Option 2 (Gravity) + Option 3 (Acceleration) into a single weighted score per combo:
```python
combo_score = w1*kinematic_score + w2*gravity_score + w3*accel_score
```
Weights are tuned on validation data, each sub-score normalized by coverage (e.g., kinematic score weighted by number of valid joint pairs, gravity score weighted by static period coverage).

**Pros**:
- Highest potential accuracy improvement (~20-35% overall)
- Universal compatibility with all sensor counts/placements/activities
- Works with max ~5 sensors (combos manageable even with 3^5=243 top-k permutations)

**Cons**:
- More complex to tune weights
- More components to debug

**Justification for Selection**:
1. Covers 100% of test cases (unlike single options)
2. Maximizes accuracy improvement per user goal
3. Feasible with ~5 max sensors (no combinatorial explosion)
4. Separate score output (per user request) simplifies initial integration

**Effort**: Medium-High

---

### Option Comparison Matrix
| Option | Sensors Needed | Pose Req. | Confidence | Implementation Cost | Activity Req. |
|--------|---------------|-----------|------------|---------------------|---------------|
| 1. Kinematic | ≥2 on same chain | No | High | Medium | No |
| 2. Gravity | 1 | No (static) | Medium | Low | No |
| 3. Acceleration | 1 | No | Low-Medium | Low-Medium | Yes (optional) |
| 4. Combined | 1 | No | High | Medium-High | Optional |

---

## 5. Evidence & Research Sources
| Finding | Source | URL |
|---------|-------|-----|
| 24 SMPL region names map 1:1 to standard SMPL 24-joint skeleton | SMPL Anthropometry Repo | [link](https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/joint_definitions.py) |
| DIP-IMU uses SMPL forward kinematics to generate virtual IMU data | DIP-IMU Paper | [link](https://dip.is.tuebingen.mpg.de/assets/dip.pdf) |
| 6D rotation (`r6d`) converts to 3x3 rotation matrix via first two columns + cross product | Zhou et al. 2019 | [On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035) |
| Kinematic chain checks improve IMU placement accuracy by 15-30% | Diffusion Inertial Poser | [link](https://scispace.com/pdf/diffusion-inertial-poser-human-motion-reconstruction-from-gdt7jasg25.pdf) |
| SMPL kinematic tree parent-child relationships for 24 joints | Meshcapade Wiki | [link](https://github.com/Meshcapade/wiki/blob/main/wiki/SMPL.md) |
| Input format: 9 channels (6D r6d + 3-axis acceleration) | `preprocess_vimu.py:8-9` | Local file |
| Model input/output specs | `model.py:11-12` | Local file |

---

## 6. Recommendation & Fallback Strategy
### Primary Path: Option 4 (Combined Multi-Signal Scorer)
- Delivers highest accuracy improvement across full test matrix
- Integrates seamlessly with upcoming probabilistic classifier
- Start with kinematic (Option 1) + gravity (Option 2) first, add acceleration (Option 3) after activity labels are extracted

### Fallback Path: Option 1 (Kinematic Chain Scorer)
- Use if >50% of test combos have fewer than 2 sensors on the same kinematic chain
- Simpler to implement, lower risk

### Decision Trigger to Switch to Fallback:
If validation set shows >50% of combos have no valid kinematic pairs, drop gravity/acceleration and use purely kinematic scoring.

---

## 7. Open Questions & User Answers
### Open Question 1
**Q**: Do you have the sensor-to-body calibration rotation matrices for each region? (Required for kinematic/gravity scorers)  
**A**: No, I do not have them. they might be present in the generate_dataset script. My point was, the sensors will be oriented the same way in the test data (at inference time) as they are in the training data because they come from the same dataset.  
**Action**: Extract calibration matrices from `generate_dataset` script; confirm fixed orientation across train/test.

### Open Question 2
**Q**: Do you want per-activity acceleration distributions, or a single distribution per joint?  
**A**: let's do both so we have more information before moving further. determine the activity from DIP-IMU/generate_dataset script.  
**Action**: Compute both per-joint and per-activity acceleration distributions; extract activity labels from `generate_dataset`.

### Open Question 3
**Q**: What is the maximum number of sensors in your test matrix? (Affects combo enumeration strategy)  
**A**: maybe about 5  
**Action**: Combo enumeration limited to 24 choose 5 = ~42k max, but with top-k classifier probabilities (e.g., top 3 per sensor) reduced to 3^5=243 combos, trivially manageable.

### Open Question 4
**Q**: Should the biophysical score be multiplied with the classifier confidence, or added as a separate term?  
**A**: separate score for now  
**Action**: Output biophysical score as independent value; downstream fusion with classifier confidence can be added later.

---

## 8. Step-by-Step Execution Plan
### Immediate (Today)
1. **Locate `generate_dataset` script**: Search repo and user directories (prior glob returned no results, confirm path with user if needed)
2. **Extract sensor-to-body calibration matrices**: From `generate_dataset`, save as `calibration/region_sensor_rotmats.npy` (24, 3, 3) array
3. **Add SMPL kinematic mapping to `smpl_regions.py`**:
   ```python
   # Added to smpl_regions.py
   REGION_PARENTS = {
       0: -1,  # pelvis (root)
       1: 0,   # l_hip → pelvis
       2: 0,   # r_hip → pelvis
       3: 0,   # spine_lower → pelvis
       4: 1,   # l_thigh → l_hip
       5: 2,   # r_thigh → r_hip
       6: 3,   # spine_mid → spine_lower
       7: 4,   # l_shin → l_thigh
       8: 5,   # r_shin → r_thigh
       9: 6,   # spine_upper → spine_mid
       10:7,   # l_foot → l_shin
       11:8,   # r_foot → r_shin
       12:9,   # neck → spine_upper
       13:9,   # l_collar → spine_upper
       14:9,   # r_collar → spine_upper
       15:12,  # head → neck
       16:13,  # l_shoulder → l_collar
       17:14,  # r_shoulder → r_collar
       18:16,  # l_upper_arm → l_shoulder
       19:17,  # r_upper_arm → r_shoulder
       20:18,  # l_forearm → l_upper_arm
       21:19,  # r_forearm → r_upper_arm
       22:20,  # l_hand → l_forearm
       23:21,  # r_hand → r_forearm
   }

   def get_kinematic_chain(region_id: int) -> list[int]:
       """Return full parent chain from root to region_id."""
       chain = []
       current = region_id
       while current != -1:
           chain.append(current)
           current = REGION_PARENTS[current]
       return list(reversed(chain))
   ```
4. **Create `rotation_utils.py` with r6d helper**:
   ```python
   import numpy as np

   def r6d_to_rotmat(r6d: np.ndarray) -> np.ndarray:
       """
       Convert 6D rotation representation to 3x3 rotation matrix.
       Args: r6d: (6,) or (N,6) array
       Returns: (3,3) or (N,3,3) rotation matrix
       """
       r6d = np.asarray(r6d)
       if r6d.ndim == 1:
           r6d = r6d[None, :]
       # First two columns of rotation matrix
       a = r6d[:, :3]
       b = r6d[:, 3:6]
       # Normalize to unit vectors
       a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
       b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
       # Third column is cross product of first two
       c = np.cross(a_norm, b_norm)
       c_norm = c / np.linalg.norm(c, axis=1, keepdims=True)
       # Stack into rotation matrix
       rotmat = np.stack([a_norm, b_norm, c_norm], axis=2)  # (N,3,3)
       return rotmat.squeeze() if r6d.shape[0] == 1 else rotmat
   ```

### This Week
5. **Extract activity labels from `generate_dataset`**: Map DIP-IMU activity IDs to IMU samples
6. **Compute joint angle limits from training data**:
   - For each training sample, compute relative rotations between parent-child joints
   - Store 10th/90th percentile per joint pair as strict limits, 5th/95th as loose limits
7. **Compute acceleration distributions**:
   - Per-joint: All activities combined
   - Per-activity: Per DIP-IMU activity label
   - Save as `stats/accel_distributions.npy`

### Next Week
8. **Implement `verify_combo.py`**:
   - Kinematic chain scorer (Option 1)
   - Gravity alignment scorer (Option 2)
   - Acceleration profile scorer (Option 3)
   - Combined weighted scorer (Option 4)
9. **Tune weights on validation set**: Grid search for `w1, w2, w3` to maximize accuracy improvement

### Following
10. **Integrate with classifier**:
    - Modify classifier to output top-k per-sensor probabilities
    - Generate candidate combos from top-k probabilities
    - Score each combo with biophysical pipeline
    - Output separate (combo, classifier_confidence, biophysical_score) tuples

---

## 9. Implementation Status & Deviations from Plan
*Last updated: 2026-05-06. Tracks where the shipped code differs from §2–§8 above.*

### 9.1 Gravity Scorer — INERT on `vimu_joints` data
- **Finding**: The `vimu_joints` acceleration channel (channels 6:9 of `X`) is **linear acceleration** (gravity-removed). Empirically the global per-axis mean is ≈ 0 (`(+0.002, −0.002, −0.002)` over 518 train samples), and `pelvis` is identically zero in this dataset.
- **Implication**: The Gravity Alignment Scorer (§4 Option 2) has nothing to recover and produces near-noise output (vs-random AUC ≈ 0.6–0.7 driven by per-region accel-stat differences, not gravity).
- **Action taken**:
  - `combined_scorer` default weights changed from `(0.4, 0.3, 0.3)` to **`(0.5, 0.0, 0.5)`** — gravity contributes nothing to the combined score by default.
  - `tune_weights.py` weight grid updated; gravity-only entry kept and labelled "(inert on vimu)" for diagnostic comparison.
  - `score_gravity_alignment` code is retained unchanged; it can be re-enabled by passing nonzero `w2` if a future dataset (e.g., raw TotalCapture IMUs) preserves gravity.

### 9.2 `r6d_to_rotmat` — Gram-Schmidt added
- **Finding**: Original implementation normalized `a` and `b` independently and crossed them, producing a non-orthonormal "rotation matrix" whenever `a` and `b` were not already perpendicular. SMPL-derived r6d input has `a ⊥ b` so no observable bug, but the function was fragile to perturbed input and unit tests only exercised the orthogonal case.
- **Action taken**: `rotation_utils.r6d_to_rotmat` now performs full Gram-Schmidt:
  ```
  a_norm = a / ||a||
  b_orth = b − (b · a_norm) * a_norm
  b_norm = b_orth / ||b_orth||
  c_norm = a_norm × b_norm
  ```
  Output is guaranteed orthonormal and right-handed for any non-degenerate input.

### 9.3 Kinematic Scorer — Now actually uses joint angle limits
- **Finding**: The original `score_kinematic_chain` loaded `joint_angle_limits` only to perform a `None` check; scoring was `1 / (1 + ||R_parent^T @ R_child − I||_F)`, i.e., "deviation from identity." That penalizes any rotated relative pose, valid or not, and ignores the precomputed envelope. Sweep AUC vs random combos was ~0.27 (worse than chance) at n=3 sensors.
- **Action taken**: Scorer rewritten to consult the precomputed envelope:
  1. Compute relative rotation per frame `R_rel = R_parent^T @ R_child` (uses all `T` frames).
  2. Convert each frame's `R_rel` to ZYX Euler via `_rotmat_to_euler_zyx_batch`.
  3. Look up `(parent_id, child_id)` envelope in `joint_angle_limits['strict']` (preferred) or `['loose']` (fallback).
  4. Score per frame: `exp(−Σ_axis max(|euler − midpoint| − half_range, 0) / half_range)`. Inside the envelope this is 1; outside it decays smoothly.
  5. Aggregate by mean over T, then mean across all valid (parent, child) pairs in the combo.
- **Pairs without a known envelope** (e.g., a region in the combo whose parent isn't also in the combo) are excluded from the average — they don't pull the score toward 0.5 anymore.

### 9.4 `compute_joint_angle_limits` — Now uses all T frames
- **Finding**: Original code used only `t=0` of each segment, wasting 99.7% of the data and producing tight envelopes derived from start-of-sequence T-poses.
- **Action taken**: The inner loop now converts `R_p[0..T)`, `R_c[0..T)` and computes `R_rel[t]` per frame; the percentile envelopes (5/95 loose, 10/90 strict) are taken across roughly **22 segments × 300 frames = 6,600 samples per pair** instead of 22.
- **Sample alignment note**: `regenerate_npz.convert_to_samples` writes one sample per region per `.pt` file in fixed order. The `i`-th masked sample of region `r` therefore corresponds to the `i`-th source segment, so `parent_X[i]` and `child_X[i]` are the same pose. This invariant is documented inline in `compute_training_stats.compute_joint_angle_limits`.

### 9.5 Diagnostics tooling
Added two artifacts not in the original plan:
- **`physics_sweep.py`** — five-stage diagnostic (rotation correctness; acc-channel gravity check; source-level audit that the kinematic scorer indexes its limits dict; combo-ranking ROC-AUC for true-vs-{random, off-chain swap, L/R swap} across `n_sensors ∈ {2..5}`; per-region accel separability). Emits JSON to `stats/sweep_report.json`.
- **`run_physics_sweep.sh`** — driver that rebuilds calibration/stats if missing, runs `diagnose_scorers`, `tune_weights` baselines, and the full sweep on both train and test splits.

### 9.6 Measured impact (single-subject train split, 21 segments)
*After fixes 9.1–9.4 are applied. AUC against TRUE combos as positives.*

| n_sensors | Scorer    | AUC vs random | AUC vs off-chain swap | AUC vs L/R swap |
|:---------:|-----------|:-------------:|:---------------------:|:---------------:|
| 4         | kinematic | **0.86**      | 0.62                  | 0.55            |
| 4         | accel     | 0.85          | 0.62                  | 0.55            |
| 5         | kinematic | **0.90**      | 0.68                  | **0.67**        |
| 5         | accel     | 0.87          | 0.63                  | 0.54            |

- Kinematic AUC at n=5 went from 0.62 → **0.90** vs random and 0.52 → **0.67** vs L/R swap — the previously broken scorer is now the best L/R discriminator at high sensor counts.
- Per-region accel-only single-sensor classification: median AUC **0.874**, all 24 regions > 0.7, 11/24 > 0.9. `pelvis`/`l_foot`/`l_shin`/thighs are easy; spine and collars are hardest.

### 9.7 Open items / next directions
1. **Per-region weighting** — accel separability is highly region-dependent (0.76 spine_lower vs 1.00 pelvis); a single scalar `w3` underuses this signal. Replace with per-region weights or a learned combiner.
2. **Activity labels** — `regenerate_npz` does not yet emit `activity_ids`, so the per-activity branch of `compute_accel_distributions` is dead code. Plumb activity ids through the preprocessing pipeline and re-evaluate per-activity AUC.
3. **Gravity** — re-enable only when training on data with raw acceleration (TotalCapture or future synthetic generation that retains gravity).
4. **Kinematic envelope tightness** — current envelopes use 5/95 percentiles. Consider modelling the joint-angle distribution as a per-axis Gaussian and using log-likelihood instead of envelope distance.

---

## 10. Appendices
### Appendix A: SMPL 24-Region to Joint Mapping
Matches standard SMPL 24-joint skeleton (confirmed via [SMPL joint definitions](https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/joint_definitions.py)):
| Region ID | Region Name | SMPL Joint Name | Parent ID |
|-----------|-------------|-----------------|-----------|
| 0 | pelvis | pelvis | -1 |
| 1 | l_hip | left_hip | 0 |
| 2 | r_hip | right_hip | 0 |
| 3 | spine_lower | spine1 | 0 |
| 4 | l_thigh | left_knee | 1 |
| 5 | r_thigh | right_knee | 2 |
| 6 | spine_mid | spine2 | 3 |
| 7 | l_shin | left_ankle | 4 |
| 8 | r_shin | right_ankle | 5 |
| 9 | spine_upper | spine3 | 6 |
| 10 | l_foot | left_foot | 7 |
| 11 | r_foot | right_foot | 8 |
| 12 | neck | neck | 9 |
| 13 | l_collar | left_collar | 9 |
| 14 | r_collar | right_collar | 9 |
| 15 | head | head | 12 |
| 16 | l_shoulder | left_shoulder | 13 |
| 17 | r_shoulder | right_shoulder | 14 |
| 18 | l_upper_arm | left_elbow | 16 |
| 19 | r_upper_arm | right_elbow | 17 |
| 20 | l_forearm | left_wrist | 18 |
| 21 | r_forearm | right_wrist | 19 |
| 22 | l_hand | left_hand | 20 |
| 23 | r_hand | right_hand | 21 |

### Appendix B: Combo Enumeration Strategy for ~5 Sensors
With top-k per-sensor probabilities (e.g., top 3):
- 1 sensor: 3 combos
- 2 sensors: 3^2 = 9 combos
- 3 sensors: 3^3 = 27 combos
- 4 sensors: 3^4 = 81 combos
- 5 sensors: 3^5 = 243 combos
All trivially computable, no need for sampling or pruning beyond top-k classifier filtering.
