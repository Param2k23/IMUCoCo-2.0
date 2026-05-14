# Physics Integration Plan
*Top-k Probabilities × Physics Verification — Combo Re-ranking*

---

## 1. Context

The classifier now emits a per-window top-k softmax distribution
(`predict_all_topk`, commit `12b716d`). The physics verification scorer
(commit `cdbd523`) consumes a fixed `combo: list[region_id]` and returns a
plausibility score in [0, 1]. The two have been separate.

This plan integrates them as the user described:

> *Data stream provided of n sensors → classifier → top-k probability for
> each sensor → physics pipeline tries combinations based on probabilities
> and re-ranks based on which are physically feasible. Also: if a sensor is
> identified as shoulder, check if it is rotated beyond that joint's
> physical limits — a human-informed eliminator.*

Two changes ship together:

1. **Top-k → combo enumeration → physics re-ranking** integration layer
   (`topk_combo_rerank.py`) wired into `evaluate.py --physics_rerank`.
2. **Per-region absolute orientation eliminator** — a fourth scorer
   (`score_per_region_orientation`) that flags sensors whose body-frame
   ZYX-Euler orientation falls outside that region's training-derived
   envelope. Soft by default; opt-in hard mode via `--joint_limits_hard`.

Score fusion is multiplicative: `final = P(classifier) × P(physics)`.
This supersedes `PHYSICS_VERIFICATION_PLAN.md §7-Q4` ("keep separate"),
which was answered before a real top-k re-ranking use case existed.

---

## 2. Pipeline (end-to-end)

```
   raw IMU stream of n sensors
              │
              ▼
   model(.)  → predict_all_topk     (per-window top-k softmax)
              │
              ▼
   aggregate_topk_per_sensor        (collapse n_windows → per-sensor top-k)
              │
              ▼
   enumerate_combos                 (Cartesian product, drops repeats)
              │
              ▼
   rerank_combos                    (combined_scorer per combo)
       ├── kinematic chain          (parent↔child Euler envelope)
       ├── gravity (disabled)       (linear-acc data has no gravity)
       ├── acceleration profile     (per-region 7-dim Gaussian)
       └── per-region orientation   (NEW — anatomical eliminator)
              │
              ▼
   final_score = classifier_prob × physics_score
              │
              ▼
   sorted candidates, top-1 = prediction
```

---

## 3. New Components

### 3.1 `compute_per_region_orientation_limits`
Added to `compute_training_stats.py`. For each region, applies the same
calibration as `verify_combo.score_kinematic_chain`:
```
rot_body[t] = r6d_to_rotmat(r6d[t]) @ calibration[region]
```
then converts each frame to ZYX Euler and stores per-axis percentile
envelopes in `stats/per_region_orientation_limits.npy`:

```python
{
  "loose":  {region_id: {"min": (3,) float32, "max": (3,) float32}},   # 5/95
  "strict": {region_id: {"min": (3,) float32, "max": (3,) float32}},   # 10/90
}
```

### 3.2 `score_per_region_orientation`
Added to `verify_combo.py`. Per-frame penalty mirrors the existing
`score_kinematic_chain` formulation so the scorer lives on [0, 1] and
combines naturally:
```
outside_axis = max(|euler_axis − midpoint| − half_range, 0) / half_range
per_frame    = exp(−Σ_axis outside_axis)
sensor_score = mean_t per_frame
combo_score  = mean over sensors
```
**Hard mode**: with `hard_threshold` set, any sensor whose
`frac_violations > threshold` (frames with at least one axis outside its
envelope) zero-scores the entire combo. The combo is dropped from the
top-1 contender pool.

### 3.3 `topk_combo_rerank.py`
Public surface:
- `aggregate_topk_per_sensor(topk_indices, topk_probs, k)` — average
  per-sensor probability over n_windows, return per-sensor top-k.
- `enumerate_combos(per_sensor_topk_idx, per_sensor_topk_probs, distinct_regions=True)`
  — Cartesian product, drops combos with repeated regions.
- `rerank_combos(combos, X_per_sensor, calibration, stats, weights, joint_limits_hard, eliminator_only)`
  — runs `combined_scorer` per combo, returns sorted candidates.
- `predict_with_topk_physics(...)` — one-shot wrapper.

Reuses `temporal_rerank._build_prob_matrix` for the dense conversion.

### 3.4 `evaluate.py --physics_rerank`
After the existing `predict_all_topk`, the eval loop optionally runs
`evaluate_physics_rerank(...)` which:
1. Indexes test windows by ground-truth region.
2. For each `x ∈ --rerank_n_sensors`, samples `--rerank_n_trials` synthetic
   trials of x distinct regions.
3. For each trial, draws `--rerank_n_windows` windows per region and runs
   the full top-k → combo → re-rank pipeline.
4. Reports `exact_match_acc` (whole region tuple correct) and
   `per_sensor_acc` (mean over sensors of `pred[i] == true[i]`).
5. If `--joint_limits_hard <frac>` is given, repeats the evaluation with
   the hard eliminator active and stores it in
   `physics_rerank_summary_hard`.

Both blocks land in `eval_summary.json` so a single eval run produces both
soft and hard pipeline numbers side-by-side.

---

## 4. Per-Region Rotation Values (extracted from training data)

Computed on `data/single_subject_train.npz` (single subject, ~6.6k frames
per region for rotational axes). Values are body-frame ZYX-Euler degrees
after applying `calibration[r]` to the r6d sensor reading.

`pitch` (axis Y) is mathematically bounded to ±90°; `roll` and `yaw`
(axes X and Z) cover the full ±180° circle and are subject to wrap-around
ambiguity for unconstrained rotations.

**Strict envelope (10/90 percentile, used by default):**

| ID | Region        | min (deg) [x, y, z]            | max (deg) [x, y, z]            | range (deg) |
|---:|---------------|---------------------------------|---------------------------------|--------------|
|  0 | pelvis        | [-177.2, -30.0,   -4.2]         | [ 176.0,  84.2,  173.9]         | [353.2, 114.1, 178.1] |
|  1 | l_hip         | [-151.1, -26.9, -139.4]         | [ 158.2,  72.4,  163.2]         | [309.3,  99.3, 302.6] |
|  2 | r_hip         | [-167.8, -29.5, -163.7]         | [ 169.8,  78.3,  172.7]         | [337.6, 107.8, 336.3] |
|  3 | spine_lower   | [-175.8, -29.4,   -3.1]         | [ 174.8,  84.1,  173.8]         | [350.6, 113.5, 176.9] |
|  4 | l_thigh       | [-154.1, -23.7,  -34.1]         | [ 161.1,  77.8,  172.8]         | [315.2, 101.5, 206.9] |
|  5 | r_thigh       | [-153.6, -25.9, -129.1]         | [ 161.3,  82.5,  169.3]         | [314.9, 108.5, 298.4] |
|  6 | spine_mid     | [  -4.8, -27.8,   -2.0]         | [ 174.6,  82.7,  172.7]         | [179.4, 110.5, 174.7] |
|  7 | l_shin        | [-124.8, -24.0, -158.9]         | [ 174.7,  82.6,  175.7]         | [299.5, 106.6, 334.6] |
|  8 | r_shin        | [ -96.7, -27.5,   -9.9]         | [ 158.2,  72.6,  163.7]         | [254.9, 100.1, 173.6] |
|  9 | spine_upper   | [  -2.0, -27.0,   -1.9]         | [ 175.5,  82.3,  172.3]         | [177.6, 109.3, 174.2] |
| 10 | l_foot        | [-145.0, -30.1,    4.8]         | [ 160.5,  73.6,  165.1]         | [305.5, 103.7, 160.3] |
| 11 | r_foot        | [-151.4, -20.3, -168.5]         | [ 159.5,  71.3,  -12.3]         | [310.9,  91.6, 156.1] |
| 12 | neck          | [-175.5, -29.8,   -5.8]         | [ 170.5,  78.2,  174.0]         | [346.0, 108.1, 179.8] |
| 13 | l_collar      | [-165.6, -28.6, -170.6]         | [ 142.2,  78.4,  -16.6]         | [307.8, 107.0, 154.0] |
| 14 | r_collar      | [ -26.7, -29.7,   15.0]         | [ 158.5,  66.3,  153.3]         | [185.1,  96.0, 138.2] |
| 15 | head          | [-170.1, -32.4,   -0.4]         | [  95.2,  74.1,  169.8]         | [265.3, 106.5, 170.2] |
| 16 | l_shoulder    | [-125.4,  -8.8, -107.9]         | [ 107.9,  19.0,  -78.1]         | [233.3,  27.8,  29.8] |
| 17 | r_shoulder    | [ -60.2, -10.1,   74.0]         | [ 133.2,  14.9,   99.5]         | [193.3,  25.1,  25.6] |
| 18 | l_upper_arm   | [-123.9,  -9.4, -105.0]         | [  69.6,  17.7,  -75.2]         | [193.4,  27.1,  29.7] |
| 19 | r_upper_arm   | [-124.8, -17.7,   73.2]         | [ 147.4,  19.9,  106.8]         | [272.2,  37.7,  33.5] |
| 20 | l_forearm     | [-120.5, -16.5, -102.0]         | [  65.2,  18.5,  -70.1]         | [185.7,  35.1,  31.9] |
| 21 | r_forearm     | [-129.2, -20.1,   75.6]         | [ 152.9,  19.1,  110.3]         | [282.1,  39.2,  34.7] |
| 22 | l_hand        | [-122.1, -14.0, -103.1]         | [  66.6,  11.8,  -79.3]         | [188.7,  25.7,  23.8] |
| 23 | r_hand        | [-126.3, -22.7,   71.0]         | [ 145.9,  17.9,  106.7]         | [272.2,  40.7,  35.8] |

**Loose envelope (5/95 percentile, fallback when strict is missing):**

| ID | Region        | min (deg)                       | max (deg)                       | range (deg) |
|---:|---------------|---------------------------------|---------------------------------|--------------|
|  0 | pelvis        | [-178.9, -60.6,   -7.2]         | [ 178.7,  86.6,  175.5]         | [357.6, 147.2, 182.8] |
|  1 | l_hip         | [-168.5, -57.4, -172.5]         | [ 168.9,  75.9,  171.3]         | [337.5, 133.2, 343.7] |
|  2 | r_hip         | [-174.4, -59.2, -174.1]         | [ 175.1,  82.3,  176.3]         | [349.5, 141.5, 350.4] |
|  3 | spine_lower   | [-178.0, -61.3,   -5.1]         | [ 177.8,  86.0,  175.3]         | [355.8, 147.3, 180.4] |
|  4 | l_thigh       | [-167.4, -57.5, -173.4]         | [ 171.3,  81.4,  175.4]         | [338.7, 138.9, 348.8] |
|  5 | r_thigh       | [-168.7, -55.6, -169.9]         | [ 170.8,  84.7,  173.1]         | [339.4, 140.4, 343.0] |
|  6 | spine_mid     | [ -55.4, -59.3,   -3.6]         | [ 177.1,  84.5,  174.7]         | [232.5, 143.8, 178.3] |
|  7 | l_shin        | [-159.9, -55.5, -175.2]         | [ 176.2,  84.6,  177.2]         | [336.1, 140.1, 352.3] |
|  8 | r_shin        | [-161.4, -61.1,  -29.0]         | [ 172.0,  77.3,  168.5]         | [333.3, 138.4, 197.5] |
|  9 | spine_upper   | [ -42.1, -58.4,   -3.7]         | [ 177.3,  83.5,  174.8]         | [219.4, 141.9, 178.4] |
| 10 | l_foot        | [-167.7, -56.5,   -0.4]         | [ 174.3,  77.0,  167.5]         | [342.1, 133.5, 167.9] |
| 11 | r_foot        | [-166.8, -49.5, -173.5]         | [ 175.1,  74.8,   -5.0]         | [341.9, 124.3, 168.5] |
| 12 | neck          | [-178.0, -63.3,  -17.8]         | [ 177.2,  83.9,  176.6]         | [355.1, 147.2, 194.4] |
| 13 | l_collar      | [-174.2, -60.7, -172.8]         | [ 175.8,  80.3,  -14.5]         | [350.0, 140.9, 158.3] |
| 14 | r_collar      | [-151.9, -52.6,   13.2]         | [ 174.0,  68.6,  157.2]         | [325.9, 121.2, 144.1] |
| 15 | head          | [-174.5, -65.1,   -5.9]         | [ 167.4,  80.3,  173.7]         | [342.0, 145.3, 179.6] |
| 16 | l_shoulder    | [-150.0, -13.7, -112.0]         | [ 161.6,  20.4,  -75.7]         | [311.7,  34.1,  36.3] |
| 17 | r_shoulder    | [-107.5, -15.0,   71.3]         | [ 164.3,  16.1,  102.4]         | [271.8,  31.1,  31.1] |
| 18 | l_upper_arm   | [-149.1, -15.8, -110.9]         | [ 136.3,  20.6,  -69.9]         | [285.5,  36.4,  40.9] |
| 19 | r_upper_arm   | [-150.0, -24.0,   64.9]         | [ 159.9,  23.6,  109.6]         | [309.9,  47.6,  44.7] |
| 20 | l_forearm     | [-136.3, -24.3, -111.8]         | [ 131.8,  25.3,  -63.6]         | [268.1,  49.6,  48.2] |
| 21 | r_forearm     | [-149.3, -26.2,   67.3]         | [ 163.6,  24.0,  112.9]         | [312.9,  50.2,  45.6] |
| 22 | l_hand        | [-144.7, -21.2, -106.1]         | [ 130.6,  18.7,  -73.3]         | [275.3,  39.9,  32.8] |
| 23 | r_hand        | [-149.7, -28.8,   67.7]         | [ 159.7,  21.7,  109.2]         | [309.4,  50.5,  41.5] |

### 4.1 Literature-vs-training observations
- **Arm chain (16–23)** has the tightest envelopes — pitch (axis Y) ranges
  ~25–50°, well below human shoulder/elbow ROM (shoulder flexion is
  literature 0–180°, abduction 0–180°). This is consistent with DIP-IMU
  activities being mostly upright "everyday" motion, not extremes. The
  scorer can therefore catch grossly mislabelled sensors (e.g. predicting
  shoulder when the data is from foot) but won't fire on extreme but
  legal motion outside the training distribution.
- **Pelvis / spine / shins** have wide-but-still-bounded yaw (≥170°)
  reflecting the full directional span of locomotion captures.
- **Pitch axis (Y)** is mathematically bounded to [−90°, 90°]. All
  observed values respect this — no axis-ordering bug.
- **r_foot Z, l_collar Z, r_upper_arm/r_forearm/r_hand Z**: the displayed
  `max` is numerically smaller than `min` because of yaw wrap at ±180°.
  The envelope check uses `(min, max)` as a two-sided interval — for these
  regions the interval wraps the antipode, which is benign for the
  exp-decay penalty as long as both endpoints are correct percentiles.
  (A future improvement: switch to circular percentiles for axes 0 and 2.)

These envelopes are saved verbatim in
`stats/per_region_orientation_limits.npy` (24 entries × 2 (loose/strict))
and consumed by `score_per_region_orientation`.

---

## 5. Verification Plan

Run **both** soft and hard pipelines from a single eval invocation. Set
`--joint_limits_hard 0.3` and the second block (`physics_rerank_summary_hard`)
populates automatically alongside `physics_rerank_summary`.

### 5.1 Unit / wiring smokes
1. `python topk_combo_rerank.py` — synthetic top-k passes through
   aggregate / enumerate / rerank without errors. Confirms imports.
2. `python verify_combo.py --combo 18 20 22 --data data/single_subject_test.npz \
   --stats stats --calibration calibration --weights 0.4 0.0 0.4 0.2`
   — confirms the new fourth scorer fires and the diagnostics dict carries
   `joint_limits_score` and `joint_limits` block.
3. `python verify_combo.py --combo 18 20 22 ... --joint_limits_hard 0.3`
   — confirms the hard path returns 0.0 when the combo would be eliminated
   (and reports `hard_eliminated=True`, `violator=<region_name>`).

### 5.2 End-to-end (BOTH pipelines in one run)
```
python evaluate.py \
  --checkpoint checkpoints/penalty_sweep/best_final/fold_0/best_model_fold0.pt \
  --data data/single_subject_test.npz \
  --out_dir results/physics_integration \
  --physics_rerank \
  --physics_calibration calibration \
  --physics_stats stats \
  --rerank_n_sensors 2,3,4,5 \
  --rerank_n_windows 1 \
  --rerank_n_trials 200 \
  --physics_k 3 \
  --physics_weights 0.4 0.0 0.4 0.2 \
  --joint_limits_hard 0.3
```

The resulting `eval_summary.json` will contain TWO blocks:

```jsonc
"physics_rerank_summary": {            // SOFT (no hard elimination)
  "2": {"exact_match_acc": ..., "per_sensor_acc": ..., "n_eliminated_mean": 0.0, ...},
  "3": {...},
  "4": {...},
  "5": {...}
},
"physics_rerank_summary_hard": {       // HARD (eliminate if frac > 0.3)
  "2": {"exact_match_acc": ..., "per_sensor_acc": ..., "n_eliminated_mean": >0, ...},
  "3": {...},
  "4": {...},
  "5": {...}
}
```

### 5.3 Acceptance criteria
- **Wiring**: both blocks populated; `n_eliminated_mean` is 0.0 in the
  soft block and strictly > 0 in the hard block.
- **Soft signal**: `physics_rerank_summary["4"]["per_sensor_acc"] >=
  topk_accuracy["top1"]` on the same fold (PHYSICS_RESULTS.md §9.6 shows
  the underlying physics scorer reaches AUC 0.86–0.90 vs random at n=4–5,
  so a positive ranking gain on real top-k candidates is expected).
- **Hard vs soft**: the hard block's per-sensor accuracy should be no
  worse than soft except at thresholds so aggressive that
  `n_eliminated_mean` approaches the total candidate count (in which case
  the policy is dropping correct combos). A reasonable threshold (~0.3)
  should *match or improve* per-sensor accuracy — the eliminator's job is
  to drop wrong arms-as-feet style combos that were beating the right one
  on classifier confidence alone.

### 5.4 Threshold sweep (optional follow-up)
Re-run the eval at thresholds `0.1, 0.2, 0.3, 0.5, 0.7` and chart
`per_sensor_acc` vs `n_eliminated_mean` to pick a knee. Threshold = 1.0
is equivalent to "no hard mode" (purely soft); threshold = 0.0 eliminates
any combo with even one out-of-envelope frame.

---

## 6. Files Touched / Added

| File                             | Change                                              |
|----------------------------------|-----------------------------------------------------|
| `smpl_regions.py`                | Added `REGION_BY_NAME` reverse map.                 |
| `compute_training_stats.py`      | Added `compute_per_region_orientation_limits` + main wiring. Added batched `_rotmat_to_euler_zyx_batch`. |
| `verify_combo.py`                | Added `score_per_region_orientation`. Extended `load_stats` and `combined_scorer` (now 4-tuple weights, hard-mode kwarg). Updated CLI. |
| `topk_combo_rerank.py` *(new)*   | Top-k → combo enumeration → physics rerank.         |
| `evaluate.py`                    | Added `evaluate_physics_rerank` + CLI flags. Both soft and hard pipelines reported in `eval_summary.json`. |
| `stats/per_region_orientation_limits.npy` *(new)* | 24 regions × loose/strict × {min,max}. |
| `PHYSICS_INTEGRATION_PLAN.md` *(this file)* | Plan + extracted rotation table. |

---

## 7. Backwards Compatibility Notes
- `combined_scorer` accepts both 3-tuple and 4-tuple weights. The 3-tuple
  form silently sets `w_joint_limits = 0`, so existing callers
  (`physics_sweep.py`, `tune_weights.py`) continue to work unchanged.
- If `per_region_orientation_limits.npy` is absent, the new scorer
  returns a neutral 0.5 score and `combined_scorer` skips its term — the
  existing three-scorer behaviour is preserved.
- `verify_combo.py --weights` now accepts variadic floats; the old
  3-arg form (`--weights 0.4 0.3 0.3`) still parses.
