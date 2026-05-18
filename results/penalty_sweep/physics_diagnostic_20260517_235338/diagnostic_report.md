# Physics Diagnostic Report

_Source: `results/penalty_sweep/physics_diagnostic_20260517_235338`_  
_Configs analyzed: baseline_classifier_only, pure_kin, pure_acc, pure_jointlim_soft, default_blend_

## 1. Ceiling at `physics_k = 3`

Upper bound that any rescoring (physics or otherwise) can achieve, given the classifier's top-k candidate set per sensor.

| n_sensors | exact-match ceiling | trials |
|---:|---:|---:|
| 2 | 0.9400 | 200 |
| 3 | 0.9300 | 200 |
| 4 | 0.8750 | 200 |
| 5 | 0.8850 | 200 |

## 2. Per-scorer attribution vs `baseline_classifier_only`

Paired trial-by-trial; same seed, same trial sampling. `net_fix = fixed − broken` is the marginal correction count.

| config | Δ per_sensor_acc | Δ exact_match | n_fixed | n_broken | net_fix |
|---|---:|---:|---:|---:|---:|
| pure_kin | -0.0243 | -0.0587 | 44 | 112 | -68 |
| pure_acc | -0.0014 | -0.0062 | 17 | 21 | -4 |
| pure_jointlim_soft | +0.0011 | +0.0000 | 4 | 1 | +3 |
| default_blend | +0.0039 | +0.0051 | 22 | 11 | +11 |

### 2a. `default_blend` breakdown by n_sensors

| n_sensors | base per_sensor | blend per_sensor | net_fix | base exact | blend exact |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.8275 | 0.8325 | +2 | 0.6800 | 0.6900 |
| 3 | 0.8267 | 0.8317 | +3 | 0.5650 | 0.5700 |
| 4 | 0.8200 | 0.8225 | +2 | 0.4750 | 0.4700 |
| 5 | 0.8220 | 0.8260 | +4 | 0.4050 | 0.4150 |

### 2b. Top confusion pairs `default_blend` *fixed* (baseline was wrong, blend is right)

| true region | wrongly predicted by baseline | count |
|---|---|---:|
| r_hip | l_hip | 4 |
| neck | spine_upper | 3 |
| l_forearm | l_upper_arm | 3 |
| r_upper_arm | r_shoulder | 3 |
| r_shin | l_foot | 2 |
| l_thigh | r_thigh | 2 |
| r_hand | r_upper_arm | 1 |
| head | spine_upper | 1 |
| spine_upper | r_collar | 1 |
| head | r_collar | 1 |

### 2c. Top confusion pairs `default_blend` *broke* (baseline was right, blend is wrong)

| true region | newly predicted by blend | count |
|---|---|---:|
| l_upper_arm | l_forearm | 3 |
| r_shin | l_foot | 2 |
| l_hand | l_hip | 1 |
| l_forearm | l_shoulder | 1 |
| spine_lower | spine_mid | 1 |
| l_thigh | r_foot | 1 |
| r_hand | r_forearm | 1 |
| r_hip | l_thigh | 1 |

## 3. Headline answer

Region difficulty threshold: classifier top1 ≥ 0.95 → 'easy', else 'hard'.

- `default_blend` vs baseline: Δ per_sensor_acc = **+0.0039**, net_fix = **+11** (22 fixed − 11 broken over 2800 sensor obs). **Physics helps**.
- Best single scorer Δ per_sensor_acc = **+0.0011**; blend Δ = +0.0039 (additive).

## 4. Stratified by region difficulty (appended, granular output above is preserved)

For each candidate config, attribution split by whether the *true* sensor region is 'easy' (classifier already gets it right ≥95% of the time) or 'hard'. Reveals where physics actually earns its keep.

### pure_kin

| bucket | n_obs | base per_sensor | new per_sensor | n_fixed | n_broken | net_fix |
|---|---:|---:|---:|---:|---:|---:|
| easy | 966 | 0.9886 | 0.9420 | 1 | 46 | -45 |
| hard | 1834 | 0.7361 | 0.7236 | 43 | 66 | -23 |

### pure_acc

| bucket | n_obs | base per_sensor | new per_sensor | n_fixed | n_broken | net_fix |
|---|---:|---:|---:|---:|---:|---:|
| easy | 966 | 0.9886 | 0.9855 | 1 | 4 | -3 |
| hard | 1834 | 0.7361 | 0.7356 | 16 | 17 | -1 |

### pure_jointlim_soft

| bucket | n_obs | base per_sensor | new per_sensor | n_fixed | n_broken | net_fix |
|---|---:|---:|---:|---:|---:|---:|
| easy | 966 | 0.9886 | 0.9886 | 0 | 0 | +0 |
| hard | 1834 | 0.7361 | 0.7377 | 4 | 1 | +3 |

### default_blend

| bucket | n_obs | base per_sensor | new per_sensor | n_fixed | n_broken | net_fix |
|---|---:|---:|---:|---:|---:|---:|
| easy | 966 | 0.9886 | 0.9886 | 1 | 1 | +0 |
| hard | 1834 | 0.7361 | 0.7421 | 21 | 10 | +11 |
