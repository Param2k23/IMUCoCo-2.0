# Extended Research & Development Narrative (Pre-IMUCoCo Phase)

The work conducted during this semester began as a continuation and expansion of the original CSE570 course project, **RegenHAR: Robust Human Activity Recognition via Transformer-Based Sensor Regeneration**. While the original project successfully demonstrated robust Human Activity Recognition (HAR) under missing sensor scenarios, the primary goal of this semester’s research was to investigate whether the same ideas could generalize beyond activity classification into more complex downstream tasks, particularly full-body pose estimation from sparse IMUs.

This phase of the research was exploratory and foundational in nature. Before the IMUCoCo repository and framework were formally developed, significant effort was spent studying related work, evaluating the limitations of existing approaches, experimenting with multiple datasets, and identifying the key practical challenges that remained unsolved in sparse IMU systems.

## 1. Motivation for Extending Beyond HAR

The original RegenHAR project established that Transformer-based regeneration could successfully reconstruct missing IMU sensor streams for activity recognition tasks. The system achieved strong robustness even under severe sensor failure conditions by learning correlations between multiple body-worn sensors.

However, during post-project analysis, an important limitation became evident:

- HAR is fundamentally a low-dimensional downstream task.
- Many activities can still be recognized even when a large amount of motion detail is missing.
- Therefore, strong HAR performance does not necessarily imply that the regeneration framework preserves fine-grained motion fidelity.

This raised an important research question:

> Can sparse-sensor regeneration methods generalize to higher-fidelity motion understanding tasks such as human pose estimation?

Pose estimation from IMUs is significantly more difficult than HAR because:

- the output space is continuous rather than categorical,
- temporal dependencies are substantially more complex,
- body-part relationships are highly non-linear,
- and small sensor inaccuracies can propagate into large skeletal reconstruction errors.

Because of this, the project direction expanded from:

> “Can we recognize activities with missing sensors?”

to:

> “Can we recover physically meaningful motion information from sparse IMU setups?”

This shift motivated the exploration of additional datasets and architectures outside traditional HAR pipelines.

## 2. Survey of Related Work

Before implementing new models, an extensive review of existing sparse-sensor and missing-sensor literature was conducted.

### 2.1 Competitive Analysis: SAM (Sensor-Aware iMputation)

One of the major systems analyzed was **SAM (Sensor-Aware iMputation)**.

#### Core Methodology

SAM approaches missing sensor reconstruction using:

- unsupervised clustering,
- learned sensor co-occurrence patterns,
- and precomputed lookup-table (LUT) based imputation.

The workflow operates as follows:

1. Sensor streams are clustered into representative motion groups.
2. Relationships between clusters across sensors are learned offline.
3. During inference, the available sensors are mapped to clusters.
4. The missing sensor values are reconstructed using precomputed LUT mappings.

This design is computationally efficient during inference because it avoids expensive generative reconstruction.

### 2.2 Assumptions Identified in SAM

During analysis, several assumptions in SAM were identified.

#### Assumption 1 — Repetitive Motion Structure

SAM assumes:

- human motion falls into a relatively small number of recurring sensor patterns,
- and these patterns remain stable across users and activities.

This assumption allows clustering-based generalization.

#### Assumption 2 — Static Imputation Sufficiency

The system assumes:

- classifiers can tolerate approximation errors introduced by static imputation,
- even if the reconstructed signals are not physically accurate.

#### Assumption 3 — Limited Sensor Complexity

The method implicitly assumes:

- a manageable number of sensor combinations,
- because LUT complexity grows combinatorially with the number of sensors and clusters.

### 2.3 Weaknesses Identified

Through comparative analysis, several limitations became clear.

#### Scalability Problems

The LUT-based approach scales poorly when:

- sensor counts increase,
- sensor placements vary,
- or the downstream task requires high-fidelity reconstruction.

For pose estimation, the number of possible sensor interaction combinations becomes extremely large.

#### Lack of Temporal Modeling

Static lookup approaches do not effectively capture:

- long-range temporal dependencies,
- dynamic motion transitions,
- or coordinated full-body motion patterns.

This becomes especially problematic for:

- locomotion,
- turning motions,
- transitions,
- and asymmetric body movement.

#### Incompatibility with High-Fidelity Reconstruction

For tasks like skeletal pose estimation:

- small angular inaccuracies compound over time,
- and static cluster averages are insufficient to preserve biomechanical consistency.

These observations reinforced the hypothesis that:

> generative sequence models and Transformer-based architectures are fundamentally better suited for sparse IMU reconstruction than static clustering approaches.

## 3. Transition Toward Pose Estimation

After identifying the limitations of HAR-only evaluation, the project expanded into sparse IMU pose estimation.

The reasoning behind this transition was:

- HAR evaluates semantic understanding.
- Pose estimation evaluates physical motion reconstruction.
- A regeneration framework that succeeds on pose estimation is likely to generalize more robustly to other downstream tasks.

This led to experimentation with the DIP-IMU dataset and the TransPose framework.

## 4. TransPose Baseline Investigation

The first major experimental phase involved reproducing and stress-testing the TransPose system.

### 4.1 Paper and Framework

The following work was selected:

- **TransPose: Real-time Human Motion Reconstruction from Sparse IMUs**
- GitHub: https://github.com/Xinyu-Yi/TransPose

The system was chosen because:

- it represents a strong sparse-IMU pose estimation baseline,
- it uses only six IMUs,
- and it directly targets motion reconstruction rather than activity classification.

## 5. DIP-IMU Dataset Exploration

The DIP-IMU dataset was studied extensively before experimentation.

### Dataset Characteristics

The dataset contains:

- six IMU sensors,
- high-frequency inertial measurements,
- synchronized full-body pose ground truth.

Sensor placements include:

- wrists,
- knees,
- pelvis,
- and head.

Each temporal sequence contains:

- acceleration vectors,
- orientation representations,
- and corresponding pose information.

Compared to HAR datasets, DIP-IMU presented several new challenges:

| HAR Datasets | DIP-IMU |
|---|---|
| Low-dimensional labels | Continuous skeletal motion |
| Semantic classification | Precise kinematic reconstruction |
| Small output space | Large pose manifold |
| Robust to approximation | Sensitive to small errors |

## 6. Baseline Replication Experiments

The first implementation goal was to reproduce the original TransPose performance using all six sensors.

### Offline Full-Sensor Baseline

Using 17/18 DIP-IMU sequences for training and evaluation, the reproduced baseline achieved:

- SIP Error: 13.97° ± 6.77°
- Angular Error: 7.62° ± 4.01°
- Mesh Error: 5.83 ± 3.21 cm

These results confirmed:

- the implementation pipeline was functioning correctly,
- and the baseline architecture was capable of accurate pose reconstruction under ideal sensor conditions.

## 7. Simulated Sensor Failure Experiments

Once the baseline was validated, controlled missing-sensor experiments were introduced.

### Failure Simulation Strategy

Specific sensors were intentionally removed by:

- zeroing their input channels,
- while leaving the remaining sensors intact.

The first experiments targeted lower-body sensors:

- Left Upper Leg
- Right Lower Leg

These locations were selected because:

- they are highly informative for locomotion,
- and they strongly influence lower-body kinematic chains.

### 7.1 Observed Performance Collapse

The effect of missing sensors was severe.

The SIP error increased from:

> 13.97° → 21.18°

representing a:

> 51.6% degradation

This experiment demonstrated that:

- the baseline TransPose architecture was highly dependent on complete sensor availability,
- and sparse IMU robustness remained an unsolved problem in pose estimation systems.

This result was one of the most important findings during the semester because it showed that:

> methods that appear accurate under ideal conditions may fail catastrophically under realistic wearable deployment scenarios.

## 8. Key Research Realizations During This Phase

Several major insights emerged from these experiments.

### 8.1 Regeneration Is Dataset-Dependent

One of the earliest observations was that:

> regeneration architectures tightly depend on the number and arrangement of sensors.

Unlike image inpainting or generic sequence modeling, IMU reconstruction is highly tied to:

- body topology,
- sensor placement,
- and kinematic structure.

This meant:

> a regeneration pipeline trained on one sensor layout could not trivially generalize to another.

For example:

- HAR datasets used five smartphone locations,
- DIP-IMU used six sparse body sensors,
- and future datasets used completely different topologies.

As a result, model redesign or retraining became necessary whenever:

- sensor count changed,
- or body placement configuration changed.

This scalability problem became one of the central motivations behind later IMUCoCo research.

### 8.2 Correlation Analysis Between Sensors

To better understand inter-sensor dependencies, multiple statistical analyses were performed.

#### Pearson Correlation

Linear relationships between sensors were measured using Pearson correlation.

Some meaningful relationships emerged:

- symmetric limbs showed partial similarity,
- temporally synchronized motions produced visible correlation structures.

However:

- many relationships were weak,
- highly non-linear,
- or activity-dependent.

#### Cross-Correlation Analysis

Cross-correlation experiments were also performed to identify:

- delayed temporal relationships,
- phase-shifted motion dependencies,
- and synchronization patterns.

Although some correlations appeared interpretable, the results were ultimately insufficient for designing a generalized reconstruction strategy.

The key realization was:

> IMU relationships are not purely statistical — they are biomechanical and contextual.

This reinforced the decision to move away from shallow statistical reconstruction toward learned structural representations.

## 9. Discovery of IMUCoCo and New Research Direction

While investigating how to support:

- arbitrary sensor counts,
- arbitrary sensor combinations,
- and dynamic sensor placement,

the project eventually led to the discovery of the IMUCoCo paper.

This work directly addressed:

- sparse sensor localization,
- and flexible sensor configuration problems.

The paper became a major turning point because it introduced a framework capable of reasoning about sensors independently of a fixed topology.

However, during analysis, an important practical limitation was identified.

## 10. Critical Assumption Identified in IMUCoCo

IMUCoCo assumes that:

> the locations of the available sensors are explicitly known at inference time.

In practice, this means:

- users must specify where every sensor is located,
- or the system must already know the exact body placement configuration.

This assumption was considered unrealistic for real-world deployment.

Examples identified during discussion included:

- moving a smartwatch from wrist to pocket,
- carrying a phone in different locations,
- inconsistent wearable placement,
- or partial device availability.

Under the IMUCoCo assumption:

> every placement change would require explicit relabeling of all active sensors.

This introduced a major usability limitation.

## 11. Transition Into the IMUCoCo Research Phase

At this point, the research direction became clear:

The next phase of the project would focus on:

- identifying sensor locations automatically,
- handling arbitrary sensor combinations,
- reducing dependence on fixed sensor layouts,
- and improving robustness under realistic wearable usage conditions.

This realization directly motivated the creation of the IMUCoCo-based research repository and the subsequent work documented below.

---



# IMUCoCo‑2.0 — Project Work Report

A chronological walk through the work done in this repository, derived from
the git history (`git log --reverse`) cross‑checked against the actual logs,
checkpoints, and result JSON in‑tree. Every headline number below was
re‑verified against the file noted in parentheses.

Each section is one logical phase: what was on the table going in, what the
commits actually changed, the **observation at the time**, the **decision
that followed**, and the **measured result**. Dates are commit dates from
`git log`.

Pipeline shorthand: `preprocess_vimu.py → train.py → evaluate.py`, with
per‑phase Slurm scripts (`run_*.slurm`) wrapping the three steps on the HPC
cluster.

---

## Phase 0 — Initial scaffold (Apr 12, 2026)

Commit `0e3201a` *“Initial commit: IMU body-region classifier (IMUCoCo /
SMPL)”* — 9 files, 2 378 lines.

What lands:

- `smpl_regions.py` — 24‑region vertex map, region centroids, parent table,
  `spatial_error(...)` helper (Euclidean distance in metres between
  pred/true region centroids on the T‑pose mesh).
- `model.py` — `ResNet1D` (default) and a `CNN1D` baseline. Input `(B, 9, T)`
  → 24 logits.
- `train.py` — LOSO + fixed‑split scaffold.
- `evaluate.py` — per‑window accuracy, confusion matrix, spatial error,
  symmetry pairs, majority‑vote.
- `preprocess_amass.py` — original AMASS‑first preprocessing path.
- `kaggle/SensorLoc_train.ipynb` — Kaggle notebook prototype.

**Observation at the time**: README paths are Windows
(`C:/VS/SensorLoc/…`); the project name on disk is *SensorLoc*; AMASS is
assumed as the data source. **Decision**: pivot to a Linux+CUDA, synthetic
VIMU pipeline immediately — captured in `PLAN.md` of the very next commit.

---

## Phase 1 — Pivot to the synthetic VIMU dataset (Apr 14, 2026)

Commit `d03394c` *“changes for new synthetic dataset + development,
execution changes”* — +1 448 / −202 lines. The largest single rewrite in the
history.

New / rewritten files:

- **`preprocess_vimu.py`** — converts DIP‑style `vimu` `.pt` segments into
  the `.npz` contract `X (N,9,T) float32`, `y (N,) int`, `subject_ids (N,) int`.
  Channel layout `r6d_0..r6d_5, ax, ay, az`; one sample per
  `(segment, region)`.
- **`train.py`** rewritten (+432/−202): adds `--mode fixed_split`, a strict
  CUDA policy (`--smoke_test` is the only path that allows CPU), and
  separate `normalization_stats*.pt` files.
- **`evaluate.py`** rewritten for the new contract, with weights‑only
  checkpoint loading and fail‑fast on missing stats / channel mismatch.
- Docs: `AGENTS.md` (working agreement), `DATASET.md`, `PLAN.md`,
  `UPGRADE.md` (the seven‑step A0…A6 ablation roadmap), and a `commmands`
  runbook.

**Observation at the time** (`PLAN.md`): the team only has DIP‑style `vimu`
segments from a small subject pool; accuracy is the primary metric but
spatial error matters because adjacent regions are anatomically close.
**Decisions captured in `PLAN.md`**:

- Class taxonomy fixed at 24 regions (no merging of L/R for the baseline).
- Two parallel data pipelines: *single‑subject* (ignore source split) and
  *predefined split* (respect source train/test folders).
- CUDA required in non‑smoke runs (rationale: prevent accidental long CPU
  runs).
- `UPGRADE.md` deliberately gates augmentations / SE blocks / EMA behind a
  `[FUTURE]` tag — get the baseline working first.

---

## Phase 2 — Smoke test + plumbing (Apr 17, 2026)

Commit `e340016` *“update + bug fixes + smoke test command”* — +471/−162.

- `./smoke_commands` — runs the full pipeline on a tiny slice, the fastest
  way to verify a new machine.
- `normalization.py` — centralises z‑score stats so train and eval cannot
  drift apart.
- README rewritten as the IMUCoCo / SMPL story (current form).
- `train.py` / `evaluate.py` hardened: weights‑only checkpoints, normalization
  stats saved separately, fail‑fast on missing stats or channel mismatch.

**Decision**: every later runbook (`run_*.slurm`) is required to keep the
smoke path working — the smoke command becomes the canonical sanity check.

---

## Phase 3 — Hugging Face Parquet ingestion (Apr 18, 2026)

Three commits the same afternoon (`347782b`, `8f0aac1`, `12b0348`).

The DIP `.pt` path stays, but `preprocess_vimu.py --mode hf_parquet` is added
to read `train-*.parquet` / `test-*.parquet` shards directly. The same
`(X, y, subject_ids) .npz` contract is produced.

- `smoke_commands` gains `SMOKE_MODE={pt,hf,both}`.
- Subject‑id parsing made smarter (`--subject_column`, robust filename
  fallback, single summary warning). `pyarrow` added to requirements.

**Observation at the time**: parquet rows have variable `T`, and shard joins
sometimes produce mixed‑length stacks. **Decision** (commit `f30fe91` four
days later): crop/pad every row to `window_length=300` *before* stacking,
and drop rows that can't be normalised.

By end of Apr 18 the data path is dataset‑agnostic: every downstream step
consumes the same `.npz` regardless of source.

---

## Phase 4 — First baseline on the cluster (Apr 22, 2026)

Commits `f30fe91`, `58c8e7e`, `cbde868`, `d62ce4d`, `01286d2`, `62a297e`.

Wires up the first cluster pipeline (`run_full_pipelines.slurm`, h200x4, 8 h
walltime) and produces the first real number.

Run characteristics (`logs/train_hf_full_split.log`):

- Train: `X=(21240, 9, 300)`, 24 classes, **8 subjects**.
- Test:  `X=(456, 9, 300)`, 24 classes, **2 subjects**.
- 80 epoch budget but **early stopping at epoch 30** (patience=15), best
  `val_acc=0.7171` reached at epoch 15.

Eval result (`results/full_split_full/eval_summary.json`,
`logs/eval_hf_full_split.log`):

| metric                       | value |
|------------------------------|-------|
| per‑window accuracy          | **0.7171** |
| majority‑vote acc. (k=5)     | 0.2857 |
| locked fraction              | 4.6 % |
| windows wrong                | 129 / 456 (28.3 %) |
| spatial error mean / std (m) | 0.4064 / 0.3131 |

**Observation at the time** (visible in the eval log): the majority‑vote
metric was essentially broken — only 4.6 % of windows ever locked. The
classification report shows L/R errors on hip / shin / hand at 5.3 % each;
arm chain (`l_upper_arm`, `l_forearm`, `l_hand`) recall sits at 0.42 / 0.42 /
0.47 — the *adjacent‑region* arm‑chain failures that dominate every later
result are already visible at 71 % overall accuracy.

**Decision** (`62a297e`): expose `--weight_decay` and add
`run_ablation_sweep.slurm` so the next iteration can ablate around the
baseline rather than re‑run blind. Ablations in
`results/ablations/summary.csv` are computed on this fixed split (not on
LOSO):

| run                          | val per‑window acc |
|------------------------------|-------------------:|
| baseline_seed42              | 0.7719 |
| baseline_seed43              | 0.7303 |
| baseline_seed44              | 0.7654 |
| **fusion_blend01_seed42** (imu_blend=0.1) | **0.7807** |
| fusion_blend02_seed42 (imu_blend=0.2) | 0.7434 |
| reg_batch64_wd3e4_seed42     | 0.7346 |

Best ablation row is the small **imu_blend=0.1** fusion of `vimu` and
`imu`; the larger 0.2 blend regresses, so the decision is to keep `vimu`
as the primary input.

---

## Phase 5 — LOSO baseline + tuning (Apr 23, 2026)

Commits `3c5c359`, `b21ec9d`, `7906e78`, `b1bcd2d`, `058ad88`.

LOSO becomes the primary research split. Merged dataset is
`X=(21696, 9, 300)`, **10 subjects**, IDs 1..10.

- `run_loso.slurm` — full LOSO Slurm pipeline with `SMOKE_TEST=1` shortcut.
- `b21ec9d` — small fix: smoke mode copies `loso_dataset.npz` to
  `loso_smoke_dataset.npz` so `train.py`'s smoke filename rewrite finds it.

**LOSO baseline** (`checkpoints/loso_full/loso_results.txt`):

| fold | subject | val_acc | fold | subject | val_acc |
|-----:|--------:|--------:|-----:|--------:|--------:|
| 1    | 1       | 0.7342  | 6    | 6       | 0.7061  |
| 2    | 2       | 0.7241  | 7    | 7       | **0.8425** |
| 3    | 3       | 0.7507  | 8    | 8       | 0.7877  |
| 4    | 4       | 0.7512  | 9    | 9       | 0.7315  |
| 5    | 5       | 0.8018  | 10   | 10      | **0.6875** |

**Mean LOSO 0.7517** (std 0.0444, range 0.155 from worst to best fold).

**Observation at the time** (`MEETING_30MIN_BRIEF.md` shipped with
`b1bcd2d`): fold variance is 15 pp from worst to best — strong evidence
that subject identity dominates the remaining error. **Decision**: tune
hyperparameters per‑fold rather than per‑run; build a sweep that uses the
first five folds as a proxy and reruns the winner on all ten.

`run_loso_tune.slurm` runs five trials on folds 1‑5:

| trial | epochs | bs  | lr   | wd   | patience | sweep acc |
|-------|-------:|----:|-----:|-----:|---------:|----------:|
| a     | 80     | 128 | 1e‑3 | 1e‑4 | 15       | 0.7393    |
| b     | 100    | 128 | 1e‑3 | 3e‑4 | 20       | 0.7800    |
| c     | 80     | 64  | 1e‑3 | 1e‑4 | 15       | 0.7658    |
| d     | 80     | 128 | 5e‑4 | 1e‑4 | 20       | 0.7575    |
| **e** | 120    | 128 | 7e‑4 | 3e‑4 | 25       | **0.7810** ← winner |

**Observation**: `trial_b` and `trial_e` both win by way of stronger weight
decay (`3e-4`) and longer patience — the smaller LR (`7e-4`) plus 120
epochs in `trial_e` is the marginal win. **Decision**: rerun `trial_e` on
all ten folds.

Tuned LOSO (`checkpoints/loso_tune/best_final/loso_results.txt`): **mean
0.7817**, +3.0 pp over the baseline. The worst fold (subject 6) drops to
0.6875 — i.e. tuning *helps the easy folds more than the hard ones*. This
becomes a recurring theme.

---

## Phase 6 — Epoch sweep + a 6‑step pipeline (Apr 27, 2026)

Commits `3b8855f`, `f0c85b7`, `16617f7`, `a0d9d8d`.

`run_epoch_sweep.slurm` (389 lines) is the workhorse. It clones the LOSO
script into a 6‑step pipeline: preprocess → smoke → 5‑fold sweep → pick
best → train all 10 folds → evaluate all 10 folds. `evaluate.py` is
extended in the same commit (+99) to write per‑fold confusion matrices and
the L/R confusion aggregate.

First epoch sweep result on 5 folds
(`results/epoch_sweep/leaderboard.csv` / `best_config.env`):

| trial            | epochs | bs  | lr   | wd   | patience | loss   | lrw  | mean (5 folds) |
|------------------|-------:|----:|-----:|-----:|---------:|--------|-----:|---------------:|
| ep300_custom     | 300    | 128 | 1e‑3 | 3e‑4 | 50       | custom | 0.5  | 0.8106 |
| **ep400_custom** | **400**| 128 | 1e‑3 | 3e‑4 | 60       | custom | 0.5  | **0.8480** ← winner |
| ep500_custom     | 500    | 128 | 1e‑3 | 3e‑4 | 60       | custom | 0.5  | 0.8272 |
| ep600_custom     | 600    | 128 | 1e‑3 | 3e‑4 | 60       | custom | 0.5  | 0.8157 |
| ep800_custom     | 800    | 128 | 1e‑3 | 3e‑4 | 60       | custom | 0.5  | 0.8296 |

**Observation at the time**: the curve is non‑monotone — 400 epochs beats
both 500 and 800. **Decision**: 400 is the sweet spot; stop pushing epochs
and start changing the loss.

### Phase 6b — SpatialNeighborLoss + bigger net (Apr 28, 2026)

Commit `bf49976` *“SpatialNeighborLoss, class weights, base_filters=128
for 95%+ target”* — +195/−99 in `train.py` alone.

Three things ship together:

1. **`SpatialNeighborLoss`** — adds two physics‑informed penalty terms on
   top of cross‑entropy: a *spatial* term (distance between predicted and
   true region centroids on the SMPL T‑pose), and an *L/R mirror* term
   (extra cost when the wrong side of the body is predicted, controlled by
   `--lr_weight`, default 0.5).
2. **Inverse‑frequency class weights**, computed per‑fold from the train
   split.
3. **Wider model** (`--base_filters 128`, ~5 M params) so the spatial
   penalty has more parameters to push against.

`5623745` *“v3 epoch sweep …”* runs this configuration. `ep400_custom`
holds as the winner on the 5‑fold sweep, and the full LOSO rerun
(`checkpoints/epoch_sweep/best_final/loso_results.txt`):

| fold | subj | val_acc | fold | subj | val_acc |
|-----:|-----:|--------:|-----:|-----:|--------:|
| 1    | 1    | 0.8315  | 6    | 6    | 0.8180  |
| 2    | 2    | 0.8303  | 7    | 7    | 0.8798  |
| 3    | 3    | **0.8933** | 8 | 8 | 0.8648 |
| 4    | 4    | 0.8476  | 9    | 9    | 0.8843  |
| 5    | 5    | 0.8736  | 10   | 10   | 0.8875  |

**Mean LOSO 0.8611** — +8.0 pp over the LOSO baseline, +7.9 pp over the
tuned LOSO, and the largest single jump in the project.

**Observation**: subject 6 (worst in Phase 5) jumps from 0.6875 → 0.8180.
The spatial penalty is doing exactly what the design intended — it
converts what would be a wrong‑arm‑region error into a *near* arm‑region
error, which still counts as wrong for top‑1 but is much easier for the
class‑weighted CE to learn out.

---

## Phase 7 — Penalty sweep & non‑L/R confusion analysis (May 6–7, 2026)

Commits `839affe`, `6c9300b`, `49343ae`, `d169e71`, `0751b3b`, `1c0e2f6`.

By now the L/R confusion rate is small. The next question is *what else* is
producing errors. `run_penalty_sweep.slurm` (493 lines) sweeps three
phases:

- **Phase 1** — `neighbor_weight ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` at fixed
  `lr=1e-3`, `lrw=0.5` (isolates the spatial penalty effect).
- **Phase 2** — `lr ∈ {3e-4, 7e-4, 2e-3} × nbw ∈ {0.0, 0.3, 0.5}` grid.
- **Phase 3** — `lr_weight ∈ {0.0, 0.3, 1.0, 2.0}` at fixed `lr=1e-3`,
  `nbw=0.3`.

Best trial is auto‑picked from `leaderboard.csv` and rerun on all ten
folds. (The committed `results/penalty_sweep/leaderboard.csv` is the
header‑only file — the body lives in cluster scratch — but the final
`loso_results.txt` is in‑tree.)

To find the non‑L/R failure modes, `evaluate.py` is extended (`6c9300b`):
the full 24×24 confusion matrix and top‑10 *non*‑L/R pairs are written to
each per‑fold `eval_summary.json`. A 10‑fold aggregate is in
`results/penalty_sweep/eval_best_final/confusion_aggregate.json`:

```
total_windows  : 21 696
total_errors   :  2 931   (top-1 = 86.5%)
L/R errors     :    243   ( 8.3% of errors)
non-L/R errors :  2 688   (91.7% of errors)
```

Top non‑L/R pairs (all *adjacent* on the same arm chain):

| pair                            | count | % of errors |
|---------------------------------|------:|------------:|
| r_forearm → r_upper_arm         | 211   | 7.20 |
| l_forearm → l_upper_arm         | 143   | 4.88 |
| r_upper_arm → r_forearm         | 139   | 4.74 |
| l_forearm → l_hand              | 133   | 4.54 |
| l_upper_arm → l_forearm         | 116   | 3.96 |
| r_forearm → r_hand              | 107   | 3.65 |
| r_hand → r_forearm              |  97   | 3.31 |
| l_hand → l_forearm              |  90   | 3.07 |

**Observation**: 91.7 % of all remaining errors are non‑L/R, and the
top‑8 pairs are all on the arm chain (`forearm ↔ upper_arm`, `forearm
↔ hand`). The architecture is already doing a good job on L/R — the work
to do is on adjacent‑region arm‑chain disambiguation. **Decision**:
treat the arm chain as the main target for *post‑hoc* re‑ranking (the
loss change has run its course; the next gain has to come from outside
the model).

Best‑final LOSO mean (`checkpoints/penalty_sweep/best_final/loso_results.txt`):
**0.8683** across all ten folds (range 0.8235 … 0.9071). This is the
checkpoint set every later phase re‑uses.

A small but necessary infra fix lands in the same window (`0751b3b`):
checkpoints saved with `base_filters=128` were being rebuilt by
`evaluate.py` at the default `64`, causing every layer to mismatch.
`_save_norm_stats` now writes `base_filters` into the normalization stats
file and `load_model` reads it back (CLI > stats > 64). `run_eval_only.slurm`
(`d169e71`) lets these checkpoints be re‑evaluated without retraining — it
becomes the harness for every subsequent phase.

---

## Phase 8 — Top‑k accuracy (May 12, 2026)

Commit `12b716d` *“top-k accuracy evaluation (k=1,2,3,5,10)”*.

`evaluate.py` gains a single forward pass that yields top‑k softmax probs
(`predict_all_topk`) plus `topk_accuracy` /
`topk_accuracy_per_class`. `run_topk_eval.slurm` re‑evaluates the
penalty‑sweep checkpoints. Result
(`results/penalty_sweep/topk_eval/topk_summary.json`):

| k   | mean acc   | gain vs top‑1 |
|----:|-----------:|--------------:|
| 1   | **0.8684** | —             |
| 2   | 0.9494     | +0.0810       |
| 3   | **0.9708** | +0.1024       |
| 5   | 0.9823     | +0.1139       |
| 10  | 0.9886     | +0.1202       |

(0.8684 vs 0.8683 from `loso_results.txt` is a per‑fold averaging
artifact — same checkpoints, same data.)

**Observation**: top‑3 already includes the correct label 97 % of the
time, but top‑1 misses 13 % of the time. **~10 pp of headroom is sitting
inside the candidate set**, waiting for a re‑ranker. **Decision**: build
re‑rankers. Two are tried in parallel — temporal (next phase) and
biophysical (Phase 10+).

Per‑fold top‑1 spans 0.8238 (subject 5) … 0.8958 (subject 2); per‑fold
top‑3 spans 0.9452 … 0.9825. The *worst* fold's top‑3 still beats the
best fold's top‑1 — a textbook signal that re‑ranking is the right next
move.

---

## Phase 9 — Temporal re‑ranking (May 12, 2026)

Commits `cc6aeab`, `f88bc0c`, `02cdff0`, `af6d5e9`.

`temporal_rerank.py` (319 lines) implements three post‑processing
strategies on the `(N, k_max)` top‑k softmax output — no retraining:

- `prob_sum` — sliding sum of raw probs (best when calibrated).
- `topk_vote` — weighted votes for the top‑`k_vote` classes per window.
- `majority` — hard majority on top‑1.

Both causal (past‑only) and centred windows. Sweeps all `(strategy,
window)` combos and writes the best config + per‑class gains to
`eval_summary.json["temporal_reranking"]`.

**First result (commit `f88bc0c`) was nonsense**: every config — every
strategy, every window — produced *exactly* 0.8684. **Observation**: the
`penalty_sweep` `.npz` is *class‑sorted*, not time‑sorted. A window of
size 3 spanned two different classes ~66 % of the time, collapsing
accuracy from 86 % to 28 %. The reranker was destroying its own input.

**Decision** (commit `02cdff0`): segment‑aware re‑ranking.
`_get_segments(y_true)` walks `y_true` to find contiguous same‑label
runs, and `_sliding_sum` applies the window *independently per segment*.
A simulated class‑sorted block test in the commit message shows
`prob_sum w=7 → +17 pp` on a 83 % top‑1 baseline, confirming the fix
works on synthetic data.

**Second result (commit `af6d5e9`, in‑tree at
`results/penalty_sweep/rerank_eval/rerank_summary.json`)**: every config
still produces exactly `0.8684`, gain = 0. With the fix in place the
reranker *behaves correctly* (it no longer corrupts top‑1) — but every
window within a class block already has the same top‑1 prediction, so
there is nothing to smooth. **Conclusion logged in‑repo**: temporal
smoothing alone cannot close the top‑1‑to‑top‑3 gap on this evaluation
set; the dataset has no temporal adjacency to exploit. To get gain the
candidate set has to be reordered using a non‑temporal signal — and that
signal is biomechanics.

---

## Phase 10 — Physics verification scorers (May 11, 2026)

Commit `cdbd523` *“physics verification plan impl”* — +13 679 lines. The
second‑largest commit in the history and the start of the physics work.

Three new docs land alongside the code:

- `PHYSICS_VERIFICATION_PLAN.md` (404 lines, with an **implementation log §9**
  that documents every deviation from the plan as it happened).
- `PHYSICS_RESULTS.md` — plain‑English write‑up of what was built and what
  it found.
- `verify_combo.py` (610 lines) — the four scorers + CLI.

### What was planned (`PHYSICS_VERIFICATION_PLAN.md §6, §7`)

Three independent scorers combined with tunable weights:

1. **Kinematic** — parent‑child relative rotation vs a precomputed
   envelope of plausible joint angles from training data.
2. **Gravity alignment** — accel direction during still frames should
   match the region's typical gravity orientation.
3. **Acceleration profile** — per‑region 7‑dim Gaussian over accel stats.

Key decisions made in the Q&A section before code was written:

- **Max 5 sensors per combo** (Q3): combo enumeration limited to 3^5=243
  candidates per scoring call.
- **Separate biophysical score, not fused with classifier confidence
  yet** (Q4): physics is a second opinion, downstream fusion comes later.
- **Both per‑joint and per‑activity accel distributions** (Q2): get more
  information before committing.
- **Decision trigger** (`§6`): if >50 % of validation combos have no
  valid kinematic pair, fall back to kinematic‑only and drop the other
  two scorers.

### What the implementation log (`§9`) actually shows happened

This subsection is the most useful contemporaneous record in the
repository — the physics design *changed during implementation* and the
plan file records why:

- **§9.1 — Gravity scorer is INERT on this dataset.** Discovered
  empirically: the `vimu_joints` accel channel is *linear acceleration*
  (gravity already subtracted). Global per‑axis mean ≈ `(+0.002, −0.002,
  −0.002)` over 518 train samples; pelvis identically zero. **Decision**:
  default combined weights changed from `(0.4, 0.3, 0.3)` to **`(0.5,
  0.0, 0.5)`** — gravity contributes nothing. Code retained for future
  raw‑IMU datasets.
- **§9.2 — `r6d_to_rotmat` was fragile.** Original normalised `a` and
  `b` independently and crossed them — not orthonormal whenever `a`
  was not already perpendicular to `b`. SMPL inputs *are* orthogonal so
  no observable bug, but the function was undertested. **Fix**: full
  Gram‑Schmidt (`b_orth = b − (b · a_norm) * a_norm`, then normalise).
- **§9.3 — Kinematic scorer was using none of the envelope.** Original
  `score_kinematic_chain` loaded `joint_angle_limits` just for a `None`
  check; the actual score was `1 / (1 + ‖R_parent^T·R_child − I‖_F)`,
  i.e. "how far from identity," which has no biological meaning. **Sweep
  AUC vs random combos was 0.27** (worse than chance) at n=3 sensors.
  **Fix**: per‑frame ZYX Euler, look up envelope, exp‑decay penalty
  outside the envelope, mean over frames and over valid parent‑child
  pairs.
- **§9.4 — Joint angle envelope was undertrained.** Original code used
  only `t=0` of each segment (basically T‑pose), wasting 99.7 % of the
  data and producing absurdly tight envelopes. **Fix**: use all `T` frames
  → ~22 segments × 300 frames = 6 600 samples per pair.

### Measured impact after the four fixes (`§9.6`)

AUC against TRUE combos as positives, single‑subject train split:

| n_sensors | scorer    | AUC vs random | vs off‑chain swap | vs L/R swap |
|----------:|-----------|--------------:|------------------:|------------:|
| 4         | kinematic | **0.86**      | 0.62              | 0.55        |
| 4         | accel     | 0.85          | 0.62              | 0.55        |
| 5         | kinematic | **0.90**      | 0.68              | **0.67**    |
| 5         | accel     | 0.87          | 0.63              | 0.54        |

**Observation logged in `§9.6`**: kinematic AUC at n=5 went from 0.62 →
**0.90** vs random and 0.52 → **0.67** vs L/R swap. The kinematic
scorer is the L/R discriminator at high sensor counts; the accel scorer
is the better off‑chain discriminator (matches kinematic vs random but
collapses vs L/R swaps, which by symmetry have nearly identical accel
profiles).

Per‑region accel‑only single‑sensor AUC: median **0.874**, all 24 regions
> 0.7, 11/24 > 0.9. Pelvis / l_foot / l_shin / thighs are easy; spine and
collars are hardest. **Open item flagged in `§9.7`**: a single scalar
`w3` underuses this heterogeneity — per‑region weights are a natural
next step.

### Headline findings transferred to `PHYSICS_RESULTS.md §5`

| regime                                | non‑L/R recoverable | L/R recoverable |
|---------------------------------------|--------------------:|----------------:|
| Best case (kin. neighbors present)    | ~1 476 / 1 522 (97 %) | ~243 / 243 (~100 %) |
| Realistic, n=3 random co‑sensors      |  ~249 / 1 522 (16 %)  | ~61 / 243 (25 %) |

Two kinds of wins inside the "97 %": most are *structural* (swap turns a
valid chain into garbage with no parent‑child pair to score, so truth
≈ 0.85 vs swap ≈ 0.5 default). The strong cases are *envelope* wins
where both labelings produce valid pairs but the truth fits and the swap
doesn't — e.g. `l_hip ↔ r_hip` with `pelvis + thigh` in the combo (truth
0.86, swap 0.10). **The decisive limitation, in one sentence**: the
kinematic scorer's power depends entirely on whether the surrounding
sensors give it kinematic context.

Supporting files added in this phase: `compute_training_stats.py`,
`extract_calibration.py`, `pair_discrimination.py`, `physics_sweep.py`,
`tune_weights.py`, `verify_kinematic.py`, `diagnose_scorers.py`,
`regenerate_npz.py`, `rotation_utils.py`. Stats artifacts:
`stats/joint_angle_limits.npy`, `accel_distributions.npy`,
`gravity_distributions.npy` (inert), `best_weights.npy`,
`sweep_report.json`, plus per‑pair discrimination JSONs.

---

## Phase 11 — Top‑k × physics integration (May 14, 2026)

Commit `b67625f` *“top‑k classification + physics-based filter & rerank”*.

`topk_combo_rerank.py` (273 lines) and `PHYSICS_INTEGRATION_PLAN.md` (319
lines) wire Phase 8 and Phase 10 together:

```
raw IMU stream (n sensors)
        │
        ▼
classifier → predict_all_topk          (per-window top-k softmax)
        │
        ▼
aggregate_topk_per_sensor              (collapse n_windows → per-sensor top-k)
        │
        ▼
enumerate_combos                       (Cartesian product, drop repeats)
        │
        ▼
rerank_combos
  ├─ kinematic chain                   (parent ↔ child envelope)
  ├─ gravity                           (DISABLED — w2 = 0)
  ├─ acceleration profile              (per-region 7-dim Gaussian)
  └─ per-region orientation            (NEW — anatomical eliminator)
        │
        ▼
final = P(classifier) × P(physics)
```

The Phase 10 Q&A had said *“separate score for now”*; this commit's plan
file explicitly **supersedes** that — the new use case is top‑k
re‑ranking and multiplicative fusion is the right operator:
`final = P(classifier) × P(physics)`. The plan section spells it out:
*“this supersedes `PHYSICS_VERIFICATION_PLAN.md §7-Q4` ("keep separate"),
which was answered before a real top‑k re‑ranking use case existed.”*

### New fourth scorer: per‑region absolute orientation

For every region, the training data implies a ZYX‑Euler envelope (after
applying `calibration[region]`). A sensor whose body‑frame orientation
falls outside that envelope is anatomically implausible regardless of
its parent/child neighbours. The envelopes are stored at
`stats/per_region_orientation_limits.npy` with two thresholds:
**strict (10/90 percentile, default)** and **loose (5/95 percentile,
fallback)**.

Two modes:

- **Soft** — per‑frame exp‑decay penalty (same shape as the kinematic
  scorer so it composes naturally on `[0,1]`).
- **Hard** (`--joint_limits_hard <frac>`) — any sensor with > `frac`
  out‑of‑envelope frames *zeroes the entire combo*, dropping it from the
  candidate pool before top‑1 selection.

`PHYSICS_INTEGRATION_PLAN.md §4` includes the **extracted per‑region
rotation table** — strict and loose envelopes for all 24 regions. Notable:

- Arm chain (16–23) has the tightest pitch ROM (~25–50°), well below the
  literature human shoulder/elbow ROM (0–180°). Consistent with DIP‑IMU
  being mostly upright daily‑motion captures, not athletic extremes.
  The scorer can catch grossly mislabelled sensors but won't fire on
  legal motion outside the training distribution.
- Pelvis / spine / shins have wide yaw (≥170°) reflecting full
  locomotion direction span.
- Pitch axis Y is mathematically bounded to ±90° — every observed value
  respects it (no axis‑ordering bug).
- For r_foot Z, l_collar Z, and several arm Z axes, the displayed `max`
  is numerically *smaller* than `min` because of yaw wrap at ±180°. The
  envelope check uses `(min, max)` as a two‑sided interval; for these
  regions the interval wraps the antipode (benign for exp decay).
  **Flagged as an open item**: switch axes 0/2 to *circular* percentiles
  later.

`evaluate.py` gains `--physics_rerank`, `--rerank_n_sensors`,
`--rerank_n_trials`, `--rerank_n_windows`, `--physics_k`,
`--physics_weights`, `--joint_limits_hard`, `--physics_calibration`,
`--physics_stats`. One eval run produces *both* soft and hard pipeline
numbers (`physics_rerank_summary` and `physics_rerank_summary_hard`) so
they can be compared side by side.

---

## Phase 12 — Slurm wrapper and base_filters fix (May 14, 2026)

Commits `17bd4cc`, `ba531e4`, `2c5cb11`.

- `run_physics_topk_sweep.slurm` + the Python driver
  `run_physics_topk_sweep.py` (538 lines) — sweep harness for the new
  re‑ranker. Walltime tweaked in `ba531e4`.
- `2c5cb11` — recurrence of the Phase 7 bug: the sweep was forgetting to
  pass `--base_filters 128` to `evaluate.py`, so checkpoints failed to
  load. Re‑added.

---

## Phase 13 — First partial physics sweep (May 17, 2026)

Commit `6237e2d` *“Incomplete physics results added”*.

A Stage A subset of the sweep finishes:
`results/penalty_sweep/physics_topk_sweep_20260514_025010/`. `sweep.log`
shows: 180 configurations planned, each run ~58 minutes for the harder
ones (`hard0p2` = 3 570 s = ~60 min). Only the `w-classifier_only` × `k=3`
configurations completed (soft + hard at thresholds 0.1, 0.2, 0.3, 0.5)
plus `k=5 soft` partially. **Observation at the time**: at 60 min per
config and 180 configs, a full sweep is 7+ days of cluster time. **Decision**
(next commit): replace the brute‑force sweep with a *paired diagnostic*
that fixes the trial seed across configs.

---

## Phase 14 — Physics diagnostic + paired attribution (May 18, 2026)

Commits `d1128c2`, `6ec41ff`.

`d1128c2` adds **`analyze_physics_diagnostic.py`** (461 lines) and
`run_physics_diagnostic.slurm`. The harness flips from *“try lots of
weight blends and rank by mean accuracy”* to **paired trial‑by‑trial
attribution**: every config sees the same seed and the same sampled
trials, so the marginal contribution of each scorer is `n_fixed −
n_broken` rather than the difference of two noisy means.

`6ec41ff` commits the outputs at
`results/penalty_sweep/physics_diagnostic_20260517_235338/` covering five
configs: `baseline_classifier_only`, `pure_kin`, `pure_acc`,
`pure_jointlim_soft`, `default_blend`.

### Ceiling at `physics_k = 3` (`diagnostic_report.md §1`)

Upper bound any re‑ranker can reach given the per‑sensor top‑3 candidate
set:

| n_sensors | exact‑match ceiling | trials |
|----------:|--------------------:|-------:|
| 2         | 0.9400              | 200    |
| 3         | 0.9300              | 200    |
| 4         | 0.8750              | 200    |
| 5         | 0.8850              | 200    |

**Observation**: the ceiling *decreases* with more sensors at `k=3` —
each additional sensor introduces an independent chance that the true
region falls outside the top‑3. **Decision** (open item): if `k=5`
ceilings are higher, the rerank pipeline should pick `k` based on
`n_sensors`.

### Per‑scorer paired attribution vs `baseline_classifier_only` (`§2`)

| config              | Δ per‑sensor | Δ exact | n_fixed | n_broken | net_fix |
|---------------------|-------------:|--------:|--------:|---------:|--------:|
| **pure_kin**        | **−0.0243**  | −0.0587 | 44      | 112      | **−68** |
| pure_acc            | −0.0014      | −0.0062 | 17      | 21       | −4      |
| pure_jointlim_soft  | +0.0011      |  0.0000 |  4      |  1       | +3      |
| **default_blend**   | **+0.0039**  | +0.0051 | 22      | 11       | **+11** |

`default_blend` breakdown by `n_sensors` (`§2a`):

| n  | base per‑sensor | blend per‑sensor | net_fix |
|---:|----------------:|-----------------:|--------:|
| 2  | 0.8275          | 0.8325           | +2      |
| 3  | 0.8267          | 0.8317           | +3      |
| 4  | 0.8200          | 0.8225           | +2      |
| 5  | 0.8220          | 0.8260           | +4      |

Stratified by region difficulty (`§4`, "easy" = classifier top‑1 ≥ 0.95):

| config              | easy net_fix | hard net_fix |
|---------------------|-------------:|-------------:|
| pure_kin            | **−45** (1 fixed / 46 broken) | −23 |
| pure_acc            | −3           | −1           |
| pure_jointlim_soft  | +0           | +3           |
| default_blend       | +0           | **+11** (21 fixed / 10 broken) |

**Observations at the time** (lifted from the diagnostic report and what
it implies for the project):

- **`pure_kin` regresses on its own** because in random combos the
  scorer is silent on most cases — and on the easy ones it has just
  enough wrong signal to flip 46 already‑correct predictions for the 1
  it gets right. This is exactly the *“depends entirely on kinematic
  context”* limit from `PHYSICS_RESULTS.md`: scoring arbitrary candidate
  combos puts the system in the realistic regime (16 % recovery) rather
  than the best case (97 % recovery).
- **`pure_jointlim_soft` and `default_blend` are net positive**, and the
  blend's gain (+0.0039) is more than the best single scorer (+0.0011) —
  i.e. **the signals are additive, not redundant**.
- **All of `default_blend`'s gain comes from hard cases** (+11 net on
  the 1 834 hard observations, ±0 net on the 966 easy ones). The
  threshold for "easy" is generous (≥95 %), so the blend is not breaking
  the model where it's already confident.
- **Confusion pairs fixed by the blend** (`§2b`): exactly the pairs
  Phase 7 identified as the dominant error sources — `r_hip → l_hip`
  (4), `l_forearm → l_upper_arm` (3), `r_upper_arm → r_shoulder` (3),
  `neck → spine_upper` (3), `l_thigh → r_thigh` (2).

### Decisions captured

- The kinematic scorer cannot be applied to arbitrary candidate combos
  without breaking more than it fixes. **Use it only when the combo
  includes kinematic neighbors**, or down‑weight it heavily in
  general‑case blends.
- The blend wins by *adding* the joint‑limit eliminator on top, not by
  pushing the kinematic weight higher.
- The diagnostic harness (paired trial sampling) is the right
  evaluation protocol for any future scorer work — mean‑of‑means is too
  noisy at the magnitudes involved (gains of 0.001‑0.005 per sensor).

This is where the project stands at HEAD.

---

## Accuracy progression at a glance (all numbers verified in‑tree)

| Phase | Setup                                               | Headline metric           | Source |
|-------|-----------------------------------------------------|---------------------------|--------|
| 4     | Fixed split, ResNet1D, plain CE                     | **0.7171** per‑window     | `results/full_split_full/eval_summary.json` |
| 4     | Best ablation (`fusion_blend01_seed42`)             | 0.7807 per‑window         | `results/ablations/summary.csv` |
| 5     | LOSO baseline, 10 folds                             | **0.7517** mean LOSO      | `checkpoints/loso_full/loso_results.txt` |
| 5     | LOSO tuned (`trial_e`)                              | **0.7817** mean LOSO      | `checkpoints/loso_tune/best_final/loso_results.txt` |
| 6     | Epoch sweep best (no SpatialNeighborLoss)           | 0.8480 (5 folds, sweep)   | `results/epoch_sweep/leaderboard.csv` |
| 6b    | SpatialNeighborLoss + class weights + bf=128        | **0.8611** mean LOSO      | `checkpoints/epoch_sweep/best_final/loso_results.txt` |
| 7     | Penalty sweep best‑final, all 10 folds              | **0.8683** mean LOSO      | `checkpoints/penalty_sweep/best_final/loso_results.txt` |
| 8     | Same checkpoints, top‑k ceiling                     | top‑3 0.9708, top‑5 0.9823 | `results/penalty_sweep/topk_eval/topk_summary.json` |
| 9     | Temporal re‑ranking, segment‑aware                  | +0.000 vs top‑1            | `results/penalty_sweep/rerank_eval/rerank_summary.json` |
| 14    | Physics `default_blend`, paired diagnostic          | **+0.0039 per‑sensor**, net_fix +11 | `results/penalty_sweep/physics_diagnostic_20260517_235338/diagnostic_report.md` |

The remaining headroom from top‑1 (0.8684) to top‑3 (0.9708) is the
~10‑point pool the physics work targets. Phase 14 demonstrates the
mechanics work — soft blended physics is a net‑positive re‑ranker on
random combos — but realising the full gain depends on *deliberately
seeding kinematic neighbors into the candidate combos*, exactly the
design lever called out at the end of `PHYSICS_RESULTS.md §6`.

---

## Open threads at HEAD

- Hard‑mode threshold sweep for the per‑region orientation eliminator
  (`PHYSICS_INTEGRATION_PLAN.md §5.4`).
- Switch the per‑region orientation envelope to circular percentiles for
  yaw/roll (axes 0 and 2 currently wrap at ±180°; handled benignly by
  the exp decay but a known follow‑up).
- A physics re‑ranker variant that explicitly seeds kinematic neighbors
  rather than scoring arbitrary candidate combos — the path from the
  +0.4 pp realistic‑regime diagnostic toward the +7.9 pp best‑case
  ceiling.
- Per‑region weights for the accel scorer (median per‑region AUC 0.874
  but `pelvis`=1.00 vs `spine_lower`=0.76 — a single scalar `w3`
  underuses the signal).
- Rerun temporal re‑ranking on a *time‑ordered* eval set so the
  segment‑aware fix from `02cdff0` actually has a signal to work with.
- `k=5` ceiling sweep — Phase 14 shows the `k=3` ceiling *decreases*
  with more sensors; the rerank pipeline should likely pick `k` based on
  `n_sensors`.
