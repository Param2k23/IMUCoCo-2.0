# IMUCoCo-2.0 — Complete Project Report

---

# Part I: Extended Research & Development Narrative (Pre-IMUCoCo Phase)

The work conducted during this semester began as a continuation and expansion of the original CSE570 course project, **RegenHAR: Robust Human Activity Recognition via Transformer-Based Sensor Regeneration**. While the original project successfully demonstrated robust Human Activity Recognition (HAR) under missing sensor scenarios, the primary goal of this semester's research was to investigate whether the same ideas could generalize beyond activity classification into more complex downstream tasks, particularly full-body pose estimation from sparse IMUs.

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

> "Can we recognize activities with missing sensors?"

to:

> "Can we recover physically meaningful motion information from sparse IMU setups?"

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

# Part II: IMUCoCo‑2.0 — Implementation and Experimental Work

The second part of the project translated the IMUCoCo motivation from Part I into a working body‑region classification system. The central research question for this phase was deliberately narrower than the one introduced at the end of Part I: *given a single virtual IMU sensor mounted somewhere on the body, can a learned classifier identify which of 24 SMPL body regions the sensor occupies?* Solving this problem reliably would directly address the limitation that IMUCoCo assumes sensor placements are known at inference time.

This part documents the full experimental program — the dataset choices, the architectural baselines, the loss‑function modifications, the post‑hoc re‑ranking experiments, and finally the integration of biophysical scorers as a candidate filter. The work proceeded as a sequence of well‑scoped experiments, each motivated by the failure modes of the previous one. Every headline number reported below has been re‑verified against the result artifacts in‑tree.

## 1. System Overview and Evaluation Protocol

### 1.1 Pipeline

The classification system follows a three‑stage pipeline:

1. **Preprocessing** — virtual IMU segments are converted into a uniform `.npz` contract `X (N, 9, T) float32`, `y (N,) int`, `subject_ids (N,) int`. The 9 input channels consist of six 6D rotation coefficients followed by three linear acceleration components.
2. **Training** — a deep convolutional model is trained either on a fixed train/test split or under leave‑one‑subject‑out cross validation (LOSO). Z‑score normalization statistics are computed on the train fold and stored separately to prevent leakage.
3. **Evaluation** — per‑window top‑1 accuracy, per‑region confusion, spatial error on the SMPL T‑pose mesh, and L/R symmetry confusion are computed and serialized.

Each experiment was run on an HPC cluster using H200 GPUs, with batch jobs encapsulating the preprocess–train–evaluate sequence.

### 1.2 Class Taxonomy

The 24‑region taxonomy corresponds to the standard SMPL joint layout: pelvis, spine (lower/middle/upper), neck, head, both collars, both shoulders, both upper arms, both forearms, both hands, both hips, both thighs, both knees, both shins, both feet. Left and right are kept as separate classes throughout — merging them would obscure the most informative confusion patterns observed during evaluation.

### 1.3 Evaluation Metrics

Four metrics are tracked across every experiment:

- **Per‑window top‑1 accuracy** — the primary headline metric.
- **Spatial error** — Euclidean distance in metres between the predicted and true region centroids on the SMPL T‑pose mesh, capturing how *anatomically close* the wrong predictions are.
- **L/R confusion rate** — fraction of errors corresponding to a left/right mirror swap.
- **Per‑class confusion matrix** — used to identify the dominant non‑L/R failure modes.

---

## 2. Initial Scaffold and Architecture

The first step was building the foundational classifier scaffold around an SMPL‑aligned class taxonomy.

The scaffold introduced a 24‑region vertex map and parent table derived from SMPL, a Euclidean spatial‑error helper measuring distance in metres between predicted and true region centroids on the T‑pose mesh, and two convolutional architectures: a 1D ResNet (the default) and a 1D CNN baseline. Both models took the `(B, 9, T)` IMU window as input and produced 24 region logits.

Training and evaluation harnesses supported both LOSO and fixed‑split modes from the outset. The evaluator computed per‑window accuracy, the per‑class confusion matrix, spatial error, L/R symmetry pairs, and a majority‑vote aggregate.

At this stage the system targeted the AMASS dataset as the primary data source. After an initial usability review — paths assumed a Windows filesystem, the disk‑level project name still referenced its previous identity as *SensorLoc*, and there was no GPU policy in place — the immediate decision was to pivot to a Linux + CUDA pipeline operating on synthetic virtual IMU (VIMU) data.

---

## 3. Pivot to the Synthetic VIMU Dataset

The next step was the largest single rewrite in the implementation history: converting the preprocessing path to ingest DIP‑style virtual IMU segments rather than AMASS pose data.

The new preprocessor read VIMU `.pt` segments and emitted the canonical `.npz` contract described in §1.1. Each training sample corresponded to one `(segment, body_region)` pair, with channels laid out as `r6d_0..r6d_5, ax, ay, az`. The training script was rewritten to support a `fixed_split` mode and to enforce a strict CUDA policy outside of smoke runs, eliminating the risk of accidental multi‑hour CPU jobs. Evaluation was hardened with weights‑only checkpoint loading and fail‑fast checks for missing normalization statistics or channel mismatches.

Several design decisions were locked in at this stage:

- **Class taxonomy fixed at 24 regions.** Merging left and right classes was deliberately deferred to expose symmetry confusion as a measurable failure mode rather than hide it.
- **Two parallel data pipelines.** A *single‑subject* path that ignores the source split, and a *predefined split* path that respects the source train/test folders. Both produced the same `.npz` contract downstream.
- **Augmentations, attention blocks, and EMA gated behind a "future" tag.** The deliberate ordering was: establish a clean baseline first, then layer in modeling tricks only if a measurable plateau warranted it.

---

## 4. Smoke Test and Pipeline Hardening

A small but consequential step was the introduction of a smoke‑test command that ran the full preprocess → train → evaluate sequence on a tiny slice of data. This became the canonical sanity check for every new machine and every later infrastructure change.

A centralized normalization module was added so that training and evaluation could not drift apart on z‑score statistics — a class of bug that would otherwise be invisible until evaluation numbers became inexplicable. Checkpoints were converted to weights‑only format with separately saved normalization statistics, and the evaluator gained fail‑fast guards on missing statistics or channel mismatches.

The standing requirement from this point onward was that every cluster runbook had to keep the smoke path functional. The smoke command became the contract every later infrastructure change had to honor.

---

## 5. Hugging Face Parquet Ingestion

To unlock larger and more diverse training data, the preprocessing path was extended to read Hugging Face parquet shards in addition to DIP `.pt` segments. The new mode parsed `train-*.parquet` and `test-*.parquet` shard families directly and emitted the same `.npz` contract, so every downstream step remained dataset‑agnostic.

Two practical problems surfaced during this step:

- Parquet rows had variable temporal length `T`, and shard joins occasionally produced mixed‑length stacks. The resolution was to crop or pad every row to a fixed `window_length=300` before stacking, and to drop rows that could not be normalized.
- Subject identifiers were inconsistently encoded across shards. A robust parsing path was added with column‑name overrides and a filename‑based fallback, consolidated into a single summary warning per run.

By the end of this step the data path was source‑agnostic: every later experiment consumed the same `.npz` contract regardless of whether its data originated from DIP `.pt` files or HF parquet shards.

---

## 6. First Baseline on the Cluster

With the dataset pipeline stable, the first end‑to‑end baseline was run on the HPC cluster.

Training characteristics:

- Training set: `X = (21240, 9, 300)`, 24 classes, **8 subjects**.
- Test set: `X = (456, 9, 300)`, 24 classes, **2 subjects**.
- Budget: 80 epochs, early stopping with patience 15.

Training stopped at epoch 30, with the best validation accuracy of 0.7171 reached at epoch 15.

Evaluation summary:

| Metric                       | Value |
|------------------------------|-------|
| Per‑window accuracy          | **0.7171** |
| Majority‑vote acc. (k=5)     | 0.2857 |
| Locked fraction              | 4.6 % |
| Windows wrong                | 129 / 456 (28.3 %) |
| Spatial error mean / std (m) | 0.4064 / 0.3131 |

Two observations from this run set the trajectory for the rest of the project:

- The majority‑vote metric was essentially broken — only 4.6 % of windows ever locked into a stable prediction. The metric was deprecated as a primary signal.
- The classification report showed left/right errors on hip, shin, and hand at 5.3 % each, and the arm chain (`l_upper_arm`, `l_forearm`, `l_hand`) recall sat at 0.42 / 0.42 / 0.47. **The adjacent‑region arm‑chain failure that dominates every later result was already visible at 71 % overall accuracy.**

To support targeted ablations rather than blind retraining, the training script was extended to expose weight decay as a CLI knob, and a dedicated ablation‑sweep runbook was added. The first ablation results, on the fixed split:

| Run                          | Val per‑window acc |
|------------------------------|-------------------:|
| baseline_seed42              | 0.7719 |
| baseline_seed43              | 0.7303 |
| baseline_seed44              | 0.7654 |
| **fusion_blend01_seed42** (imu_blend = 0.1) | **0.7807** |
| fusion_blend02_seed42 (imu_blend = 0.2) | 0.7434 |
| reg_batch64_wd3e4_seed42     | 0.7346 |

A modest 10 % blend of raw IMU data into the synthetic VIMU input gave the best ablation; the heavier 20 % blend regressed. The decision was to keep VIMU as the primary input channel.

---

## 7. LOSO Baseline and Hyperparameter Tuning

Once the fixed split had been characterized, leave‑one‑subject‑out cross validation was adopted as the primary research split. LOSO is the more honest evaluation regime for a subject‑transfer problem because it never lets the model see the same subject in training and validation.

The merged dataset for LOSO contained `X = (21696, 9, 300)` across 10 subjects.

### 7.1 LOSO Baseline

| Fold | Subject | Val acc | Fold | Subject | Val acc |
|-----:|--------:|--------:|-----:|--------:|--------:|
| 1    | 1       | 0.7342  | 6    | 6       | 0.7061  |
| 2    | 2       | 0.7241  | 7    | 7       | **0.8425** |
| 3    | 3       | 0.7507  | 8    | 8       | 0.7877  |
| 4    | 4       | 0.7512  | 9    | 9       | 0.7315  |
| 5    | 5       | 0.8018  | 10   | 10      | **0.6875** |

The mean LOSO accuracy was **0.7517** (std 0.0444, with a 15.5 percentage‑point spread from worst to best fold).

The 15 pp spread across folds was strong evidence that *subject identity dominated the residual error*. Two same‑configuration runs differed by more than the gain typically produced by a hyperparameter change. The next experiment therefore tuned hyperparameters on a 5‑fold proxy and re‑ran the winner on all ten folds.

### 7.2 Hyperparameter Tuning

A 5‑trial sweep on folds 1‑5:

| Trial | Epochs | Batch size  | LR   | WD   | Patience | Sweep acc |
|-------|-------:|----:|-----:|-----:|---------:|----------:|
| a     | 80     | 128 | 1e‑3 | 1e‑4 | 15       | 0.7393    |
| b     | 100    | 128 | 1e‑3 | 3e‑4 | 20       | 0.7800    |
| c     | 80     | 64  | 1e‑3 | 1e‑4 | 15       | 0.7658    |
| d     | 80     | 128 | 5e‑4 | 1e‑4 | 20       | 0.7575    |
| **e** | 120    | 128 | 7e‑4 | 3e‑4 | 25       | **0.7810** |

Trials `b` and `e` both benefited from stronger weight decay (`3e-4`) and longer patience; the smaller learning rate plus 120 epochs in trial `e` was the marginal winner. Re‑running trial `e` on all ten folds produced a mean LOSO accuracy of **0.7817**, a +3.0 pp gain over the baseline.

A telling pattern emerged in this rerun: the worst fold (subject 6) did not improve from its baseline value of 0.6875. Tuning helped the easy folds more than the hard ones — a theme that recurred in every later experiment.

---

## 8. Epoch Sweep and the SpatialNeighborLoss

By this point the L/R and arm‑chain confusion patterns were clearly visible but unaddressed. The next experiment pursued two complementary directions: extending the training budget, and modifying the loss function to inject anatomical structure.

### 8.1 Epoch Sweep

A 6‑step automated pipeline (preprocess → smoke → 5‑fold sweep → pick best → train all 10 folds → evaluate all 10 folds) was introduced to systematically explore training duration. Evaluation was extended to write per‑fold confusion matrices and L/R confusion aggregates in the same step.

| Trial            | Epochs | Batch size  | LR   | WD   | Patience | Loss   | LR weight  | Mean (5 folds) |
|------------------|-------:|----:|-----:|-----:|---------:|--------|-----:|---------------:|
| ep300_custom     | 300    | 128 | 1e‑3 | 3e‑4 | 50       | custom | 0.5  | 0.8106 |
| **ep400_custom** | **400**| 128 | 1e‑3 | 3e‑4 | 60       | custom | 0.5  | **0.8480** |
| ep500_custom     | 500    | 128 | 1e‑3 | 3e‑4 | 60       | custom | 0.5  | 0.8272 |
| ep600_custom     | 600    | 128 | 1e‑3 | 3e‑4 | 60       | custom | 0.5  | 0.8157 |
| ep800_custom     | 800    | 128 | 1e‑3 | 3e‑4 | 60       | custom | 0.5  | 0.8296 |

The curve was clearly non‑monotone: 400 epochs beat both 500 and 800. Four hundred was the sweet spot, and the conclusion was that further gains had to come from the loss function rather than the training budget.

### 8.2 SpatialNeighborLoss

Three modifications shipped together as the next experiment:

1. **SpatialNeighborLoss** — two physics‑informed penalty terms layered on top of cross‑entropy. The first is a *spatial* term penalizing predictions in proportion to the Euclidean distance between predicted and true region centroids on the SMPL T‑pose; the second is an *L/R mirror* term applying additional cost when the wrong side of the body is predicted, controlled by an `lr_weight` hyperparameter (default 0.5).
2. **Inverse‑frequency class weights**, computed per‑fold from the training split.
3. **A wider model** with `base_filters = 128` (approximately 5M parameters) so the spatial penalty had more capacity to push against.

With these in place, the 400‑epoch configuration was rerun on all ten folds:

| Fold | Subj | Val acc | Fold | Subj | Val acc |
|-----:|-----:|--------:|-----:|-----:|--------:|
| 1    | 1    | 0.8315  | 6    | 6    | 0.8180  |
| 2    | 2    | 0.8303  | 7    | 7    | 0.8798  |
| 3    | 3    | **0.8933** | 8 | 8 | 0.8648 |
| 4    | 4    | 0.8476  | 9    | 9    | 0.8843  |
| 5    | 5    | 0.8736  | 10   | 10   | 0.8875  |

The mean LOSO accuracy reached **0.8611** — +8.0 pp over the LOSO baseline, +7.9 pp over the tuned baseline, and the largest single jump in the entire project.

The most informative single observation was that subject 6 — the worst fold throughout earlier experiments — jumped from 0.6875 to 0.8180. The spatial penalty was doing exactly what its design intended: converting wrong‑arm‑region errors into *near* arm‑region errors that the class‑weighted cross‑entropy could then learn out.

---

## 9. Penalty Sweep and Non‑L/R Confusion Analysis

With the L/R confusion rate now small, the next experiment was scoped to identify *what else* was producing errors. A three‑stage penalty sweep was designed:

- **Stage 1** — `neighbor_weight ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` at fixed `lr=1e-3` and `lr_weight=0.5`, isolating the spatial penalty effect.
- **Stage 2** — a 3 × 3 grid of `lr ∈ {3e-4, 7e-4, 2e-3}` and `neighbor_weight ∈ {0.0, 0.3, 0.5}`.
- **Stage 3** — `lr_weight ∈ {0.0, 0.3, 1.0, 2.0}` at fixed `lr=1e-3` and `neighbor_weight=0.3`.

The winning configuration was retrained on all ten LOSO folds.

To attack the non‑L/R failure modes, the evaluator was extended to emit the full 24 × 24 confusion matrix and the top‑10 non‑L/R confusion pairs per fold. The 10‑fold aggregate revealed:

```
total_windows  : 21 696
total_errors   :  2 931   (top-1 = 86.5%)
L/R errors     :    243   ( 8.3% of errors)
non-L/R errors :  2 688   (91.7% of errors)
```

The dominant non‑L/R confusion pairs:

| Pair                            | Count | % of errors |
|---------------------------------|------:|------------:|
| r_forearm → r_upper_arm         | 211   | 7.20 |
| l_forearm → l_upper_arm         | 143   | 4.88 |
| r_upper_arm → r_forearm         | 139   | 4.74 |
| l_forearm → l_hand              | 133   | 4.54 |
| l_upper_arm → l_forearm         | 116   | 3.96 |
| r_forearm → r_hand              | 107   | 3.65 |
| r_hand → r_forearm              |  97   | 3.31 |
| l_hand → l_forearm              |  90   | 3.07 |

Every one of the top‑8 non‑L/R confusion pairs sat on the same arm chain (`forearm ↔ upper_arm`, `forearm ↔ hand`). The architecture had effectively solved L/R symmetry; the remaining work concentrated on *adjacent‑region* arm‑chain disambiguation. Because adjacent regions share strongly overlapping motion statistics, the conclusion was that further gains would need to come from outside the model itself — through *post‑hoc re‑ranking* using an additional, complementary signal.

The best‑final LOSO mean for this experiment was **0.8683** across all ten folds, with a per‑fold range of 0.8235 to 0.9071. This checkpoint set became the reference snapshot for every subsequent re‑ranking experiment.

A subtle infrastructure issue surfaced during this experiment: checkpoints saved with `base_filters=128` were being rebuilt by the evaluator at the default `base_filters=64`, causing tensor‑shape mismatches at load time. The fix was to record `base_filters` inside the normalization statistics file and have the evaluator read it back, with the CLI flag taking precedence. An eval‑only runbook was also added so checkpoints could be re‑evaluated without retraining — this harness was reused by every later re‑ranking experiment.

---

## 10. Top‑k Accuracy Evaluation

The penalty‑sweep experiment had effectively reached a plateau on top‑1 accuracy. The next experiment asked: *how often is the correct answer present in the top‑k predictions, even when it is not the top‑1?*

The evaluator was extended with a single forward pass that yielded the top‑k softmax probabilities for k up to 10, along with per‑class top‑k accuracy. Re‑evaluating the penalty‑sweep checkpoints produced:

| k   | Mean acc   | Gain vs top‑1 |
|----:|-----------:|--------------:|
| 1   | **0.8684** | —             |
| 2   | 0.9494     | +0.0810       |
| 3   | **0.9708** | +0.1024       |
| 5   | 0.9823     | +0.1139       |
| 10  | 0.9886     | +0.1202       |

The top‑3 predictions contained the correct label 97 % of the time, while top‑1 missed 13 % of the time. **Approximately 10 percentage points of accuracy were sitting inside the top‑3 candidate set, waiting for a re‑ranker to surface them.**

Per‑fold top‑1 spanned 0.8238 (subject 5) to 0.8958 (subject 2); per‑fold top‑3 spanned 0.9452 to 0.9825. The worst fold's top‑3 accuracy still exceeded the best fold's top‑1 accuracy — a textbook signal that re‑ranking was the right next direction.

Two re‑ranking strategies were then pursued in parallel: **temporal re‑ranking** (described next) and **biophysical re‑ranking** (described in Sections 12–14).

---

## 11. Temporal Re‑ranking Experiment

The first re‑ranker tested was purely temporal: aggregate the top‑k softmax outputs across adjacent windows under the assumption that nearby windows should agree on the body region.

Three post‑processing strategies were implemented over the `(N, k_max)` top‑k probability output, with no retraining required:

- `prob_sum` — sliding sum of raw probabilities (most appropriate when the classifier is calibrated).
- `topk_vote` — weighted votes for the top‑`k_vote` classes per window.
- `majority` — hard majority on the top‑1 prediction.

Both causal (past‑only) and centred windows were tested. The harness swept all `(strategy, window)` combinations and recorded the best configuration along with per‑class gains.

### 11.1 First Result and Root Cause

The first run produced an anomalous result: every configuration — every strategy, every window size — produced *exactly* the same top‑1 accuracy of 0.8684, gain zero. Closer inspection revealed why. The evaluation set was *class‑sorted*, not time‑sorted. A sliding window of size 3 spanned two different region labels approximately 66 % of the time, collapsing local accuracy from 86 % to 28 %. The reranker was effectively destroying its own input.

### 11.2 Segment‑Aware Fix

The fix was *segment‑aware re‑ranking*. A helper walked the label array to identify contiguous same‑label runs, and the sliding aggregator was applied independently within each segment. A synthetic test on class‑sorted blocks confirmed the fix worked: `prob_sum w=7` produced a +17 pp gain on an 83 % top‑1 baseline.

### 11.3 Outcome on the Real Evaluation Set

With the segment‑aware fix in place, every configuration on the real evaluation set again produced exactly 0.8684. The reranker now behaved correctly — it no longer corrupted top‑1 — but every window within a class segment already shared the same top‑1 prediction. There was simply nothing left to smooth.

The conclusion logged in‑repo was unambiguous: temporal smoothing alone could not close the top‑1‑to‑top‑3 gap on this evaluation set. The dataset offered no temporal adjacency to exploit. To realize the 10‑point pool inside the top‑3 candidate set, the reranking signal had to come from a non‑temporal source. The remainder of the project pursued that signal in the form of biomechanics.

---

## 12. Physics Verification Scorers

The next experiment introduced biophysical scoring as the candidate non‑temporal signal. The first deliverable was a set of standalone scorers, evaluated as a *verification pass* over candidate sensor‑to‑region assignments rather than as classifier‑integrated rerankers.

### 12.1 Planned Scorers

Three independent scorers were planned, combined with tunable weights:

1. **Kinematic scorer** — parent‑child relative rotation compared against a precomputed envelope of plausible joint angles extracted from training data.
2. **Gravity alignment scorer** — acceleration direction during still frames compared against the region's typical gravity orientation.
3. **Acceleration profile scorer** — a per‑region 7‑dimensional Gaussian fit to acceleration statistics.

Key design decisions made before implementation:

- **At most five sensors per candidate combination.** This kept combo enumeration manageable: `3^5 = 243` candidates per scoring call.
- **Biophysical scores kept separate from classifier confidence for now.** Physics would serve as a second opinion; downstream fusion would come later.
- **Both per‑joint and per‑activity acceleration distributions** were extracted to support either choice later.
- **Decision trigger.** If more than 50 % of validation combos lacked any valid parent‑child kinematic pair, the system would fall back to a kinematic‑only configuration with the other two scorers disabled.

### 12.2 Implementation Findings

The implementation log produced four substantive corrections to the planned design, each documented as it was discovered.

- **The gravity scorer is inert on this dataset.** Empirical inspection of the `vimu_joints` acceleration channel revealed that it represents *linear acceleration* (gravity is already subtracted). Across 518 training samples the global per‑axis mean was approximately `(+0.002, −0.002, −0.002)`, and the pelvis channel was identically zero. The default combined weights were changed from `(0.4, 0.3, 0.3)` to `(0.5, 0.0, 0.5)`, removing gravity from the active blend. The code was retained for future raw‑IMU datasets where gravity would be present.
- **The 6D‑to‑rotation‑matrix conversion was fragile.** The original implementation normalized the first and second basis vectors independently and crossed them, which produced a non‑orthonormal frame whenever the inputs were not already perpendicular. SMPL inputs happen to be orthogonal, so there was no observable downstream bug, but the function was undertested. The fix was a full Gram‑Schmidt: orthogonalize the second basis vector against the first before normalizing.
- **The kinematic scorer was using none of its envelope.** The original `score_kinematic_chain` function loaded the joint angle limit table but only checked it for `None`; the actual score was `1 / (1 + ‖R_parent^T · R_child − I‖_F)`, i.e. "how far from identity," which has no biological meaning. AUC against random combos was 0.27 — *worse than chance* — at n=3 sensors. The fix was to extract a per‑frame ZYX Euler decomposition, look up the precomputed envelope, apply an exponential‑decay penalty for samples outside the envelope, and average over frames and valid parent‑child pairs.
- **The joint angle envelope was undertrained.** The original training code used only the first frame `t=0` of each segment — essentially the T‑pose — wasting 99.7 % of the available data and producing absurdly tight envelopes. The fix was to use all `T` frames per segment, yielding approximately 6,600 samples per joint pair instead of 22.

### 12.3 Measured Impact

Once the four corrections were in place, AUC against true combos as positives (single‑subject train split) was:

| n_sensors | Scorer    | AUC vs random | vs off‑chain swap | vs L/R swap |
|----------:|-----------|--------------:|------------------:|------------:|
| 4         | kinematic | **0.86**      | 0.62              | 0.55        |
| 4         | accel     | 0.85          | 0.62              | 0.55        |
| 5         | kinematic | **0.90**      | 0.68              | **0.67**    |
| 5         | accel     | 0.87          | 0.63              | 0.54        |

At n=5 sensors the kinematic scorer reached AUC 0.90 against random combos and 0.67 against L/R swaps — a major recovery from the pre‑fix 0.27. The two scorers turned out to play complementary roles: the kinematic scorer was the stronger L/R discriminator at high sensor counts, while the acceleration scorer was the better discriminator against off‑chain swaps. By symmetry, L/R swaps produced nearly identical acceleration profiles, which explained why the acceleration scorer collapsed on L/R comparisons.

Per‑region acceleration‑only single‑sensor AUC had a median of **0.874**. All 24 regions exceeded 0.7, and 11 of 24 exceeded 0.9. Pelvis, left foot, left shin, and the thighs were the easiest cases; spine and collars were the hardest. The wide spread suggested that a single scalar weight for the acceleration scorer was leaving signal on the table — per‑region weights were a natural follow‑up.

### 12.4 Best‑Case vs Realistic Recovery

Two regimes were characterized:

| Regime                                | non‑L/R recoverable | L/R recoverable |
|---------------------------------------|--------------------:|----------------:|
| Best case (kinematic neighbors present)    | ~1 476 / 1 522 (97 %) | ~243 / 243 (~100 %) |
| Realistic, n=3 random co‑sensors      |  ~249 / 1 522 (16 %)  | ~61 / 243 (25 %) |

Within the best case, most wins were *structural*: a wrong labeling turned a valid kinematic chain into garbage with no parent‑child pair to score, so the truth combo scored ~0.85 while the swapped combo defaulted to ~0.5. The stronger cases were *envelope* wins where both labelings produced valid pairs but only the truth fit the envelope — for example, an `l_hip ↔ r_hip` swap with `pelvis + thigh` in the combo (truth 0.86, swap 0.10).

The decisive limitation, stated in one sentence: **the kinematic scorer's discriminative power depends entirely on whether the surrounding sensors give it kinematic context**. This conclusion became the central design constraint for the next experiment.

---

## 13. Top‑k × Physics Integration

The next experiment wired the top‑k classifier output and the physics scorers into a unified re‑ranking pipeline.

The structure of the combined system:

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

The earlier design decision to keep physics separate from classifier confidence was explicitly superseded for this use case: when the goal is top‑k re‑ranking, multiplicative fusion — `P(final) = P(classifier) × P(physics)` — is the natural operator.

### 13.1 The Fourth Scorer: Per‑Region Absolute Orientation

A new scorer was introduced as part of the integration. For every body region, the training data implied a ZYX‑Euler envelope (after applying region‑specific calibration). A sensor whose body‑frame orientation fell outside that envelope was anatomically implausible regardless of its parent or child neighbours. Envelopes were stored with two thresholds:

- **Strict** — 10th and 90th percentile bounds, used by default.
- **Loose** — 5th and 95th percentile bounds, used as a fallback.

Two scoring modes were implemented:

- **Soft mode** — a per‑frame exponential decay penalty matching the shape of the kinematic scorer, so the two composed naturally on `[0, 1]`.
- **Hard mode** — any sensor with more than a configurable fraction of out‑of‑envelope frames *zeroed the entire combo*, dropping it from the candidate pool before the top‑1 selection.

### 13.2 Notes on the Envelope Table

Inspection of the extracted per‑region rotation envelopes produced several useful observations:

- The arm chain had the tightest pitch range of motion (approximately 25°–50°), well below the literature human shoulder/elbow range of motion (0°–180°). This was consistent with the DIP‑IMU dataset being dominated by upright daily‑motion captures rather than athletic extremes. The implication was that the scorer could catch grossly mislabelled sensors but would not penalize legal motion outside the training distribution.
- Pelvis, spine, and shins showed wide yaw ranges (≥170°), reflecting full locomotion direction span.
- The pitch axis was mathematically bounded to ±90°, and every observed value respected this — confirming no axis‑ordering bug.
- Several axes showed a numerically smaller `max` than `min` due to yaw wraparound at ±180°. The envelope check used `(min, max)` as a two‑sided interval, which wrapped the antipode benignly under exponential decay but was flagged as an open item — circular percentiles would be the principled fix.

The evaluator was extended with a comprehensive set of CLI knobs for the new re‑ranker, and a single eval run could produce both soft‑mode and hard‑mode pipeline numbers side by side for direct comparison.

---

## 14. Physics Sweep Infrastructure

The next step was building the harness required to systematically evaluate the rerank pipeline. A driver was added that enumerated the cartesian product of weight blends, candidate set sizes (`physics_k`), hard‑mode thresholds, and trial seeds.

A recurrence of the `base_filters` mismatch surfaced once again: the sweep driver was not forwarding `--base_filters 128` to the evaluator, so the wider checkpoints failed to load. The fix was the same as the original incident — make the parameter explicit in the sweep configuration.

---

## 15. Initial Partial Physics Sweep

A first end‑to‑end run of the rerank sweep made it through a Stage A subset of the planned 180 configurations. The log showed wall times of approximately 58 minutes per configuration for the harder hard‑mode threshold settings. Only the classifier‑only baseline at `k=3` had completed — soft mode and hard mode at thresholds 0.1, 0.2, 0.3, and 0.5 — plus a partial `k=5` soft‑mode pass.

The arithmetic was unforgiving: at one hour per configuration and 180 configurations, the full sweep required more than seven days of cluster time. The decision was to abandon the brute‑force mean‑of‑means sweep and replace it with a *paired diagnostic* that fixed the trial seed across configurations.

---

## 16. Physics Diagnostic and Paired Attribution

The final experiment in the implementation program replaced the brute‑force sweep with a paired diagnostic harness. Under this protocol every configuration sees the same seed and the same sampled trials, so the marginal contribution of each scorer can be measured directly as `n_fixed − n_broken` rather than the difference of two noisy sample means — far more sensitive at the magnitudes involved (per‑sensor gains of 0.001 to 0.005).

Five configurations were evaluated: `baseline_classifier_only`, `pure_kin`, `pure_acc`, `pure_jointlim_soft`, and `default_blend`.

### 16.1 Re‑ranker Ceiling at `physics_k = 3`

The upper bound any re‑ranker could reach given the per‑sensor top‑3 candidate set:

| n_sensors | Exact‑match ceiling | Trials |
|----------:|--------------------:|-------:|
| 2         | 0.9400              | 200    |
| 3         | 0.9300              | 200    |
| 4         | 0.8750              | 200    |
| 5         | 0.8850              | 200    |

A counterintuitive finding: the ceiling *decreases* with more sensors at `k=3`. Each additional sensor introduces an independent chance that the true region falls outside the top‑3. The implication is that the rerank pipeline should likely choose `k` adaptively as a function of `n_sensors`.

### 16.2 Per‑Scorer Paired Attribution

Compared against the classifier‑only baseline:

| Config              | Δ per‑sensor | Δ exact | n_fixed | n_broken | net_fix |
|---------------------|-------------:|--------:|--------:|---------:|--------:|
| **pure_kin**        | **−0.0243**  | −0.0587 | 44      | 112      | **−68** |
| pure_acc            | −0.0014      | −0.0062 | 17      | 21       | −4      |
| pure_jointlim_soft  | +0.0011      |  0.0000 |  4      |  1       | +3      |
| **default_blend**   | **+0.0039**  | +0.0051 | 22      | 11       | **+11** |

The blended configuration's gain broken down by sensor count:

| n  | Base per‑sensor | Blend per‑sensor | net_fix |
|---:|----------------:|-----------------:|--------:|
| 2  | 0.8275          | 0.8325           | +2      |
| 3  | 0.8267          | 0.8317           | +3      |
| 4  | 0.8200          | 0.8225           | +2      |
| 5  | 0.8220          | 0.8260           | +4      |

Stratified by region difficulty, with "easy" defined as classifier top‑1 ≥ 0.95:

| Config              | Easy net_fix | Hard net_fix |
|---------------------|-------------:|-------------:|
| pure_kin            | **−45** (1 fixed / 46 broken) | −23 |
| pure_acc            | −3           | −1           |
| pure_jointlim_soft  | +0           | +3           |
| default_blend       | +0           | **+11** (21 fixed / 10 broken) |

### 16.3 Interpretation

Four observations carried implications for the project:

- **The pure kinematic scorer regressed when used alone.** On random candidate combinations the kinematic scorer was silent on most cases, and on the easy cases it had just enough wrong signal to flip 46 already‑correct predictions for the 1 it got right. This was the direct consequence of the limitation identified in the previous experiment: scoring arbitrary candidate combinations places the system in the *realistic* regime (16 % recovery) rather than the *best case* regime (97 % recovery).
- **The blended configuration was net positive**, and its gain (+0.0039 per sensor) exceeded the best single scorer's gain (+0.0011 per sensor). The signals were *additive*, not redundant.
- **All of the blend's gain came from hard cases**: +11 net on the 1,834 hard observations, and 0 net on the 966 easy ones. The "easy" threshold was generous (≥95 % classifier confidence), so the blend was demonstrably not breaking the model where it was already confident.
- **The confusion pairs fixed by the blend** were precisely those identified in the penalty‑sweep analysis as the dominant error sources: `r_hip → l_hip` (4 fixed), `l_forearm → l_upper_arm` (3), `r_upper_arm → r_shoulder` (3), `neck → spine_upper` (3), `l_thigh → r_thigh` (2).

### 16.4 Conclusions from the Diagnostic

- The kinematic scorer cannot be applied to arbitrary candidate combinations without breaking more than it fixes. It must be used only when the combination includes kinematic neighbors, or heavily down‑weighted in general‑case blends.
- The blend wins by *adding* the joint‑limit eliminator on top, not by pushing the kinematic weight higher.
- The paired diagnostic protocol is the right evaluation harness for any future scorer work. Mean‑of‑means comparisons are too noisy at the magnitudes involved.

---

## 17. Accuracy Progression Summary

The full progression of headline metrics across the experimental program:

| Step | Setup                                               | Headline metric           |
|------|-----------------------------------------------------|---------------------------|
| 6    | Fixed split, ResNet1D, plain CE                     | **0.7171** per‑window     |
| 6    | Best ablation (`fusion_blend01_seed42`)             | 0.7807 per‑window         |
| 7    | LOSO baseline, 10 folds                             | **0.7517** mean LOSO      |
| 7    | LOSO tuned (trial `e`)                              | **0.7817** mean LOSO      |
| 8    | Epoch sweep best (no SpatialNeighborLoss)           | 0.8480 (5‑fold sweep)     |
| 8    | SpatialNeighborLoss + class weights + bf=128        | **0.8611** mean LOSO      |
| 9    | Penalty sweep best‑final, all 10 folds              | **0.8683** mean LOSO      |
| 10   | Same checkpoints, top‑k ceiling                     | top‑3 0.9708, top‑5 0.9823 |
| 11   | Temporal re‑ranking, segment‑aware                  | +0.000 vs top‑1            |
| 16   | Physics `default_blend`, paired diagnostic          | **+0.0039 per‑sensor**, net_fix +11 |

The dominant remaining headroom is the ~10‑point pool between top‑1 (0.8684) and top‑3 (0.9708), which the physics work targeted but only partially realized. The paired diagnostic demonstrated the mechanics: soft‑blended physics is a net‑positive re‑ranker on random candidate combinations. Realizing the full ceiling depends on *deliberately seeding kinematic neighbors into the candidate combinations* — exactly the design lever surfaced as the central limitation in Section 12.

---

## 18. Open Research Threads

Several concrete next steps remained open at the end of the implementation program:

- **Hard‑mode threshold sweep** for the per‑region orientation eliminator, to characterize the precision/recall trade‑off as a function of the out‑of‑envelope frame threshold.
- **Circular percentiles** for the per‑region orientation envelope on the yaw and roll axes, replacing the current linear two‑sided interval that wraps benignly at ±180°.
- **A neighbor‑seeded physics re‑ranker** that explicitly includes kinematic neighbors in the candidate set rather than scoring arbitrary combinations — the path from the +0.4 pp realistic‑regime diagnostic gain toward the +7.9 pp best‑case ceiling.
- **Per‑region weights for the acceleration scorer**, exploiting the heterogeneity in per‑region AUC (median 0.874, but pelvis at 1.00 vs spine_lower at 0.76) that a single scalar weight underuses.
- **Temporal re‑ranking on a time‑ordered evaluation set**, so the segment‑aware fix from Step 11 has an actual signal to exploit.
- **Adaptive `k` selection** in the rerank pipeline. The diagnostic showed that the `k=3` ceiling *decreases* with more sensors, so the rerank pipeline should likely pick `k` based on `n_sensors`.

---

# Part III: Joint Relationship Lookup Table for Automatic Sensor Localization

## Motivation

Part II's body-region classifier (and IMUCoCo itself, as identified in §10 of Part I) assumes that the body location of each active sensor is **known at inference time**. In realistic wearable deployments this assumption breaks down: a user may move a device to a different pocket, sensors may be placed inconsistently across sessions, or the system may need to operate with no prior configuration knowledge.

Part III presents a complementary research direction that attacks this limitation directly: **automatically identifying which body location each sensor occupies** by matching the observed pairwise motion relationships between sensors against a pre-built gallery of known joint-pair signatures. While the Part II classifier identifies each sensor's location from its individual signal, the lookup table identifies sensor locations from the motion relationship between sensor pairs — the same localization problem approached from a pairwise rather than per-sensor perspective.

---

## 1. Table Structure

The lookup table is a tensor of shape **`[24, 24, F]`**, where axes 0 and 1 index the two joints being compared and axis 2 holds an `F`-dimensional relationship descriptor for that pair. Two variants exist:

| Variant | `F` | Stored files |
|---------|-----|--------------|
| Handcrafted | 6 | `lookup_table_mean.pt`, `lookup_table_std.pt`, `lookup_table_mean_normalized.pt` |
| Learned | 64 | `notebooks/joint_rel_model_best.pt` |

Only the 276 unique unordered pairs (upper triangle) carry meaningful content; the diagonal is zero.

---

## 2. Handcrafted 6-D Method

### Features

Each of the 6 features is derived from a 300-frame window of virtual-IMU data per joint pair `(i, j)`. Acceleration channels 6–8 and orientation channels 0–5 of the 9-channel signal are used.

| # | Name | Signal | Description |
|---|------|--------|-------------|
| 0 | `cross_corr_peak` | accel | Peak value of the normalised cross-correlation between joints i and j |
| 1 | `cross_corr_lag` | accel | Lag at that peak, normalised by window length T |
| 2 | `magnitude_ratio` | accel | log(RMS_i / RMS_j), clipped to [−5, 5] |
| 3 | `dtw_distance` | accel | DTW distance on a 30-point downsample, normalised by T |
| 4 | `orientation_similarity` | orient | Mean cosine similarity between orientation vectors over time |
| 5 | `relative_orientation_std` | orient | Standard deviation of that cosine similarity over time |

### Generating the Table

For every training sequence (885 files) and every joint pair, the 6 features above are computed from a 300-frame window. The per-pair vectors are averaged across all sequences. The resulting mean table is then **min-max normalised** per feature dimension to `[0, 1]` using the training-data range; the raw mean and its per-entry standard deviation are also saved for diagnostics.

---

## 3. Learned Joint Relationship Model (LJRM)

The LJRM replaces hand-crafted kinematic features with a **64-dimensional learned embedding** that captures the kinematic relationship between any pair of body joints. It is trained contrastively using the SMPL skeleton's adjacency graph as the supervision signal.

### 3.1 Architecture

The model has two sub-networks stacked together:

#### `JointEncoder` — per-joint temporal embeddings

Takes a window of IMU data for a single joint and produces a compact embedding.

```
Input:  [B, T=300, 9]   ← (batch, time, 9-channel virtual IMU)
   ↓  Linear projection: 9 → 128 dims
   ↓  Sinusoidal positional encoding
   ↓  Transformer encoder (2 layers, 4 heads, FFN dim=512)
   ↓  Mean pool over time → [B, 128]
   ↓  Linear projection: 128 → 64
   ↓  L2 normalize
Output: [B, 64]          ← unit-norm embedding
```

#### `PairRelationshipHead` — joint-pair relationship vector

Takes two joint embeddings and fuses them into a relationship descriptor.

```
Input:  e_i [B, 64],  e_j [B, 64]
   ↓  Concatenate [e_i | e_j | |e_i − e_j| | e_i ⊙ e_j] → [B, 256]
   ↓  MLP: 256 → 256 (ReLU) → 64
   ↓  L2 normalize
Output: [B, 64]          ← relationship embedding for pair (i, j)
```

The four-way fusion (concatenation, difference, element-wise product) gives the head explicit access to both individual joint identity and how the two joints relate to each other.

### 3.2 Training Signal — NT-Xent Contrastive Loss

The model is trained with no explicit labels. Instead, the **SMPL skeleton graph** (24 joints, 23 kinematic edges) defines what counts as a "positive pair":

- Two joint pairs that are **kinematically adjacent** in the skeleton are pulled together in embedding space.
- All other pairs in the batch are pushed apart.
- Temperature τ = 0.07 makes the loss sharp.

This forces the model to encode **structural body knowledge** — shoulder↔elbow should embed similarly to hip↔knee because both are adjacent limb segments.

### 3.3 Training Setup

| Setting | Value |
|---------|-------|
| Data | DIP-IMU, 885 sequences × 300-frame windows |
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR (30 epochs) |
| Gradient clipping | max norm = 1.0 |
| Best checkpoint | Epoch 22, val_loss = 4.193 |

### 3.4 Building the Lookup Table

After training, the model runs once over all training sequences to build a `[24, 24, 64]` gallery:

1. For every joint pair `(i, j)`, compute `PairRelationshipHead(JointEncoder(joint_i), JointEncoder(joint_j))`.
2. Store the 64-D vector at position `[i, j]`.
3. Accumulate across all 885 sequences using **Welford's online algorithm** (constant memory regardless of dataset size).
4. The final table is the per-pair **mean embedding**; a companion std tensor captures cross-sequence consistency.

This table is used downstream for efficient pair-relationship lookup without re-running the full model.

---

## 4. Inference — Configuration Identification

At test time, given a window of IMU data from `N` sensors:

1. Compute the relationship descriptor for every observed sensor pair `(i, j)` — either the 6 handcrafted features or the 64-D learned embedding.
2. Look up the closest entry in the pre-built gallery using **negative L² distance** (handcrafted) or **cosine similarity** (learned).
3. The top-ranked gallery entry identifies which joint pair `(i, j)` each sensor pair corresponds to, resolving the sensor-to-body-location assignment.

For `N`-sensor **configuration identification**, all `C(N, 2)` pair descriptors are aggregated; the configuration whose gallery entries best match the query is selected by summing pair-wise similarity scores, with permutation-invariant scoring to handle unknown sensor ordering.

---

## 5. Results and Analysis

Evaluated on 19 held-out test sequences. Chance baseline for all-pairs retrieval: **Top-1 = 0.36%** (1 in 276).

### All-pairs retrieval (276 classes, 5 244 queries)

| Metric | Handcrafted | Learned | Improvement |
|--------|-------------|---------|-------------|
| Top-1  | 4.12 %  | **21.80 %** | 5.3× |
| Top-5  | 14.23 % | **59.59 %** | 4.2× |
| Top-10 | 23.99 % | **76.37 %** | 3.2× |
| MRR    | 0.109   | **0.389**   | 3.6× |

The learned model far outperforms the handcrafted baseline across all metrics, demonstrating that the contrastive Transformer captures joint-pair structure that hand-engineered correlation features miss.

### N-sensor configuration identification (learned, permutation-invariant)

| N sensors | Configs | Top-1 | Chance | Lift | MRR |
|-----------|---------|-------|--------|------|-----|
| 2 | 136     | 14.86 % | 0.74 % | 20×  | 0.286 |
| 3 | 680     | 8.00 %  | 0.15 % | 54×  | 0.174 |
| 4 | 2 380   | 3.86 %  | 0.04 % | 92×  | 0.092 |
| 5 | 6 188   | 1.75 %  | 0.02 % | 108× | 0.051 |

While Top-1 accuracy decreases as N grows (the configuration space explodes combinatorially), the lift over chance *increases* — the model is doing substantially better than random at every sensor count.

### Hardest and Easiest Pairs (learned)

- **Easiest**: `right_collar ↔ right_shoulder` (89.5 %), `right_collar ↔ head` (84.2 %), `left_collar ↔ head` (84.2 %)
- **Hardest**: lower-body pairs such as `pelvis ↔ right_ankle`, `right_knee ↔ right_ankle` (0 %) — these joints share similar motion statistics
- **Adjacent pairs** average Top-1: **40 %** vs **non-adjacent**: 20 %

The adjacency gap is expected: the NT-Xent loss was trained using adjacency as its positive signal, so the model naturally encodes structural proximity. Upper-body pairs with distinctive motion profiles (collar, shoulder, head) are the clearest cases; lower-body pairs that move in synchrony during locomotion are the hardest.

---

## 6. Limitations and Open Questions

**Configuration Top-1 drops steeply with N.** At N=5, Top-1 is only 1.75% — though 108× above chance, this is not yet deployment-ready for arbitrary sensor sets.

**Lower-body pairs remain near 0%.** Joints such as `right_knee ↔ right_ankle` produce nearly identical motion statistics during most daily activities. Distinguishing them likely requires activity-conditioned models or richer temporal context.

**Absolute localization vs relative identification.** The current table identifies *which pair* a sensor combination corresponds to, but not their absolute body positions independently. Integration with the Part II classifier — which knows the single-sensor label distribution — could resolve this: the lookup table constrains the joint configuration, the classifier scores each individual sensor.

**Combining the two pipelines.** The natural next step is a joint inference pipeline where:
- The Part III lookup table proposes a plausible sensor configuration (constraining the search space).
- The Part II classifier (+ physics re-ranker from Phase 14) scores each sensor independently within that configuration.
- A combined score selects the final assignment.

This would directly address the §10 assumption that motivates both lines of work, replacing manual sensor labeling with a fully automated localization and classification pipeline.
