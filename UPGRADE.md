# UPGRADE.md

This document captures the current pipeline state, why it is vulnerable, and a concrete upgrade path for IMUCoCo with DIP-style `vimu` data using 9 channels (`r6d_0..r6d_5, ax, ay, az`).

## Baseline-first scope (non-future)

The first baseline training run should include only:
- Data conversion from `.pt` to `.npz` using `vimu.vimu_joints` with 9 channels (`:9`) and 24-class labels.
- Two data pipelines:
  - single-subject experiment (ignore source split and create a train/test split from one subject)
  - full-dataset experiment (respect predefined train/test folders)
- Training pipeline updates required by current decisions:
  - CUDA policy: fallback to CPU only when `--smoke_test`; otherwise error if CUDA is unavailable
  - train/eval compatibility for converted vimu-based `.npz` datasets

Everything else below marked `[FUTURE]` is not required for the first baseline run.

## 1) Current state (from code) and what it implies

### 1.1 Current training paradigm
- `train.py` trains with leave-one-subject-out (LOSO) CV only.
- For each fold:
  - split by `subject_ids` (one subject held out for val)
  - z-score normalize with train-only stats (`mean/std` over `(N,T)` per channel)
  - train either `resnet` (default) or `cnn`
  - optimize validation loss with early stopping (`patience`)
  - save best checkpoint by minimum val loss

### 1.2 Current model architecture
- `model.py` default (`resnet`) = 1D ResNet classifier:
  - input `(B, 9, T)`
  - stem `Conv1d(9->64, k=7) + BN + ReLU`
  - residual blocks:
    - block1 `64->64` stride 1
    - block2 `64->128` stride 2
    - block3 `128->256` stride 2
    - block4 `256->256` stride 1
  - global average pool + dropout(0.3) + FC(256->24)
- baseline `cnn` = 3 conv blocks with maxpool, GAP, FC.

### 1.3 Current loss metric and objective
- Optimization loss: `CrossEntropyLoss` (multiclass CE).
- Model-selection metric in training: **minimum validation CE loss**.
- Reported metric in training summary: per-fold **validation accuracy**.

### 1.4 Current evaluation paradigm
- `evaluate.py` performs per-window inference on a `.npz` dataset.
- Metrics and outputs:
  - per-window accuracy
  - optional majority-vote stream accuracy (`vote_k`)
  - confusion matrix image
  - symmetry confusion analysis (left/right region pairs)
  - spatial error (`smpl_regions.spatial_error`) on wrong predictions
  - sklearn classification report
  - JSON summary artifact

### 1.5 Why current setup is susceptible to improvement
- No data augmentation in training -> overfitting risk on subject-specific signatures.
- Loss is plain CE with no imbalance handling -> minority regions can underperform.
- Fixed architecture has no channel-attention despite strong axis/channel structure in IMU.
- No top-k/per-class spatial error trend tracking in eval -> harder to diagnose where gains come from.
- Checkpoints do not yet capture full preprocessing config for reproducible external inference.

---

## 2) Data augmentations / transformations (detailed)

Assume input is `(C=9, T)` per sample from `vimu_joints[:, region, :9]`.

For each method below:
- **Current susceptibility**: why current code needs it
- **Implementation**: exact practical recipe
- **Expected improvement**: realistic range and where
- **Why it should work here**: data characteristic
- **Test to verify it worked**: concrete checks

### 2.1 [FUTURE] Channel jitter (additive Gaussian noise per channel)
- Current susceptibility:
  - No sensor noise modeling in training; model may key off clean synthetic artifacts.
- Implementation:
  - During training only, for each sample/channel:
    - `x[c] = x[c] + eps`, `eps ~ N(0, sigma_c^2)`
  - Start with:
    - r6d channels `sigma=0.01 * std_r6d_channel`
    - accel channels `sigma=0.03 * std_acc_channel`
  - Keep deterministic by seeding torch RNG.
- Expected improvement:
  - +0.5 to +2.0 points top-1 on held-out subjects (larger if overfitting baseline).
- Why it should work here:
  - IMU signals contain hardware noise, calibration drift, and quantization; additive noise improves robustness to these perturbations.
- Test to verify:
  1. Train baseline vs jitter-only with same seed and split.
  2. Compare val/test top-1, macro-F1, and spatial error mean.
  3. Add robustness probe: evaluate with synthetic noise injected at test-time (`sigma_test` sweep 0 to 0.05 std); jitter model should degrade slower.

### 2.2 [FUTURE] Time masking (SpecAugment-style time dropout)
- Current susceptibility:
  - Model may over-rely on short discriminative bursts; real signals have packet drops / transient corruption.
- Implementation:
  - Training only, sample `m` masks (e.g., 1 to 2).
  - For each mask:
    - choose width `w ~ Uniform(0, W_max)` where `W_max = 0.1*T`
    - choose start `t0`
    - set `x[:, t0:t0+w] = 0` or channel mean.
- Expected improvement:
  - +0.5 to +1.5 points top-1; bigger stability for majority-vote metric.
- Why it should work here:
  - IMU streams are temporal and often partially missing/noisy; forcing temporal redundancy improves continuity-based representation.
- Test to verify:
  1. Baseline vs masking-only.
  2. Evaluate on artificially masked test copies (5%, 10%, 20% drop windows).
  3. Check majority-vote locked fraction and voted accuracy improve under dropouts.

### 2.3 [FUTURE] Small time-warp (mild temporal distortion)
- Current susceptibility:
  - Motion speed differs across subjects/actions; model may overfit absolute tempo.
- Implementation:
  - Training only.
  - Choose warp factor `r ~ Uniform(0.9, 1.1)`.
  - Resample signal to `round(T*r)` by interpolation, then crop/pad back to `T`.
  - Keep warps small to preserve label semantics.
- Expected improvement:
  - +0.3 to +1.2 top-1, often more macro-F1 gain on dynamic classes.
- Why it should work here:
  - Same body location can produce similar motion patterns at different speeds (subject gait cadence, action tempo).
- Test to verify:
  1. Baseline vs warp-only.
  2. Test-time speed perturbation benchmark (`r_test = 0.85, 0.9, 1.1, 1.15`).
  3. Improved model should retain higher accuracy under speed perturbations.

### 2.4 [FUTURE] Sensor-axis scale + bias perturbation (calibration simulation)
- Current susceptibility:
  - No calibration perturbation in current training; deployment sensors have scale/bias errors.
- Implementation:
  - Training only, per sample/channel:
    - `x[c] = a_c * x[c] + b_c`
    - `a_c ~ Uniform(0.97, 1.03)`
    - `b_c ~ Uniform(-0.02*std_c, 0.02*std_c)`
  - Use different ranges for r6d/acc if needed.
- Expected improvement:
  - +0.5 to +2.5 top-1 in cross-device transfer; may be neutral on in-domain clean test.
- Why it should work here:
  - IMU data is extremely sensitive to calibration and placement; this directly models those nuisance factors.
- Test to verify:
  1. Calibration-stress evaluation: scale/bias perturb test set.
  2. Compare absolute drop from clean to stressed test; upgraded model should drop less.

### 2.5 [FUTURE] Optional transform: mixup (sequence-level)
- Current susceptibility:
  - Hard decision boundaries between nearby/symmetric regions can overfit.
- Implementation:
  - Sample pair `(x_i,y_i),(x_j,y_j)` and `lam~Beta(alpha,alpha)`, `alpha=0.2`.
  - `x = lam*x_i + (1-lam)*x_j`
  - loss: `lam*CE(logits,y_i)+(1-lam)*CE(logits,y_j)`.
- Expected improvement:
  - +0.2 to +1.0 top-1; often better calibration (ECE) and robustness.
- Why it should work here:
  - Regions with similar kinematics (especially symmetry pairs) benefit from smoother class boundaries.
- Test to verify:
  1. Compare expected calibration error (ECE) and NLL besides accuracy.
  2. Confusion in symmetry pairs should reduce modestly.

### [FUTURE] Augmentation rollout recommendation
- Start with `channel jitter + time masking` (highest ROI / lowest risk).
- Add `scale+bias` next for robustness.
- Add time-warp last (easy to overdo; keep warps small).

---

## 3) Model upgrades (detailed)

### 3.1 [FUTURE] Add channel attention (SE blocks) to ResNet1D
- Current susceptibility:
  - Current residual blocks treat channels uniformly after BN+conv; no explicit channel reweighting.
- Implementation:
  - In each residual block output `u`:
    - squeeze: `s = GAP(u)` -> `(B,C)`
    - excitation: `z = sigmoid(W2(relu(W1(s))))`
    - scale: `u_se = u * z.unsqueeze(-1)`
  - Use reduction ratio `r=16`.
- Expected improvement:
  - +0.5 to +1.8 top-1 with minimal compute increase.
- Why it should work here:
  - IMU channels have unequal informativeness by class/action/time; adaptive reweighting is beneficial.
- Test to verify:
  1. A/B compare ResNet vs ResNet+SE with equal training settings.
  2. Track per-class gains and spatial error reduction.

### 3.2 [FUTURE] Temporal dilations in deeper blocks
- Current susceptibility:
  - Current receptive field may under-capture longer temporal dependencies for `T=300` segments.
- Implementation:
  - Keep first blocks undilated.
  - Use dilations in block3/block4 (e.g., `d=2,4` with adjusted padding).
- Expected improvement:
  - +0.3 to +1.5 top-1 if long-range structure matters.
- Why it should work here:
  - Body-location signatures include both short transients and broader periodic motion context.
- Test to verify:
  1. Compare on long-segment validation.
  2. Analyze gains by motion tempo buckets (slow vs fast sequences).

### 3.3 [FUTURE] Multi-branch r6d/acc stem (optional)
- Current susceptibility:
  - Orientation and acceleration statistics differ; shared first conv may be suboptimal.
- Implementation:
  - Split input into r6d(6ch) and acc(3ch) stems, then concat and continue residual trunk.
- Expected improvement:
  - +0.2 to +1.0 top-1, sometimes better stability.
- Why it should work here:
  - Orientation and acceleration channels have different semantics and dynamics, so early specialization can help.
- Test to verify:
  1. Compare branch vs shared-stem using same params budget.
  2. Check improvement is consistent across seeds (>=3).

---

## 4) Training upgrades (detailed)

### 4.1 Device policy hardening
- Current susceptibility:
  - Current `auto` can silently use CPU in non-smoke runs, causing unexpected long runs.
- Implementation:
  - If CUDA unavailable and `--smoke_test` set -> warn + use CPU.
  - If CUDA unavailable and not smoke -> raise error and exit.
- Expected improvement:
  - No direct accuracy gain; major reliability and experiment consistency gain.
- Why it should work here:
  - Prevent accidental full-length CPU runs and inconsistent runtime conditions.
- Test to verify:
  1. Run without CUDA and with/without `--smoke_test`.
  2. Confirm fallback/error behavior matches spec.

### 4.2 [FUTURE] Imbalance-aware loss: class-weighted CE (first) or focal loss (second)
- Current susceptibility:
  - 24 classes can be imbalanced by subject/action sampling.
- Implementation:
  - Weighted CE:
    - compute class counts on training fold only
    - weights `w_k = (count_k + eps)^(-gamma)` with `gamma~0.5`
    - normalize weights to mean 1.
  - Focal loss option (`gamma=1.5 to 2.0`) if hard-negative focus is needed.
- Expected improvement:
  - +0.5 to +2.0 macro-F1; top-1 may rise modestly.
- Why it should work here:
  - Minority regions otherwise contribute less to gradient updates.
- Test to verify:
  1. Compare macro-F1 and worst-5-class accuracy.
  2. Check confusion rows of minority classes improve.

### 4.3 [FUTURE] AdamW + warmup + cosine schedule
- Current susceptibility:
  - Current Adam + cosine works, but decoupled weight decay typically generalizes better.
- Implementation:
  - Optimizer: `AdamW(lr=3e-4, weight_decay=1e-2)`.
  - Warmup first 5 epochs from `lr/10` to `lr`.
  - Cosine decay to `lr_min = 1e-6`.
- Expected improvement:
  - +0.3 to +1.5 top-1 and smoother convergence.
- Why it should work here:
  - Small/mid-size IMU datasets are regularization-sensitive; decoupled decay helps.
- Test to verify:
  1. Compare validation-loss curves (stability and best epoch).
  2. Confirm same/better generalization at matched train loss.

### 4.4 [FUTURE] EMA of model weights
- Current susceptibility:
  - Final checkpoint may reflect noisy local minima.
- Implementation:
  - Maintain EMA shadow params:
    - `ema = decay*ema + (1-decay)*theta`, `decay=0.999`
  - Evaluate/save EMA weights.
- Expected improvement:
  - +0.2 to +1.0 top-1, usually reduced variance across seeds.
- Why it should work here:
  - IMU datasets with noisy batches benefit from temporal parameter averaging.
- Test to verify:
  1. Report non-EMA vs EMA metrics at each checkpoint.
  2. EMA should improve average and reduce std across 3 seeds.

### 4.5 [FUTURE] Reproducibility controls
- Current susceptibility:
  - No explicit seed/config checkpointing; hard to compare ablations reliably.
- Implementation:
  - Add `--seed`, set Python/Numpy/Torch seeds.
  - Optional deterministic flags.
  - Save full run config + normalization stats + channel schema in checkpoint.
- Expected improvement:
  - No direct accuracy gain; high confidence in comparison quality.
- Why it should work here:
  - You are planning many ablations; uncontrolled randomness can hide true effects.
- Test to verify:
  1. Repeat same run 3 times with fixed seed -> near-identical metrics.

---

## 5) Evaluation upgrades (detailed)

### 5.1 [FUTURE] Add top-k accuracy (k=3,5)
- Current susceptibility:
  - Current top-1 alone hides near-miss behavior among anatomically nearby regions.
- Implementation:
  - Use softmax logits and compute top-k hits.
  - Save in summary JSON.
- Expected improvement:
  - Not a training improvement directly; better observability and model-selection quality.
- Why it should work here:
  - Adjacent/symmetric regions are semantically close; top-k captures ranking quality.
- Test to verify:
  1. Ensure `top5 >= top3 >= top1` always.

### 5.2 [FUTURE] Per-class spatial error + symmetry trend tracking
- Current susceptibility:
  - Current spatial error is global only; hard to see where geometry improves.
- Implementation:
  - Compute per-true-class mean/std spatial error.
  - Track symmetry confusion aggregate over runs in CSV/JSON.
- Expected improvement:
  - Better diagnosis and targeted improvements; faster iteration.
- Why it should work here:
  - Task has explicit geometry and known symmetry failure modes.
- Test to verify:
  1. Compare run-to-run reduction in worst symmetry pair confusion.

### 5.3 [FUTURE] Calibration metrics (ECE, NLL)
- Current susceptibility:
  - Majority vote and downstream use depend on confidence quality, not just accuracy.
- Implementation:
  - Compute ECE (10-15 bins) and NLL from predicted probabilities.
- Expected improvement:
  - Better confidence reliability; supports thresholding and abstention later.
- Why it should work here:
  - Near-class confusions are common; calibrated probabilities improve decision policies.
- Test to verify:
  1. Reliability diagram and ECE reduction after label smoothing/mixup.

---

## 6) [FUTURE] V1 improved baseline config (exact hyperparameters)

This is a pragmatic first upgrade with low engineering risk.

### Data
- Input source: `vimu.vimu_joints` only.
- Channels: all 9 dims (`r6d_0..r6d_5, ax, ay, az`).
- Sample construction: one sample per `(segment, region)`.
- Shape: `X (N,9,T)`, `y (N,)`, `subject_ids (N,)`.

### Train
- Arch: `resnet` (current) + optional SE blocks if implemented.
- Batch size: `128` (CUDA), `32` for smoke/CPU fallback.
- Epochs: `80` (full), `5` smoke.
- Optimizer: `AdamW`.
- LR: `3e-4` base.
- Weight decay: `1e-2`.
- LR schedule: warmup 5 epochs -> cosine to `1e-6`.
- Early stopping patience: `15`.
- Dropout: `0.3`.
- Label smoothing: `0.05`.
- Gradient clipping: `1.0`.
- EMA decay: `0.999`.
- Seed: `42`.

### Augmentations (train only)
- Channel jitter: r6d `0.01*std`, accel `0.03*std`.
- Time masking: 1 to 2 masks, max width `0.1*T`.
- Scale+bias perturb: scale `[0.97,1.03]`, bias `+-0.02*std`.
- Time-warp: disabled initially; enable at `r in [0.9,1.1]` only if stable.

### Why this config
- Strong but safe regularization for IMU domain shift.
- AdamW + warmup/cosine is robust across many medium-sized sensor tasks.
- Keeps architecture close to existing code to preserve debugging velocity.

### Expected net effect vs current baseline
- Top-1 accuracy: typically +1 to +4 points (dataset-dependent).
- Macro-F1: typically +2 to +6 points if imbalance exists.
- Spatial error mean: modest decrease when confusion shifts to nearby classes.
- Lower run-to-run variance due to EMA + seed discipline.

---

## 7) Validation and test plan (what to run)

### 7.1 Required experiment table
- A0: current baseline (no aug, CE)
- A1: A0 + jitter
- A2: A1 + time masking
- A3: A2 + scale+bias
- A4: A3 + AdamW/warmup/cosine
- A5: A4 + weighted CE
- A6: A5 + EMA

Track for each:
- top-1, top-3, top-5
- macro-F1
- spatial error mean/std
- symmetry confusion top pairs
- ECE, NLL

### 7.2 Success criteria
- V1 accepted if:
  - top-1 improves >= 1.0 point on primary split
  - macro-F1 improves >= 2.0 points OR worst-5-class avg improves >= 3.0 points
  - no regression > 0.5 point on smoke sanity runs

### 7.3 What was executed now
- Attempted smoke test in this environment:
  - `python3 evaluate.py --smoke`
- Result:
  - failed due to missing dependency: `ModuleNotFoundError: No module named 'numpy'`.
- Therefore, runtime validation is specified above but could not be executed in this session.

---

## 8) References (verified lookup)

Core model/training references:
- He et al., "Deep Residual Learning for Image Recognition", arXiv:1512.03385, 2015. DOI: 10.48550/arXiv.1512.03385
- Hu et al., "Squeeze-and-Excitation Networks", arXiv:1709.01507, 2017. DOI: 10.48550/arXiv.1709.01507
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW), arXiv:1711.05101, 2017/2019. DOI: 10.48550/arXiv.1711.05101
- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", arXiv:1608.03983, 2016/2017. DOI: 10.48550/arXiv.1608.03983

Augmentation / regularization references:
- DeVries & Taylor, "Improved Regularization ... with Cutout", arXiv:1708.04552, 2017. DOI: 10.48550/arXiv.1708.04552
- Park et al., "SpecAugment", arXiv:1904.08779, 2019. DOI: 10.48550/arXiv.1904.08779
- Zhang et al., "mixup: Beyond Empirical Risk Minimization", arXiv:1710.09412, 2017/2018. DOI: 10.48550/arXiv.1710.09412
- Lin et al., "Focal Loss for Dense Object Detection", arXiv:1708.02002, 2017/2018. DOI: 10.48550/arXiv.1708.02002

Time-series background:
- Fawaz et al., "Deep learning for time series classification: a review", arXiv:1809.04356, 2018/2019. DOI: 10.48550/arXiv.1809.04356
