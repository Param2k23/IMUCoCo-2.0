# Sensor Location Project - Technical Notes

## 1) Approach

Goal: classify IMU placement into 24 body regions (labels `0..23`) from inertial time windows.

Pipeline used:
1. `preprocess_vimu.py` converts parquet to model-ready `.npz`
2. `train.py` trains classifier (`resnet` used in reported runs)
3. `evaluate.py` reports accuracy, confusion, symmetry confusion, and spatial error

Data source type:
- Hugging Face parquet shards (`train-*.parquet`, `test-*.parquet`)
- Subject IDs explicitly parsed with `SUBJECT_COLUMN=subject_id`

---

## 2) Dataset Format and Meaning of `N`

Training/evaluation files are `.npz` with:
- `X`: `(N, 9, T)` float32
- `y`: `(N,)` int64
- `subject_ids`: `(N,)` int64

Meaning:
- `N` = total number of windows (samples)
- `9` = channels per window: `r6d_0..r6d_5, ax, ay, az`
- `T` = time steps per window (`T=300` in current runs)

Each sample is one window:
- input: `X[i]` with shape `(9,300)`
- label: `y[i]` in `[0,23]`
- subject: `subject_ids[i]`

---

## 3) What Data Looked Like in This Run

From preprocessing and training logs:

- Fixed-split train:
  - `X=(21240, 9, 300)`, classes=24, subjects=8
- Fixed-split test:
  - `X=(456, 9, 300)`, classes=24, subjects=2

- LOSO merged dataset:
  - `X=(21696, 9, 300)`
  - `y=(21696,)`, classes=24
  - `subject_ids=(21696,)`, subjects=10, subject range `1..10`

---

## 4) Methods Used

Model:
- `ResNet1D` from `model.py`
- Conv stem -> residual blocks -> global average pooling -> dropout -> FC(24)

Training setup (`train.py`):
- Loss: CrossEntropy
- Optimizer: Adam
- LR schedule: CosineAnnealingLR
- Early stopping (patience)
- Normalization:
  - mean/std computed from training split only
  - saved to `normalization_stats*.pt`
  - reused for evaluation

Evaluation (`evaluate.py`):
- Per-window accuracy
- Majority vote accuracy (`vote_k=5`)
- Confusion matrix
- Symmetry confusion (left/right pairs)
- Spatial error (meters on wrong predictions)

---

## 5) Results (Printed)

## Fixed-split evaluation (`logs/eval_hf_full_split.log`)

```text
Per-window accuracy: 0.7171
Majority-vote accuracy: 0.2857  (locked 4.6% of windows)
N wrong: 129 / 456 (28.3%)
Mean spatial error: 0.4064 m
Std spatial error: 0.3131 m
```

## LOSO results (`checkpoints/loso_full/loso_results.txt`)

```text
Fold  Subject  Val_Acc
-----------------------------------
1         1    0.7342
2         2    0.7241
3         3    0.7507
4         4    0.7512
5         5    0.8018
6         6    0.7061
7         7    0.8425
8         8    0.7877
9         9    0.7315
10       10    0.6875
-----------------------------------
Mean LOSO Acc: 0.7517
```

LOSO summary statistics:
- Mean: `0.7517`
- Std: `0.0444`
- Min: `0.6875`
- Max: `0.8425`

---

## 6) Direct Interpretation

- Fixed-split baseline: `71.71%` per-window accuracy.
- LOSO mean: `75.17%` across 10 unseen-subject folds.
- Subject variability exists (range `15.5` points from worst to best fold).
- Majority vote (`k=5`) is currently not useful in this setup because lock rate is low (`4.6%`).

---

## 7) Future Work

1. Add per-fold LOSO evaluation artifacts (confusion + spatial error for each fold).
2. Improve temporal smoothing (smaller `k`, confidence-based smoothing, or segment-aware voting).
3. Address hard classes and subject shift via targeted augmentation/loss tuning.
