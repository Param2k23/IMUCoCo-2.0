# Sensor Location Identification — IMUCoCo / SMPL

Classifies which of the **24 IMUCoCo body regions** a wearable IMU is placed on, using 9-channel virtual IMU features: **6D orientation (r6d) + 3-axis acceleration**.

```
c:\VS\SensorLoc\
├── smpl_regions.py       ← 24-region vertex map, centroids, spatial-error helper
├── preprocess_vimu.py    ← Convert DIP-style vimu .pt segments to .npz
├── model.py              ← ResNet-1D + CNN-1D (PyTorch)
├── train.py              ← LOSO and fixed-split training loop
├── evaluate.py           ← Confusion matrix, spatial error, majority-vote filter
└── data/                 ← output dataset goes here
```

---

## Requirements

```
pip install torch scipy scikit-learn smplx seaborn matplotlib numpy
```

---

## Paths (already configured for this machine)

| Resource | Path |
|---|---|
| AMASS root | `C:/VS/TransPose/data/dataset_raw/AMASS` |
| SMPL model | `C:/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` |
| Dataset output | `C:/VS/SensorLoc/data/dataset.npz` |
| Checkpoints | `C:/VS/SensorLoc/checkpoints/` |
| Results | `C:/VS/SensorLoc/results/` |

---

## Quick Start (vimu r6d+acc)

```powershell
cd C:\VS\SensorLoc

# 1) Convert single-subject experiment dataset (ignores source train/test split)
python preprocess_vimu.py \
  --mode single_subject \
  --train_dir data/DIP_IMU_train_real_imu_position_only \
  --test_dir data/DIP_IMU_test_real_imu_position_only \
  --train_csv data/DIP_IMU_train_real_imu_position_only.csv \
  --subject_id 1 \
  --out_train data/processed/single_subject_train.npz \
  --out_test data/processed/single_subject_test.npz

# 2) Train fixed split
python train.py \
  --mode fixed_split \
  --train_data data/processed/single_subject_train.npz \
  --val_data data/processed/single_subject_test.npz \
  --out_dir checkpoints/single_subject_full \
  --arch resnet \
  --epochs 80 \
  --device cuda

# 3) Evaluate
python evaluate.py \
  --checkpoint checkpoints/single_subject_full/best_model.pt \
  --data data/processed/single_subject_test.npz \
  --out_dir results/single_subject_full
```

---

## Full Predefined-Split Run

```powershell
# Step 1 — Preprocess predefined train/test split
python preprocess_vimu.py \
  --mode predefined_split \
  --train_dir data/DIP_IMU_train_real_imu_position_only \
  --test_dir data/DIP_IMU_test_real_imu_position_only \
  --train_csv data/DIP_IMU_train_real_imu_position_only.csv \
  --out_train data/processed/full_train.npz \
  --out_test data/processed/full_test.npz

# Step 2 — Train fixed split
python train.py \
  --mode fixed_split \
  --train_data data/processed/full_train.npz \
  --val_data data/processed/full_test.npz \
  --out_dir checkpoints/full_split_full \
  --arch resnet \
  --epochs 80 \
  --device cuda

# Step 3 — Evaluate
python evaluate.py \
  --checkpoint checkpoints/full_split_full/best_model.pt \
  --data data/processed/full_test.npz \
  --out_dir results/full_split_full \
  --vote_k 5
```

---

## Model Architecture (ResNet-1D)

```
Input      (B, 9, T)          — 6D orientation (r6d) + 3-axis acceleration
Stem       Conv1d(9→64, k=7)  + BN + ReLU
Layer 1    ResBlock(64→64,  stride=1)
Layer 2    ResBlock(64→128, stride=2)
Layer 3    ResBlock(128→256, stride=2)
Layer 4    ResBlock(256→256, stride=1)
GAP        AdaptiveAvgPool1d(1)
Dropout    p=0.3
FC         Linear(256, 24)
Output     (B, 24) logits
```

---

## Outputs

| File | Description |
|---|---|
| `checkpoints/best_model_fold{k}.pt` | Best **weights-only** checkpoint per LOSO fold |
| `checkpoints/normalization_stats_fold{k}.pt` | Per-fold normalization stats (mean/std) + metadata |
| `checkpoints/best_model.pt` | Best **weights-only** checkpoint for fixed-split mode |
| `checkpoints/normalization_stats.pt` | Normalization stats for fixed-split mode |
| `checkpoints/loso_results.txt` | Per-fold and mean accuracy |
| `results/confusion_matrix.png` | Normalised 24×24 confusion matrix |
| `results/eval_summary.json` | Accuracy, spatial error, symmetry confusion |

---

## Key Design Choices

- **Orientation Invariance**: Random SO(3) rotation applied per sequence at preprocessing time, so the model never sees a fixed orientation.
- **Spatial Error**: When a prediction is wrong, reports Euclidean distance (metres) between the centroids of the predicted and true body regions on the SMPL T-pose mesh.
- **Symmetry Confusion**: Flags left/right pairs (e.g. l_thigh ↔ r_thigh) that are commonly confused due to near-identical motion signatures.
- **Majority Vote**: Locks a prediction only if ≥ ⌊k/2⌋+1 of the last k windows agree (default k=5).
