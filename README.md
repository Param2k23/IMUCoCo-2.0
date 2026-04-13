# Sensor Location Identification — IMUCoCo / SMPL

Classifies which of the **24 IMUCoCo body regions** a wearable IMU is placed on, using only raw 3-axis accelerometer + 3-axis gyroscope data.

```
c:\VS\SensorLoc\
├── smpl_regions.py       ← 24-region vertex map, centroids, spatial-error helper
├── preprocess_amass.py   ← Extract labelled IMU windows from AMASS .npz files
├── model.py              ← ResNet-1D + CNN-1D (PyTorch)
├── train.py              ← LOSO cross-validation training loop
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

## Quick Start (Smoke Test — no AMASS/SMPL needed)

```powershell
cd C:\VS\SensorLoc

# 1. Generate synthetic dataset (480 windows, 20 per region)
python preprocess_amass.py --smoke_test

# 2. Train one LOSO fold (5 epochs)
python train.py --data data/smoke_dataset.npz --smoke_test --epochs 5

# 3. Evaluate
python evaluate.py \
  --checkpoint checkpoints/best_model_fold0.pt \
  --data data/smoke_dataset.npz
```

---

## Full AMASS Run

```powershell
# Step 1 — Preprocess  (processes up to 5 sequences per AMASS sub-dataset)
python preprocess_amass.py `
  --amass_root C:/VS/TransPose/data/dataset_raw/AMASS `
  --smpl_model C:/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl `
  --output     C:/VS/SensorLoc/data/dataset.npz `
  --max_seqs   5

# Step 2 — Train (LOSO, 50 epochs, ResNet-1D)
python train.py `
  --data    C:/VS/SensorLoc/data/dataset.npz `
  --arch    resnet `
  --epochs  50

# Step 3 — Evaluate best fold
python evaluate.py `
  --checkpoint C:/VS/SensorLoc/checkpoints/best_model_fold0.pt `
  --data       C:/VS/SensorLoc/data/dataset.npz `
  --vote_k     5
```

---

## Model Architecture (ResNet-1D)

```
Input      (B, 6, 120)        — 3-axis Accel + 3-axis Gyro
Stem       Conv1d(6→64, k=7)  + BN + ReLU
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
| `checkpoints/best_model_fold{k}.pt` | Best checkpoint per LOSO fold |
| `checkpoints/loso_results.txt` | Per-fold and mean accuracy |
| `results/confusion_matrix.png` | Normalised 24×24 confusion matrix |
| `results/eval_summary.json` | Accuracy, spatial error, symmetry confusion |

---

## Key Design Choices

- **Orientation Invariance**: Random SO(3) rotation applied per sequence at preprocessing time, so the model never sees a fixed orientation.
- **Spatial Error**: When a prediction is wrong, reports Euclidean distance (metres) between the centroids of the predicted and true body regions on the SMPL T-pose mesh.
- **Symmetry Confusion**: Flags left/right pairs (e.g. l_thigh ↔ r_thigh) that are commonly confused due to near-identical motion signatures.
- **Majority Vote**: Locks a prediction only if ≥ ⌊k/2⌋+1 of the last k windows agree (default k=5).
