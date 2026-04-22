#!/bin/bash
#SBATCH --job-name=imu_hf_parquet_full
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# Optional: uncomment/adjust for your cluster
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p logs checkpoints/single_subject_full checkpoints/full_split_full results/single_subject_full results/full_split_full data/processed

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[slurm] repo: $ROOT_DIR"
echo "[slurm] python: $PYTHON_BIN"
echo "[slurm] started: $(date)"

PARQUET_DIR="${HF_PARQUET_DIR:-data/raw_dip/data}"
MERGE_MODE="${MERGE_INPUTS:-vimu_only}"
IMU_BLEND="${IMU_BLEND:-0.0}"

echo "[slurm] parquet_dir: $PARQUET_DIR"
echo "[slurm] merge_inputs: $MERGE_MODE (imu_blend=$IMU_BLEND)"

echo "[slurm] hf_parquet full-split: preprocess"
"$PYTHON_BIN" preprocess_vimu.py \
  --mode hf_parquet \
  --parquet_dir "$PARQUET_DIR" \
  --train_glob "train-*.parquet" \
  --test_glob "test-*.parquet" \
  --hf_split predefined \
  --window_length 300 \
  --crop_mode center \
  --pad_mode edge \
  --merge_inputs "$MERGE_MODE" \
  --imu_blend "$IMU_BLEND" \
  --out_train data/processed/full_train.npz \
  --out_test data/processed/full_test.npz \
  2>&1 | tee logs/preprocess_hf_full_split.log

echo "[slurm] hf_parquet full-split: sanity-check"
"$PYTHON_BIN" - <<'PY'
import numpy as np
d = np.load("data/processed/full_train.npz")
print("full_train X shape:", d["X"].shape)
assert d["X"].shape[1] == 9
PY

echo "[slurm] hf_parquet full-split: train"
"$PYTHON_BIN" train.py \
  --mode fixed_split \
  --train_data data/processed/full_train.npz \
  --val_data data/processed/full_test.npz \
  --out_dir checkpoints/full_split_full \
  --arch resnet \
  --epochs 80 \
  --batch_size 128 \
  --lr 1e-3 \
  --patience 15 \
  --device cuda \
  2>&1 | tee logs/train_hf_full_split.log

echo "[slurm] hf_parquet full-split: evaluate"
"$PYTHON_BIN" evaluate.py \
  --checkpoint checkpoints/full_split_full/best_model.pt \
  --data data/processed/full_test.npz \
  --out_dir results/full_split_full \
  --vote_k 5 \
  --batch_size 256 \
  2>&1 | tee logs/eval_hf_full_split.log

echo "[slurm] finished: $(date)"
echo "[slurm] done"
