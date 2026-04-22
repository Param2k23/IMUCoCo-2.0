#!/bin/bash
#SBATCH --job-name=imu_ablation_sweep
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p h200x4

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p \
  logs \
  checkpoints/ablations \
  results/ablations \
  data/processed

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

PARQUET_DIR="${HF_PARQUET_DIR:-data/raw_dip/data}"

echo "[ablation] repo: $ROOT_DIR"
echo "[ablation] python: $PYTHON_BIN"
echo "[ablation] parquet_dir: $PARQUET_DIR"
echo "[ablation] started: $(date)"

echo "[ablation] preprocess once (hf_parquet predefined)"
"$PYTHON_BIN" preprocess_vimu.py \
  --mode hf_parquet \
  --parquet_dir "$PARQUET_DIR" \
  --train_glob "train-*.parquet" \
  --test_glob "test-*.parquet" \
  --hf_split predefined \
  --window_length 300 \
  --crop_mode center \
  --pad_mode edge \
  --merge_inputs vimu_only \
  --imu_blend 0.0 \
  --out_train data/processed/full_train.npz \
  --out_test data/processed/full_test.npz \
  2>&1 | tee logs/preprocess_ablation_full_split.log

echo "[ablation] sanity check"
"$PYTHON_BIN" - <<'PY'
import numpy as np
d = np.load("data/processed/full_train.npz")
print("full_train X shape:", d["X"].shape)
assert d["X"].shape[1] == 9
PY

SUMMARY_CSV="results/ablations/summary.csv"
echo "run_name,seed,merge_inputs,imu_blend,batch_size,weight_decay,val_per_window_acc,vote_acc,locked_fraction,spatial_mean_m,spatial_std_m,n_wrong" > "$SUMMARY_CSV"

run_case () {
  local RUN_NAME="$1"
  local SEED="$2"
  local MERGE="$3"
  local IMU_BLEND="$4"
  local BATCH_SIZE="$5"
  local WEIGHT_DECAY="$6"

  local TRAIN_NPZ="data/processed/${RUN_NAME}_train.npz"
  local TEST_NPZ="data/processed/${RUN_NAME}_test.npz"
  local CKPT_DIR="checkpoints/ablations/${RUN_NAME}"
  local RES_DIR="results/ablations/${RUN_NAME}"

  echo "[ablation] ====================================================="
  echo "[ablation] run=$RUN_NAME seed=$SEED merge=$MERGE imu_blend=$IMU_BLEND batch=$BATCH_SIZE wd=$WEIGHT_DECAY"

  "$PYTHON_BIN" preprocess_vimu.py \
    --mode hf_parquet \
    --parquet_dir "$PARQUET_DIR" \
    --train_glob "train-*.parquet" \
    --test_glob "test-*.parquet" \
    --hf_split predefined \
    --window_length 300 \
    --crop_mode center \
    --pad_mode edge \
    --merge_inputs "$MERGE" \
    --imu_blend "$IMU_BLEND" \
    --seed "$SEED" \
    --out_train "$TRAIN_NPZ" \
    --out_test "$TEST_NPZ" \
    2>&1 | tee "logs/preprocess_${RUN_NAME}.log"

  "$PYTHON_BIN" train.py \
    --mode fixed_split \
    --train_data "$TRAIN_NPZ" \
    --val_data "$TEST_NPZ" \
    --out_dir "$CKPT_DIR" \
    --arch resnet \
    --epochs 80 \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-3 \
    --weight_decay "$WEIGHT_DECAY" \
    --patience 15 \
    --device cuda \
    2>&1 | tee "logs/train_${RUN_NAME}.log"

  "$PYTHON_BIN" evaluate.py \
    --checkpoint "$CKPT_DIR/best_model.pt" \
    --data "$TEST_NPZ" \
    --out_dir "$RES_DIR" \
    --vote_k 5 \
    --batch_size 256 \
    2>&1 | tee "logs/eval_${RUN_NAME}.log"

  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

summary_path = Path("$RES_DIR") / "eval_summary.json"
data = json.loads(summary_path.read_text())
with open("$SUMMARY_CSV", "a", encoding="utf-8") as f:
    f.write(
        ",".join(
            [
                "$RUN_NAME",
                "$SEED",
                "$MERGE",
                "$IMU_BLEND",
                "$BATCH_SIZE",
                "$WEIGHT_DECAY",
                str(data.get("per_window_accuracy", "")),
                str(data.get("majority_vote_accuracy", "")),
                str(data.get("locked_fraction", "")),
                str(data.get("spatial_error", {}).get("mean_m", "")),
                str(data.get("spatial_error", {}).get("std_m", "")),
                str(data.get("spatial_error", {}).get("n_wrong", "")),
            ]
        )
        + "\\n"
    )
PY
}

# 1) Baseline repeats (seed variance)
run_case "baseline_seed42" 42 "vimu_only" 0.0 128 1e-4
run_case "baseline_seed43" 43 "vimu_only" 0.0 128 1e-4
run_case "baseline_seed44" 44 "vimu_only" 0.0 128 1e-4

# 2) Fusion ablations
run_case "fusion_blend01_seed42" 42 "blend_global_imu" 0.1 128 1e-4
run_case "fusion_blend02_seed42" 42 "blend_global_imu" 0.2 128 1e-4

# 3) Regularization tweak
run_case "reg_batch64_wd3e4_seed42" 42 "blend_global_imu" 0.1 64 3e-4

echo "[ablation] finished: $(date)"
echo "[ablation] summary csv -> $SUMMARY_CSV"
