#!/bin/bash
#SBATCH --job-name=imu_loso_tune
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -p h200x4

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p logs data/processed checkpoints/loso_tune results/loso_tune

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

PARQUET_DIR="${HF_PARQUET_DIR:-data/raw_dip/data}"
SUBJECT_COLUMN="${SUBJECT_COLUMN:-subject_id}"
MERGE_MODE="${MERGE_INPUTS:-vimu_only}"
IMU_BLEND="${IMU_BLEND:-0.0}"
WINDOW_LENGTH="${WINDOW_LENGTH:-300}"
CROP_MODE="${CROP_MODE:-center}"
PAD_MODE="${PAD_MODE:-edge}"
ARCH="${ARCH:-resnet}"

# Smoke mode: fast validation of full pipeline wiring.
SMOKE_TEST="${SMOKE_TEST:-0}"

# Sweep control.
MAX_FOLDS_SWEEP="${MAX_FOLDS_SWEEP:-5}"   # quick ranking folds
MAX_FOLDS_FINAL="${MAX_FOLDS_FINAL:-0}"   # 0 = all folds

if [[ "$SMOKE_TEST" == "1" ]]; then
  MAX_FOLDS_SWEEP=1
  MAX_FOLDS_FINAL=1
  echo "[tune] SMOKE_TEST=1 -> forcing 1-fold smoke behavior"
fi

echo "[tune] repo: $ROOT_DIR"
echo "[tune] python: $PYTHON_BIN"
echo "[tune] parquet_dir: $PARQUET_DIR"
echo "[tune] subject_column: $SUBJECT_COLUMN"
echo "[tune] merge_inputs: $MERGE_MODE (imu_blend=$IMU_BLEND)"
echo "[tune] max_folds_sweep: $MAX_FOLDS_SWEEP"
echo "[tune] max_folds_final: $MAX_FOLDS_FINAL"
echo "[tune] started: $(date)"

echo "[tune] step 1/4: preprocess parquet once"
"$PYTHON_BIN" preprocess_vimu.py \
  --mode hf_parquet \
  --parquet_dir "$PARQUET_DIR" \
  --train_glob "train-*.parquet" \
  --test_glob "test-*.parquet" \
  --hf_split predefined \
  --subject_column "$SUBJECT_COLUMN" \
  --window_length "$WINDOW_LENGTH" \
  --crop_mode "$CROP_MODE" \
  --pad_mode "$PAD_MODE" \
  --merge_inputs "$MERGE_MODE" \
  --imu_blend "$IMU_BLEND" \
  --out_train data/processed/loso_tune_train.npz \
  --out_test data/processed/loso_tune_test.npz \
  2>&1 | tee logs/preprocess_loso_tune.log

echo "[tune] step 2/4: merge to single LOSO dataset"
"$PYTHON_BIN" - <<'PY'
import numpy as np

tr = np.load("data/processed/loso_tune_train.npz")
te = np.load("data/processed/loso_tune_test.npz")
X = np.concatenate([tr["X"], te["X"]], axis=0).astype(np.float32)
y = np.concatenate([tr["y"], te["y"]], axis=0).astype(np.int64)
sids = np.concatenate([tr["subject_ids"], te["subject_ids"]], axis=0).astype(np.int64)

if X.ndim != 3 or X.shape[1] != 9:
    raise ValueError(f"Expected X shape (N,9,T), got {X.shape}")
if len(np.unique(sids)) <= 1:
    raise ValueError("Need >1 unique subject_id for LOSO")

np.savez_compressed("data/processed/loso_tune_dataset.npz", X=X, y=y, subject_ids=sids)
print("[tune] merged X:", X.shape)
print("[tune] classes:", len(np.unique(y)))
print("[tune] subjects:", len(np.unique(sids)))
PY

RESULTS_CSV="results/loso_tune/leaderboard.csv"
echo "trial,epochs,batch_size,lr,weight_decay,patience,max_folds,mean_loso_acc" > "$RESULTS_CSV"

run_trial () {
  local NAME="$1"
  local EPOCHS="$2"
  local BS="$3"
  local LR="$4"
  local WD="$5"
  local PAT="$6"
  local FOLDS="$7"

  local OUT_DIR="checkpoints/loso_tune/${NAME}"
  mkdir -p "$OUT_DIR"

  echo "[tune] -----------------------------------------------------"
  echo "[tune] trial=$NAME epochs=$EPOCHS bs=$BS lr=$LR wd=$WD pat=$PAT folds=$FOLDS"

  local -a TRAIN_ARGS=(
    --mode loso
    --data data/processed/loso_tune_dataset.npz
    --out_dir "$OUT_DIR"
    --arch "$ARCH"
    --epochs "$EPOCHS"
    --batch_size "$BS"
    --lr "$LR"
    --weight_decay "$WD"
    --patience "$PAT"
    --device cuda
    --max_folds "$FOLDS"
  )
  if [[ "$SMOKE_TEST" == "1" ]]; then
    cp data/processed/loso_tune_dataset.npz data/processed/loso_tune_smoke_dataset.npz
    TRAIN_ARGS+=(--smoke_test)
  fi

  "$PYTHON_BIN" train.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee "logs/train_${NAME}.log"

  local MEAN
  MEAN=$("$PYTHON_BIN" - <<PY
import re
from pathlib import Path
p = Path("$OUT_DIR/loso_results.txt")
txt = p.read_text()
m = re.search(r"Mean LOSO Acc:\s*([0-9.]+)", txt)
if not m:
    raise RuntimeError(f"Could not parse mean from {p}")
print(m.group(1))
PY
)
  echo "[tune] trial=$NAME mean_loso_acc=$MEAN"
  echo "$NAME,$EPOCHS,$BS,$LR,$WD,$PAT,$FOLDS,$MEAN" >> "$RESULTS_CSV"
}

echo "[tune] step 3/4: LOSO hyperparameter sweep"
if [[ "$SMOKE_TEST" == "1" ]]; then
  run_trial "smoke_baseline" 5 64 1e-3 1e-4 5 "$MAX_FOLDS_SWEEP"
  run_trial "smoke_alt_lr" 5 64 5e-4 1e-4 5 "$MAX_FOLDS_SWEEP"
else
  run_trial "trial_a" 80 128 1e-3 1e-4 15 "$MAX_FOLDS_SWEEP"
  run_trial "trial_b" 100 128 1e-3 3e-4 20 "$MAX_FOLDS_SWEEP"
  run_trial "trial_c" 80 64 1e-3 1e-4 15 "$MAX_FOLDS_SWEEP"
  run_trial "trial_d" 80 128 5e-4 1e-4 20 "$MAX_FOLDS_SWEEP"
  run_trial "trial_e" 120 128 7e-4 3e-4 25 "$MAX_FOLDS_SWEEP"
fi

echo "[tune] current leaderboard:"
cat "$RESULTS_CSV"

echo "[tune] step 4/4: rerun best trial on final folds"
BEST_TRIAL=$("$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("results/loso_tune/leaderboard.csv").open()))
if not rows:
    raise RuntimeError("Empty leaderboard.csv")
best = max(rows, key=lambda r: float(r["mean_loso_acc"]))
print(best["trial"])
PY
)
echo "[tune] best trial from sweep: $BEST_TRIAL"

"$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("results/loso_tune/leaderboard.csv").open()))
best = max(rows, key=lambda r: float(r["mean_loso_acc"]))
Path("results/loso_tune/best_config.env").write_text(
    "\n".join([
        f"EPOCHS={best['epochs']}",
        f"BATCH_SIZE={best['batch_size']}",
        f"LR={best['lr']}",
        f"WEIGHT_DECAY={best['weight_decay']}",
        f"PATIENCE={best['patience']}",
    ]) + "\n"
)
print("[tune] wrote results/loso_tune/best_config.env")
PY

# shellcheck disable=SC1091
source results/loso_tune/best_config.env
BEST_OUT="checkpoints/loso_tune/best_final"
mkdir -p "$BEST_OUT"

FINAL_ARGS=(
  --mode loso
  --data data/processed/loso_tune_dataset.npz
  --out_dir "$BEST_OUT"
  --arch "$ARCH"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --patience "$PATIENCE"
  --device cuda
)
if [[ "$MAX_FOLDS_FINAL" != "0" ]]; then
  FINAL_ARGS+=(--max_folds "$MAX_FOLDS_FINAL")
fi
if [[ "$SMOKE_TEST" == "1" ]]; then
  cp data/processed/loso_tune_dataset.npz data/processed/loso_tune_smoke_dataset.npz
  FINAL_ARGS+=(--smoke_test)
fi

"$PYTHON_BIN" train.py "${FINAL_ARGS[@]}" \
  2>&1 | tee logs/train_loso_tune_best_final.log

echo "[tune] done. artifacts:"
echo "  leaderboard: results/loso_tune/leaderboard.csv"
echo "  best run:    checkpoints/loso_tune/best_final/loso_results.txt"
echo "[tune] finished: $(date)"
