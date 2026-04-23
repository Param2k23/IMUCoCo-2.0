#!/bin/bash
#SBATCH --job-name=imu_hf_parquet_loso
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -p h200x4

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p logs checkpoints/loso_full results/loso_full data/processed

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

PARQUET_DIR="${HF_PARQUET_DIR:-data/raw_dip/data}"
MERGE_MODE="${MERGE_INPUTS:-vimu_only}"
IMU_BLEND="${IMU_BLEND:-0.0}"
SUBJECT_COLUMN="${SUBJECT_COLUMN:-}"
WINDOW_LENGTH="${WINDOW_LENGTH:-300}"
CROP_MODE="${CROP_MODE:-center}"
PAD_MODE="${PAD_MODE:-edge}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PATIENCE="${PATIENCE:-15}"
MAX_FOLDS="${MAX_FOLDS:-0}"
ARCH="${ARCH:-resnet}"
SMOKE_TEST="${SMOKE_TEST:-0}"

if [[ "$SMOKE_TEST" == "1" ]]; then
  EPOCHS=5
  PATIENCE=5
  MAX_FOLDS=1
  echo "[loso] SMOKE_TEST=1 -> forcing epochs=$EPOCHS patience=$PATIENCE max_folds=$MAX_FOLDS"
fi

echo "[loso] repo: $ROOT_DIR"
echo "[loso] python: $PYTHON_BIN"
echo "[loso] parquet_dir: $PARQUET_DIR"
echo "[loso] merge_inputs: $MERGE_MODE (imu_blend=$IMU_BLEND)"
echo "[loso] subject_column: ${SUBJECT_COLUMN:-<auto-detect>}"
echo "[loso] smoke_test: $SMOKE_TEST"
echo "[loso] started: $(date)"

PREPROCESS_ARGS=(
  --mode hf_parquet
  --parquet_dir "$PARQUET_DIR"
  --train_glob "train-*.parquet"
  --test_glob "test-*.parquet"
  --hf_split predefined
  --window_length "$WINDOW_LENGTH"
  --crop_mode "$CROP_MODE"
  --pad_mode "$PAD_MODE"
  --merge_inputs "$MERGE_MODE"
  --imu_blend "$IMU_BLEND"
  --out_train data/processed/loso_train.npz
  --out_test data/processed/loso_test.npz
)
if [[ -n "$SUBJECT_COLUMN" ]]; then
  PREPROCESS_ARGS+=(--subject_column "$SUBJECT_COLUMN")
fi

echo "[loso] step 1/4: preprocess hf_parquet (predefined)"
"$PYTHON_BIN" preprocess_vimu.py "${PREPROCESS_ARGS[@]}" \
  2>&1 | tee logs/preprocess_hf_loso.log

echo "[loso] step 2/4: merge train+test into one LOSO dataset"
"$PYTHON_BIN" - <<'PY'
import numpy as np
from pathlib import Path

tr = np.load("data/processed/loso_train.npz")
te = np.load("data/processed/loso_test.npz")

X = np.concatenate([tr["X"], te["X"]], axis=0).astype(np.float32)
y = np.concatenate([tr["y"], te["y"]], axis=0).astype(np.int64)
sids = np.concatenate([tr["subject_ids"], te["subject_ids"]], axis=0).astype(np.int64)

if X.ndim != 3 or X.shape[1] != 9:
    raise ValueError(f"Expected X shape (N,9,T), got {X.shape}")
if len(X) != len(y) or len(y) != len(sids):
    raise ValueError("Length mismatch among X, y, subject_ids")
if y.min() < 0 or y.max() > 23:
    raise ValueError(f"Labels out of range [0,23]: min={y.min()} max={y.max()}")

subjects = np.unique(sids)
if len(subjects) <= 1:
    raise ValueError(
        "LOSO requires >1 unique subject_id, but only "
        f"{len(subjects)} found: {subjects.tolist()}. "
        "If Parquet has subject metadata, pass SUBJECT_COLUMN=<colname>."
    )

out_path = Path("data/processed/loso_dataset.npz")
np.savez_compressed(out_path, X=X, y=y, subject_ids=sids)

print("[loso] merged dataset:", out_path)
print("[loso] X shape:", X.shape)
print("[loso] y shape:", y.shape, "classes:", len(np.unique(y)))
print("[loso] subject_ids shape:", sids.shape, "subjects:", len(subjects))
print("[loso] subject id range:", int(subjects.min()), "to", int(subjects.max()))
PY

echo "[loso] step 3/4: train LOSO"
TRAIN_ARGS=(
  --mode loso
  --data data/processed/loso_dataset.npz
  --out_dir checkpoints/loso_full
  --arch "$ARCH"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --patience "$PATIENCE"
  --device cuda
)
if [[ "$MAX_FOLDS" != "0" ]]; then
  TRAIN_ARGS+=(--max_folds "$MAX_FOLDS")
fi
if [[ "$SMOKE_TEST" == "1" ]]; then
  TRAIN_ARGS+=(--smoke_test)
fi

"$PYTHON_BIN" train.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee logs/train_hf_loso.log

echo "[loso] step 4/4: summarize"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

results_path = Path("checkpoints/loso_full/loso_results.txt")
if not results_path.exists():
    raise FileNotFoundError(f"Missing LOSO results file: {results_path}")

print("[loso] results file:", results_path)
print(results_path.read_text())
PY

echo "[loso] finished: $(date)"
echo "[loso] done"
