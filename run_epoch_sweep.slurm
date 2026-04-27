#!/bin/bash
# =============================================================================
# run_epoch_sweep.slurm
# =============================================================================
# Purpose
# -------
# Previous run_loso_tune showed accuracy mildly increasing as epochs grew
# (80 → 120).  This job pushes further — 150 / 200 / 250 / 300 — and answers:
#   "Does accuracy keep climbing, or does it plateau/overfit?"
#
# New features compared to run_loso_tune
# ----------------------------------------
#   • Tests 4 deeper epoch budgets (150 / 200 / 250 / 300)
#   • Uses the new SensorLocLoss (--loss_fn custom, --lr_weight 0.5) by default
#     — set LOSS_FN=crossentropy to disable and compare against plain CE
#   • Patience is scaled with epochs (floor = epochs / 6) so early-stop
#     doesn't cut the run too short on high-epoch trials
#   • Produces a leaderboard CSV, a learning-curve summary, and automatically
#     reruns the winner on all LOSO folds
#   • All intermediate artefacts land in checkpoints/epoch_sweep/<trial>/
#
# Usage
# -----
#   sbatch run_epoch_sweep.slurm
#
# Override any setting without editing the file:
#   MAX_FOLDS_SWEEP=3 LOSS_FN=crossentropy sbatch run_epoch_sweep.slurm
#   SMOKE_TEST=1 sbatch run_epoch_sweep.slurm
# =============================================================================

#SBATCH --job-name=imu_epoch_sweep
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -p h200x4

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p logs data/processed checkpoints/epoch_sweep results/epoch_sweep

# ── Python binary ─────────────────────────────────────────────────────────────
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

# ── Configurable overrides ────────────────────────────────────────────────────
PARQUET_DIR="${HF_PARQUET_DIR:-data/raw_dip/data}"
SUBJECT_COLUMN="${SUBJECT_COLUMN:-subject_id}"
MERGE_MODE="${MERGE_INPUTS:-vimu_only}"
IMU_BLEND="${IMU_BLEND:-0.0}"
WINDOW_LENGTH="${WINDOW_LENGTH:-300}"
CROP_MODE="${CROP_MODE:-center}"
PAD_MODE="${PAD_MODE:-edge}"
ARCH="${ARCH:-resnet}"

# Loss function: 'custom' (CE + L/R penalty) or 'crossentropy'
LOSS_FN="${LOSS_FN:-custom}"
# Weight of the left-right symmetry penalty in SensorLocLoss
LR_WEIGHT="${LR_WEIGHT:-0.5}"

# Fixed hyperparams (same as the best trial from run_loso_tune)
BATCH_SIZE="${BATCH_SIZE:-128}"
BASE_LR="${BASE_LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"

# How many LOSO folds to run in the SWEEP phase (faster ranking)
MAX_FOLDS_SWEEP="${MAX_FOLDS_SWEEP:-5}"
# How many folds for the final rerun (0 = all)
MAX_FOLDS_FINAL="${MAX_FOLDS_FINAL:-0}"

SMOKE_TEST="${SMOKE_TEST:-0}"

# ── Smoke overrides ───────────────────────────────────────────────────────────
if [[ "$SMOKE_TEST" == "1" ]]; then
  MAX_FOLDS_SWEEP=1
  MAX_FOLDS_FINAL=1
  echo "[sweep] SMOKE_TEST=1 → forcing 1-fold"
fi

echo "============================================================"
echo "[sweep] job         : ${SLURM_JOB_ID:-local}  (${SLURM_JOB_NAME:-bash_run})"
echo "[sweep] repo        : $ROOT_DIR"
echo "[sweep] python      : $PYTHON_BIN"
echo "[sweep] parquet_dir : $PARQUET_DIR"
echo "[sweep] arch        : $ARCH"
echo "[sweep] loss_fn     : $LOSS_FN  (lr_weight=$LR_WEIGHT)"
echo "[sweep] batch_size  : $BATCH_SIZE  lr=$BASE_LR  wd=$WEIGHT_DECAY"
echo "[sweep] sweep folds : $MAX_FOLDS_SWEEP"
echo "[sweep] final folds : ${MAX_FOLDS_FINAL} (0=all)"
echo "[sweep] started     : $(date)"
echo "============================================================"

# =============================================================================
# STEP 1 — Preprocess raw Parquet → two .npz files
# =============================================================================
echo ""
echo "[sweep] ── STEP 1/4: preprocess hf_parquet ──────────────────────────────"
"$PYTHON_BIN" preprocess_vimu.py \
  --mode hf_parquet \
  --parquet_dir "$PARQUET_DIR" \
  --train_glob "train-*.parquet" \
  --test_glob  "test-*.parquet" \
  --hf_split predefined \
  --subject_column "$SUBJECT_COLUMN" \
  --window_length "$WINDOW_LENGTH" \
  --crop_mode "$CROP_MODE" \
  --pad_mode  "$PAD_MODE" \
  --merge_inputs "$MERGE_MODE" \
  --imu_blend "$IMU_BLEND" \
  --out_train data/processed/epoch_sweep_train.npz \
  --out_test  data/processed/epoch_sweep_test.npz \
  2>&1 | tee logs/preprocess_epoch_sweep.log

# =============================================================================
# STEP 2 — Merge train + test into one LOSO dataset
# =============================================================================
echo ""
echo "[sweep] ── STEP 2/4: merge into LOSO dataset ────────────────────────────"
"$PYTHON_BIN" - <<'PY'
import numpy as np

tr = np.load("data/processed/epoch_sweep_train.npz")
te = np.load("data/processed/epoch_sweep_test.npz")

X    = np.concatenate([tr["X"],           te["X"]],           axis=0).astype(np.float32)
y    = np.concatenate([tr["y"],           te["y"]],           axis=0).astype(np.int64)
sids = np.concatenate([tr["subject_ids"], te["subject_ids"]], axis=0).astype(np.int64)

if X.ndim != 3 or X.shape[1] != 9:
    raise ValueError(f"Expected X shape (N,9,T), got {X.shape}")
subjects = np.unique(sids)
if len(subjects) <= 1:
    raise ValueError(
        f"LOSO requires >1 unique subject_id, found: {subjects.tolist()}. "
        "Set SUBJECT_COLUMN=<col> if the column name differs."
    )

np.savez_compressed("data/processed/epoch_sweep_dataset.npz", X=X, y=y, subject_ids=sids)
print("[sweep] merged X:", X.shape)
print("[sweep] classes :", len(np.unique(y)))
print("[sweep] subjects:", len(subjects))
PY

# =============================================================================
# STEP 3 — Epoch-depth sweep (the key experiment)
# =============================================================================

RESULTS_CSV="results/epoch_sweep/leaderboard.csv"
echo "trial,epochs,batch_size,lr,weight_decay,patience,loss_fn,lr_weight,max_folds,mean_loso_acc" \
  > "$RESULTS_CSV"

# Helper: run one training trial and append its result to the leaderboard
run_trial () {
  local NAME="$1"
  local EPOCHS="$2"
  local BS="$3"
  local LR="$4"
  local WD="$5"
  # Scale patience with epoch budget: floor(epochs / 6), min 15, max 50
  local PAT=$(( EPOCHS / 6 ))
  PAT=$(( PAT < 15 ? 15 : PAT ))
  PAT=$(( PAT > 50 ? 50 : PAT ))
  local FOLDS="$6"
  local LOSS="$7"
  local LRW="$8"

  local OUT_DIR="checkpoints/epoch_sweep/${NAME}"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "[sweep] ──────────────────────────────────────────────────────────"
  echo "[sweep] trial  = $NAME"
  echo "[sweep] epochs = $EPOCHS  bs=$BS  lr=$LR  wd=$WD  patience=$PAT"
  echo "[sweep] loss   = $LOSS  lr_weight=$LRW  folds=$FOLDS"

  local -a TRAIN_ARGS=(
    --mode loso
    --data data/processed/epoch_sweep_dataset.npz
    --out_dir "$OUT_DIR"
    --arch "$ARCH"
    --epochs "$EPOCHS"
    --batch_size "$BS"
    --lr "$LR"
    --weight_decay "$WD"
    --patience "$PAT"
    --loss_fn "$LOSS"
    --lr_weight "$LRW"
    --device cuda
    --max_folds "$FOLDS"
  )
  if [[ "$SMOKE_TEST" == "1" ]]; then
    cp data/processed/epoch_sweep_dataset.npz \
       data/processed/epoch_sweep_smoke_dataset.npz
    TRAIN_ARGS+=(--smoke_test)
  fi

  "$PYTHON_BIN" train.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee "logs/train_${NAME}.log"

  # Parse mean LOSO acc from results file
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
  echo "[sweep] trial=$NAME  epochs=$EPOCHS  mean_loso_acc=$MEAN"
  echo "$NAME,$EPOCHS,$BS,$LR,$WD,$PAT,$LOSS,$LRW,$FOLDS,$MEAN" >> "$RESULTS_CSV"
}

echo ""
echo "[sweep] ── STEP 3/4: epoch-depth sweep ──────────────────────────────────"
echo "[sweep] Previous run_loso_tune: best seen at ~120 epochs."
echo "[sweep] Now testing 150 / 200 / 250 / 300 to see if accuracy keeps rising."

if [[ "$SMOKE_TEST" == "1" ]]; then
  # ---------- smoke: two tiny trials, 1 fold each ----------------------------
  run_trial "smoke_ep5"  5  64  1e-3  1e-4  "$MAX_FOLDS_SWEEP"  "$LOSS_FN"  "$LR_WEIGHT"
  run_trial "smoke_ep10" 10 64  1e-3  1e-4  "$MAX_FOLDS_SWEEP"  "$LOSS_FN"  "$LR_WEIGHT"
else
  # ---------- real sweep -----------------------------------------------------
  # Naming convention: ep<N>_<loss_short>
  #   Each trial keeps batch/lr/wd from the best run_loso_tune config.
  #   Only the epoch budget changes, so the comparison is clean.

  run_trial "ep150_${LOSS_FN}" 150 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"

  run_trial "ep200_${LOSS_FN}" 200 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"

  run_trial "ep250_${LOSS_FN}" 250 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"

  run_trial "ep300_${LOSS_FN}" 300 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"
fi

echo ""
echo "[sweep] ── Leaderboard so far ──────────────────────────────────────────"
column -t -s',' "$RESULTS_CSV"

# =============================================================================
# Summarise the trend: is accuracy still climbing or flattening?
# =============================================================================
echo ""
echo "[sweep] ── Epoch-vs-accuracy trend ─────────────────────────────────────"
"$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("results/epoch_sweep/leaderboard.csv").open()))
if not rows:
    raise RuntimeError("Empty leaderboard.csv")

# Sort by epoch count
rows.sort(key=lambda r: int(r["epochs"]))

print(f"{'Trial':<25} {'Epochs':>6} {'Mean LOSO Acc':>14}")
print("-" * 50)
prev_acc = None
for r in rows:
    acc = float(r["mean_loso_acc"])
    delta = ""
    if prev_acc is not None:
        diff = acc - prev_acc
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
        delta = f"  {arrow} {abs(diff)*100:+.3f}pp"
    print(f"  {r['trial']:<23} {r['epochs']:>6} {acc:>14.4f}{delta}")
    prev_acc = acc

best = max(rows, key=lambda r: float(r["mean_loso_acc"]))
print()
print(f"  Best trial : {best['trial']}  ({float(best['mean_loso_acc']):.4f})")

# Check trend: is accuracy monotonically increasing?
accs = [float(r["mean_loso_acc"]) for r in rows]
if all(accs[i] <= accs[i+1] for i in range(len(accs)-1)):
    print("  Trend      : STILL RISING — consider even more epochs.")
elif all(accs[i] >= accs[i+1] for i in range(len(accs)-1)):
    print("  Trend      : DECLINING   — earlier epoch count was optimal.")
else:
    print("  Trend      : MIXED       — accuracy peaked mid-sweep; best epoch found.")
PY

# =============================================================================
# STEP 4 — Rerun the winning epoch count on ALL LOSO folds
# =============================================================================
echo ""
echo "[sweep] ── STEP 4/4: full LOSO run with best config ─────────────────────"

# Write best_config.env
"$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("results/epoch_sweep/leaderboard.csv").open()))
best = max(rows, key=lambda r: float(r["mean_loso_acc"]))
env = Path("results/epoch_sweep/best_config.env")
env.write_text(
    "\n".join([
        f"BEST_EPOCHS={best['epochs']}",
        f"BEST_BATCH_SIZE={best['batch_size']}",
        f"BEST_LR={best['lr']}",
        f"BEST_WEIGHT_DECAY={best['weight_decay']}",
        f"BEST_PATIENCE={best['patience']}",
        f"BEST_LOSS_FN={best['loss_fn']}",
        f"BEST_LR_WEIGHT={best['lr_weight']}",
        f"BEST_TRIAL={best['trial']}",
        f"BEST_MEAN_ACC={best['mean_loso_acc']}",
    ]) + "\n"
)
print("[sweep] wrote results/epoch_sweep/best_config.env")
print(env.read_text())
PY

# shellcheck disable=SC1091
source results/epoch_sweep/best_config.env
BEST_OUT="checkpoints/epoch_sweep/best_final"
mkdir -p "$BEST_OUT"

echo "[sweep] best trial  : $BEST_TRIAL"
echo "[sweep] best epochs : $BEST_EPOCHS"
echo "[sweep] best loss   : $BEST_LOSS_FN  (lr_weight=$BEST_LR_WEIGHT)"

FINAL_ARGS=(
  --mode loso
  --data data/processed/epoch_sweep_dataset.npz
  --out_dir "$BEST_OUT"
  --arch "$ARCH"
  --epochs "$BEST_EPOCHS"
  --batch_size "$BEST_BATCH_SIZE"
  --lr "$BEST_LR"
  --weight_decay "$BEST_WEIGHT_DECAY"
  --patience "$BEST_PATIENCE"
  --loss_fn "$BEST_LOSS_FN"
  --lr_weight "$BEST_LR_WEIGHT"
  --device cuda
)
if [[ "$MAX_FOLDS_FINAL" != "0" ]]; then
  FINAL_ARGS+=(--max_folds "$MAX_FOLDS_FINAL")
fi
if [[ "$SMOKE_TEST" == "1" ]]; then
  cp data/processed/epoch_sweep_dataset.npz \
     data/processed/epoch_sweep_smoke_dataset.npz
  FINAL_ARGS+=(--smoke_test)
fi

"$PYTHON_BIN" train.py "${FINAL_ARGS[@]}" \
  2>&1 | tee logs/train_epoch_sweep_best_final.log

# Print the full result
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
p = Path("checkpoints/epoch_sweep/best_final/loso_results.txt")
if not p.exists():
    raise FileNotFoundError(f"Missing: {p}")
print("[sweep] ── Full LOSO results (best config) ──────────────────────────────")
print(p.read_text())
PY

echo ""
echo "============================================================"
echo "[sweep] Artefacts:"
echo "  Leaderboard CSV  : results/epoch_sweep/leaderboard.csv"
echo "  Best config env  : results/epoch_sweep/best_config.env"
echo "  Sweep checkpoints: checkpoints/epoch_sweep/<trial>/"
echo "  Final model      : checkpoints/epoch_sweep/best_final/"
echo "  Train logs       : logs/train_ep*.log"
echo "[sweep] finished: $(date)"
echo "============================================================"
