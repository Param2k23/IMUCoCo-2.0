#!/bin/bash
# =============================================================================
# run_epoch_sweep.slurm   (v2)
# =============================================================================
# What it does
# ------------
#   1. Preprocess raw Parquet → LOSO dataset
#   2. Epoch-depth sweep  : 150 / 200 / 300 / 350 / 400 epochs
#      + one high-LR-weight variant (300 ep, lr_weight=1.0) to test stronger
#        left-right penalty
#   3. Pick the best trial automatically
#   4. Rerun best config on ALL LOSO folds (full training)
#   5. Run evaluate.py on every saved fold checkpoint  ← NEW
#      → per-window accuracy, majority-vote accuracy, L/R confusion breakdown,
#        confusion matrix PNG, spatial error, full eval_summary.json
#   6. Aggregate L/R confusion across all folds        ← NEW
#
# Smoke vs Real
# -------------
#   Smoke  : SMOKE_TEST=1 bash run_epoch_sweep.slurm
#            2 tiny trials (5 + 10 epochs), 1 fold, CPU fallback, ~5 min
#   Real   : sbatch run_epoch_sweep.slurm
#            5 sweep trials × 5 folds + full LOSO + eval, ~12-16 h on GPU
#
# Overrides (no file edits needed)
# ---------------------------------
#   MAX_FOLDS_SWEEP=3 sbatch run_epoch_sweep.slurm
#   LOSS_FN=crossentropy sbatch run_epoch_sweep.slurm
#   LR_WEIGHT=1.0 sbatch run_epoch_sweep.slurm
# =============================================================================

#SBATCH --job-name=imu_epoch_sweep
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=20:00:00
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

LOSS_FN="${LOSS_FN:-custom}"       # 'custom' or 'crossentropy'
LR_WEIGHT="${LR_WEIGHT:-0.5}"      # penalty weight in SensorLocLoss

BATCH_SIZE="${BATCH_SIZE:-128}"
BASE_LR="${BASE_LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"

MAX_FOLDS_SWEEP="${MAX_FOLDS_SWEEP:-5}"   # folds used during ranking sweep
MAX_FOLDS_FINAL="${MAX_FOLDS_FINAL:-0}"   # 0 = all folds for final run

VOTE_K="${VOTE_K:-5}"              # majority-vote window size in evaluate.py

SMOKE_TEST="${SMOKE_TEST:-0}"

# ── Smoke overrides ───────────────────────────────────────────────────────────
if [[ "$SMOKE_TEST" == "1" ]]; then
  MAX_FOLDS_SWEEP=1
  MAX_FOLDS_FINAL=1
  echo "[sweep] SMOKE_TEST=1 → 1-fold, tiny epochs, CPU fallback"
fi

echo "============================================================"
echo "[sweep] job         : ${SLURM_JOB_ID:-local}  (${SLURM_JOB_NAME:-bash_run})"
echo "[sweep] repo        : $ROOT_DIR"
echo "[sweep] python      : $PYTHON_BIN"
echo "[sweep] parquet_dir : $PARQUET_DIR"
echo "[sweep] arch        : $ARCH"
echo "[sweep] loss_fn     : $LOSS_FN  (lr_weight=$LR_WEIGHT)"
echo "[sweep] batch_size  : $BATCH_SIZE  lr=$BASE_LR  wd=$WEIGHT_DECAY"
echo "[sweep] sweep folds : $MAX_FOLDS_SWEEP  |  final folds: ${MAX_FOLDS_FINAL} (0=all)"
echo "[sweep] vote_k      : $VOTE_K"
echo "[sweep] started     : $(date)"
echo "============================================================"

# =============================================================================
# STEP 1 — Preprocess
# =============================================================================
echo ""
echo "[sweep] ── STEP 1/6: preprocess hf_parquet ──────────────────────────────"
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
# STEP 2 — Merge into single LOSO dataset
# =============================================================================
echo ""
echo "[sweep] ── STEP 2/6: merge into LOSO dataset ────────────────────────────"
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
    raise ValueError(f"LOSO requires >1 unique subject_id, found: {subjects.tolist()}")

np.savez_compressed("data/processed/epoch_sweep_dataset.npz", X=X, y=y, subject_ids=sids)
print("[sweep] merged X:", X.shape, "| classes:", len(np.unique(y)), "| subjects:", len(subjects))
PY

# =============================================================================
# STEP 3 — Epoch-depth sweep
# =============================================================================
RESULTS_CSV="results/epoch_sweep/leaderboard.csv"
echo "trial,epochs,batch_size,lr,weight_decay,patience,loss_fn,lr_weight,max_folds,mean_loso_acc" \
  > "$RESULTS_CSV"

# ------------------------------------------------------------------
# Helper: run_trial <name> <epochs> <bs> <lr> <wd> <folds> <loss> <lrw>
# Patience is auto-scaled: floor(epochs/6), clamped [15, 60]
# ------------------------------------------------------------------
run_trial () {
  local NAME="$1"  EPOCHS="$2"  BS="$3"  LR="$4"  WD="$5"  FOLDS="$6"
  local LOSS="$7"  LRW="$8"

  local PAT=$(( EPOCHS / 6 ))
  PAT=$(( PAT < 15 ? 15 : PAT ))
  PAT=$(( PAT > 60 ? 60 : PAT ))

  local OUT_DIR="checkpoints/epoch_sweep/${NAME}"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "[sweep] ── trial=$NAME  epochs=$EPOCHS  bs=$BS  lr=$LR  wd=$WD  pat=$PAT  loss=$LOSS  lrw=$LRW  folds=$FOLDS"

  local -a ARGS=(
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
       data/processed/epoch_sweep_smoke_dataset.npz 2>/dev/null || true
    ARGS+=(--smoke_test)
  fi

  "$PYTHON_BIN" train.py "${ARGS[@]}" 2>&1 | tee "logs/train_${NAME}.log"

  local MEAN
  MEAN=$("$PYTHON_BIN" - <<PY
import re
from pathlib import Path
p = Path("$OUT_DIR/loso_results.txt")
m = re.search(r"Mean LOSO Acc:\s*([0-9.]+)", p.read_text())
if not m:
    raise RuntimeError(f"Could not parse mean from {p}")
print(m.group(1))
PY
)
  echo "[sweep] trial=$NAME  mean_loso_acc=$MEAN"
  echo "$NAME,$EPOCHS,$BS,$LR,$WD,$PAT,$LOSS,$LRW,$FOLDS,$MEAN" >> "$RESULTS_CSV"
}

echo ""
echo "[sweep] ── STEP 3/6: epoch-depth sweep ──────────────────────────────────"
if [[ "$SMOKE_TEST" == "1" ]]; then
  # ── Smoke: 2 tiny trials, 1 fold ─────────────────────────────────────────
  run_trial "smoke_ep5"  5  64 1e-3 1e-4 "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"
  run_trial "smoke_ep10" 10 64 1e-3 1e-4 "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"
else
  # ── Real sweep: push beyond the 300-ep peak seen last run ────────────────
  # Trial 1-4: epoch depth (same lr_weight=0.5)
  run_trial "ep150_custom" 150 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"

  run_trial "ep200_custom" 200 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"

  run_trial "ep300_custom" 300 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"

  run_trial "ep350_custom" 350 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"

  run_trial "ep400_custom" 400 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "$LOSS_FN" "$LR_WEIGHT"

  # Trial 5: stronger L/R penalty (lr_weight=1.0) at best previous epoch
  # Tests whether doubling the symmetry penalty reduces L/R confusion errors
  run_trial "ep300_lrw1" 300 "$BATCH_SIZE" "$BASE_LR" "$WEIGHT_DECAY" \
            "$MAX_FOLDS_SWEEP" "custom" "1.0"
fi

# ── Print leaderboard ─────────────────────────────────────────────────────
echo ""
echo "[sweep] ── Leaderboard ──────────────────────────────────────────────────"
column -t -s',' "$RESULTS_CSV"

# ── Epoch-vs-accuracy trend ───────────────────────────────────────────────
echo ""
echo "[sweep] ── Epoch-vs-accuracy trend ─────────────────────────────────────"
"$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("results/epoch_sweep/leaderboard.csv").open()))
if not rows:
    raise RuntimeError("Empty leaderboard.csv")

rows.sort(key=lambda r: int(r["epochs"]))
print(f"{'Trial':<25} {'Epochs':>6} {'Mean LOSO Acc':>14}")
print("-" * 52)
prev = None
for r in rows:
    acc = float(r["mean_loso_acc"])
    delta = ""
    if prev is not None:
        d = acc - prev
        delta = f"  {'▲' if d>0 else '▼'} {abs(d)*100:+.3f}pp"
    print(f"  {r['trial']:<23} {r['epochs']:>6} {acc:>14.4f}{delta}")
    prev = acc

best = max(rows, key=lambda r: float(r["mean_loso_acc"]))
print()
print(f"  Best : {best['trial']}  →  {float(best['mean_loso_acc']):.4f}")

accs = [float(r["mean_loso_acc"]) for r in rows]
if all(accs[i] <= accs[i+1] for i in range(len(accs)-1)):
    print("  Trend: STILL RISING — consider even more epochs.")
elif all(accs[i] >= accs[i+1] for i in range(len(accs)-1)):
    print("  Trend: DECLINING   — earlier epoch count was best.")
else:
    peak_ep = rows[accs.index(max(accs))]["epochs"]
    print(f"  Trend: MIXED — peaked at ep={peak_ep}; plateau region identified.")
PY

# =============================================================================
# STEP 4 — Full LOSO rerun with best config
# =============================================================================
echo ""
echo "[sweep] ── STEP 4/6: write best_config.env ──────────────────────────────"
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
print("[sweep] best config →", best['trial'], f"acc={best['mean_loso_acc']}")
print(env.read_text())
PY

# shellcheck disable=SC1091
source results/epoch_sweep/best_config.env

BEST_OUT="checkpoints/epoch_sweep/best_final"
mkdir -p "$BEST_OUT"

echo "[sweep] ── STEP 4/6: full LOSO (best=$BEST_TRIAL, epochs=$BEST_EPOCHS, lrw=$BEST_LR_WEIGHT)"

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
     data/processed/epoch_sweep_smoke_dataset.npz 2>/dev/null || true
  FINAL_ARGS+=(--smoke_test)
fi

"$PYTHON_BIN" train.py "${FINAL_ARGS[@]}" \
  2>&1 | tee logs/train_epoch_sweep_best_final.log

# Print training result
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
p = Path("checkpoints/epoch_sweep/best_final/loso_results.txt")
print("[sweep] ── Full LOSO training results ────────────────────────────────────")
print(p.read_text())
PY

# =============================================================================
# STEP 5 — evaluate.py on every fold checkpoint  (L/R confusion + confusion matrix)
# =============================================================================
echo ""
echo "[sweep] ── STEP 5/6: evaluate.py on all fold checkpoints ────────────────"

# Count how many fold checkpoints were saved
N_CKPTS=$(ls "$BEST_OUT"/best_model_fold*.pt 2>/dev/null | wc -l)
echo "[sweep] found $N_CKPTS fold checkpoints in $BEST_OUT"

if [[ "$N_CKPTS" -eq 0 ]]; then
  echo "[sweep] WARNING: no checkpoints found — skipping evaluate step"
else
  for CKPT in "$BEST_OUT"/best_model_fold*.pt; do
    FOLD_ID=$(echo "$CKPT" | grep -oP '(?<=fold)\d+')
    EVAL_OUT="results/epoch_sweep/eval_fold${FOLD_ID}"
    mkdir -p "$EVAL_OUT"
    echo "[sweep]   evaluating fold $FOLD_ID → $EVAL_OUT"

    "$PYTHON_BIN" evaluate.py \
      --checkpoint "$CKPT" \
      --data       data/processed/epoch_sweep_dataset.npz \
      --out_dir    "$EVAL_OUT" \
      --vote_k     "$VOTE_K" \
      --batch_size 512 \
      2>&1 | tee "logs/eval_fold${FOLD_ID}.log"
  done
fi

# =============================================================================
# STEP 6 — Aggregate L/R confusion across all folds
# =============================================================================
echo ""
echo "[sweep] ── STEP 6/6: aggregate L/R confusion across folds ───────────────"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

eval_dirs = sorted(Path("results/epoch_sweep").glob("eval_fold*"))
if not eval_dirs:
    print("[sweep] No eval_fold* dirs found — skipping aggregation.")
    raise SystemExit(0)

total_windows   = 0
total_errors    = 0
total_lr_errors = 0
pair_counts     = {}   # pair_name -> total_lr_errors across folds
fold_summaries  = []

for d in eval_dirs:
    j = d / "eval_summary.json"
    if not j.exists():
        print(f"  [warn] missing {j}")
        continue
    data = json.loads(j.read_text())
    lr   = data.get("left_right_confusion", {})
    n_w  = data.get("n_windows", 0)
    n_e  = lr.get("n_total_errors", 0)
    n_lr = lr.get("n_lr_errors", 0)

    total_windows   += n_w
    total_errors    += n_e
    total_lr_errors += n_lr

    fold_summaries.append({
        "fold"       : d.name,
        "windows"    : n_w,
        "acc"        : data.get("per_window_accuracy", 0),
        "total_err"  : n_e,
        "lr_err"     : n_lr,
        "lr_pct"     : lr.get("lr_error_pct_of_all_errors", 0),
    })

    for pp in lr.get("per_pair", []):
        name = pp["pair"]
        pair_counts[name] = pair_counts.get(name, 0) + pp["total_lr_errors"]

# ── Print per-fold table ─────────────────────────────────────────────────────
print()
print(f"{'Fold':<15} {'Windows':>8} {'Acc':>7} {'Errors':>8} {'L/R Err':>8} {'L/R%':>7}")
print("-" * 58)
for fs in fold_summaries:
    print(f"  {fs['fold']:<13} {fs['windows']:>8} {fs['acc']:>7.4f} "
          f"{fs['total_err']:>8} {fs['lr_err']:>8} {fs['lr_pct']:>7.1f}%")

# ── Overall aggregate ────────────────────────────────────────────────────────
overall_lr_pct = 100.0 * total_lr_errors / max(1, total_errors)
print("-" * 58)
print(f"  {'TOTAL':<13} {total_windows:>8}          "
      f"{total_errors:>8} {total_lr_errors:>8} {overall_lr_pct:>7.1f}%")
print()
print(f"  ► {total_lr_errors} out of {total_errors} misclassifications "
      f"({overall_lr_pct:.1f}%) are pure left↔right mirror swaps")
print()

# ── Per symmetric-pair ranking ────────────────────────────────────────────────
if pair_counts:
    print(f"  {'Symmetric pair':<36} {'Total L/R errors':>18}")
    print("  " + "-" * 55)
    for pair, cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        print(f"    {pair:<34} {cnt:>18}")

# ── Save aggregate JSON ───────────────────────────────────────────────────────
agg = {
    "total_windows"     : total_windows,
    "total_errors"      : total_errors,
    "total_lr_errors"   : total_lr_errors,
    "overall_lr_pct"    : round(overall_lr_pct, 2),
    "per_fold"          : fold_summaries,
    "per_pair_total"    : {k: v for k, v in
                           sorted(pair_counts.items(), key=lambda x: -x[1])},
}
out = Path("results/epoch_sweep/lr_confusion_aggregate.json")
out.write_text(json.dumps(agg, indent=2))
print(f"\n  Aggregate saved → {out}")
PY

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "[sweep] Artefacts:"
echo "  Leaderboard       : results/epoch_sweep/leaderboard.csv"
echo "  Best config       : results/epoch_sweep/best_config.env"
echo "  Eval per fold     : results/epoch_sweep/eval_fold*/"
echo "  L/R aggregate     : results/epoch_sweep/lr_confusion_aggregate.json"
echo "  Confusion matrices: results/epoch_sweep/eval_fold*/confusion_matrix.png"
echo "  Train logs        : logs/train_ep*.log"
echo "  Eval logs         : logs/eval_fold*.log"
echo "[sweep] finished: $(date)"
echo "============================================================"
