#!/bin/bash
# =============================================================================
# run_eval_only.slurm
# =============================================================================
# Runs evaluate.py on ALL existing fold checkpoints found under a given
# checkpoint directory. No training — just inference + confusion matrix.
#
# Use this after any training run to get:
#   eval_summary.json  (per_window_acc, confusion_matrix 24×24,
#                       top_confused_pairs, left_right_confusion)
#   confusion_matrix.png
#
# Default: scans checkpoints/penalty_sweep/best_final/
# Override: CKPT_DIR=checkpoints/epoch_sweep/best_final sbatch run_eval_only.slurm
#
# Usage
# -----
#   sbatch run_eval_only.slurm                          # penalty_sweep results
#   CKPT_DIR=checkpoints/epoch_sweep/best_final \
#     OUT_DIR=results/epoch_sweep/eval_rerun \
#     sbatch run_eval_only.slurm
#
# Runtime: ~10-15 min for 10 folds on a single GPU
# =============================================================================

#SBATCH --job-name=imu_eval_only
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p h200x4

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

# ── Python binary ─────────────────────────────────────────────────────────────
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

# ── Configurable overrides ────────────────────────────────────────────────────
# Which checkpoint directory to evaluate
CKPT_DIR="${CKPT_DIR:-checkpoints/penalty_sweep/best_final}"

# Dataset to evaluate against (must match the one used for training)
DATA_NPZ="${DATA_NPZ:-data/processed/penalty_sweep_dataset.npz}"

# Where to write per-fold eval_summary.json and confusion_matrix.png
OUT_DIR="${OUT_DIR:-results/penalty_sweep/eval_best_final}"

VOTE_K="${VOTE_K:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
# base_filters of the trained model — 0 = auto-detect from normalization stats.
# Set to 128 for checkpoints trained with --base_filters 128 that predate the fix.
BASE_FILTERS="${BASE_FILTERS:-128}"

mkdir -p logs "$OUT_DIR"

echo "============================================================"
echo "[eval] job        : ${SLURM_JOB_ID:-local}  (${SLURM_JOB_NAME:-bash_run})"
echo "[eval] python     : $PYTHON_BIN"
echo "[eval] ckpt_dir   : $CKPT_DIR"
echo "[eval] data       : $DATA_NPZ"
echo "[eval] out_dir    : $OUT_DIR"
echo "[eval] vote_k     : $VOTE_K  batch_size=$BATCH_SIZE  base_filters=$BASE_FILTERS"
echo "[eval] started    : $(date)"
echo "============================================================"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[eval] ERROR: checkpoint directory not found: $CKPT_DIR"
  exit 1
fi
if [[ ! -f "$DATA_NPZ" ]]; then
  echo "[eval] ERROR: dataset not found: $DATA_NPZ"
  exit 1
fi

# ── Run evaluate.py on every fold checkpoint ──────────────────────────────────
N_CKPTS=$(ls "$CKPT_DIR"/best_model_fold*.pt 2>/dev/null | wc -l)
echo "[eval] found $N_CKPTS fold checkpoints"

if [[ "$N_CKPTS" -eq 0 ]]; then
  echo "[eval] ERROR: no best_model_fold*.pt files found in $CKPT_DIR"
  exit 1
fi

for CKPT in "$CKPT_DIR"/best_model_fold*.pt; do
  FOLD_ID=$(echo "$CKPT" | grep -oP '(?<=fold)\d+')
  FOLD_OUT="${OUT_DIR}/fold${FOLD_ID}"
  mkdir -p "$FOLD_OUT"
  echo ""
  echo "[eval] ── fold $FOLD_ID  →  $FOLD_OUT"

  "$PYTHON_BIN" evaluate.py \
    --checkpoint  "$CKPT" \
    --data        "$DATA_NPZ" \
    --out_dir     "$FOLD_OUT" \
    --vote_k      "$VOTE_K" \
    --batch_size  "$BATCH_SIZE" \
    --base_filters "$BASE_FILTERS" \
    2>&1 | tee "logs/eval_only_fold${FOLD_ID}.log"
done

# ── Aggregate non-L/R confused pairs + L/R breakdown ─────────────────────────
echo ""
echo "[eval] ── Aggregating confusion data across $N_CKPTS folds ──────────────"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from collections import defaultdict

out_dir = Path("$OUT_DIR")
eval_dirs = sorted(out_dir.glob("fold*"))
if not eval_dirs:
    print("[eval] No fold* dirs found — skipping aggregation.")
    raise SystemExit(0)

pair_counts = defaultdict(int)
lr_totals   = defaultdict(int)
fold_rows   = []
total_windows = 0
total_errors  = 0
total_lr_err  = 0

for d in eval_dirs:
    j = d / "eval_summary.json"
    if not j.exists():
        print(f"  [warn] missing {j}")
        continue
    data = json.loads(j.read_text())

    n_w  = data.get("n_windows", 0)
    acc  = data.get("per_window_accuracy", 0)
    lr   = data.get("left_right_confusion", {})
    n_e  = lr.get("n_total_errors", 0)
    n_lr = lr.get("n_lr_errors", 0)

    total_windows += n_w
    total_errors  += n_e
    total_lr_err  += n_lr

    for p in data.get("top_confused_pairs", []):
        key = f"{p['true']} -> {p['predicted']}"
        pair_counts[key] += p["count"]

    for pp in lr.get("per_pair", []):
        lr_totals[pp["pair"]] += pp["total_lr_errors"]

    fold_rows.append({
        "fold": d.name, "n_windows": n_w, "acc": round(acc, 4),
        "total_errors": n_e, "lr_errors": n_lr,
        "lr_pct": round(100.0 * n_lr / max(1, n_e), 1),
    })

# ── Per-fold summary ──────────────────────────────────────────────────────────
print()
print(f"{'Fold':<10} {'Windows':>8} {'Acc':>7} {'Errors':>8} {'L/R Err':>8} {'L/R%':>6}")
print("-" * 54)
for f in fold_rows:
    print(f"  {f['fold']:<8} {f['n_windows']:>8} {f['acc']:>7.4f} "
          f"{f['total_errors']:>8} {f['lr_errors']:>8} {f['lr_pct']:>5.1f}%")
nonlr = total_errors - total_lr_err
print("-" * 54)
print(f"  {'TOTAL':<8} {total_windows:>8}         {total_errors:>8} "
      f"{total_lr_err:>8} {100*total_lr_err/max(1,total_errors):>5.1f}%")
print(f"  Non-L/R errors: {nonlr} ({100*nonlr/max(1,total_errors):.1f}%)")

# ── Top-20 non-L/R confused pairs ────────────────────────────────────────────
print()
print("  Top-20 non-L/R confused pairs (aggregated across all folds):")
print(f"  {'Pair':<40} {'Count':>8} {'% of errors':>12}")
print("  " + "-" * 63)
for pair, cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:20]:
    pct = 100.0 * cnt / max(1, total_errors)
    print(f"    {pair:<38} {cnt:>8}    {pct:>8.2f}%")

# ── L/R breakdown ─────────────────────────────────────────────────────────────
if lr_totals:
    print()
    print(f"  L/R symmetric pair errors (total={total_lr_err}):")
    for pair, cnt in sorted(lr_totals.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / max(1, total_errors)
        print(f"    {pair:<38} {cnt:>8}    {pct:>8.2f}%")

# ── Save aggregate JSON ───────────────────────────────────────────────────────
agg = {
    "total_windows"  : total_windows,
    "total_errors"   : total_errors,
    "total_lr_errors": total_lr_err,
    "total_nonlr_errors": nonlr,
    "lr_pct_of_errors": round(100.0 * total_lr_err / max(1, total_errors), 2),
    "nonlr_pct_of_errors": round(100.0 * nonlr / max(1, total_errors), 2),
    "top_nonlr_pairs": [
        {"pair": k, "total_count": v,
         "pct_of_all_errors": round(100.0 * v / max(1, total_errors), 2)}
        for k, v in sorted(pair_counts.items(), key=lambda x: -x[1])
    ],
    "lr_pair_totals": dict(sorted(lr_totals.items(), key=lambda x: -x[1])),
    "per_fold": fold_rows,
}
out = Path("$OUT_DIR") / "confusion_aggregate.json"
out.write_text(json.dumps(agg, indent=2))
print(f"\n  Saved -> {out}")
PY

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "[eval] Files to pull from server:"
echo ""
echo "  ${OUT_DIR}/confusion_aggregate.json    <- top-20 non-L/R pairs + summary"
echo "  ${OUT_DIR}/fold*/eval_summary.json     <- per-fold full confusion matrix"
echo "  ${OUT_DIR}/fold*/confusion_matrix.png  <- per-fold heatmap"
echo "  logs/eval_only_fold*.log               <- per-fold inference logs"
echo "  logs/slurm_imu_eval_only_${SLURM_JOB_ID:-local}.out"
echo "============================================================"
echo "[eval] finished: $(date)"
