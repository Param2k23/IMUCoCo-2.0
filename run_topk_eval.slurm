#!/bin/bash
# =============================================================================
# run_topk_eval.slurm
# =============================================================================
# Dedicated top-k accuracy evaluation script.
#
# Runs evaluate.py on ALL fold checkpoints in a given directory and reports
# top-k accuracy (k=1,2,3,5,10 by default) per-fold and aggregated.
#
# What it produces per fold:
#   eval_summary.json  → topk_accuracy {top1..top10}
#                      → per_class_topk  (per-region breakdown)
#   confusion_matrix.png
#
# What it aggregates:
#   topk_summary.json  → mean top-k per k, per-fold table, per-class mean
#
# Usage
# -----
#   sbatch run_topk_eval.slurm                          # penalty_sweep default
#
#   CKPT_DIR=checkpoints/epoch_sweep/best_final \
#     OUT_DIR=results/epoch_sweep/topk_eval \
#     TOPK=1,2,3,5,10 \
#     sbatch run_topk_eval.slurm
#
# Runtime: ~10-15 min for 10 folds on a single GPU
# =============================================================================

#SBATCH --job-name=imu_topk_eval
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
CKPT_DIR="${CKPT_DIR:-checkpoints/penalty_sweep/best_final}"
DATA_NPZ="${DATA_NPZ:-data/processed/penalty_sweep_dataset.npz}"
OUT_DIR="${OUT_DIR:-results/penalty_sweep/topk_eval}"
VOTE_K="${VOTE_K:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BASE_FILTERS="${BASE_FILTERS:-128}"
# Top-k values to evaluate — comma-separated
TOPK="${TOPK:-1,2,3,5,10}"

mkdir -p logs "$OUT_DIR"

echo "============================================================"
echo "[topk] job        : ${SLURM_JOB_ID:-local}  (${SLURM_JOB_NAME:-bash_run})"
echo "[topk] python     : $PYTHON_BIN"
echo "[topk] ckpt_dir   : $CKPT_DIR"
echo "[topk] data       : $DATA_NPZ"
echo "[topk] out_dir    : $OUT_DIR"
echo "[topk] topk       : $TOPK"
echo "[topk] base_filters: $BASE_FILTERS"
echo "[topk] started    : $(date)"
echo "============================================================"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[topk] ERROR: checkpoint directory not found: $CKPT_DIR"
  exit 1
fi
if [[ ! -f "$DATA_NPZ" ]]; then
  echo "[topk] ERROR: dataset not found: $DATA_NPZ"
  exit 1
fi

N_CKPTS=$(ls "$CKPT_DIR"/best_model_fold*.pt 2>/dev/null | wc -l)
echo "[topk] found $N_CKPTS fold checkpoints"
if [[ "$N_CKPTS" -eq 0 ]]; then
  echo "[topk] ERROR: no best_model_fold*.pt found in $CKPT_DIR"
  exit 1
fi

# ── Run evaluate.py on every fold checkpoint ──────────────────────────────────
for CKPT in "$CKPT_DIR"/best_model_fold*.pt; do
  FOLD_ID=$(echo "$CKPT" | grep -oP '(?<=fold)\d+')
  FOLD_OUT="${OUT_DIR}/fold${FOLD_ID}"
  mkdir -p "$FOLD_OUT"
  echo ""
  echo "[topk] ── fold $FOLD_ID  →  $FOLD_OUT"

  "$PYTHON_BIN" evaluate.py \
    --checkpoint   "$CKPT" \
    --data         "$DATA_NPZ" \
    --out_dir      "$FOLD_OUT" \
    --vote_k       "$VOTE_K" \
    --batch_size   "$BATCH_SIZE" \
    --base_filters "$BASE_FILTERS" \
    --topk         "$TOPK" \
    2>&1 | tee "logs/topk_eval_fold${FOLD_ID}.log"
done

# ── Aggregate top-k across folds ─────────────────────────────────────────────
echo ""
echo "[topk] ── Aggregating top-k results across $N_CKPTS folds ──────────────"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from collections import defaultdict

out_dir  = Path("$OUT_DIR")
topk_str = "$TOPK"
k_values = [int(k) for k in topk_str.split(",")]

eval_dirs = sorted(out_dir.glob("fold*"))
if not eval_dirs:
    print("[topk] No fold* dirs found.")
    raise SystemExit(0)

fold_rows   = []
class_topk  = defaultdict(lambda: defaultdict(list))  # class -> k -> [acc, ...]
total_windows = 0

for d in eval_dirs:
    j = d / "eval_summary.json"
    if not j.exists():
        print(f"  [warn] missing {j}")
        continue
    data = json.loads(j.read_text())

    n_w   = data.get("n_windows", 0)
    total_windows += n_w
    tkac  = data.get("topk_accuracy", {})

    row = {"fold": d.name, "n_windows": n_w}
    for k in k_values:
        row[f"top{k}"] = round(tkac.get(f"top{k}", 0.0), 4)
    fold_rows.append(row)

    # Per-class aggregation
    for cls_row in data.get("per_class_topk", []):
        name = cls_row["class"]
        for k in k_values:
            val = cls_row.get(f"top{k}", None)
            if val is not None:
                class_topk[name][k].append(val)

# ── Per-fold top-k table ──────────────────────────────────────────────────────
k_header = "".join(f"  top{k:<3}" for k in k_values)
print()
print(f"  {'Fold':<10} {'Windows':>8}{k_header}")
print("  " + "-" * (20 + 7 * len(k_values)))
for f in fold_rows:
    vals = "".join(f"  {f.get(f'top{k}', 0):>6.4f}" for k in k_values)
    print(f"  {f['fold']:<10} {f['n_windows']:>8}{vals}")

# Mean row
print("  " + "-" * (20 + 7 * len(k_values)))
means = {}
for k in k_values:
    col = [f.get(f"top{k}", 0) for f in fold_rows]
    means[k] = sum(col) / max(1, len(col))
mean_vals = "".join(f"  {means[k]:>6.4f}" for k in k_values)
print(f"  {'MEAN':<10} {total_windows:>8}{mean_vals}")

# Gap to top-1
print()
print(f"  Top-k gain over top-1 (mean):")
top1_mean = means.get(1, means.get(min(k_values), 0))
for k in k_values:
    gain = means[k] - top1_mean
    print(f"    top{k:<3} = {means[k]:.4f}  ({gain:+.4f} vs top1)")

# ── Per-class top-k table ─────────────────────────────────────────────────────
print()
print(f"  Per-class mean top-k accuracy (averaged across folds):")
k_hdr = "".join(f"  top{k:<2}" for k in k_values)
print(f"  {'Class':<20} {'N (total)':>10}{k_hdr}")
print("  " + "-" * (32 + 6 * len(k_values)))
rows_sorted = sorted(
    class_topk.items(),
    key=lambda x: -sum(x[1].get(min(k_values), [0])) / max(1, len(x[1].get(min(k_values), [1])))
)
for name, kdict in rows_sorted:
    n_total = sum(len(kdict.get(k, [])) for k in k_values) // max(1, len(k_values))
    vals = "".join(
        f"  {sum(kdict.get(k, [0])) / max(1, len(kdict.get(k, [1]))):>5.3f}"
        for k in k_values
    )
    print(f"  {name:<20} {n_total:>10}{vals}")

# ── Save aggregate JSON ───────────────────────────────────────────────────────
agg = {
    "topk_values": k_values,
    "mean_topk": {f"top{k}": round(means[k], 4) for k in k_values},
    "topk_gain_vs_top1": {
        f"top{k}": round(means[k] - means.get(1, means.get(min(k_values), 0)), 4)
        for k in k_values
    },
    "per_fold": fold_rows,
    "per_class_mean": {
        name: {
            f"top{k}": round(
                sum(kdict.get(k, [0])) / max(1, len(kdict.get(k, [1]))), 4
            )
            for k in k_values
        }
        for name, kdict in class_topk.items()
    },
}
out = Path("$OUT_DIR") / "topk_summary.json"
out.write_text(json.dumps(agg, indent=2))
print(f"\n  Saved -> {out}")
PY

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "[topk] Files to pull from server:"
echo ""
echo "  ${OUT_DIR}/topk_summary.json          <- mean top-k + per-class"
echo "  ${OUT_DIR}/fold*/eval_summary.json    <- per-fold topk_accuracy + per_class_topk"
echo "  ${OUT_DIR}/fold*/confusion_matrix.png"
echo "  logs/topk_eval_fold*.log"
echo "  logs/slurm_imu_topk_eval_${SLURM_JOB_ID:-local}.out"
echo "============================================================"
echo "[topk] finished: $(date)"
