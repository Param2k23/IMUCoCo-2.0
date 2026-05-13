#!/bin/bash
# =============================================================================
# run_rerank_eval.slurm
# =============================================================================
# Temporal re-ranking evaluation script.
#
# Runs evaluate.py on ALL fold checkpoints and sweeps over:
#   - window sizes  : 1, 3, 5, 7, 10  (configurable via RERANK_WINDOWS)
#   - strategies    : prob_sum, topk_vote, majority  (configurable)
#
# For each fold, eval_summary.json gains a "temporal_reranking" block:
#   best_acc, best_config, gain_vs_top1, all_configs, per_class_best_config
#
# Produces:
#   results/.../rerank_summary.json   <- aggregated best config per fold + mean
#
# Usage
# -----
#   sbatch run_rerank_eval.slurm                         # penalty_sweep default
#
#   CKPT_DIR=checkpoints/epoch_sweep/best_final \
#     OUT_DIR=results/epoch_sweep/rerank_eval \
#     RERANK_WINDOWS=1,3,5,7,10 \
#     RERANK_STRATEGY=prob_sum,majority \
#     sbatch run_rerank_eval.slurm
#
# Runtime: ~15-25 min for 10 folds x 3 strategies x 5 window sizes
# =============================================================================

#SBATCH --job-name=imu_rerank_eval
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=01:30:00
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
OUT_DIR="${OUT_DIR:-results/penalty_sweep/rerank_eval}"
VOTE_K="${VOTE_K:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BASE_FILTERS="${BASE_FILTERS:-128}"
TOPK="${TOPK:-1,2,3,5,10}"

# Re-ranking specific
RERANK_WINDOWS="${RERANK_WINDOWS:-1,3,5,7,10}"
RERANK_STRATEGY="${RERANK_STRATEGY:-prob_sum,topk_vote,majority}"
RERANK_K_VOTE="${RERANK_K_VOTE:-3}"
# Set to "--rerank_causal" to use past-only window (streaming mode)
RERANK_CAUSAL_FLAG="${RERANK_CAUSAL_FLAG:-}"

mkdir -p logs "$OUT_DIR"

echo "============================================================"
echo "[rerank] job         : ${SLURM_JOB_ID:-local}  (${SLURM_JOB_NAME:-bash_run})"
echo "[rerank] python      : $PYTHON_BIN"
echo "[rerank] ckpt_dir    : $CKPT_DIR"
echo "[rerank] data        : $DATA_NPZ"
echo "[rerank] out_dir     : $OUT_DIR"
echo "[rerank] topk        : $TOPK"
echo "[rerank] rr_windows  : $RERANK_WINDOWS"
echo "[rerank] rr_strategy : $RERANK_STRATEGY"
echo "[rerank] rr_k_vote   : $RERANK_K_VOTE"
echo "[rerank] causal      : ${RERANK_CAUSAL_FLAG:-offline (centred)}"
echo "[rerank] started     : $(date)"
echo "============================================================"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[rerank] ERROR: checkpoint directory not found: $CKPT_DIR"
  exit 1
fi
if [[ ! -f "$DATA_NPZ" ]]; then
  echo "[rerank] ERROR: dataset not found: $DATA_NPZ"
  exit 1
fi

N_CKPTS=$(ls "$CKPT_DIR"/best_model_fold*.pt 2>/dev/null | wc -l)
echo "[rerank] found $N_CKPTS fold checkpoints"
if [[ "$N_CKPTS" -eq 0 ]]; then
  echo "[rerank] ERROR: no best_model_fold*.pt found in $CKPT_DIR"
  exit 1
fi

# ── Run evaluate.py on every fold checkpoint ──────────────────────────────────
for CKPT in "$CKPT_DIR"/best_model_fold*.pt; do
  FOLD_ID=$(echo "$CKPT" | grep -oP '(?<=fold)\d+')
  FOLD_OUT="${OUT_DIR}/fold${FOLD_ID}"
  mkdir -p "$FOLD_OUT"
  echo ""
  echo "[rerank] ── fold $FOLD_ID  →  $FOLD_OUT"

  "$PYTHON_BIN" evaluate.py \
    --checkpoint     "$CKPT" \
    --data           "$DATA_NPZ" \
    --out_dir        "$FOLD_OUT" \
    --vote_k         "$VOTE_K" \
    --batch_size     "$BATCH_SIZE" \
    --base_filters   "$BASE_FILTERS" \
    --topk           "$TOPK" \
    --rerank_windows "$RERANK_WINDOWS" \
    --rerank_strategy "$RERANK_STRATEGY" \
    --rerank_k_vote  "$RERANK_K_VOTE" \
    $RERANK_CAUSAL_FLAG \
    2>&1 | tee "logs/rerank_eval_fold${FOLD_ID}.log"
done

# ── Aggregate re-ranking results across folds ─────────────────────────────────
echo ""
echo "[rerank] ── Aggregating re-ranking results across $N_CKPTS folds ──────────"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from collections import defaultdict

out_dir    = Path("$OUT_DIR")
eval_dirs  = sorted(out_dir.glob("fold*"))
if not eval_dirs:
    print("[rerank] No fold* dirs found.")
    raise SystemExit(0)

fold_rows   = []
config_accs = defaultdict(list)   # config_name -> [acc per fold]
top1_accs   = []

for d in eval_dirs:
    j = d / "eval_summary.json"
    if not j.exists():
        print(f"  [warn] missing {j}")
        continue
    data  = json.loads(j.read_text())
    n_w   = data.get("n_windows", 0)
    top1  = data.get("per_window_accuracy", 0)
    rr    = data.get("temporal_reranking", {})

    top1_accs.append(top1)

    row = {
        "fold"        : d.name,
        "n_windows"   : n_w,
        "top1_acc"    : round(top1, 4),
        "best_rr_acc" : round(rr.get("best_acc", top1), 4),
        "best_config" : rr.get("best_config", "n/a"),
        "gain"        : round(rr.get("gain_vs_top1", 0), 4),
    }

    for cfg, acc in rr.get("all_configs", {}).items():
        config_accs[cfg].append(acc)

    fold_rows.append(row)

# ── Per-fold summary table ────────────────────────────────────────────────────
print()
print(f"  {'Fold':<10} {'Windows':>8} {'top-1':>7} {'best_rr':>8} {'gain':>7} {'best_config'}")
print("  " + "-" * 64)
for f in fold_rows:
    print(f"  {f['fold']:<10} {f['n_windows']:>8} {f['top1_acc']:>7.4f} "
          f"{f['best_rr_acc']:>8.4f} {f['gain']:>+7.4f}  {f['best_config']}")

mean_top1 = sum(top1_accs) / max(1, len(top1_accs))
print("  " + "-" * 64)
print(f"  {'MEAN':<10} {'':>8} {mean_top1:>7.4f}")

# ── Per-config mean accuracy table ───────────────────────────────────────────
print()
print("  Mean accuracy per (strategy, window) config — sorted best → worst:")
print(f"  {'Config':<22} {'Mean Acc':>10} {'vs top-1':>10} {'# folds':>8}")
print("  " + "-" * 54)
config_means = {
    cfg: sum(vals) / len(vals)
    for cfg, vals in config_accs.items()
}
for cfg, mean_acc in sorted(config_means.items(), key=lambda x: -x[1]):
    delta = mean_acc - mean_top1
    n_f   = len(config_accs[cfg])
    print(f"  {cfg:<22} {mean_acc:>10.4f} {delta:>+10.4f} {n_f:>8}")

# ── Best overall config ───────────────────────────────────────────────────────
if config_means:
    best_cfg = max(config_means, key=config_means.get)
    best_acc = config_means[best_cfg]
    print()
    print(f"  Best overall config : {best_cfg}")
    print(f"  Best mean accuracy  : {best_acc:.4f}  ({best_acc - mean_top1:+.4f} vs top-1)")

# ── Save aggregate JSON ───────────────────────────────────────────────────────
agg = {
    "mean_top1_acc"   : round(mean_top1, 4),
    "best_config"     : max(config_means, key=config_means.get) if config_means else None,
    "best_config_acc" : round(max(config_means.values()) if config_means else mean_top1, 4),
    "best_gain_vs_top1": round(
        max(config_means.values(), default=mean_top1) - mean_top1, 4
    ),
    "per_config_mean" : {k: round(v, 4) for k, v in
                         sorted(config_means.items(), key=lambda x: -x[1])},
    "per_fold"        : fold_rows,
}
out = Path("$OUT_DIR") / "rerank_summary.json"
out.write_text(json.dumps(agg, indent=2))
print(f"\n  Saved -> {out}")
PY

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "[rerank] Files to pull from server:"
echo ""
echo "  ${OUT_DIR}/rerank_summary.json         <- best config + mean table"
echo "  ${OUT_DIR}/fold*/eval_summary.json     <- per-fold temporal_reranking block"
echo "  logs/rerank_eval_fold*.log             <- per-fold detail"
echo "  logs/slurm_imu_rerank_eval_${SLURM_JOB_ID:-local}.out"
echo "============================================================"
echo "[rerank] finished: $(date)"
