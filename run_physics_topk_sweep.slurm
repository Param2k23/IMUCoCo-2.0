#!/bin/bash
# =============================================================================
# run_physics_topk_sweep.slurm
# =============================================================================
# Overnight CUDA sweep over (weights × physics_k × hard_threshold × n_windows)
# for the top-k + physics combo re-ranking pipeline.
#
# Wraps run_physics_topk_sweep.py — the 3-stage orchestrator (Stage A coarse
# grid → Stage B n_windows refinement → Stage C high-trial final eval).
#
# Each config reuses cached eval_summary.json, so re-submitting after a
# timeout / preemption resumes from the last completed config.
#
# What it produces:
#   ${OUT_BASE}/stageA/<tag>/eval_summary.json    per-config raw metrics
#   ${OUT_BASE}/stageA_summary.csv                aggregated table (sortable)
#   ${OUT_BASE}/stageA_winners.json               top-5 by per_sensor / exact
#   ${OUT_BASE}/stageB_summary.csv                refinement results
#   ${OUT_BASE}/stageC/...                        high-trial final eval
#   ${OUT_BASE}/final_report.json                 selected config + metrics
#   ${OUT_BASE}/sweep.log                         orchestrator stdout
#
# Usage
# -----
#   sbatch run_physics_topk_sweep.slurm                           # defaults
#
#   CKPT=checkpoints/epoch_sweep/best_final/best_model_fold0.pt \
#     DATA_NPZ=data/processed/epoch_sweep_dataset.npz \
#     OUT_BASE=results/epoch_sweep/physics_topk_sweep \
#     N_TRIALS=1000 N_TRIALS_FINAL=5000 \
#     sbatch run_physics_topk_sweep.slurm
#
#   # Limit to one stage:
#   SKIP_STAGES=B,C sbatch run_physics_topk_sweep.slurm
#
# Runtime: ~2-4h on a single H200 at default N_TRIALS=500.
#          Stage A dominates (180 configs); stages B+C are <30 min combined.
# =============================================================================

#SBATCH --job-name=imu_physics_sweep
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

# ── Python binary ─────────────────────────────────────────────────────────────
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

# ── Configurable overrides ────────────────────────────────────────────────────
CKPT="${CKPT:-checkpoints/penalty_sweep/best_final/best_model_fold0.pt}"
DATA_NPZ="${DATA_NPZ:-data/processed/penalty_sweep_dataset.npz}"
OUT_BASE="${OUT_BASE:-results/penalty_sweep/physics_topk_sweep_$(date +%Y%m%d_%H%M%S)}"
N_TRIALS="${N_TRIALS:-500}"
N_TRIALS_FINAL="${N_TRIALS_FINAL:-3000}"
N_SENSORS="${N_SENSORS:-2,3,4,5}"
SEED="${SEED:-0}"
SKIP_STAGES="${SKIP_STAGES:-}"

mkdir -p logs "$OUT_BASE"

echo "============================================================"
echo "[sweep] job          : ${SLURM_JOB_ID:-local}  (${SLURM_JOB_NAME:-bash_run})"
echo "[sweep] python       : $PYTHON_BIN"
echo "[sweep] checkpoint   : $CKPT"
echo "[sweep] data         : $DATA_NPZ"
echo "[sweep] out_base     : $OUT_BASE"
echo "[sweep] n_trials     : $N_TRIALS  (final: $N_TRIALS_FINAL)"
echo "[sweep] n_sensors    : $N_SENSORS"
echo "[sweep] skip_stages  : ${SKIP_STAGES:-<none>}"
echo "[sweep] started      : $(date)"
echo "============================================================"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "$CKPT" ]]; then
  echo "[sweep] ERROR: checkpoint not found: $CKPT"
  exit 1
fi
if [[ ! -f "$DATA_NPZ" ]]; then
  echo "[sweep] ERROR: dataset not found: $DATA_NPZ"
  exit 1
fi

# Confirm CUDA visibility
"$PYTHON_BIN" -c "import torch; print('[sweep] CUDA available :', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

# ── Run the orchestrator ──────────────────────────────────────────────────────
"$PYTHON_BIN" run_physics_topk_sweep.py \
    --checkpoint     "$CKPT" \
    --data           "$DATA_NPZ" \
    --out_base       "$OUT_BASE" \
    --n_trials       "$N_TRIALS" \
    --n_trials_final "$N_TRIALS_FINAL" \
    --n_sensors      "$N_SENSORS" \
    --seed           "$SEED" \
    ${SKIP_STAGES:+--skip_stages "$SKIP_STAGES"} \
    2>&1 | tee "${OUT_BASE}/sweep.log"

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "[sweep] Files to pull from server:"
echo ""
echo "  ${OUT_BASE}/final_report.json          <- recommended config + metrics"
echo "  ${OUT_BASE}/stageA_summary.csv         <- coarse-grid table"
echo "  ${OUT_BASE}/stageA_winners.json        <- top-5 per metric"
echo "  ${OUT_BASE}/stageB_summary.csv         <- n_windows refinement"
echo "  ${OUT_BASE}/stageC/*/eval_summary.json <- final high-trial eval"
echo "  ${OUT_BASE}/sweep.log"
echo "  logs/slurm_imu_physics_sweep_${SLURM_JOB_ID:-local}.out"
echo "============================================================"
echo "[sweep] finished: $(date)"
