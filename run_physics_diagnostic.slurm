#!/bin/bash
# =============================================================================
# run_physics_diagnostic.slurm
# =============================================================================
# Diagnostic sweep: does the physics pipeline help, and which scorer
# contributes? Runs 5 evaluate.py configs sequentially inside one GPU job,
# all sharing the same seed/trial sampling so per-trial outputs across
# configs are directly comparable.
#
# Configs (k=3, n_trials=200, n_windows=1, soft mode):
#   1. baseline_classifier_only   weights 0   0 0   0      (pure classifier)
#   2. pure_kin                   weights 1.0 0 0   0      (kinematic only)
#   3. pure_acc                   weights 0   0 1.0 0      (acceleration only)
#   4. pure_jointlim_soft         weights 0   0 0   1.0    (anatomical only)
#   5. default_blend              weights 0.4 0 0.4 0.2    (combined)
#
# Outputs:
#   ${OUT_BASE}/<tag>/eval_summary.json
#   ${OUT_BASE}/<tag>/physics_rerank_predictions.jsonl   (per-trial diagnostics)
#   ${OUT_BASE}/<tag>/run.log
#
# Run analyze_physics_diagnostic.py over ${OUT_BASE} afterwards to produce
# the attribution + ceiling + stratified-by-difficulty report.
#
# Each config caches its eval_summary.json — resubmitting after a timeout
# resumes from the last completed config.
#
# Usage
# -----
#   sbatch run_physics_diagnostic.slurm
#
# Expected wall time: ~30 min/config * 5 = ~2.5h, well under 5h budget.
# =============================================================================

#SBATCH --job-name=imu_physics_diag
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p h200x4

# Note: deliberately NOT using `set -e` — we want a failed individual
# config to leave its partial output and let the analyzer still run on
# whatever did succeed. We still use `set -u` (unset vars) and `pipefail`.
set -uo pipefail

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
OUT_BASE="${OUT_BASE:-results/penalty_sweep/physics_diagnostic_$(date +%Y%m%d_%H%M%S)}"
N_TRIALS="${N_TRIALS:-200}"
PHYSICS_K="${PHYSICS_K:-3}"
N_SENSORS="${N_SENSORS:-2,3,4,5}"
SEED="${SEED:-0}"
BASE_FILTERS="${BASE_FILTERS:-128}"
CALIBRATION_DIR="${CALIBRATION_DIR:-calibration}"
STATS_DIR="${STATS_DIR:-stats}"

mkdir -p logs "$OUT_BASE"

echo "============================================================"
echo "[diag] job          : ${SLURM_JOB_ID:-local}"
echo "[diag] python       : $PYTHON_BIN"
echo "[diag] checkpoint   : $CKPT"
echo "[diag] data         : $DATA_NPZ"
echo "[diag] out_base     : $OUT_BASE"
echo "[diag] n_trials     : $N_TRIALS"
echo "[diag] physics_k    : $PHYSICS_K"
echo "[diag] n_sensors    : $N_SENSORS"
echo "[diag] base_filters : $BASE_FILTERS"
echo "[diag] started      : $(date)"
echo "============================================================"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "$CKPT" ]]; then
  echo "[diag] ERROR: checkpoint not found: $CKPT"; exit 1
fi
if [[ ! -f "$DATA_NPZ" ]]; then
  echo "[diag] ERROR: dataset not found: $DATA_NPZ"; exit 1
fi
"$PYTHON_BIN" -c "import torch; print('[diag] CUDA available :', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

# ── Config table ──────────────────────────────────────────────────────────────
# Each row: "tag|w_kin w_grav w_acc w_joint"
CONFIGS=(
  "baseline_classifier_only|0.0 0.0 0.0 0.0"
  "pure_kin|1.0 0.0 0.0 0.0"
  "pure_acc|0.0 0.0 1.0 0.0"
  "pure_jointlim_soft|0.0 0.0 0.0 1.0"
  "default_blend|0.4 0.0 0.4 0.2"
)

run_config () {
  local tag="$1"
  local weights="$2"
  local out_dir="${OUT_BASE}/${tag}"
  mkdir -p "$out_dir"

  if [[ -f "${out_dir}/eval_summary.json" ]]; then
    echo "[diag] CACHED  ${tag}  (eval_summary.json exists, skipping)"
    return 0
  fi

  echo ""
  echo "[diag] -------------------------------------------------------------"
  echo "[diag] RUNNING  ${tag}   weights=(${weights})"
  echo "[diag] started  $(date)"
  local t0=$(date +%s)

  # shellcheck disable=SC2086
  "$PYTHON_BIN" evaluate.py \
      --checkpoint "$CKPT" \
      --data       "$DATA_NPZ" \
      --out_dir    "$out_dir" \
      --rerank_windows 0 \
      --physics_rerank \
      --physics_calibration "$CALIBRATION_DIR" \
      --physics_stats       "$STATS_DIR" \
      --rerank_n_sensors    "$N_SENSORS" \
      --rerank_n_windows    1 \
      --rerank_n_trials     "$N_TRIALS" \
      --physics_k           "$PHYSICS_K" \
      --rerank_seed         "$SEED" \
      --physics_weights     $weights \
      --base_filters        "$BASE_FILTERS" \
      >"${out_dir}/run.log" 2>&1
  local rc=$?

  local t1=$(date +%s)
  if [[ $rc -ne 0 ]]; then
    echo "[diag] FAILED  ${tag}   rc=${rc}  ($((t1 - t0))s)  log: ${out_dir}/run.log"
    tail -n 30 "${out_dir}/run.log" | sed 's/^/[diag][failtail] /'
    return $rc
  fi
  echo "[diag] done     ${tag}   ($((t1 - t0))s)"
  return 0
}

# ── Run all configs ───────────────────────────────────────────────────────────
FAILED=()
for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"
  weights="${entry##*|}"
  run_config "$tag" "$weights" || FAILED+=("$tag")
done

if (( ${#FAILED[@]} > 0 )); then
  echo ""
  echo "[diag] WARNING: ${#FAILED[@]} config(s) failed: ${FAILED[*]}"
  echo "[diag] Continuing to analyzer with whatever output is present."
fi

# ── Run the analyzer ──────────────────────────────────────────────────────────
echo ""
echo "[diag] -------------------------------------------------------------"
echo "[diag] Running analyzer over ${OUT_BASE}"
"$PYTHON_BIN" analyze_physics_diagnostic.py "$OUT_BASE" || {
  echo "[diag] WARNING: analyzer failed (results still on disk for manual inspection)"
}

echo ""
echo "============================================================"
echo "[diag] Files to pull from server:"
echo "  ${OUT_BASE}/diagnostic_report.json    <- attribution + ceiling + stratified"
echo "  ${OUT_BASE}/diagnostic_report.md      <- printed markdown summary"
echo "  ${OUT_BASE}/<tag>/eval_summary.json   <- raw per-config summaries"
echo "  ${OUT_BASE}/<tag>/physics_rerank_predictions.jsonl"
echo "  logs/slurm_imu_physics_diag_${SLURM_JOB_ID:-local}.out"
echo "============================================================"
echo "[diag] finished: $(date)"
