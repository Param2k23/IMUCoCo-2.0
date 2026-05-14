#!/usr/bin/env bash
# run_physics_topk_sweep.sh
# ========================
# Overnight launcher for the physics + top-k rerank sweep.
# Activates the venv, runs the orchestrator under nohup, prints PID + tail
# command. Resumable — re-running this script picks up where it left off
# (configs whose eval_summary.json already exists are skipped).
#
# Usage:
#   ./run_physics_topk_sweep.sh                                 # defaults
#   CKPT=path/to/best_model.pt ./run_physics_topk_sweep.sh      # override ckpt
#   N_TRIALS=1000 N_TRIALS_FINAL=5000 ./run_physics_topk_sweep.sh
#
# Stop:    kill the printed PID
# Inspect: tail -f sweep.log
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-${ROOT}/.venv/bin/python}"
CKPT="${CKPT:-checkpoints/smoke_single_subject/best_model.pt}"
DATA="${DATA:-data/single_subject_test.npz}"
OUT_BASE="${OUT_BASE:-results/physics_topk_sweep_$(date +%Y%m%d_%H%M%S)}"
N_TRIALS="${N_TRIALS:-500}"
N_TRIALS_FINAL="${N_TRIALS_FINAL:-3000}"
N_SENSORS="${N_SENSORS:-2,3,4,5}"
SEED="${SEED:-0}"
SKIP_STAGES="${SKIP_STAGES:-}"

LOG="${OUT_BASE}/sweep.log"
mkdir -p "${OUT_BASE}"

echo "Sweep configuration:"
echo "  checkpoint     = ${CKPT}"
echo "  data           = ${DATA}"
echo "  out_base       = ${OUT_BASE}"
echo "  n_trials       = ${N_TRIALS}  (final: ${N_TRIALS_FINAL})"
echo "  n_sensors      = ${N_SENSORS}"
echo "  log            = ${LOG}"
echo

# Sanity: GPU?
"${PY}" -c "import torch; print('CUDA available:', torch.cuda.is_available(), \
  '| device count:', torch.cuda.device_count())"

nohup "${PY}" run_physics_topk_sweep.py \
    --checkpoint    "${CKPT}" \
    --data          "${DATA}" \
    --out_base      "${OUT_BASE}" \
    --n_trials      "${N_TRIALS}" \
    --n_trials_final "${N_TRIALS_FINAL}" \
    --n_sensors     "${N_SENSORS}" \
    --seed          "${SEED}" \
    ${SKIP_STAGES:+--skip_stages "$SKIP_STAGES"} \
    > "${LOG}" 2>&1 &

PID=$!
echo "Sweep PID: ${PID}"
echo "Tail with:  tail -f ${LOG}"
echo "Stop with:  kill ${PID}"
echo "${PID}" > "${OUT_BASE}/sweep.pid"
