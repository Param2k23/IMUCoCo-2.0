#!/usr/bin/env bash
# run_physics_sweep.sh
# ====================
# End-to-end physics-verification diagnostics. Answers:
#   - Are the precomputed stats internally consistent?
#   - Do the three scorers actually separate true vs negative combos?
#   - Which scorer / sensor count carries signal worth pursuing?
#   - What is the next direction (per-region weights, fix kinematic, drop gravity)?
#
# Usage:
#   ./run_physics_sweep.sh [TRAIN_NPZ] [TEST_NPZ]
# Defaults to data/single_subject_train.npz and data/single_subject_test.npz.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-${ROOT}/.venv/bin/python}"
TRAIN_NPZ="${1:-data/single_subject_train.npz}"
TEST_NPZ="${2:-data/single_subject_test.npz}"
STATS_DIR="${STATS_DIR:-stats}"
CALIB_DIR="${CALIB_DIR:-calibration}"

mkdir -p "${STATS_DIR}" "${CALIB_DIR}" logs

ts() { date +%H:%M:%S; }
header() { echo; echo "================================================================"; echo "[$(ts)] $*"; echo "================================================================"; }

header "0. Environment"
"${PY}" --version
"${PY}" -c "import numpy, torch; print('numpy', numpy.__version__, ' torch', torch.__version__)"
echo "TRAIN_NPZ = ${TRAIN_NPZ}"
echo "TEST_NPZ  = ${TEST_NPZ}"
echo "STATS_DIR = ${STATS_DIR}"
echo "CALIB_DIR = ${CALIB_DIR}"

header "1. Rotation utility self-test"
"${PY}" rotation_utils.py

header "2. (Re)build calibration if missing"
if [[ ! -f "${CALIB_DIR}/region_sensor_rotmats.npy" ]]; then
  "${PY}" extract_calibration.py
else
  echo "  Calibration present; skipping."
fi

header "3. (Re)build training stats if missing"
need_stats=0
for f in joint_angle_limits.npy accel_distributions.npy gravity_distributions.npy; do
  if [[ ! -f "${STATS_DIR}/${f}" ]]; then
    need_stats=1
    echo "  missing: ${STATS_DIR}/${f}"
  fi
done
if [[ "${need_stats}" -eq 1 ]]; then
  "${PY}" compute_training_stats.py --train_data "${TRAIN_NPZ}" --output_dir "${STATS_DIR}"
else
  echo "  All training stats present; skipping."
fi

header "4. Stats sanity"
"${PY}" - <<'PYEOF'
import numpy as np, os
from smpl_regions import REGION_NAMES
sd = os.environ.get("STATS_DIR", "stats")
for f in ["joint_angle_limits.npy", "accel_distributions.npy", "gravity_distributions.npy"]:
    p = os.path.join(sd, f)
    obj = np.load(p, allow_pickle=True).item()
    if "joint_angle_limits" in f:
        print(f"  {f}: loose pairs={len(obj.get('loose',{}))} strict pairs={len(obj.get('strict',{}))}")
        # Are limits trivial (min == max)?
        trivial = 0
        for (p_,c_), lim in obj.get('loose',{}).items():
            if np.allclose(lim['min'], lim['max']):
                trivial += 1
        print(f"    pairs with trivial (min==max) limits: {trivial}")
    elif "accel" in f:
        pj = obj.get("per_joint", {})
        print(f"  {f}: per_joint regions={len(pj)} per_activity entries={len(obj.get('per_activity',{}))}")
        # singular cov?
        bad = []
        for r, d in pj.items():
            cov = d.get("cov")
            if cov is None: continue
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                bad.append(REGION_NAMES[r])
        if bad: print(f"    singular cov for: {bad}")
    else:
        print(f"  {f}: regions with gravity dist={len(obj)}")
PYEOF

header "5. Diagnostic snapshot (read-only inspection of scorer outputs)"
"${PY}" diagnose_scorers.py --data "${TEST_NPZ}" --stats_dir "${STATS_DIR}" --calibration "${CALIB_DIR}" || true

header "6. Existing weight-tuning (current implementation, baseline gap)"
"${PY}" tune_weights.py --data "${TEST_NPZ}" --stats_dir "${STATS_DIR}" --calibration "${CALIB_DIR}" \
  --n_sensors 3 --n_samples 30 | tee logs/tune_weights_baseline.log
"${PY}" tune_weights.py --data "${TEST_NPZ}" --stats_dir "${STATS_DIR}" --calibration "${CALIB_DIR}" \
  --n_sensors 5 --n_samples 30 | tee logs/tune_weights_5sensor.log

header "7. Full physics sweep + diagnostics"
# Use the train file so we have enough segments to compute stable AUCs.
"${PY}" physics_sweep.py \
  --data "${TRAIN_NPZ}" \
  --stats_dir "${STATS_DIR}" \
  --calibration "${CALIB_DIR}" \
  --n_sensors_sweep 2 3 4 5 \
  --n_segments_used 200 \
  --out_json "${STATS_DIR}/sweep_report.json" | tee logs/physics_sweep.log

header "8. Repeat sweep on TEST split (out-of-distribution sanity)"
"${PY}" physics_sweep.py \
  --data "${TEST_NPZ}" \
  --stats_dir "${STATS_DIR}" \
  --calibration "${CALIB_DIR}" \
  --n_sensors_sweep 2 3 4 5 \
  --n_segments_used 200 \
  --out_json "${STATS_DIR}/sweep_report_test.json" | tee logs/physics_sweep_test.log

header "9. Compact summary"
"${PY}" - <<'PYEOF'
import json, os, numpy as np
sd = os.environ.get("STATS_DIR", "stats")
for split, path in [("TRAIN", f"{sd}/sweep_report.json"), ("TEST", f"{sd}/sweep_report_test.json")]:
    if not os.path.exists(path):
        continue
    r = json.load(open(path))
    print(f"\n--- {split} ---")
    cr = r.get("combo_ranking", {})
    print(f"{'n':>2} | {'scorer':9s} | {'AUC rand':>9s} {'AUC offch':>10s} {'AUC LR':>7s} | {'gap rand':>9s}")
    for n in sorted(cr.keys(), key=int):
        for s in ("kinematic","gravity","accel"):
            x = cr[n][s]
            ar = x["vs_random"]["auc"]
            ao = x["vs_offchain_swap"]["auc"]
            al = x["vs_lr_swap"]["auc"]
            gr = x["vs_random"]["gap"]
            print(f"{n:>2} | {s:9s} | {ar:9.3f} {ao:10.3f} {al:7.3f} | {gr:+9.3f}")
PYEOF

header "DONE"
echo "Reports:"
echo "  - ${STATS_DIR}/sweep_report.json (train)"
echo "  - ${STATS_DIR}/sweep_report_test.json (test)"
echo "  - logs/physics_sweep.log"
echo "  - logs/physics_sweep_test.log"
echo "  - logs/tune_weights_baseline.log"
