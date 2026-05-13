#!/bin/bash
# =============================================================================
# run_penalty_sweep.slurm
# =============================================================================
# Purpose
# -------
#   Systematically explores the effect of spatial penalty weights and learning
#   rate on LOSO accuracy, starting from the 86% fixed-penalty baseline
#   (neighbor_weight=0.3, lr_weight=0.5, lr=1e-3).
#
#   Axes swept:
#     A) neighbor_weight (spatial neighbor penalty): 0.0, 0.1, 0.2, 0.3, 0.5, 0.7
#     B) lr_weight       (left-right mirror penalty): 0.0, 0.3, 0.5, 1.0
#     C) learning rate   (LR): 3e-4, 7e-4, 1e-3, 2e-3
#
#   Trials are organised as:
#     Phase 1 — penalty_vs_baseline: vary neighbor_weight at fixed LR=1e-3
#               → isolates the spatial penalty effect vs plain CE
#     Phase 2 — lr_vs_penalty: cross neighbour_weight × LR at fixed lr_weight
#               → finds the best (penalty, LR) combination
#     Phase 3 — lrw_ablation: vary lr_weight at the best (nbw, lr) from phase 2
#               → checks if stronger L/R penalty helps further
#
#   All phases use 5 LOSO folds (configurable) for ranking.
#   After the sweep the best config reruns on all folds.
#
# Usage
# -----
#   sbatch run_penalty_sweep.slurm          # real run (~10-14 h on h200x4)
#   SMOKE_TEST=1 bash run_penalty_sweep.slurm  # smoke: 1-fold, tiny epochs, ~10 min
#
# Env overrides (no file edits needed)
# -------------------------------------
#   MAX_FOLDS_SWEEP=3 sbatch ...            # fewer ranking folds (faster)
#   MAX_FOLDS_FINAL=5 sbatch ...            # limit final run too
#   EPOCHS=200        sbatch ...
#   BATCH_SIZE=64     sbatch ...
#   ARCH=resnet       sbatch ...
# =============================================================================

#SBATCH --job-name=imu_penalty_sweep
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -p h200x4

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p logs data/processed checkpoints/penalty_sweep results/penalty_sweep

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
BASE_FILTERS="${BASE_FILTERS:-128}"       # 128 = 5M-param model used for 86% baseline
CLASS_WEIGHTS="${CLASS_WEIGHTS:-sqrt_inv}"

EPOCHS="${EPOCHS:-300}"                  # same epoch depth as best prior run
BATCH_SIZE="${BATCH_SIZE:-128}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"

# Baseline values that achieved 86%
BASELINE_NBW="0.3"   # neighbor_weight
BASELINE_LRW="0.5"   # lr_weight
BASELINE_LR="1e-3"   # learning rate

MAX_FOLDS_SWEEP="${MAX_FOLDS_SWEEP:-5}"  # folds for ranking (5 = reliable signal)
MAX_FOLDS_FINAL="${MAX_FOLDS_FINAL:-0}"  # 0 = all folds for final run

VOTE_K="${VOTE_K:-5}"

SMOKE_TEST="${SMOKE_TEST:-0}"

# ── Smoke overrides ───────────────────────────────────────────────────────────
if [[ "$SMOKE_TEST" == "1" ]]; then
  MAX_FOLDS_SWEEP=1
  MAX_FOLDS_FINAL=1
  EPOCHS=10
  echo "[penalty] SMOKE_TEST=1 → 1-fold, ${EPOCHS} epochs, CPU fallback"
fi

echo "============================================================"
echo "[penalty] job          : ${SLURM_JOB_ID:-local}  (${SLURM_JOB_NAME:-bash_run})"
echo "[penalty] repo         : $ROOT_DIR"
echo "[penalty] python       : $PYTHON_BIN"
echo "[penalty] parquet_dir  : $PARQUET_DIR"
echo "[penalty] arch         : $ARCH  (base_filters=$BASE_FILTERS)"
echo "[penalty] class_weights: $CLASS_WEIGHTS"
echo "[penalty] epochs       : $EPOCHS  batch=$BATCH_SIZE  wd=$WEIGHT_DECAY"
echo "[penalty] sweep folds  : $MAX_FOLDS_SWEEP  |  final folds: ${MAX_FOLDS_FINAL} (0=all)"
echo "[penalty] baseline     : nbw=$BASELINE_NBW  lrw=$BASELINE_LRW  lr=$BASELINE_LR → 86% acc"
echo "[penalty] started      : $(date)"
echo "============================================================"

# =============================================================================
# STEP 1 — Preprocess (once)
# =============================================================================
echo ""
echo "[penalty] ── STEP 1/5: preprocess hf_parquet ───────────────────────────"
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
  --out_train data/processed/penalty_sweep_train.npz \
  --out_test  data/processed/penalty_sweep_test.npz \
  2>&1 | tee logs/preprocess_penalty_sweep.log

# =============================================================================
# STEP 2 — Merge into single LOSO dataset
# =============================================================================
echo ""
echo "[penalty] ── STEP 2/5: merge into LOSO dataset ─────────────────────────"
"$PYTHON_BIN" - <<'PY'
import numpy as np

tr = np.load("data/processed/penalty_sweep_train.npz")
te = np.load("data/processed/penalty_sweep_test.npz")
X    = np.concatenate([tr["X"],           te["X"]],           axis=0).astype(np.float32)
y    = np.concatenate([tr["y"],           te["y"]],           axis=0).astype(np.int64)
sids = np.concatenate([tr["subject_ids"], te["subject_ids"]], axis=0).astype(np.int64)

if X.ndim != 3 or X.shape[1] != 9:
    raise ValueError(f"Expected X shape (N,9,T), got {X.shape}")
subjects = np.unique(sids)
if len(subjects) <= 1:
    raise ValueError(f"LOSO requires >1 unique subject_id, found: {subjects.tolist()}")

np.savez_compressed("data/processed/penalty_sweep_dataset.npz", X=X, y=y, subject_ids=sids)
print("[penalty] merged X:", X.shape, "| classes:", len(np.unique(y)), "| subjects:", len(subjects))
PY

# =============================================================================
# STEP 3 — Penalty sweep
# =============================================================================
# CSV header: includes neighbor_weight and lr_weight as first-class columns
RESULTS_CSV="results/penalty_sweep/leaderboard.csv"
echo "trial,epochs,batch_size,lr,weight_decay,patience,loss_fn,lr_weight,neighbor_weight,class_weights,max_folds,mean_loso_acc" \
  > "$RESULTS_CSV"

# ------------------------------------------------------------------
# Helper: run_trial <name> <epochs> <bs> <lr> <wd> <loss> <lrw> <nbw> <cw> <bf> <folds>
#   Patience auto-scaled: floor(epochs/6), clamped [15, 90]
# ------------------------------------------------------------------
run_trial () {
  local NAME="$1"  EPOCHS_T="$2"  BS="$3"  LR="$4"  WD="$5"
  local LOSS="$6"  LRW="$7"  NBW="$8"  CW="$9"  BF="${10}"  FOLDS="${11}"

  local PAT=$(( EPOCHS_T / 6 ))
  PAT=$(( PAT < 15 ? 15 : PAT ))
  PAT=$(( PAT > 90 ? 90 : PAT ))

  local OUT_DIR="checkpoints/penalty_sweep/${NAME}"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "[penalty] ── trial=$NAME"
  echo "[penalty]    epochs=$EPOCHS_T  bs=$BS  lr=$LR  wd=$WD  pat=$PAT  folds=$FOLDS"
  echo "[penalty]    loss=$LOSS  lrw=$LRW  nbw=$NBW  cw=$CW  bf=$BF"

  local -a ARGS=(
    --mode loso
    --data data/processed/penalty_sweep_dataset.npz
    --out_dir "$OUT_DIR"
    --arch "$ARCH"
    --epochs "$EPOCHS_T"
    --batch_size "$BS"
    --lr "$LR"
    --weight_decay "$WD"
    --patience "$PAT"
    --loss_fn "$LOSS"
    --lr_weight "$LRW"
    --neighbor_weight "$NBW"
    --class_weights "$CW"
    --base_filters "$BF"
    --device cuda
    --max_folds "$FOLDS"
  )
  if [[ "$SMOKE_TEST" == "1" ]]; then
    cp data/processed/penalty_sweep_dataset.npz \
       data/processed/penalty_sweep_smoke_dataset.npz 2>/dev/null || true
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
  echo "[penalty] trial=$NAME  mean_loso_acc=$MEAN"
  echo "$NAME,$EPOCHS_T,$BS,$LR,$WD,$PAT,$LOSS,$LRW,$NBW,$CW,$FOLDS,$MEAN" >> "$RESULTS_CSV"
}

echo ""
echo "[penalty] ── STEP 3/5: penalty × LR sweep ──────────────────────────────"

if [[ "$SMOKE_TEST" == "1" ]]; then
  # ── Smoke: 2 tiny trials to verify wiring ──────────────────────────────────
  run_trial "smoke_baseline" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "$BASELINE_NBW" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "smoke_no_penalty" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "crossentropy" "0.0" "0.0" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"
else
  # ===========================================================================
  # Phase 1: neighbor_weight ablation at fixed LR=1e-3, lrw=0.5
  # Goal: isolate the effect of the spatial neighbor penalty (hip/thigh fix)
  # Baseline (nbw=0.3) should reproduce ~86%; nbw=0.0 is plain CE reference.
  # ===========================================================================
  echo ""
  echo "[penalty] ── Phase 1: neighbor_weight ablation (lr=${BASELINE_LR}, lrw=${BASELINE_LRW}) ──"

  # Reference: plain CrossEntropy (no custom loss at all)
  run_trial "ph1_ce_only" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "crossentropy" "0.0" "0.0" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # Custom loss but neighbor penalty OFF → only L/R mirror term
  run_trial "ph1_nbw0.0" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.0" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # Light neighbor penalty
  run_trial "ph1_nbw0.1" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.1" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph1_nbw0.2" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.2" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # Baseline (should reproduce ~86%)
  run_trial "ph1_nbw0.3_baseline" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.3" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # Stronger neighbor penalties — does pushing harder help?
  run_trial "ph1_nbw0.5" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.5" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph1_nbw0.7" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.7" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # ===========================================================================
  # Phase 2: LR × neighbor_weight grid
  # Best nbw from Phase 1 × three new LR values to see if LR shift helps
  # We test all four LR values against the baseline nbw and two key alternates
  # ===========================================================================
  echo ""
  echo "[penalty] ── Phase 2: LR × neighbor_weight grid ────────────────────────"

  # Low LR (3e-4) — sometimes helps avoid overshooting with a penalised loss
  run_trial "ph2_lr3e4_nbw0.0" "$EPOCHS" "$BATCH_SIZE" "3e-4" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.0" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph2_lr3e4_nbw0.3" "$EPOCHS" "$BATCH_SIZE" "3e-4" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.3" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph2_lr3e4_nbw0.5" "$EPOCHS" "$BATCH_SIZE" "3e-4" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.5" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # Medium LR (7e-4)
  run_trial "ph2_lr7e4_nbw0.0" "$EPOCHS" "$BATCH_SIZE" "7e-4" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.0" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph2_lr7e4_nbw0.3" "$EPOCHS" "$BATCH_SIZE" "7e-4" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.3" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph2_lr7e4_nbw0.5" "$EPOCHS" "$BATCH_SIZE" "7e-4" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.5" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # Higher LR (2e-3) — faster early progress but may overfit with heavy penalty
  run_trial "ph2_lr2e3_nbw0.0" "$EPOCHS" "$BATCH_SIZE" "2e-3" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.0" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph2_lr2e3_nbw0.3" "$EPOCHS" "$BATCH_SIZE" "2e-3" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.3" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph2_lr2e3_nbw0.5" "$EPOCHS" "$BATCH_SIZE" "2e-3" "$WEIGHT_DECAY" \
            "custom" "$BASELINE_LRW" "0.5" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # ===========================================================================
  # Phase 3: lr_weight ablation at the baseline (nbw=0.3, lr=1e-3)
  # Goal: check if increasing/decreasing the L/R mirror penalty changes anything
  # ===========================================================================
  echo ""
  echo "[penalty] ── Phase 3: lr_weight ablation (lr=${BASELINE_LR}, nbw=${BASELINE_NBW}) ──"

  # No L/R penalty (lrw=0), only neighbor penalty
  run_trial "ph3_lrw0.0" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "0.0" "$BASELINE_NBW" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # Lighter L/R penalty
  run_trial "ph3_lrw0.3" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "0.3" "$BASELINE_NBW" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  # Stronger L/R penalty — does pushing L/R discrimination harder help?
  run_trial "ph3_lrw1.0" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "1.0" "$BASELINE_NBW" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"

  run_trial "ph3_lrw2.0" "$EPOCHS" "$BATCH_SIZE" "$BASELINE_LR" "$WEIGHT_DECAY" \
            "custom" "2.0" "$BASELINE_NBW" "$CLASS_WEIGHTS" "$BASE_FILTERS" \
            "$MAX_FOLDS_SWEEP"
fi   # end smoke/real branch

# =============================================================================
# STEP 4 — Leaderboard + analysis
# =============================================================================
echo ""
echo "[penalty] ── STEP 4/5: leaderboard ─────────────────────────────────────"
column -t -s',' "$RESULTS_CSV" || cat "$RESULTS_CSV"

echo ""
echo "[penalty] ── Penalty × LR analysis ─────────────────────────────────────"
"$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("results/penalty_sweep/leaderboard.csv").open()))
if not rows:
    raise RuntimeError("Empty leaderboard.csv")

rows.sort(key=lambda r: float(r["mean_loso_acc"]), reverse=True)

print(f"\n{'Rank':<5} {'Trial':<30} {'LR':<8} {'nbw':<6} {'lrw':<6} {'Loss':<14} {'Mean LOSO Acc':>14}")
print("-" * 90)
for i, r in enumerate(rows, 1):
    print(f"  {i:<3} {r['trial']:<30} {r['lr']:<8} {r['neighbor_weight']:<6} "
          f"{r['lr_weight']:<6} {r['loss_fn']:<14} {float(r['mean_loso_acc']):>14.4f}")

best = rows[0]
baseline = next((r for r in rows if "baseline" in r["trial"]), None)
print()
print(f"  ► Best overall : {best['trial']}  acc={float(best['mean_loso_acc']):.4f}")
print(f"    nbw={best['neighbor_weight']}  lrw={best['lr_weight']}  lr={best['lr']}")
if baseline:
    delta = float(best["mean_loso_acc"]) - float(baseline["mean_loso_acc"])
    print(f"  ► vs baseline  : {'▲' if delta >= 0 else '▼'} {abs(delta)*100:+.3f}pp  "
          f"(baseline={float(baseline['mean_loso_acc']):.4f})")

# Phase summaries
print()
for phase in ["ph1", "ph2", "ph3"]:
    phase_rows = [r for r in rows if r["trial"].startswith(phase)]
    if not phase_rows:
        continue
    top = max(phase_rows, key=lambda r: float(r["mean_loso_acc"]))
    print(f"  Phase {phase[-1]} best: {top['trial']:<30} acc={float(top['mean_loso_acc']):.4f}"
          f"  nbw={top['neighbor_weight']}  lrw={top['lr_weight']}  lr={top['lr']}")
PY

# =============================================================================
# STEP 5 — Rerun best config on all folds
# =============================================================================
echo ""
echo "[penalty] ── STEP 5/5: write best_config.env ───────────────────────────"
"$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("results/penalty_sweep/leaderboard.csv").open()))
best = max(rows, key=lambda r: float(r["mean_loso_acc"]))
env = Path("results/penalty_sweep/best_config.env")
env.write_text(
    "\n".join([
        f"BEST_TRIAL={best['trial']}",
        f"BEST_EPOCHS={best['epochs']}",
        f"BEST_BATCH_SIZE={best['batch_size']}",
        f"BEST_LR={best['lr']}",
        f"BEST_WEIGHT_DECAY={best['weight_decay']}",
        f"BEST_PATIENCE={best['patience']}",
        f"BEST_LOSS_FN={best['loss_fn']}",
        f"BEST_LR_WEIGHT={best['lr_weight']}",
        f"BEST_NEIGHBOR_WEIGHT={best['neighbor_weight']}",
        f"BEST_CLASS_WEIGHTS={best['class_weights']}",
        f"BEST_MEAN_ACC={best['mean_loso_acc']}",
    ]) + "\n"
)
print("[penalty] best config →", best['trial'], f"acc={best['mean_loso_acc']}")
print(env.read_text())
PY

# shellcheck disable=SC1091
source results/penalty_sweep/best_config.env

BEST_OUT="checkpoints/penalty_sweep/best_final"
mkdir -p "$BEST_OUT"

echo "[penalty] full LOSO with best config: trial=$BEST_TRIAL  lr=$BEST_LR  nbw=$BEST_NEIGHBOR_WEIGHT  lrw=$BEST_LR_WEIGHT  acc=${BEST_MEAN_ACC}"

FINAL_ARGS=(
  --mode loso
  --data data/processed/penalty_sweep_dataset.npz
  --out_dir "$BEST_OUT"
  --arch "$ARCH"
  --epochs "$BEST_EPOCHS"
  --batch_size "$BEST_BATCH_SIZE"
  --lr "$BEST_LR"
  --weight_decay "$BEST_WEIGHT_DECAY"
  --patience "$BEST_PATIENCE"
  --loss_fn "$BEST_LOSS_FN"
  --lr_weight "$BEST_LR_WEIGHT"
  --neighbor_weight "$BEST_NEIGHBOR_WEIGHT"
  --class_weights "$BEST_CLASS_WEIGHTS"
  --base_filters "$BASE_FILTERS"
  --device cuda
)
if [[ "$MAX_FOLDS_FINAL" != "0" ]]; then
  FINAL_ARGS+=(--max_folds "$MAX_FOLDS_FINAL")
fi
if [[ "$SMOKE_TEST" == "1" ]]; then
  cp data/processed/penalty_sweep_dataset.npz \
     data/processed/penalty_sweep_smoke_dataset.npz 2>/dev/null || true
  FINAL_ARGS+=(--smoke_test)
fi

"$PYTHON_BIN" train.py "${FINAL_ARGS[@]}" \
  2>&1 | tee logs/train_penalty_sweep_best_final.log

# Print final training result
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
p = Path("checkpoints/penalty_sweep/best_final/loso_results.txt")
print("[penalty] ── Full LOSO training results ─────────────────────────────────")
print(p.read_text())
PY

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "[penalty] Artefacts:"
echo "  Leaderboard  : results/penalty_sweep/leaderboard.csv"
echo "  Best config  : results/penalty_sweep/best_config.env"
echo "  Final ckpts  : checkpoints/penalty_sweep/best_final/"
echo "  Train logs   : logs/train_ph*.log"
echo "  Slurm log    : logs/slurm_imu_penalty_sweep_${SLURM_JOB_ID:-local}.out"
echo "[penalty] finished: $(date)"
echo "============================================================"
