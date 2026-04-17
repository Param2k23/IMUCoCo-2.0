# AGENTS.md

## Repository snapshot (verified)
- Single-package Python project for IMU sensor-location classification across 24 body regions.
- Core pipeline is script-based, not a package: `preprocess_vimu.py` -> `train.py` -> `evaluate.py`.
- `preprocess_amass.py` is a legacy/older path and is not aligned with the current target workflow.
- Model definitions live in `model.py` (`resnet` default, `cnn` baseline); label schema and spatial-error helpers live in `smpl_regions.py`.
- No CI, no linter/typecheck/test config files in repo; validation is by running scripts.

## Environment and dependencies
- `requirements.txt` intentionally excludes PyTorch; install `torch` separately for your CUDA/CPU target.
- README paths are Windows-specific defaults, but scripts run on Linux if you pass explicit CLI paths.
- Generated artifacts are ignored by git: `data/`, `checkpoints/`, `results/`.

## High-value command patterns (Linux/CUDA)
- Full sequential commands for both current pipelines are in `commmands`.
- For a new environment/machine setup, run `./smoke_commands` first; a successful smoke run is the fastest verification that preprocessing, training, checkpointing, normalization-stat loading, and evaluation are all working end-to-end.
- `./smoke_commands` does not require GPU access; smoke training is compatible with CPU fallback.
- Single-subject conversion (ignore source split):
  - `python preprocess_vimu.py --mode single_subject ... --out_train data/processed/single_subject_train.npz --out_test data/processed/single_subject_test.npz`
- Predefined train/test conversion:
  - `python preprocess_vimu.py --mode predefined_split ... --out_train data/processed/full_train.npz --out_test data/processed/full_test.npz`
- Train fixed split:
  - `python train.py --mode fixed_split --train_data <train.npz> --val_data <test.npz> --out_dir <ckpt_dir> --device cuda`
- Evaluate:
  - `python evaluate.py --checkpoint <ckpt_dir>/best_model.pt --data <test.npz> --out_dir <results_dir> --vote_k 5`

## Data contract and interfaces
- Training/eval `.npz` must contain: `X` `(N,9,T)` float32 (`r6d_0..r6d_5, ax, ay, az`), `y` `(N,)` int labels `0..23`, `subject_ids` `(N,)` int for LOSO split.
- `train.py` performs z-score normalization using train-split stats and writes:
  - weights-only checkpoint: `best_model.pt` or `best_model_fold{k}.pt`
  - separate normalization stats: `normalization_stats.pt` or `normalization_stats_fold{k}.pt`
- `evaluate.py` loads model weights with `weights_only=True` and loads normalization stats from the corresponding `normalization_stats*.pt` file.

## Gotchas agents usually miss
- `--smoke_test` in `train.py` caps epochs to 5 and allows CPU fallback when CUDA is unavailable; non-smoke runs require CUDA.
- `train.py --mode fixed_split` expects `--train_data` and `--val_data`; `--mode loso` uses `--data` (+ optional `--max_folds`).
- `evaluate.py` now fails fast if normalization stats file is missing/malformed or if channel count mismatches.
- `evaluate.py --smoke` only tests majority-vote logic; it does not run model inference.

## Current direction (from user guidance)
- Near-term work is **not** AMASS-first: prioritize Linux+CUDA training with `preprocess_vimu.py` and custom dataset integration.
- Primary optimization targets are research baseline quality using both accuracy and spatial error.
