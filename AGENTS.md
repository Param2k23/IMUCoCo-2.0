# AGENTS.md

## Repository snapshot (verified)
- Single-package Python project for IMU sensor-location classification across 24 body regions.
- Core pipeline is script-based, not a package: `preprocess_amass.py` -> `train.py` -> `evaluate.py`.
- Model definitions live in `model.py` (`resnet` default, `cnn` baseline); label schema and spatial-error helpers live in `smpl_regions.py`.
- No CI, no linter/typecheck/test config files in repo; validation is by running scripts.

## Environment and dependencies
- `requirements.txt` intentionally excludes PyTorch; install `torch` separately for your CUDA/CPU target.
- README paths are Windows-specific defaults, but scripts run on Linux if you pass explicit CLI paths.
- Generated artifacts are ignored by git: `data/`, `checkpoints/`, `results/`.

## High-value command patterns (Linux/CUDA)
- Smoke data generation: `python preprocess_amass.py --smoke_test --output data/dataset.npz` (writes `data/smoke_dataset.npz` via string replacement).
- Train on smoke/custom data: `python train.py --data data/smoke_dataset.npz --out_dir checkpoints --device cuda --epochs 5`.
- Evaluate checkpoint: `python evaluate.py --checkpoint checkpoints/best_model_fold0.pt --data data/smoke_dataset.npz --out_dir results --vote_k 5`.
- Preprocess timing-only mode exists for AMASS: `python preprocess_amass.py --estimate_only ...`.

## Data contract and interfaces
- Training/eval `.npz` must contain: `X` `(N,6,T)` float32, `y` `(N,)` int labels `0..23`, `subject_ids` `(N,)` int for LOSO split.
- `train.py` performs per-fold z-score normalization using train split stats and saves checkpoints with keys: `model`, `arch`, `fold`, `test_subj`, metrics.
- `evaluate.py` expects checkpoint `model` state dict (without `module.` prefix); this is handled in `train.py` by `_state_dict_to_save`.

## Gotchas agents usually miss
- `--smoke_test` in `train.py` caps epochs to 5 but does **not** force a single LOSO fold; actual fold count depends on `subject_ids` in the dataset.
- `preprocess_amass.py --smoke_test` rewrites output path by replacing `dataset.npz`; if your `--output` does not contain that substring, smoke output may not be renamed as expected.
- `evaluate.py --smoke` only tests majority-vote logic; it does not run model inference.

## Current direction (from user guidance)
- Near-term work is **not** AMASS-first: prioritize a Linux+CUDA training pipeline for smoke tests + custom dataset integration.
- Primary optimization targets are research baseline quality using both accuracy and spatial error.
