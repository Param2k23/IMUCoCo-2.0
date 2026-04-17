# PLAN.md

## Goal (confirmed)
- Build a Linux+CUDA-first training pipeline for research baselines.
- Prioritize smoke tests and **custom dataset integration** (not AMASS-first).
- Optimize for both classification accuracy and spatial error.
- Use `vimu.vimu_joints` as input source with 9 channels (`r6d_0..r6d_5, ax, ay, az`).

## Purpose questions raised (answered + still open)
- Answered: immediate focus is baseline training quality, not deployment packaging.
- Answered: key metrics are per-window accuracy + spatial error.
- Answered: default runtime target is Linux with CUDA GPUs.
- Answered: `subject_ids` should represent the parsed subject token from filename (e.g., `s_01` -> 1).
- Answered: class taxonomy stays fixed at 24 regions (`0..23`) for baseline.
- Answered: baseline includes two data/training pipelines:
  1. single-subject split experiment (ignore source train/test partition)
  2. full-dataset predefined-split experiment (respect source train/test folders)
- Answered: CUDA policy: if CUDA is unavailable, allow CPU only when `--smoke_test`; otherwise raise an error.

## Next steps
1. Implement data conversion pipeline for `.pt` segments -> `.npz` using `vimu.vimu_joints[:, :, :9]`:
   - output contract: `X (N,9,T)` float32, `y (N,)` int `0..23`, `subject_ids (N,)` int.
   - one sample per `(segment, region)` where region index maps directly to class label.
2. Add single-subject pipeline:
   - collect one subject's files from both source splits,
   - ignore original train/test split,
   - create a smaller multi-class train/test experiment for that subject.
3. Add full-dataset predefined-split pipeline:
   - build train `.npz` from source train folder,
   - build test `.npz` from source test folder,
   - keep split boundaries unchanged.
4. Add dataset validation checks in conversion or validator utility:
   - keys/shapes/dtypes/range checks,
   - class coverage and subject count summary,
   - CSV deduplication by `file_name` before indexing.
5. Update `train.py` device behavior:
   - if CUDA unavailable and `--smoke_test`, fallback to CPU with warning,
   - if CUDA unavailable and not smoke, throw a clear error and exit.
6. Ensure training/evaluation support both experiment modes:
   - LOSO for subject-grouped workflows,
   - fixed train/test evaluation for predefined split workflow.
7. Create Linux runbook command chain for both pipelines: convert -> validate -> train -> evaluate.

## Potential improvements
- [FUTURE] Add augmentation stack (channel jitter, time masking, scale+bias perturbation, optional time-warp/mixup).
- [FUTURE] Add model upgrades (SE attention blocks, temporal dilations, optional split r6d/acc stem).
- [FUTURE] Add training upgrades (class-weighted/focal loss, AdamW warmup-cosine, EMA, stronger reproducibility controls).
- [FUTURE] Add richer evaluation outputs (top-k, per-class spatial error trends, calibration metrics ECE/NLL).
- [FUTURE] Add split controls beyond LOSO where subject identity may be weak/noisy.
- [FUTURE] Save normalization stats and full preprocessing config in checkpoints for external reproducibility.
- [FUTURE] Add lightweight automation (`run_pipeline.sh` or Make targets) for one-command smoke-to-eval flow on Linux.
- [FUTURE] Add minimal regression checks (data contract, checkpoint load, majority-vote logic).
