# DATASET.md

## Dataset layout

### Original local layout (`.pt` + CSV)
- Data is organized by split (`train` / `test`) and variant (for example `*_real_imu_position_only`).
- Both split folders are expected to contain similarly structured `.pt` segment files in the full dataset.
- A companion CSV index (for example `DIP_IMU_train_real_imu_position_only.csv`) lists segment metadata.

### Hugging Face Parquet export (e.g. `spongie01/DIP-IMU-position-only`)
After `snapshot_download`, data often lives under `data/raw_dip/data/` as:
- `train-00000-of-00004.parquet`, … (all `train-*.parquet` shards are **read and merged** in sorted order)
- `test-00000-of-00001.parquet` (or any `test-*.parquet` matching `--test_glob`)

Each row is one segment. Expected columns (nested dicts / structs are supported):
- **`vimu`**: dict with `vimu` or `vimu_joints` → virtual IMU `(T, 24, 9)` (r6d + acc), same semantics as `.pt` `vimu_joints`.
- **`imu`**: optional dict with `imu` → real sensors `(T, 17, 9)`. Used only if `--merge_inputs blend_global_imu` (blended with a global mean over the 17 sensors; output remains 9 channels).
- **`joint`**: may contain `asp_position` etc.; not required for the current classifier (labels are region indices `0..23`).

Subject id for LOSO: use a column such as `subject_id`, `subject`, or `file_name` matching `s_<id>_...`. If none match, rows get `subject_id=0` (see logs).

Use:
`python preprocess_vimu.py --mode hf_parquet --parquet_dir data/raw_dip/data ...`

## Segment file naming
- Segment files follow: `s_<subject>_<sequence>_seg<id>.pt`.
- Example: `s_01_03_seg12.pt`
  - `01` -> subject id (string in filename)
  - `03` -> source sequence/recording id
  - `seg12` -> segmented chunk index

## `.csv` index file
- File: `data/DIP_IMU_train_real_imu_position_only.csv`
- Columns:
  - `dataset_name`
  - `file_name`
  - `length` (frames per segment; 300 in this subset)
  - `kinematic_energy` (motion magnitude proxy)
- Beware of duplicate CSV rows: this subset has **62 rows but only 31 unique `file_name` values** (so **31 duplicate rows**).

## `.pt` segment structure (observed schema)
Each `.pt` is a nested dictionary (PyTorch serialization) with this structure:

- `joint`
  - `orientation`: float tensor `(T, 24, 3, 3)`
  - `velocity`: float tensor `(T, 24, 3)`
  - `position`: float tensor `(T, 24, 3)`
  - `asp_position`: float tensor `(T, 24, 3)`
  - `asp_velocity`: float tensor `(T, 24, 3)`
- `imu`
  - `imu`: float tensor `(T, 17, 9)`
- `vimu`
  - `vimu_joints`: float tensor `(T, 24, 9)`
  - `vimu_mesh`: `None` in inspected sample
- `gt`
  - `pose_local`: float tensor `(T, 24, 3, 3)`
  - `tran`: `None` in inspected sample
  - `ft_contact`: int tensor `(T, 2)`

`T` is segment length (300 frames in this subset).

## Relationship to the 24-region classifier goal
- Your classifier target is one of 24 body regions (`0..23`), matching `smpl_regions.py`.
- Training/evaluation code in this repo (`train.py`, `evaluate.py`) expects a prebuilt `.npz` with:
  - `X`: float32 `(N, 9, window)`
  - `y`: int labels `(N,)` in `0..23`
  - `subject_ids`: int `(N,)`
- Therefore, this dataset is a **source format** and must be converted to that `.npz` contract before using the current pipeline.

## Recommended conversion contract
When building your custom-data adapter, produce:
- `X[n]`: one IMU window with 9 channels (`r6d_0..r6d_5, ax, ay, az`) and fixed `window` length.
- `y[n]`: region class id (`0..23`) for that window.
- `subject_ids[n]`: subject/group id used for LOSO split.

## Minimal loader example
```python
import torch

sample = torch.load("data/DIP_IMU_train_real_imu_position_only/s_01_03_seg0.pt", map_location="cpu")
vimu = sample["vimu"]["vimu_joints"]  # (T, 24, 9) -> r6d(6) + acc(3)
jpos = sample["joint"]["position"]    # (T, 24, 3)

print(vimu.shape, jpos.shape)
```

## Practical notes for this repo
- `data/`, `checkpoints/`, and `results/` are git-ignored outputs.
- De-duplicate CSV records by `file_name` before building your training index/sampler.
- Before training, validate the generated `.npz` shapes/dtypes against `train.py` expectations.
