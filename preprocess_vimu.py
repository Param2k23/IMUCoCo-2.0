"""
preprocess_vimu.py
==================
Convert DIP-style segmented .pt files (vimu_joints) into train/eval .npz files
for IMUCoCo classification using 9 channels from vimu_joints.

Important channel semantics in this repository:
- vimu_joints has 9 features per region: [orientation_r6d(6), acceleration_xyz(3)]
- The preprocessed X keeps all 9 channels (r6d + acc)

Two modes:
1) single_subject  -> build train/test split from one subject only
2) predefined_split -> build train/test using source train/test folders
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)

SUBJECT_RE = re.compile(r"^s_(\d+)_")


@dataclass
class SampleRecord:
    file_path: str
    subject_id: int


def parse_subject_id(file_name: str) -> int:
    m = SUBJECT_RE.match(file_name)
    if not m:
        raise ValueError(f"Cannot parse subject id from filename: {file_name}")
    return int(m.group(1))


def list_pt_files(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted(
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if name.endswith(".pt")
    )


def dedup_csv_filenames(csv_path: Optional[str]) -> Optional[set[str]]:
    if not csv_path:
        return None
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV index not found: {csv_path}")

    seen: set[str] = set()
    unique: set[str] = set()
    total = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "file_name" not in (reader.fieldnames or []):
            raise ValueError(f"CSV missing 'file_name' column: {csv_path}")
        for row in reader:
            total += 1
            fname = row["file_name"].strip()
            if not fname:
                continue
            if fname in seen:
                continue
            seen.add(fname)
            unique.add(fname)

    log.info(
        "CSV %s rows=%d unique_file_name=%d deduped=%d",
        csv_path,
        total,
        len(unique),
        total - len(unique),
    )
    return unique


def build_records(
    data_dir: str,
    allowed_filenames: Optional[set[str]] = None,
    subject_filter: Optional[int] = None,
) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for p in list_pt_files(data_dir):
        fname = os.path.basename(p)
        if allowed_filenames is not None and fname not in allowed_filenames:
            continue
        sid = parse_subject_id(fname)
        if subject_filter is not None and sid != subject_filter:
            continue
        records.append(SampleRecord(file_path=p, subject_id=sid))
    return records


def extract_segment_tensor(file_path: str) -> np.ndarray:
    data = torch.load(file_path, map_location="cpu")
    vimu = data["vimu"]["vimu_joints"]
    if not torch.is_tensor(vimu):
        raise ValueError(f"vimu_joints is not tensor in {file_path}")
    if vimu.ndim != 3:
        raise ValueError(
            f"vimu_joints must be (T,24,9), got {tuple(vimu.shape)} in {file_path}"
        )
    t, n_regions, n_feat = vimu.shape
    if n_regions != 24:
        raise ValueError(f"Expected 24 regions, got {n_regions} in {file_path}")
    if n_feat < 9:
        raise ValueError(f"Expected at least 9 features, got {n_feat} in {file_path}")
    arr = (
        vimu[:, :, :9].detach().cpu().numpy().astype(np.float32)
    )  # (T, 24, 9) = orientation_r6d + acceleration_xyz
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {file_path}")
    if t <= 0:
        raise ValueError(f"Empty segment length in {file_path}")
    return arr


def records_to_npz_arrays(
    records: Sequence[SampleRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    sid_list: List[int] = []

    for rec in records:
        seg = extract_segment_tensor(rec.file_path)  # (T, 24, 9)
        # one sample per region
        for region in range(24):
            sample = np.transpose(seg[:, region, :], (1, 0))  # (9, T)
            X_list.append(sample)
            y_list.append(region)
            sid_list.append(rec.subject_id)

    if not X_list:
        raise ValueError("No samples collected for this split")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    subject_ids = np.asarray(sid_list, dtype=np.int64)
    return X, y, subject_ids


def validate_arrays(
    X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, name: str
) -> None:
    if X.ndim != 3:
        raise ValueError(f"{name}: X must be 3D (N,9,T), got shape={X.shape}")
    if X.shape[1] != 9:
        raise ValueError(f"{name}: X second dim must be 9, got {X.shape[1]}")
    if y.ndim != 1 or subject_ids.ndim != 1:
        raise ValueError(f"{name}: y and subject_ids must be 1D")
    if len(X) != len(y) or len(y) != len(subject_ids):
        raise ValueError(
            f"{name}: length mismatch X={len(X)} y={len(y)} subject_ids={len(subject_ids)}"
        )
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError(f"{name}: X must be float dtype, got {X.dtype}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"{name}: y must be integer dtype, got {y.dtype}")
    if not np.issubdtype(subject_ids.dtype, np.integer):
        raise ValueError(
            f"{name}: subject_ids must be integer dtype, got {subject_ids.dtype}"
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: X contains non-finite values")
    if y.min() < 0 or y.max() > 23:
        raise ValueError(
            f"{name}: y must be in [0, 23], got min={y.min()} max={y.max()}"
        )


def save_npz(
    path: str, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, label: str
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, X=X, y=y, subject_ids=subject_ids)
    classes = np.unique(y)
    subjects = np.unique(subject_ids)
    log.info("Saved %s -> %s", label, path)
    log.info("  X shape=%s dtype=%s", X.shape, X.dtype)
    log.info("  y shape=%s classes=%d", y.shape, len(classes))
    log.info("  subject_ids shape=%s subjects=%d", subject_ids.shape, len(subjects))


def stratified_split_indices(
    y: np.ndarray, test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    per_class: Dict[int, List[int]] = {}
    for idx, cls in enumerate(y.tolist()):
        per_class.setdefault(int(cls), []).append(idx)

    test_idx: List[int] = []
    train_idx: List[int] = []
    for cls, idxs in per_class.items():
        idxs_copy = list(idxs)
        rng.shuffle(idxs_copy)
        n_test = max(1, int(round(len(idxs_copy) * test_ratio)))
        n_test = min(n_test, len(idxs_copy) - 1) if len(idxs_copy) > 1 else 1
        if n_test <= 0:
            n_test = 1
        test_part = idxs_copy[:n_test]
        train_part = idxs_copy[n_test:]
        if not train_part:
            train_part = test_part[:1]
            test_part = test_part[1:]
        test_idx.extend(test_part)
        train_idx.extend(train_part)

    return np.array(sorted(train_idx), dtype=np.int64), np.array(
        sorted(test_idx), dtype=np.int64
    )


def run_single_subject(args: argparse.Namespace) -> None:
    allowed_train = dedup_csv_filenames(args.train_csv)
    allowed_test = dedup_csv_filenames(args.test_csv)

    records = []
    records.extend(
        build_records(args.train_dir, allowed_train, subject_filter=args.subject_id)
    )
    records.extend(
        build_records(args.test_dir, allowed_test, subject_filter=args.subject_id)
    )

    if not records:
        raise ValueError(f"No records found for subject {args.subject_id}")

    X_all, y_all, sid_all = records_to_npz_arrays(records)
    validate_arrays(X_all, y_all, sid_all, "single_subject_all")

    tr_idx, te_idx = stratified_split_indices(
        y_all, test_ratio=args.test_ratio, seed=args.seed
    )
    X_tr, y_tr, sid_tr = X_all[tr_idx], y_all[tr_idx], sid_all[tr_idx]
    X_te, y_te, sid_te = X_all[te_idx], y_all[te_idx], sid_all[te_idx]

    validate_arrays(X_tr, y_tr, sid_tr, "single_subject_train")
    validate_arrays(X_te, y_te, sid_te, "single_subject_test")

    save_npz(args.out_train, X_tr, y_tr, sid_tr, "single-subject train")
    save_npz(args.out_test, X_te, y_te, sid_te, "single-subject test")


def run_predefined_split(args: argparse.Namespace) -> None:
    allowed_train = dedup_csv_filenames(args.train_csv)
    allowed_test = dedup_csv_filenames(args.test_csv)

    train_records = build_records(args.train_dir, allowed_train, subject_filter=None)
    test_records = build_records(args.test_dir, allowed_test, subject_filter=None)

    if not train_records:
        raise ValueError("No training records found for predefined split")
    if not test_records:
        raise ValueError("No test records found for predefined split")

    X_tr, y_tr, sid_tr = records_to_npz_arrays(train_records)
    X_te, y_te, sid_te = records_to_npz_arrays(test_records)

    validate_arrays(X_tr, y_tr, sid_tr, "predefined_train")
    validate_arrays(X_te, y_te, sid_te, "predefined_test")

    save_npz(args.out_train, X_tr, y_tr, sid_tr, "predefined train")
    save_npz(args.out_test, X_te, y_te, sid_te, "predefined test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert vimu .pt segments to train/eval .npz using 9-channel r6d+acc"
    )
    p.add_argument(
        "--mode", choices=["single_subject", "predefined_split"], required=True
    )

    p.add_argument("--train_dir", default="data/DIP_IMU_train_real_imu_position_only")
    p.add_argument("--test_dir", default="data/DIP_IMU_test_real_imu_position_only")
    p.add_argument(
        "--train_csv", default="data/DIP_IMU_train_real_imu_position_only.csv"
    )
    p.add_argument("--test_csv", default="")

    p.add_argument(
        "--subject_id",
        type=int,
        default=1,
        help="Used in single_subject mode (e.g. 1 for s_01)",
    )
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_train", default="data/single_subject_train.npz")
    p.add_argument("--out_test", default="data/single_subject_test.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "single_subject":
        run_single_subject(args)
    else:
        run_predefined_split(args)


if __name__ == "__main__":
    main()
