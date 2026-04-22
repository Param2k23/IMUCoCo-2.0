"""
preprocess_vimu.py
==================
Convert DIP-style segmented .pt files (vimu_joints) into train/eval .npz files
for IMUCoCo classification using 9 channels from vimu_joints.

Important channel semantics in this repository:
- vimu_joints has 9 features per region: [orientation_r6d(6), acceleration_xyz(3)]
- The preprocessed X keeps all 9 channels (r6d + acc)

Modes:
1) single_subject  -> build train/test split from one subject only
2) predefined_split -> build train/test using source train/test folders
3) hf_parquet       -> Hugging Face Parquet export (e.g. spongie01/DIP-IMU-position-only):
   nested dict columns joint / imu / vimu; train shards are merged automatically.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


@dataclass
class NumpySegmentRecord:
    """One motion segment as (T, 24, 9) float32 (already merged if applicable)."""

    subject_id: int
    segment: np.ndarray


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


def _to_float32_array(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=np.float32)
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().astype(np.float32)
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj, dtype=np.float32)
    raise TypeError(f"Cannot convert to float32 array: {type(obj)}")


def _unwrap_dict_value(d: Dict[str, Any], preferred_keys: Tuple[str, ...]) -> Any:
    for k in preferred_keys:
        if k in d:
            return d[k]
    if len(d) == 1:
        return next(iter(d.values()))
    raise KeyError(
        f"Expected one of keys {preferred_keys}, got keys={list(d.keys())}"
    )


def _squeeze_leading_time(arr: np.ndarray, expected_last: int = 9) -> np.ndarray:
    """Handle [[[...]]] or (1, T, 24, 9) from HF exports."""
    a = arr
    while a.ndim > 3 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)
    if a.ndim == 4 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)
    if a.ndim != 3:
        raise ValueError(
            f"Expected segment rank 3 (T,24,9) after squeeze, got shape={a.shape}"
        )
    t, n_regions, n_feat = a.shape
    if n_regions != 24:
        raise ValueError(f"Expected 24 regions, got {n_regions}")
    if n_feat < 9:
        raise ValueError(f"Expected at least 9 features, got {n_feat}")
    return a[:, :, :9].astype(np.float32)


def extract_vimu_numpy(vimu_field: Any) -> np.ndarray:
    """Parse vimu column: dict with 'vimu' or 'vimu_joints', or raw array."""
    if isinstance(vimu_field, dict):
        inner = _unwrap_dict_value(
            vimu_field, ("vimu_joints", "vimu", "data", "values")
        )
    else:
        inner = vimu_field
    arr = _to_float32_array(inner)
    return _squeeze_leading_time(arr)


def extract_imu_numpy(imu_field: Any) -> Optional[np.ndarray]:
    """Parse imu column: dict with 'imu', shape (T, 17, 9). Optional."""
    if imu_field is None:
        return None
    if isinstance(imu_field, dict):
        inner = _unwrap_dict_value(imu_field, ("imu", "data", "values"))
    else:
        inner = imu_field
    arr = _to_float32_array(inner)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 3:
        raise ValueError(f"imu must be rank 3 (T,17,9), got shape={arr.shape}")
    return arr.astype(np.float32)


def merge_vimu_imu_channels(
    vimu: np.ndarray,
    imu: Optional[np.ndarray],
    merge: str,
    imu_blend: float,
) -> np.ndarray:
    """
    Combine virtual IMU per region with real IMU stream. Output stays (T, 24, 9).

    - vimu_only: use vimu only (same as legacy .pt pipeline).
    - blend_global_imu: vimu * (1 - a) + (mean over 17 sensors, tiled to 24) * a.
    """
    if merge == "vimu_only":
        return vimu
    if merge == "blend_global_imu":
        if imu is None:
            log.warning("merge=blend_global_imu but imu missing; using vimu_only")
            return vimu
        if imu_blend <= 0.0:
            return vimu
        # imu: (T, 17, 9) -> (T, 1, 9) broadcast to (T, 24, 9)
        pooled = imu.mean(axis=1, keepdims=True)
        broadcast = np.broadcast_to(pooled, vimu.shape).astype(np.float32)
        out = (1.0 - imu_blend) * vimu + imu_blend * broadcast
        return out.astype(np.float32)
    raise ValueError(f"Unknown merge mode: {merge}")


SUBJECT_RE_ROW = re.compile(r"s_(\d+)_", re.IGNORECASE)
# Columns holding large motion tensors — skip for generic string scan (performance).
_PARQUET_SKIP_SCAN_KEYS = frozenset({"vimu", "imu", "joint"})


def _subject_from_value(val: Any) -> Optional[int]:
    """Parse subject id from a cell value (int, float, str, or shallow dict)."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float) and np.isfinite(val) and val == int(val) and val >= 0:
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        m = SUBJECT_RE_ROW.search(s)
        if m:
            return int(m.group(1))
        if s.isdigit():
            return int(s)
        return None
    if isinstance(val, dict):
        for subk in (
            "file_name",
            "filename",
            "path",
            "name",
            "segment_id",
            "subject_id",
            "subject",
            "Subject",
            "dataset_name",
        ):
            if subk in val:
                got = _subject_from_value(val[subk])
                if got is not None:
                    return got
        return None


def try_parse_subject_from_row(
    row: Dict[str, Any], subject_column: Optional[str]
) -> Optional[int]:
    """
    Infer subject id from Parquet row metadata.
    Many HF exports only store joint/imu/vimu — then this returns None (caller uses 0).
    """
    if subject_column:
        if subject_column not in row:
            raise KeyError(
                f"--subject_column {subject_column!r} missing; "
                f"row has: {sorted(row.keys())}"
            )
        got = _subject_from_value(row[subject_column])
        if got is None:
            raise ValueError(
                f"Could not parse subject from column {subject_column!r}: {row[subject_column]!r}"
            )
        return got

    for key in (
        "subject_id",
        "subject",
        "Subject",
        "file_name",
        "filename",
        "segment_id",
        "name",
        "path",
        "filepath",
        "pt_path",
        "source",
        "dataset_name",
    ):
        if key not in row or row[key] is None:
            continue
        got = _subject_from_value(row[key])
        if got is not None:
            return got

    for key, val in row.items():
        if key in _PARQUET_SKIP_SCAN_KEYS:
            continue
        got = _subject_from_value(val)
        if got is not None:
            return got
    return None


def npz_from_segments(
    segments: Sequence[np.ndarray],
    subject_ids: Sequence[int],
    window_length: int = 0,
    crop_mode: str = "center",
    pad_mode: str = "edge",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(segments) != len(subject_ids):
        raise ValueError("segments and subject_ids length mismatch")

    lengths = [int(seg.shape[0]) for seg in segments]
    if any(seg.ndim != 3 or seg.shape[1] != 24 or seg.shape[2] != 9 for seg in segments):
        bad = next(seg for seg in segments if seg.ndim != 3 or seg.shape[1] != 24 or seg.shape[2] != 9)
        raise ValueError(
            f"segment must be (T,24,9), got shape={getattr(bad, 'shape', None)}"
        )

    if window_length > 0:
        target_t = int(window_length)
        if target_t <= 0:
            raise ValueError("--window_length must be > 0 when provided")
    else:
        target_t = min(lengths)
        if min(lengths) != max(lengths):
            log.warning(
                "Variable segment lengths detected (min=%d, max=%d). "
                "Auto-aligning to min length T=%d by truncating longer segments. "
                "For robust training, set --window_length (e.g. 300) to enable crop/pad.",
                min(lengths),
                max(lengths),
                target_t,
            )

    def _normalize_length(seg: np.ndarray) -> np.ndarray:
        t = int(seg.shape[0])
        if t == target_t:
            return seg
        if t > target_t:
            if crop_mode == "center":
                start = (t - target_t) // 2
            else:
                start = 0
            end = start + target_t
            return seg[start:end, :, :]

        pad_t = target_t - t
        if pad_mode == "edge":
            if t == 0:
                raise ValueError("Cannot edge-pad empty segment")
            pad_block = np.repeat(seg[-1:, :, :], pad_t, axis=0)
        else:
            pad_block = np.zeros((pad_t, seg.shape[1], seg.shape[2]), dtype=seg.dtype)
        return np.concatenate([seg, pad_block], axis=0)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    sid_list: List[int] = []
    n_cropped = 0
    n_padded = 0

    for seg, sid in zip(segments, subject_ids):
        t = int(seg.shape[0])
        if t > target_t:
            n_cropped += 1
        elif t < target_t:
            n_padded += 1
        seg_use = _normalize_length(seg)
        for region in range(24):
            sample = np.transpose(seg_use[:, region, :], (1, 0))  # (9, T)
            X_list.append(sample)
            y_list.append(region)
            sid_list.append(int(sid))

    if not X_list:
        raise ValueError("No samples collected for this split")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    subject_ids_arr = np.asarray(sid_list, dtype=np.int64)
    if n_cropped or n_padded:
        log.info(
            "Length normalization: target_T=%d cropped=%d padded=%d total_segments=%d "
            "(crop_mode=%s pad_mode=%s)",
            target_t,
            n_cropped,
            n_padded,
            len(segments),
            crop_mode,
            pad_mode,
        )
    return X, y, subject_ids_arr


def records_to_npz_arrays(
    records: Sequence[SampleRecord],
    window_length: int = 0,
    crop_mode: str = "center",
    pad_mode: str = "edge",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    segs = [extract_segment_tensor(r.file_path) for r in records]
    sids = [r.subject_id for r in records]
    return npz_from_segments(
        segs,
        sids,
        window_length=window_length,
        crop_mode=crop_mode,
        pad_mode=pad_mode,
    )


def numpy_segment_records_to_npz_arrays(
    records: Sequence[NumpySegmentRecord],
    window_length: int = 0,
    crop_mode: str = "center",
    pad_mode: str = "edge",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    segs = [r.segment for r in records]
    sids = [r.subject_id for r in records]
    return npz_from_segments(
        segs,
        sids,
        window_length=window_length,
        crop_mode=crop_mode,
        pad_mode=pad_mode,
    )


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "Reading Parquet requires pyarrow. Install with: pip install pyarrow"
        ) from e
    return pq


def glob_parquet(parquet_dir: str, pattern: str) -> List[str]:
    parquet_dir = os.path.normpath(parquet_dir)
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    paths = sorted(
        glob.glob(os.path.join(parquet_dir, pattern)),
        key=lambda p: os.path.basename(p),
    )
    if not paths:
        raise FileNotFoundError(
            f"No files matching {pattern!r} under {parquet_dir}"
        )
    return paths


def iter_parquet_rows(paths: Sequence[str]) -> Iterable[Tuple[Dict[str, Any], str]]:
    pq = _import_pyarrow()
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=32):
            names = batch.schema.names
            cols = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
            for row_idx in range(batch.num_rows):
                row = {names[i]: cols[i][row_idx] for i in range(len(names))}
                yield row, path


def load_parquet_segments(
    paths: Sequence[str],
    merge: str,
    imu_blend: float,
    subject_filter: Optional[int] = None,
    subject_column: Optional[str] = None,
) -> Tuple[List[NumpySegmentRecord], int, int]:
    """Returns (records, n_rows_missing_subject, n_rows_read)."""
    out: List[NumpySegmentRecord] = []
    total_rows = 0
    missing_subject = 0
    logged_columns = False

    for row, path in iter_parquet_rows(paths):
        total_rows += 1
        hint = f"{path}#row{total_rows - 1}"
        if "vimu" not in row:
            raise KeyError(f"Row missing 'vimu' column ({hint})")
        vimu = extract_vimu_numpy(row["vimu"])
        imu = None
        if row.get("imu") is not None:
            try:
                imu = extract_imu_numpy(row["imu"])
            except (TypeError, ValueError) as e:
                log.warning("Skipping imu parse for %s: %s", hint, e)
        seg = merge_vimu_imu_channels(vimu, imu, merge, imu_blend)
        if not np.isfinite(seg).all():
            raise ValueError(f"Non-finite values in segment ({hint})")
        if seg.shape[0] <= 0:
            raise ValueError(f"Empty segment ({hint})")

        sid_opt = try_parse_subject_from_row(row, subject_column)
        if sid_opt is None:
            missing_subject += 1
            if not logged_columns:
                log.info(
                    "Parquet row columns (no subject parsed yet): %s",
                    sorted(row.keys()),
                )
                logged_columns = True
            sid = 0
        else:
            sid = sid_opt

        if subject_filter is not None and sid != subject_filter:
            continue
        out.append(NumpySegmentRecord(subject_id=sid, segment=seg))

    return out, missing_subject, total_rows


def _warn_parquet_subject_fallback(
    missing: int, total: int, subject_column: Optional[str]
) -> None:
    if missing and not subject_column:
        log.warning(
            "Assigned subject_id=0 to %d / %d Parquet rows (no subject metadata). "
            "Fixed-split training and smoke tests are fine; LOSO / per-subject analysis "
            "will not separate people. Re-export with file_name/subject_id, or set "
            "--subject_column.",
            missing,
            total,
        )


def run_hf_parquet(args: argparse.Namespace) -> None:
    train_paths = glob_parquet(args.parquet_dir, args.train_glob)
    test_paths = glob_parquet(args.parquet_dir, args.test_glob)
    log.info(
        "Parquet: train files=%d test files=%d (dir=%s)",
        len(train_paths),
        len(test_paths),
        args.parquet_dir,
    )

    merge = args.merge_inputs
    imu_blend = float(args.imu_blend)

    subj_col = (args.subject_column or "").strip() or None

    if args.hf_split == "stratified":
        train_recs, miss_tr, ntr = load_parquet_segments(
            train_paths,
            merge,
            imu_blend,
            subject_filter=args.subject_id,
            subject_column=subj_col,
        )
        test_recs, miss_te, nte = load_parquet_segments(
            test_paths,
            merge,
            imu_blend,
            subject_filter=args.subject_id,
            subject_column=subj_col,
        )
        _warn_parquet_subject_fallback(miss_tr + miss_te, ntr + nte, subj_col)
        records = train_recs + test_recs
        if not records:
            raise ValueError(f"No parquet rows for subject {args.subject_id}")

        X_all, y_all, sid_all = numpy_segment_records_to_npz_arrays(
            records,
            window_length=args.window_length,
            crop_mode=args.crop_mode,
            pad_mode=args.pad_mode,
        )
        validate_arrays(X_all, y_all, sid_all, "parquet_single_all")

        tr_idx, te_idx = stratified_split_indices(
            y_all, test_ratio=args.test_ratio, seed=args.seed
        )
        X_tr, y_tr, sid_tr = X_all[tr_idx], y_all[tr_idx], sid_all[tr_idx]
        X_te, y_te, sid_te = X_all[te_idx], y_all[te_idx], sid_all[te_idx]

        validate_arrays(X_tr, y_tr, sid_tr, "parquet_single_train")
        validate_arrays(X_te, y_te, sid_te, "parquet_single_test")

        save_npz(args.out_train, X_tr, y_tr, sid_tr, "HF parquet single-subject train")
        save_npz(args.out_test, X_te, y_te, sid_te, "HF parquet single-subject test")
        return

    # hf_split == "predefined"
    train_recs, miss_tr, ntr = load_parquet_segments(
        train_paths,
        merge,
        imu_blend,
        subject_filter=None,
        subject_column=subj_col,
    )
    test_recs, miss_te, nte = load_parquet_segments(
        test_paths,
        merge,
        imu_blend,
        subject_filter=None,
        subject_column=subj_col,
    )
    _warn_parquet_subject_fallback(miss_tr + miss_te, ntr + nte, subj_col)
    if not train_recs:
        raise ValueError("No training segments loaded from Parquet")
    if not test_recs:
        raise ValueError("No test segments loaded from Parquet")

    X_tr, y_tr, sid_tr = numpy_segment_records_to_npz_arrays(
        train_recs,
        window_length=args.window_length,
        crop_mode=args.crop_mode,
        pad_mode=args.pad_mode,
    )
    X_te, y_te, sid_te = numpy_segment_records_to_npz_arrays(
        test_recs,
        window_length=args.window_length,
        crop_mode=args.crop_mode,
        pad_mode=args.pad_mode,
    )

    validate_arrays(X_tr, y_tr, sid_tr, "parquet_predefined_train")
    validate_arrays(X_te, y_te, sid_te, "parquet_predefined_test")

    save_npz(args.out_train, X_tr, y_tr, sid_tr, "HF parquet train")
    save_npz(args.out_test, X_te, y_te, sid_te, "HF parquet test")


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

    X_all, y_all, sid_all = records_to_npz_arrays(
        records,
        window_length=args.window_length,
        crop_mode=args.crop_mode,
        pad_mode=args.pad_mode,
    )
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

    X_tr, y_tr, sid_tr = records_to_npz_arrays(
        train_records,
        window_length=args.window_length,
        crop_mode=args.crop_mode,
        pad_mode=args.pad_mode,
    )
    X_te, y_te, sid_te = records_to_npz_arrays(
        test_records,
        window_length=args.window_length,
        crop_mode=args.crop_mode,
        pad_mode=args.pad_mode,
    )

    validate_arrays(X_tr, y_tr, sid_tr, "predefined_train")
    validate_arrays(X_te, y_te, sid_te, "predefined_test")

    save_npz(args.out_train, X_tr, y_tr, sid_tr, "predefined train")
    save_npz(args.out_test, X_te, y_te, sid_te, "predefined test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert vimu .pt segments to train/eval .npz using 9-channel r6d+acc"
    )
    p.add_argument(
        "--mode",
        choices=["single_subject", "predefined_split", "hf_parquet"],
        required=True,
    )

    p.add_argument("--train_dir", default="data/DIP_IMU_train_real_imu_position_only")
    p.add_argument("--test_dir", default="data/DIP_IMU_test_real_imu_position_only")
    p.add_argument(
        "--train_csv", default="data/DIP_IMU_train_real_imu_position_only.csv"
    )
    p.add_argument("--test_csv", default="")

    p.add_argument(
        "--parquet_dir",
        default="data/raw_dip/data",
        help="Directory with train-*.parquet / test-*.parquet (HF snapshot_download layout)",
    )
    p.add_argument(
        "--train_glob",
        default="train-*.parquet",
        help="Glob under parquet_dir for training shards (merged in order)",
    )
    p.add_argument(
        "--test_glob",
        default="test-*.parquet",
        help="Glob under parquet_dir for the test split",
    )
    p.add_argument(
        "--hf_split",
        choices=["predefined", "stratified"],
        default="predefined",
        help="hf_parquet: predefined = train shards -> train npz, test shards -> test npz; "
        "stratified = filter one subject then random stratified train/test split",
    )
    p.add_argument(
        "--merge_inputs",
        choices=["vimu_only", "blend_global_imu"],
        default="vimu_only",
        help="vimu_only: use nested vimu only (legacy). blend_global_imu: mix per-region vimu "
        "with global mean of real IMU sensors (see --imu_blend)",
    )
    p.add_argument(
        "--imu_blend",
        type=float,
        default=0.0,
        help="When merge_inputs=blend_global_imu: weight of global IMU term in [0,1]",
    )
    p.add_argument(
        "--subject_column",
        default="",
        help="hf_parquet: Parquet column name for subject id (int or s_XX string). "
        "If unset, heuristics scan metadata columns; HF exports with only joint/imu/vimu "
        "get subject_id=0 (see logs).",
    )

    p.add_argument(
        "--subject_id",
        type=int,
        default=1,
        help="Used in single_subject mode (e.g. 1 for s_01); also hf_parquet + hf_split=stratified",
    )
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Target temporal length T for all segments before stacking. "
        "0 = auto (use per-split minimum T and truncate longer segments).",
    )
    p.add_argument(
        "--crop_mode",
        choices=["center", "start"],
        default="center",
        help="When a segment is longer than --window_length, crop from center or start.",
    )
    p.add_argument(
        "--pad_mode",
        choices=["edge", "zero"],
        default="edge",
        help="When a segment is shorter than --window_length, pad by repeating last frame "
        "(edge) or with zeros.",
    )

    p.add_argument("--out_train", default="data/single_subject_train.npz")
    p.add_argument("--out_test", default="data/single_subject_test.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "single_subject":
        run_single_subject(args)
    elif args.mode == "predefined_split":
        run_predefined_split(args)
    else:
        run_hf_parquet(args)


if __name__ == "__main__":
    main()
