"""
regenerate_npz.py
===============
Simple script to regenerate .npz files from .pt files with correct 9 features.

Usage:
    python regenerate_npz.py --input_dir <dir> --output <output.npz>
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch


def extract_segment_tensor(file_path: str) -> np.ndarray:
    """Extract (T, 24, 9) from .pt file."""
    data = torch.load(file_path, map_location="cpu")
    vimu = data["vimu"]["vimu_joints"]
    if not torch.is_tensor(vimu):
        raise ValueError(f"vimu_joints is not tensor in {file_path}")
    if vimu.ndim != 3:
        raise ValueError(f"Expected (T,24,9), got {tuple(vimu.shape)}")
    arr = vimu[:, :, :9].detach().cpu().numpy().astype(np.float32)
    return arr


def convert_to_samples(
    pt_files: List[str], subject_id: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert list of .pt files to X, y, subject_ids."""
    X_list = []
    y_list = []
    sid_list = []

    for f in pt_files:
        arr = extract_segment_tensor(f)  # (T, 24, 9)
        T = arr.shape[0]

        for region in range(24):
            sample = np.transpose(arr[:, region, :], (1, 0))  # (9, T)
            X_list.append(sample)
            y_list.append(region)
            sid_list.append(subject_id)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subject_ids = np.array(sid_list, dtype=np.int64)

    return X, y, subject_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with .pt files")
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--subject_id", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading .pt files from {args.input_dir}")

    pt_files = [
        os.path.join(args.input_dir, f)
        for f in sorted(os.listdir(args.input_dir))
        if f.endswith(".pt")
    ]

    print(f"Found {len(pt_files)} .pt files")

    if len(pt_files) == 0:
        print("No .pt files found!")
        return

    X, y, subject_ids = convert_to_samples(pt_files, args.subject_id)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"subject_ids shape: {subject_ids.shape}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, X=X, y=y, subject_ids=subject_ids)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
