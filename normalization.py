"""
normalization.py
================
Shared normalization helpers for train/eval parity.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


EPS = 1e-8
EXPECTED_CHANNELS = 9


def validate_input_array(
    X: np.ndarray, expected_channels: int = EXPECTED_CHANNELS
) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,{expected_channels},T), got shape={X.shape}")
    if X.shape[1] != expected_channels:
        raise ValueError(f"X must have {expected_channels} channels, got {X.shape[1]}")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")


def compute_channel_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-channel mean/std with shape (1, C, 1)."""
    validate_input_array(X_train)
    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = std + EPS
    return mean, std


def apply_channel_stats(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    validate_input_array(X)
    if mean.shape != (1, X.shape[1], 1):
        raise ValueError(f"mean shape must be (1,{X.shape[1]},1), got {mean.shape}")
    if std.shape != (1, X.shape[1], 1):
        raise ValueError(f"std shape must be (1,{X.shape[1]},1), got {std.shape}")
    Xn = (X - mean) / std
    if not np.isfinite(Xn).all():
        raise ValueError("Normalized X contains non-finite values")
    return Xn.astype(np.float32)
