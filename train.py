"""
train.py
========
Training entrypoint for the IMU sensor-location classifier.
Supports LOSO and fixed train/test split modes.

Usage
-----
  python train.py --data C:/VS/SensorLoc/data/dataset.npz --epochs 50
  python train.py --data C:/VS/SensorLoc/data/smoke_dataset.npz --epochs 5 --smoke_test
  python train.py --data ... --multi_gpu   # uses all visible CUDA GPUs (DataParallel)

Output
------
  checkpoints/best_model_fold<k>.pt  — best val-loss checkpoint per fold
  checkpoints/loso_results.txt       — per-fold and average accuracy
"""

from __future__ import annotations
import argparse
import os
import sys
import logging
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import build_model
from smpl_regions import REGION_NAMES, NUM_REGIONS, SYMMETRY_PAIRS
from normalization import (
    compute_channel_stats,
    apply_channel_stats,
    validate_input_array,
)


# ---------------------------------------------------------------------------
# Custom losses
# ---------------------------------------------------------------------------

from smpl_regions import REGION_CENTROIDS  # (24, 3) T-pose centroids in metres


class SpatialNeighborLoss(nn.Module):
    """
    Spatial-aware loss that penalises the model for being confident in ANY
    nearby body region, not just left-right mirrors.

    = CrossEntropy(logits, y)
    + lr_weight      × mean P(mirror_class | x)          [left-right penalty]
    + neighbor_weight × mean Σ_j  w_j · P(class_j | x)  [neighbor penalty]

    where w_j = max(0, 1 - dist(centroid_y, centroid_j) / radius)
    for all j ≠ y within `neighbor_radius` metres.

    Intuition
    ---------
    The model confuses l_hip with pelvis and l_thigh (all ~10-20 cm apart)
    because CE treats those errors identically to confusing l_hip with l_hand.
    The neighbor penalty adds extra cost proportional to *how close* the wrong
    prediction is — the closer, the harder we penalise.

    This is a strict generalisation of SensorLocLoss:
      SensorLocLoss ≡ SpatialNeighborLoss(neighbor_weight=0, radius=0)

    Parameters
    ----------
    lr_weight       : weight for the left-right mirror penalty (0 = off)
    neighbor_weight : weight for the spatial proximity penalty (0.3 recommended)
    neighbor_radius : distance threshold in metres; regions beyond this are
                      not penalised (default 0.35 m ≈ hip-to-knee distance)
    label_smoothing : passed to CrossEntropyLoss
    class_weights   : optional (C,) tensor for weighted CE (upweight hard classes)
    """

    def __init__(
        self,
        lr_weight: float = 0.5,
        neighbor_weight: float = 0.3,
        neighbor_radius: float = 0.35,
        label_smoothing: float = 0.05,
        num_classes: int = NUM_REGIONS,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lr_weight       = lr_weight
        self.neighbor_weight = neighbor_weight

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing
        )

        # ── Left-right mirror map ─────────────────────────────────────────
        mirror = list(range(num_classes))
        for l_id, r_id in SYMMETRY_PAIRS:
            mirror[l_id] = r_id
            mirror[r_id] = l_id
        self.register_buffer("mirror_map", torch.tensor(mirror, dtype=torch.long))

        # ── Spatial proximity weight matrix  (C, C) ───────────────────────
        # prox[i, j] = how strongly to penalise class j when true class is i
        # = max(0, 1 - dist(i,j)/radius),  0 on diagonal
        centroids = torch.tensor(REGION_CENTROIDS, dtype=torch.float32)  # (C, 3)
        diffs = centroids.unsqueeze(0) - centroids.unsqueeze(1)          # (C, C, 3)
        dists = diffs.norm(dim=2)                                         # (C, C)
        prox  = (1.0 - dists / neighbor_radius).clamp(min=0.0)           # (C, C)
        prox.fill_diagonal_(0.0)                                          # zero self
        self.register_buffer("prox_matrix", prox)                        # (C, C)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs   = F.softmax(logits, dim=1)           # (B, C)

        penalties = ce_loss * 0.0  # scalar zero on correct device
        n_terms   = 0

        # ── Left-right mirror penalty ─────────────────────────────────────
        if self.lr_weight > 0.0:
            mirror_targets = self.mirror_map[targets]     # (B,)
            has_mirror     = mirror_targets != targets    # (B,)
            if has_mirror.any():
                mirror_probs = probs[has_mirror].gather(
                    1, mirror_targets[has_mirror].unsqueeze(1)
                ).squeeze(1)
                penalties = penalties + self.lr_weight * mirror_probs.mean()
                n_terms  += 1

        # ── Spatial neighbor penalty ──────────────────────────────────────
        if self.neighbor_weight > 0.0:
            # For each sample, get the proximity weights for its true class
            # prox_row: (B, C)  — how close each other class is to true class
            prox_row = self.prox_matrix[targets]          # (B, C)
            # Penalise: sum of (prox_weight × prob_of_that_class) per sample
            neighbor_pen = (prox_row * probs).sum(dim=1).mean()  # scalar
            penalties    = penalties + self.neighbor_weight * neighbor_pen
            n_terms     += 1

        return ce_loss + penalties


def build_criterion(
    loss_fn: str,
    lr_weight: float = 0.5,
    neighbor_weight: float = 0.3,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Factory: 'crossentropy' or 'custom' (SpatialNeighborLoss)."""
    if loss_fn == "custom":
        log.info(
            "Using SpatialNeighborLoss  "
            "(CE + L/R penalty lr_w=%.2f + neighbor penalty nb_w=%.2f)",
            lr_weight, neighbor_weight,
        )
        return SpatialNeighborLoss(
            lr_weight=lr_weight,
            neighbor_weight=neighbor_weight,
            label_smoothing=0.05,
            class_weights=class_weights,
        )
    else:
        log.info("Using standard CrossEntropyLoss%s",
                 " (class-weighted)" if class_weights is not None else "")
        return nn.CrossEntropyLoss(weight=class_weights)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    X = data["X"].astype(np.float32)  # (N, C, T)
    y = data["y"].astype(np.int64)  # (N,)
    sids = data["subject_ids"].astype(np.int64)  # (N,)
    validate_input_array(X)
    log.info(
        "Loaded %s  |  X=%s  classes=%d  subjects=%d",
        path,
        X.shape,
        len(np.unique(y)),
        len(np.unique(sids)),
    )
    return X, y, sids


def normalise(X_train: np.ndarray, X_val: np.ndarray):
    """Z-score normalise per channel using shared helper."""
    mean, std = compute_channel_stats(X_train)
    return (
        apply_channel_stats(X_train, mean, std),
        apply_channel_stats(X_val, mean, std),
        mean,
        std,
    )


def _state_dict_to_save(model: nn.Module) -> dict:
    """Checkpoint dict without 'module.' prefix (compatible with evaluate.py)."""
    return (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )


def _save_norm_stats(
    path: str,
    mean: np.ndarray,
    std: np.ndarray,
    in_channels: int,
    arch: str,
    fold: int,
    test_subj: int,
) -> None:
    torch.save(
        {
            "norm_mean": mean.astype(np.float32),
            "norm_std": std.astype(np.float32),
            "in_channels": int(in_channels),
            "arch": arch,
            "fold": int(fold),
            "test_subj": int(test_subj),
        },
        path,
    )


def wrap_data_parallel(
    model: nn.Module, enabled: bool, device: torch.device
) -> nn.Module:
    """Replicates the model on all visible CUDA devices (batch split per step)."""
    if not enabled or device.type != "cuda":
        return model
    n = torch.cuda.device_count()
    if n < 2:
        log.info("multi_gpu set but only %d CUDA device(s) — using single GPU.", n)
        return model
    log.info(
        "nn.DataParallel across %d GPU(s) — per-step batch still %s total samples",
        n,
        "(split across devices)",
    )
    return nn.DataParallel(model)


def make_loaders(X_tr, y_tr, X_val, y_val, batch_size: int):
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_dl = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    return tr_dl, val_dl


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------


def _build_class_weights(
    y: np.ndarray,
    device: torch.device,
    mode: str = "none",
) -> Optional[torch.Tensor]:
    """
    Build optional class-weight tensor for CrossEntropyLoss.

    mode='none'        → no weights (standard CE)
    mode='inverse_freq' → 1/class_frequency (upweights rare / hard classes)
    mode='sqrt_inv'    → 1/sqrt(class_frequency) (softer version)
    """
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=NUM_REGIONS).astype(np.float32)
    counts = np.maximum(counts, 1)          # avoid div-by-zero
    if mode == "inverse_freq":
        w = 1.0 / counts
    elif mode == "sqrt_inv":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class_weights mode: {mode!r}")
    w = w / w.sum() * NUM_REGIONS           # normalise so mean weight = 1
    return torch.tensor(w, dtype=torch.float32).to(device)


def loso_train(
    X: np.ndarray,
    y: np.ndarray,
    sids: np.ndarray,
    arch: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    out_dir: str,
    device: torch.device,
    multi_gpu: bool = False,
    max_folds: Optional[int] = None,
    loss_fn: str = "crossentropy",
    lr_weight: float = 0.5,
    neighbor_weight: float = 0.3,
    class_weights_mode: str = "none",
    base_filters: int = 64,
) -> List[float]:
    """
    Leave-One-Subject-Out cross-validation.

    Returns list of per-fold validation accuracies.
    """
    unique_subjects = np.unique(sids)
    if max_folds is not None:
        unique_subjects = unique_subjects[:max_folds]
    n_folds = len(unique_subjects)
    fold_accs = []

    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "loso_results.txt")

    with open(results_path, "w") as rf:
        rf.write("Fold  Subject  Val_Acc\n")
        rf.write("-" * 35 + "\n")

        for fold_idx, test_subj in enumerate(unique_subjects):
            log.info(
                "\n── Fold %d/%d  (test subject = %d) ──",
                fold_idx + 1,
                n_folds,
                test_subj,
            )

            train_mask = sids != test_subj
            val_mask = sids == test_subj

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            if len(X_val) == 0:
                log.warning("  Fold %d: empty val set, skipping.", fold_idx)
                continue

            # Normalise
            X_tr_n, X_val_n, mean, std = normalise(X_tr, X_val)

            tr_dl, val_dl = make_loaders(
                X_tr_n,
                y_tr.astype(np.int64),
                X_val_n,
                y_val.astype(np.int64),
                batch_size,
            )

            # Build fresh model each fold
            model = build_model(
                arch, n_classes=NUM_REGIONS, in_channels=X.shape[1],
                base_filters=base_filters,
            ).to(device)
            model = wrap_data_parallel(model, multi_gpu, device)
            cw = _build_class_weights(y_tr, device, class_weights_mode)
            criterion = build_criterion(
                loss_fn, lr_weight, neighbor_weight, cw
            ).to(device)
            optimiser = Adam(
                model.parameters(), lr=lr, weight_decay=float(weight_decay)
            )
            scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr / 100)

            best_val_loss = float("inf")
            best_val_acc = 0.0
            patience_ctr = 0
            ckpt_path = os.path.join(out_dir, f"best_model_fold{fold_idx}.pt")
            stats_path = os.path.join(out_dir, f"normalization_stats_fold{fold_idx}.pt")
            t0 = time.time()

            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = train_one_epoch(
                    model, tr_dl, criterion, optimiser, device
                )
                val_loss, val_acc = evaluate(model, val_dl, criterion, device)
                scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_ctr = 0
                    torch.save(
                        _state_dict_to_save(model),
                        ckpt_path,
                    )
                    _save_norm_stats(
                        stats_path,
                        mean,
                        std,
                        in_channels=int(X.shape[1]),
                        arch=arch,
                        fold=fold_idx,
                        test_subj=int(test_subj),
                    )
                else:
                    patience_ctr += 1

                if epoch % 5 == 0 or epoch == 1:
                    log.info(
                        "  ep %3d/%d  tr_loss=%.4f  tr_acc=%.4f  "
                        "val_loss=%.4f  val_acc=%.4f",
                        epoch,
                        epochs,
                        tr_loss,
                        tr_acc,
                        val_loss,
                        val_acc,
                    )

                if patience_ctr >= patience:
                    log.info(
                        "  Early stopping at epoch %d (patience=%d).", epoch, patience
                    )
                    break

            elapsed = time.time() - t0
            log.info(
                "  Fold %d done  |  best_val_acc=%.4f  |  %.1fs",
                fold_idx,
                best_val_acc,
                elapsed,
            )
            fold_accs.append(best_val_acc)
            rf.write(f"  {fold_idx + 1:2d}     {test_subj:5d}    {best_val_acc:.4f}\n")

        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        rf.write("-" * 35 + "\n")
        rf.write(f"Mean LOSO Acc: {mean_acc:.4f}\n")

    log.info("\n=== LOSO Complete ===")
    log.info("   Per-fold:  %s", [f"{a:.4f}" for a in fold_accs])
    log.info("   Mean acc:  %.4f", float(np.mean(fold_accs)) if fold_accs else 0.0)
    log.info("   Results → %s", results_path)
    return fold_accs


def fixed_split_train(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    arch: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    out_dir: str,
    device: torch.device,
    multi_gpu: bool = False,
    loss_fn: str = "crossentropy",
    lr_weight: float = 0.5,
    neighbor_weight: float = 0.3,
    class_weights_mode: str = "none",
    base_filters: int = 64,
) -> float:
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "fixed_split_results.txt")

    X_tr_n, X_val_n, mean, std = normalise(X_tr, X_val)
    tr_dl, val_dl = make_loaders(
        X_tr_n,
        y_tr.astype(np.int64),
        X_val_n,
        y_val.astype(np.int64),
        batch_size,
    )

    model = build_model(
        arch, n_classes=NUM_REGIONS, in_channels=X_tr.shape[1],
        base_filters=base_filters,
    ).to(device)
    model = wrap_data_parallel(model, multi_gpu, device)
    cw = _build_class_weights(y_tr, device, class_weights_mode)
    criterion = build_criterion(loss_fn, lr_weight, neighbor_weight, cw).to(device)
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr / 100)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_ctr = 0
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    stats_path = os.path.join(out_dir, "normalization_stats.pt")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_dl, criterion, optimiser, device)
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(
                _state_dict_to_save(model),
                ckpt_path,
            )
            _save_norm_stats(
                stats_path,
                mean,
                std,
                in_channels=int(X_tr.shape[1]),
                arch=arch,
                fold=-1,
                test_subj=-1,
            )
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "  ep %3d/%d  tr_loss=%.4f  tr_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch,
                epochs,
                tr_loss,
                tr_acc,
                val_loss,
                val_acc,
            )

        if patience_ctr >= patience:
            log.info("  Early stopping at epoch %d (patience=%d).", epoch, patience)
            break

    elapsed = time.time() - t0
    with open(results_path, "w") as rf:
        rf.write("Split  Val_Acc\n")
        rf.write("-" * 24 + "\n")
        rf.write(f"fixed  {best_val_acc:.4f}\n")

    log.info("\n=== Fixed-Split Training Complete ===")
    log.info("   best_val_acc: %.4f", best_val_acc)
    log.info("   elapsed: %.1fs", elapsed)
    log.info("   Results -> %s", results_path)
    return best_val_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="LOSO training for sensor-location classifier"
    )
    p.add_argument("--mode", default="loso", choices=["loso", "fixed_split"])
    p.add_argument("--data", default="C:/VS/SensorLoc/data/dataset.npz")
    p.add_argument("--train_data", default="")
    p.add_argument("--val_data", default="")
    p.add_argument("--out_dir", default="C:/VS/SensorLoc/checkpoints")
    p.add_argument("--arch", default="resnet", choices=["resnet", "cnn"])
    p.add_argument("--epochs", type=int, default=100,
                   help="Max training epochs per fold (default 100; was 50)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stop patience in epochs (default 20; was 10)")
    p.add_argument(
        "--loss_fn",
        default="custom",
        choices=["crossentropy", "custom"],
        help="'custom' = SpatialNeighborLoss; 'crossentropy' = standard CE",
    )
    p.add_argument(
        "--lr_weight", type=float, default=0.5,
        help="Weight for left-right mirror penalty in SpatialNeighborLoss (default 0.5)",
    )
    p.add_argument(
        "--neighbor_weight", type=float, default=0.3,
        help="Weight for spatial neighbor penalty — tackles hip/thigh confusion (default 0.3)",
    )
    p.add_argument(
        "--class_weights",
        default="none",
        choices=["none", "inverse_freq", "sqrt_inv"],
        help="Class weighting for CE: 'inverse_freq' upweights hard/rare classes (default none)",
    )
    p.add_argument(
        "--base_filters", type=int, default=64,
        help="ResNet base filter count — 64 (default, 1.3M params) or 128 (5M params, more capacity)",
    )
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use smoke_dataset.npz and run only 1 fold for 5 epochs",
    )
    p.add_argument("--device", default="auto", help="cuda / mps / cpu / auto")
    p.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Use nn.DataParallel on every visible CUDA GPU (e.g. 2× T4)",
    )
    p.add_argument(
        "--max_folds",
        type=int,
        default=0,
        help="Maximum LOSO folds to run (0 = all folds)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.smoke_test:
        raise RuntimeError(
            "CUDA is required for non-smoke runs. Re-run with --smoke_test or on a CUDA machine."
        )

    if args.device == "auto":
        device = torch.device("cuda" if cuda_available else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not cuda_available:
            if args.smoke_test:
                log.warning(
                    "CUDA requested but unavailable; --smoke_test active, falling back to CPU."
                )
                device = torch.device("cpu")
            else:
                raise RuntimeError("CUDA device requested but unavailable.")

    if not cuda_available and args.smoke_test and device.type != "cpu":
        log.warning("CUDA unavailable; --smoke_test active, falling back to CPU.")
        device = torch.device("cpu")
    log.info("Using device: %s", device)

    if args.smoke_test:
        args.epochs = min(args.epochs, 5)

    if args.mode == "loso":
        data_path = args.data
        if args.smoke_test:
            data_path = data_path.replace("dataset.npz", "smoke_dataset.npz")

        X, y, sids = load_dataset(data_path)
        max_folds = args.max_folds if args.max_folds > 0 else None
        if args.smoke_test and max_folds is None:
            max_folds = 1

        loso_train(
            X,
            y,
            sids,
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            out_dir=args.out_dir,
            device=device,
            multi_gpu=args.multi_gpu,
            max_folds=max_folds,
            loss_fn=args.loss_fn,
            lr_weight=args.lr_weight,
            neighbor_weight=args.neighbor_weight,
            class_weights_mode=args.class_weights,
            base_filters=args.base_filters,
        )
    else:
        if not args.train_data or not args.val_data:
            raise ValueError("--mode fixed_split requires --train_data and --val_data")

        train_path = args.train_data
        val_path = args.val_data
        X_tr, y_tr, _ = load_dataset(train_path)
        X_val, y_val, _ = load_dataset(val_path)
        fixed_split_train(
            X_tr,
            y_tr,
            X_val,
            y_val,
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            out_dir=args.out_dir,
            device=device,
            multi_gpu=args.multi_gpu,
            loss_fn=args.loss_fn,
            lr_weight=args.lr_weight,
            neighbor_weight=args.neighbor_weight,
            class_weights_mode=args.class_weights,
            base_filters=args.base_filters,
        )
