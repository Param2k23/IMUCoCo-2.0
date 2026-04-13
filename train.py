"""
train.py
========
Leave-One-Subject-Out (LOSO) training for the IMU sensor-location classifier.

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
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import build_model
from smpl_regions import REGION_NAMES, NUM_REGIONS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    X    = data['X'].astype(np.float32)   # (N, 6, 120)
    y    = data['y'].astype(np.int64)     # (N,)
    sids = data['subject_ids'].astype(np.int64)  # (N,)
    log.info("Loaded %s  |  X=%s  classes=%d  subjects=%d",
             path, X.shape, len(np.unique(y)), len(np.unique(sids)))
    return X, y, sids


def normalise(X_train: np.ndarray, X_val: np.ndarray):
    """Z-score normalise per channel using training set statistics."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)   # (1, 6, 1)
    std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def _state_dict_to_save(model: nn.Module) -> dict:
    """Checkpoint dict without 'module.' prefix (compatible with evaluate.py)."""
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def wrap_data_parallel(model: nn.Module, enabled: bool, device: torch.device) -> nn.Module:
    """Replicates the model on all visible CUDA devices (batch split per step)."""
    if not enabled or device.type != 'cuda':
        return model
    n = torch.cuda.device_count()
    if n < 2:
        log.info("multi_gpu set but only %d CUDA device(s) — using single GPU.", n)
        return model
    log.info("nn.DataParallel across %d GPU(s) — per-step batch still %s total samples",
             n, "(split across devices)")
    return nn.DataParallel(model)


def make_loaders(X_tr, y_tr, X_val, y_val, batch_size: int):
    tr_ds  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_dl  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)
    return tr_dl, val_dl


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimiser: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def loso_train(X: np.ndarray, y: np.ndarray, sids: np.ndarray,
               arch: str, epochs: int, batch_size: int,
               lr: float, patience: int, out_dir: str,
               device: torch.device,
               multi_gpu: bool = False) -> List[float]:
    """
    Leave-One-Subject-Out cross-validation.

    Returns list of per-fold validation accuracies.
    """
    unique_subjects = np.unique(sids)
    n_folds         = len(unique_subjects)
    fold_accs       = []

    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, 'loso_results.txt')

    with open(results_path, 'w') as rf:
        rf.write("Fold  Subject  Val_Acc\n")
        rf.write("-" * 35 + "\n")

        for fold_idx, test_subj in enumerate(unique_subjects):
            log.info("\n── Fold %d/%d  (test subject = %d) ──",
                     fold_idx + 1, n_folds, test_subj)

            train_mask = sids != test_subj
            val_mask   = sids == test_subj

            X_tr, y_tr   = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask],   y[val_mask]

            if len(X_val) == 0:
                log.warning("  Fold %d: empty val set, skipping.", fold_idx)
                continue

            # Normalise
            X_tr_n, X_val_n, _, _ = normalise(X_tr, X_val)

            tr_dl, val_dl = make_loaders(
                X_tr_n, y_tr.astype(np.int64),
                X_val_n, y_val.astype(np.int64),
                batch_size,
            )

            # Build fresh model each fold
            model     = build_model(arch, n_classes=NUM_REGIONS).to(device)
            model     = wrap_data_parallel(model, multi_gpu, device)
            criterion = nn.CrossEntropyLoss()
            optimiser = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr/100)

            best_val_loss = float('inf')
            best_val_acc  = 0.0
            patience_ctr  = 0
            ckpt_path     = os.path.join(out_dir, f'best_model_fold{fold_idx}.pt')
            t0            = time.time()

            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc   = train_one_epoch(model, tr_dl, criterion,
                                                     optimiser, device)
                val_loss, val_acc = evaluate(model, val_dl, criterion, device)
                scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc  = val_acc
                    patience_ctr  = 0
                    torch.save({
                        'epoch':     epoch,
                        'model':     _state_dict_to_save(model),
                        'val_acc':   val_acc,
                        'val_loss':  val_loss,
                        'arch':      arch,
                        'fold':      fold_idx,
                        'test_subj': int(test_subj),
                    }, ckpt_path)
                else:
                    patience_ctr += 1

                if epoch % 5 == 0 or epoch == 1:
                    log.info("  ep %3d/%d  tr_loss=%.4f  tr_acc=%.4f  "
                             "val_loss=%.4f  val_acc=%.4f",
                             epoch, epochs, tr_loss, tr_acc, val_loss, val_acc)

                if patience_ctr >= patience:
                    log.info("  Early stopping at epoch %d (patience=%d).",
                             epoch, patience)
                    break

            elapsed = time.time() - t0
            log.info("  Fold %d done  |  best_val_acc=%.4f  |  %.1fs",
                     fold_idx, best_val_acc, elapsed)
            fold_accs.append(best_val_acc)
            rf.write(f"  {fold_idx+1:2d}     {test_subj:5d}    {best_val_acc:.4f}\n")

        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        rf.write("-" * 35 + "\n")
        rf.write(f"Mean LOSO Acc: {mean_acc:.4f}\n")

    log.info("\n=== LOSO Complete ===")
    log.info("   Per-fold:  %s", [f"{a:.4f}" for a in fold_accs])
    log.info("   Mean acc:  %.4f", float(np.mean(fold_accs)) if fold_accs else 0.0)
    log.info("   Results → %s", results_path)
    return fold_accs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LOSO training for sensor-location classifier")
    p.add_argument('--data',       default='C:/VS/SensorLoc/data/dataset.npz')
    p.add_argument('--out_dir',    default='C:/VS/SensorLoc/checkpoints')
    p.add_argument('--arch',       default='resnet', choices=['resnet', 'cnn'])
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--patience',   type=int, default=10)
    p.add_argument('--smoke_test', action='store_true',
                   help='Use smoke_dataset.npz and run only 1 fold for 5 epochs')
    p.add_argument('--device',     default='auto',
                   help='cuda / mps / cpu / auto')
    p.add_argument('--multi_gpu', action='store_true',
                   help='Use nn.DataParallel on every visible CUDA GPU (e.g. 2× T4)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    log.info("Using device: %s", device)

    data_path = args.data
    if args.smoke_test:
        data_path = data_path.replace('dataset.npz', 'smoke_dataset.npz')
        args.epochs  = min(args.epochs, 5)

    X, y, sids = load_dataset(data_path)

    # Smoke test: limit to 1 fold
    if args.smoke_test:
        fold_accs = loso_train(
            X[:, :, :], y, sids,
            arch=args.arch, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, out_dir=args.out_dir,
            device=device,
            multi_gpu=args.multi_gpu,
        )
        # Only run 1 fold in smoke mode
        log.info("Smoke test complete. Single fold acc: %.4f",
                 fold_accs[0] if fold_accs else 0.0)
    else:
        loso_train(
            X, y, sids,
            arch=args.arch, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, out_dir=args.out_dir,
            device=device,
            multi_gpu=args.multi_gpu,
        )
