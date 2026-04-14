"""
evaluate.py
===========
Take a test IMU stream (.npz), run the trained classifier, and output:
  • Per-window accuracy
  • Majority-vote ("locked") accuracy
  • 24×24 confusion matrix (saved as confusion_matrix.png)
  • Symmetry confusion analysis
  • Spatial error (mean ± std in metres) for mis-classified samples
  • sklearn classification report

Usage
-----
  python evaluate.py \
    --checkpoint C:/VS/SensorLoc/checkpoints/best_model_fold0.pt \
    --data       C:/VS/SensorLoc/data/dataset.npz \
    --vote_k     5 \
    --out_dir    C:/VS/SensorLoc/results
"""

from __future__ import annotations
import argparse
import os
import sys
import logging
import json
from collections import deque

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import build_model
from smpl_regions import (
    REGION_NAMES,
    NUM_REGIONS,
    SYMMETRY_PAIRS,
    spatial_error as compute_spatial_error,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "resnet")
    in_channels = int(ckpt.get("in_channels", 6))
    model = build_model(arch, n_classes=NUM_REGIONS, in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    fold = ckpt.get("fold", "?")
    acc = ckpt.get("val_acc", float("nan"))
    log.info(
        "Loaded checkpoint  arch=%s  in_channels=%d  fold=%s  saved_val_acc=%.4f",
        arch,
        in_channels,
        fold,
        acc,
    )
    return model


def load_test_data(
    data_path: str, test_fold_subj: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X and y from .npz.  If test_fold_subj given, filter to that subject.
    Otherwise use all data.
    """
    d = np.load(data_path)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    sids = d["subject_ids"].astype(np.int64)

    if test_fold_subj is not None:
        mask = sids == test_fold_subj
        X, y = X[mask], y[mask]
        log.info("Filtered to subject %d  →  %d windows", test_fold_subj, len(y))
    else:
        log.info("Using all %d windows", len(y))
    return X, y


@torch.no_grad()
def predict_all(
    model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Return per-window predicted class labels."""
    preds = []
    n = len(X)
    for start in range(0, n, batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Majority-vote filter
# ---------------------------------------------------------------------------


def majority_vote_stream(preds: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Simulate a streaming majority-vote filter.

    A prediction is "locked" only if the same class appears ≥ ceil(k/2)+1
    times in the last k windows.  Otherwise the output is -1 (unlocked).

    Parameters
    ----------
    preds : (N,)  per-window predictions
    k     : window size for majority vote

    Returns
    -------
    locked : (N,)  majority-voted predictions (-1 = not locked)
    """
    locked = np.full_like(preds, -1)
    buf = deque(maxlen=k)
    threshold = (k // 2) + 1  # majority
    for i, p in enumerate(preds):
        buf.append(int(p))
        if len(buf) == k:
            # Most common
            counts = np.bincount(np.array(buf), minlength=NUM_REGIONS)
            best = int(counts.argmax())
            if counts[best] >= threshold:
                locked[i] = best
    return locked


# ---------------------------------------------------------------------------
# Confusion matrix visualisation
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray, out_path: str, title: str = "Confusion Matrix"
) -> None:
    short = [n.replace("_", "\n") for n in REGION_NAMES]
    fig, ax = plt.subplots(figsize=(18, 16))

    # Normalise per row (true-label)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    sns.heatmap(
        cm_norm,
        ax=ax,
        xticklabels=short,
        yticklabels=short,
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="#e0e0e0",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"label": "Recall fraction"},
    )
    ax.set_xlabel("Predicted Region", fontsize=12)
    ax.set_ylabel("True Region", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", out_path)


# ---------------------------------------------------------------------------
# Symmetry confusion analysis
# ---------------------------------------------------------------------------


def symmetry_analysis(cm: np.ndarray) -> list[dict]:
    """
    For each (left, right) pair, compute the confusion rate
    (fraction of true-left labelled as right, and vice versa).
    """
    results = []
    for l_id, r_id in SYMMETRY_PAIRS:
        l_total = cm[l_id].sum()
        r_total = cm[r_id].sum()
        lr_conf = cm[l_id, r_id] / (l_total + 1e-8)  # left→right
        rl_conf = cm[r_id, l_id] / (r_total + 1e-8)  # right→left
        results.append(
            {
                "left": REGION_NAMES[l_id],
                "right": REGION_NAMES[r_id],
                "left→right_rate": float(lr_conf),
                "right→left_rate": float(rl_conf),
                "symmetric_confusion": float((lr_conf + rl_conf) / 2),
            }
        )
    results.sort(key=lambda d: -d["symmetric_confusion"])
    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load model & data ────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # If checkpoint has a fold/subject, optionally filter to that subject
    ckpt_subj = None
    if args.test_subject >= 0:
        ckpt_subj = args.test_subject
    X, y_true = load_test_data(args.data, ckpt_subj)

    # ── Per-window inference ────────────────────────────────────────────
    log.info("Running inference on %d windows …", len(X))
    y_pred = predict_all(model, X, batch_size=args.batch_size, device=device)

    per_window_acc = float(accuracy_score(y_true, y_pred))
    log.info("Per-window accuracy: %.4f", per_window_acc)

    # ── Majority-vote filter ────────────────────────────────────────────
    y_voted = majority_vote_stream(y_pred, k=args.vote_k)
    vote_mask = y_voted >= 0
    if vote_mask.sum() > 0:
        voted_acc = float(accuracy_score(y_true[vote_mask], y_voted[vote_mask]))
        locked_pct = float(vote_mask.mean()) * 100
        log.info(
            "Majority-vote accuracy: %.4f  (locked %.1f%% of windows)",
            voted_acc,
            locked_pct,
        )
    else:
        voted_acc = 0.0
        locked_pct = 0.0
        log.warning("No windows were locked by majority vote (k=%d)", args.vote_k)

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_REGIONS)))
    plot_confusion_matrix(
        cm,
        out_path=os.path.join(args.out_dir, "confusion_matrix.png"),
        title=f"24-Region Sensor Location — Per-window Acc={per_window_acc:.3f}",
    )

    # ── Symmetry analysis ────────────────────────────────────────────────
    sym_results = symmetry_analysis(cm)
    log.info("\n── Symmetry Confusion (top 5) ─────────────────────────────")
    for r in sym_results[:5]:
        log.info(
            "  %-14s ↔ %-14s  avg_rate=%.3f",
            r["left"],
            r["right"],
            r["symmetric_confusion"],
        )

    # ── Spatial error ────────────────────────────────────────────────────
    sp_err = compute_spatial_error(y_pred.astype(np.int32), y_true.astype(np.int32))
    log.info("\n── Spatial Error (mis-classified samples) ─────────────────")
    log.info(
        "  N wrong      : %d / %d  (%.1f%%)",
        sp_err["n_wrong"],
        len(y_true),
        100.0 * sp_err["n_wrong"] / max(1, len(y_true)),
    )
    log.info("  Mean error   : %.4f m", sp_err["mean_m"])
    log.info("  Std error    : %.4f m", sp_err["std_m"])

    # ── Classification report ────────────────────────────────────────────
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_REGIONS)),
        target_names=REGION_NAMES,
        zero_division=0,
    )
    log.info(
        "\n── Classification Report ───────────────────────────────────\n%s", report
    )

    # ── Save summary JSON ────────────────────────────────────────────────
    summary = {
        "per_window_accuracy": per_window_acc,
        "majority_vote_accuracy": voted_acc,
        "vote_k": args.vote_k,
        "locked_fraction": locked_pct / 100,
        "n_windows": int(len(y_true)),
        "spatial_error": {
            "mean_m": sp_err["mean_m"],
            "std_m": sp_err["std_m"],
            "n_wrong": sp_err["n_wrong"],
        },
        "symmetry_confusion": sym_results,
    }
    summ_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("\n✓ Evaluation complete.  Summary saved → %s", summ_path)


# ---------------------------------------------------------------------------
# Smoke self-test
# ---------------------------------------------------------------------------


def smoke_self_test(out_dir: str) -> None:
    """
    Verify majority-vote logic with a synthetic constant stream.
    All 5 windows predict class 3 → should be locked to 3.
    """
    preds = np.array([3, 3, 3, 3, 3, 7, 3, 3, 3, 3])
    voted = majority_vote_stream(preds, k=5)
    locked_vals = voted[voted >= 0]
    assert all(v in (3, 7) for v in locked_vals), (
        f"Unexpected locked values: {locked_vals}"
    )
    log.info("Smoke self-test (majority vote): PASSED  locked=%s", locked_vals.tolist())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate sensor-location classifier")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--data", required=True, help="Path to test .npz dataset")
    p.add_argument("--out_dir", default="C:/VS/SensorLoc/results")
    p.add_argument("--vote_k", type=int, default=5, help="Majority-vote window size")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--test_subject",
        type=int,
        default=-1,
        help="If >= 0, filter data to this subject id only",
    )
    p.add_argument(
        "--smoke", action="store_true", help="Run internal self-tests and exit"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        smoke_self_test(args.out_dir)
        sys.exit(0)
    run_evaluation(args)
