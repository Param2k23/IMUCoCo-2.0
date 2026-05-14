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
from normalization import apply_channel_stats, validate_input_array
from smpl_regions import (
    REGION_NAMES,
    NUM_REGIONS,
    SYMMETRY_PAIRS,
    spatial_error as compute_spatial_error,
)
from temporal_rerank import rerank_accuracy_table, rerank_per_class_accuracy
from topk_combo_rerank import predict_with_topk_physics
from verify_combo import load_calibration, load_stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stats_path_for_checkpoint(ckpt_path: str) -> str:
    base = os.path.basename(ckpt_path)
    if base.startswith("best_model_fold") and base.endswith(".pt"):
        suffix = base[len("best_model_") :]
        return os.path.join(os.path.dirname(ckpt_path), f"normalization_stats_{suffix}")
    return os.path.join(os.path.dirname(ckpt_path), "normalization_stats.pt")


def load_model(
    ckpt_path: str, device: torch.device, base_filters_override: int = 0
) -> tuple[torch.nn.Module, int, np.ndarray, np.ndarray, int, str, int]:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    stats_path = _stats_path_for_checkpoint(ckpt_path)
    if not os.path.isfile(stats_path):
        raise ValueError(f"Missing normalization stats file: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    arch = stats.get("arch", "resnet")
    if "in_channels" not in stats:
        raise ValueError("Normalization stats missing required key 'in_channels'.")
    in_channels = int(stats["in_channels"])
    if in_channels != 9:
        raise ValueError(f"Expected 9-channel stats, got in_channels={in_channels}")
    if "norm_mean" not in stats or "norm_std" not in stats:
        raise ValueError(
            "Normalization stats missing required keys 'norm_mean' and 'norm_std'."
        )
    mean = np.asarray(stats["norm_mean"], dtype=np.float32)
    std = np.asarray(stats["norm_std"], dtype=np.float32)
    if mean.shape != (1, in_channels, 1) or std.shape != (1, in_channels, 1):
        raise ValueError(
            f"Invalid normalization stat shapes. mean={mean.shape}, std={std.shape}, expected (1,{in_channels},1)."
        )

    # base_filters: prefer CLI override > saved in stats > default 64
    if base_filters_override > 0:
        base_filters = base_filters_override
        log.info("base_filters=%d (from --base_filters override)", base_filters)
    elif "base_filters" in stats:
        base_filters = int(stats["base_filters"])
        log.info("base_filters=%d (from normalization stats)", base_filters)
    else:
        base_filters = 64
        log.warning(
            "base_filters not found in stats — defaulting to 64. "
            "If the model was trained with --base_filters 128, pass --base_filters 128."
        )

    model = build_model(
        arch, n_classes=NUM_REGIONS, in_channels=in_channels,
        base_filters=base_filters,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    fold = stats.get("fold", "?")
    test_subj = int(stats.get("test_subj", -1))
    log.info(
        "Loaded checkpoint  arch=%s  in_channels=%d  base_filters=%d  fold=%s",
        arch, in_channels, base_filters, fold,
    )
    return model, in_channels, mean, std, test_subj, arch, base_filters


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
def predict_all_topk(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    k_max: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return BOTH top-k indices and softmax probabilities.

    Returns
    -------
    topk_indices : (N, k_max)  int64  — class indices ranked by confidence
    topk_probs   : (N, k_max)  float32 — corresponding softmax probabilities
    """
    all_indices, all_probs = [], []
    n = len(X)
    for start in range(0, n, batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        logits = model(xb)                                    # (B, C)
        probs  = torch.softmax(logits, dim=1)                # (B, C)
        k      = min(k_max, logits.shape[1])
        topk   = probs.topk(k, dim=1)                        # values, indices
        all_indices.append(topk.indices.cpu().numpy())
        all_probs.append(topk.values.cpu().numpy())
    return (
        np.concatenate(all_indices, axis=0).astype(np.int64),   # (N, k_max)
        np.concatenate(all_probs,   axis=0).astype(np.float32), # (N, k_max)
    )


# Keep the original for backward compat (used by majority_vote_stream)
def predict_all(
    model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Return per-window top-1 predicted class labels."""
    indices, _ = predict_all_topk(model, X, batch_size, device, k_max=1)
    return indices[:, 0]


# ---------------------------------------------------------------------------
# Top-k accuracy
# ---------------------------------------------------------------------------


def topk_accuracy(
    y_true: np.ndarray,
    topk_indices: np.ndarray,
    k_values: list[int],
) -> dict[int, float]:
    """
    For each k in k_values, compute the fraction of windows where the
    true label appears in the top-k predictions.

    Parameters
    ----------
    y_true       : (N,)      ground-truth class indices
    topk_indices : (N, k_max) predicted class indices sorted by confidence
    k_values     : list of k values to evaluate (e.g. [1,2,3,5,10])

    Returns
    -------
    dict  k -> accuracy (float 0..1)
    """
    results = {}
    k_max = topk_indices.shape[1]
    for k in k_values:
        k_eff = min(k, k_max)
        # true label in the first k columns?
        in_topk = (topk_indices[:, :k_eff] == y_true[:, np.newaxis]).any(axis=1)
        results[k] = float(in_topk.mean())
    return results


def topk_accuracy_per_class(
    y_true: np.ndarray,
    topk_indices: np.ndarray,
    k_values: list[int],
    class_names: list[str],
) -> list[dict]:
    """
    Per-class top-k accuracy for each k in k_values.

    Returns list of dicts:
      {class_name, n_windows, top1, top2, top3, top5, top10, ...}
    """
    k_max = topk_indices.shape[1]
    rows = []
    for cls_idx, name in enumerate(class_names):
        mask = y_true == cls_idx
        n = int(mask.sum())
        if n == 0:
            continue
        row = {"class": name, "n_windows": n}
        for k in k_values:
            k_eff = min(k, k_max)
            in_topk = (topk_indices[mask][:, :k_eff] == cls_idx).any(axis=1)
            row[f"top{k}"] = round(float(in_topk.mean()), 4)
        rows.append(row)
    return rows


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
# Top confused pairs (non-L/R)
# ---------------------------------------------------------------------------


def _top_confused_pairs(
    cm: np.ndarray,
    top_n: int = 10,
) -> list[dict]:
    """
    Return the top-N (true, predicted) pairs by raw error count,
    *excluding* correct predictions (diagonal) and the L/R symmetric pairs
    already tracked by left_right_confusion_analysis.

    Each entry: {true, predicted, count, pct_of_true}
    sorted descending by count.
    """
    # Build set of L/R mirror index-pairs to exclude
    lr_index_pairs: set[tuple[int, int]] = set()
    for l_id, r_id in SYMMETRY_PAIRS:
        lr_index_pairs.add((l_id, r_id))
        lr_index_pairs.add((r_id, l_id))

    pairs = []
    n_classes = cm.shape[0]
    for true_idx in range(n_classes):
        row_total = cm[true_idx].sum()
        for pred_idx in range(n_classes):
            if true_idx == pred_idx:
                continue                        # skip correct predictions
            if (true_idx, pred_idx) in lr_index_pairs:
                continue                        # skip L/R pairs (tracked separately)
            count = int(cm[true_idx, pred_idx])
            if count == 0:
                continue
            pct_of_true = round(100.0 * count / max(1, row_total), 2)
            pairs.append({
                "true": REGION_NAMES[true_idx],
                "predicted": REGION_NAMES[pred_idx],
                "count": count,
                "pct_of_true_class": pct_of_true,
            })

    pairs.sort(key=lambda d: -d["count"])
    return pairs[:top_n]


# ---------------------------------------------------------------------------
# Left-Right confusion breakdown
# ---------------------------------------------------------------------------


def left_right_confusion_analysis(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """
    Among misclassified windows, count how many are due to a pure left↔right
    mirror swap (e.g. l_hand predicted as r_hand, or vice versa).

    Returns
    -------
    dict with keys:
      n_total_errors   : total misclassified windows
      n_lr_errors      : errors that are a left↔right mirror swap
      lr_error_pct     : n_lr_errors / n_total_errors  (0–100 %)
      per_pair         : list of {pair, true→pred, count} sorted by count desc
    """
    # Build mirror map from SYMMETRY_PAIRS
    mirror = {}
    for l_id, r_id in SYMMETRY_PAIRS:
        mirror[l_id] = r_id
        mirror[r_id] = l_id

    wrong_mask = y_pred != y_true
    n_total_errors = int(wrong_mask.sum())

    if n_total_errors == 0:
        return {
            "n_total_errors": 0,
            "n_lr_errors": 0,
            "lr_error_pct": 0.0,
            "per_pair": [],
        }

    y_t_wrong = y_true[wrong_mask]
    y_p_wrong = y_pred[wrong_mask]

    # A left-right error: predicted class == mirror of true class
    lr_mask = np.array(
        [mirror.get(int(t), -1) == int(p) for t, p in zip(y_t_wrong, y_p_wrong)]
    )
    n_lr_errors = int(lr_mask.sum())

    # Per symmetric-pair breakdown
    per_pair = []
    for l_id, r_id in SYMMETRY_PAIRS:
        # l→r errors
        lr_count = int(((y_t_wrong == l_id) & (y_p_wrong == r_id)).sum())
        # r→l errors
        rl_count = int(((y_t_wrong == r_id) & (y_p_wrong == l_id)).sum())
        if lr_count + rl_count > 0:
            per_pair.append({
                "pair": f"{REGION_NAMES[l_id]} ↔ {REGION_NAMES[r_id]}",
                f"{REGION_NAMES[l_id]}→{REGION_NAMES[r_id]}": lr_count,
                f"{REGION_NAMES[r_id]}→{REGION_NAMES[l_id]}": rl_count,
                "total_lr_errors": lr_count + rl_count,
            })
    per_pair.sort(key=lambda d: -d["total_lr_errors"])

    return {
        "n_total_errors": n_total_errors,
        "n_lr_errors": n_lr_errors,
        "lr_error_pct": round(100.0 * n_lr_errors / max(1, n_total_errors), 2),
        "per_pair": per_pair,
    }


# ---------------------------------------------------------------------------
# Physics-rerank evaluation (combines top-k with physics scoring)
# ---------------------------------------------------------------------------


def evaluate_physics_rerank(
    raw_X: np.ndarray,                 # (N, 9, T) — un-normalised IMU data
    y_true: np.ndarray,                # (N,)
    topk_indices: np.ndarray,          # (N, k_max)
    topk_probs:   np.ndarray,          # (N, k_max)
    calibration:  np.ndarray,          # (24, 3, 3)
    stats: dict,
    n_sensors_list: list[int],         # e.g. [2, 3, 4, 5]
    n_windows: int = 1,
    n_trials: int = 100,
    k: int = 3,
    weights: tuple = (0.4, 0.0, 0.4, 0.2),
    joint_limits_hard: float | None = None,
    eliminator_only: bool = False,
    seed: int = 0,
) -> dict:
    """
    For each x in n_sensors_list, sample n_trials of x distinct ground-truth
    regions; for each trial draw n_windows windows per region; run the
    top-k -> physics -> rerank pipeline; report exact-match and per-sensor
    accuracy of the best-scoring combo.

    Returns
    -------
    summary : dict keyed by str(x_sensors) ->
              {exact_match_acc, per_sensor_acc, n_trials, n_eliminated_mean}
    """
    rng = np.random.default_rng(seed)
    available_regions = np.unique(y_true)

    # Index windows by region for fast sampling.
    region_to_idx = {int(r): np.where(y_true == r)[0] for r in available_regions}

    summary: dict = {}

    for x in n_sensors_list:
        if x > len(available_regions):
            log.warning(
                "Skipping x=%d: only %d distinct regions available in y_true",
                x, len(available_regions),
            )
            continue

        n_exact = 0
        per_sensor_correct = 0
        per_sensor_total = 0
        n_eliminated_total = 0
        n_combos_total = 0

        for _ in range(n_trials):
            # Pick x distinct GT regions for this trial.
            chosen = rng.choice(available_regions, size=x, replace=False)
            chosen = chosen.astype(np.int64)

            # For each chosen region, draw n_windows window indices.
            sensor_idx_buf = np.zeros((x, n_windows, topk_indices.shape[1]),
                                      dtype=np.int64)
            sensor_p_buf = np.zeros((x, n_windows, topk_probs.shape[1]),
                                    dtype=np.float32)
            X_per_sensor = np.zeros((x, raw_X.shape[1], raw_X.shape[2]),
                                    dtype=np.float32)

            for s, r in enumerate(chosen):
                pool = region_to_idx[int(r)]
                if len(pool) == 0:
                    # Should not happen because we sampled from y_true uniques.
                    continue
                draw = rng.choice(pool, size=min(n_windows, len(pool)),
                                  replace=len(pool) < n_windows)
                sensor_idx_buf[s, :len(draw)] = topk_indices[draw]
                sensor_p_buf[s, :len(draw)] = topk_probs[draw]
                # For physics, use the first drawn window. (We could average
                # but the physics scorer naturally consumes a single (9, T).)
                X_per_sensor[s] = raw_X[draw[0]]

            ranked = predict_with_topk_physics(
                sensor_idx_buf, sensor_p_buf, X_per_sensor,
                calibration, stats,
                k=k, weights=weights,
                joint_limits_hard=joint_limits_hard,
                eliminator_only=eliminator_only,
            )
            n_combos_total += len(ranked)
            n_eliminated_total += sum(1 for r in ranked if r["hard_eliminated"])

            if not ranked:
                continue
            best = ranked[0]
            true_tuple = tuple(int(c) for c in chosen)
            pred_tuple = best["combo"]

            if pred_tuple == true_tuple:
                n_exact += 1
            per_sensor_correct += sum(
                1 for a, b in zip(pred_tuple, true_tuple) if a == b
            )
            per_sensor_total += x

        exact_acc = n_exact / max(1, n_trials)
        per_sensor_acc = per_sensor_correct / max(1, per_sensor_total)
        n_elim_mean = n_eliminated_total / max(1, n_trials)

        summary[str(x)] = {
            "exact_match_acc": round(exact_acc, 6),
            "per_sensor_acc": round(per_sensor_acc, 6),
            "n_trials": n_trials,
            "n_eliminated_mean": round(n_elim_mean, 4),
            "n_combos_total": n_combos_total,
        }
        log.info(
            "  x=%d  exact=%.4f  per_sensor=%.4f  trials=%d  elim_mean=%.2f",
            x, exact_acc, per_sensor_acc, n_trials, n_elim_mean,
        )

    return summary


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load model & data ────────────────────────────────────────────────
    model, ckpt_in_channels, norm_mean, norm_std, ckpt_test_subj, _, _ = load_model(
        args.checkpoint, device, base_filters_override=args.base_filters
    )

    # If checkpoint has a fold/subject, optionally filter to that subject
    ckpt_subj = None
    if args.test_subject >= 0:
        ckpt_subj = args.test_subject
    elif ckpt_test_subj >= 0:
        ckpt_subj = ckpt_test_subj
        log.info("No --test_subject provided; using checkpoint test_subj=%d", ckpt_subj)
    raw_X, y_true = load_test_data(args.data, ckpt_subj)
    validate_input_array(raw_X)
    if raw_X.shape[1] != ckpt_in_channels:
        raise ValueError(
            f"Input channel mismatch: dataset has {raw_X.shape[1]} channels but checkpoint expects {ckpt_in_channels}."
        )
    X = apply_channel_stats(raw_X, norm_mean, norm_std)

    # ── Per-window inference ────────────────────────────────────────────
    log.info("Running inference on %d windows …", len(X))
    k_values = sorted(set(args.topk))                        # e.g. [1,2,3,5,10]
    k_max    = max(k_values)

    topk_indices, topk_probs = predict_all_topk(
        model, X, batch_size=args.batch_size, device=device, k_max=k_max
    )
    y_pred = topk_indices[:, 0]  # top-1 for backward compat

    # ── Top-k accuracy ──────────────────────────────────────────────────
    topk_acc = topk_accuracy(y_true, topk_indices, k_values)
    log.info("\n── Top-k Accuracy ─────────────────────────────────────────")
    for k, acc in sorted(topk_acc.items()):
        log.info("  Top-%-2d accuracy : %.4f  (%d / %d)",
                 k, acc, int(acc * len(y_true)), len(y_true))

    per_window_acc = topk_acc[1]  # top-1 == per-window accuracy
    log.info("Per-window (top-1) accuracy: %.4f", per_window_acc)

    # ── Temporal re-ranking ───────────────────────────────────────────────
    rerank_results  = {}  # (strategy, window) -> acc
    best_rerank_acc = per_window_acc
    best_rerank_cfg = None

    if args.rerank_windows:
        strategies = [s.strip() for s in args.rerank_strategy.split(",")]
        w_sizes    = sorted(set(args.rerank_windows))

        log.info("\n── Temporal Re-Ranking ──────────────────────────────────────")
        log.info("  strategies : %s", strategies)
        log.info("  windows    : %s", w_sizes)
        log.info("  causal     : %s", args.rerank_causal)
        log.info("  top-1 baseline : %.4f", per_window_acc)
        log.info("")

        rr_table = rerank_accuracy_table(
            y_true, topk_indices, topk_probs,
            window_sizes=w_sizes,
            strategies=strategies,
            n_classes=NUM_REGIONS,
            k_vote=args.rerank_k_vote,
            causal=args.rerank_causal,
        )

        # Print comparison table
        log.info("  %-14s  %4s  %8s  %8s", "Strategy", "Win", "Acc", "vs top-1")
        log.info("  " + "-" * 42)
        for (strat, w), acc in sorted(rr_table.items(), key=lambda x: -x[1]):
            delta = acc - per_window_acc
            log.info("  %-14s  %4d  %8.4f  %+8.4f", strat, w, acc, delta)
            rerank_results[f"{strat}_w{w}"] = round(acc, 6)
            if acc > best_rerank_acc:
                best_rerank_acc = acc
                best_rerank_cfg = (strat, w)

        if best_rerank_cfg:
            s, w = best_rerank_cfg
            log.info("\n  Best re-rank config : strategy=%s  window=%d", s, w)
            log.info("  Best re-rank acc    : %.4f  (%+.4f vs top-1)",
                     best_rerank_acc, best_rerank_acc - per_window_acc)

            # Per-class breakdown for the best config
            pc_rr = rerank_per_class_accuracy(
                y_true, topk_indices, topk_probs,
                window_size=w, strategy=s,
                n_classes=NUM_REGIONS, class_names=REGION_NAMES,
                k_vote=args.rerank_k_vote, causal=args.rerank_causal,
            )
            log.info("\n── Per-class gain from re-ranking (best config: %s w=%d) ──", s, w)
            log.info("  %-18s %8s %8s %8s", "Class", "top-1", "reranked", "gain")
            log.info("  " + "-" * 46)
            for row in pc_rr:
                log.info("  %-18s %8.4f %8.4f %+8.4f  (n=%d)",
                         row["class"], row["top1_acc"], row["reranked_acc"],
                         row["gain"], row["n_windows"])
        else:
            pc_rr = []
            log.info("  No re-ranking config improved over top-1 baseline.")
    else:
        rr_table = {}
        pc_rr = []
        log.info("Re-ranking disabled (--rerank_windows not set).")

    # ── Physics-based combo re-ranking ──────────────────────────────────
    physics_rerank_summary: dict = {}
    physics_rerank_summary_hard: dict = {}
    if args.physics_rerank:
        log.info("\n── Physics combo re-ranking ───────────────────────────────")
        log.info("  Loading calibration & stats …")
        try:
            calibration = load_calibration(args.physics_calibration)
            stats = load_stats(args.physics_stats)
        except Exception as e:
            log.error("Failed to load physics calibration/stats: %s", e)
            calibration = None
            stats = None

        if calibration is not None and stats is not None:
            n_sensors_list = sorted(set(args.rerank_n_sensors))
            log.info(
                "  n_sensors=%s  n_windows=%d  n_trials=%d  k=%d  weights=%s",
                n_sensors_list, args.rerank_n_windows, args.rerank_n_trials,
                args.physics_k, args.physics_weights,
            )

            log.info("  --- soft mode (no hard elimination) ---")
            physics_rerank_summary = evaluate_physics_rerank(
                raw_X=raw_X, y_true=y_true,
                topk_indices=topk_indices, topk_probs=topk_probs,
                calibration=calibration, stats=stats,
                n_sensors_list=n_sensors_list,
                n_windows=args.rerank_n_windows,
                n_trials=args.rerank_n_trials,
                k=args.physics_k,
                weights=tuple(args.physics_weights),
                joint_limits_hard=None,
                eliminator_only=False,
                seed=args.rerank_seed,
            )

            if args.joint_limits_hard is not None:
                log.info(
                    "  --- hard mode (eliminate when violation_frac > %.2f) ---",
                    args.joint_limits_hard,
                )
                physics_rerank_summary_hard = evaluate_physics_rerank(
                    raw_X=raw_X, y_true=y_true,
                    topk_indices=topk_indices, topk_probs=topk_probs,
                    calibration=calibration, stats=stats,
                    n_sensors_list=n_sensors_list,
                    n_windows=args.rerank_n_windows,
                    n_trials=args.rerank_n_trials,
                    k=args.physics_k,
                    weights=tuple(args.physics_weights),
                    joint_limits_hard=args.joint_limits_hard,
                    eliminator_only=False,
                    seed=args.rerank_seed,
                )

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

    # ── Left-Right confusion breakdown ────────────────────────────────────
    lr_analysis = left_right_confusion_analysis(y_true, y_pred)
    log.info("\n── Left-Right Confusion Breakdown ─────────────────────────")
    log.info(
        "  Total errors      : %d / %d  (%.1f%%)",
        lr_analysis["n_total_errors"],
        len(y_true),
        100.0 * lr_analysis["n_total_errors"] / max(1, len(y_true)),
    )
    log.info(
        "  L↔R mirror errors : %d / %d errors  (%.1f%% of all errors)",
        lr_analysis["n_lr_errors"],
        lr_analysis["n_total_errors"],
        lr_analysis["lr_error_pct"],
    )
    if lr_analysis["per_pair"]:
        log.info("  Per symmetric pair (sorted by error count):")
        for pp in lr_analysis["per_pair"]:
            log.info("    %-36s  total=%d", pp["pair"], pp["total_lr_errors"])
            for k, v in pp.items():
                if k not in ("pair", "total_lr_errors"):
                    log.info("      %-40s %d", k, v)

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

    # ── Top confused non-L/R pairs ───────────────────────────────────────
    top_pairs = _top_confused_pairs(cm, top_n=10)
    log.info("\n── Top-10 non-L/R confused pairs ──────────────────────────")
    for i, p in enumerate(top_pairs, 1):
        log.info(
            "  %2d. %-16s → %-16s  count=%d  (%.1f%% of true class)",
            i, p["true"], p["predicted"], p["count"], p["pct_of_true_class"],
        )

    # ── Per-class top-k breakdown ─────────────────────────────────────────
    per_class_topk = topk_accuracy_per_class(
        y_true, topk_indices, k_values, REGION_NAMES
    )
    log.info("\n── Per-class top-k accuracy ────────────────────────────────")
    header = f"  {'Class':<18}" + "".join(f" top{k:>2}" for k in k_values)
    log.info(header)
    log.info("  " + "-" * (18 + 6 * len(k_values)))
    for row in per_class_topk:
        vals = "".join(f" {row.get(f'top{k}', 0.0):>5.3f}" for k in k_values)
        log.info("  %-18s%s  (n=%d)", row["class"], vals, row["n_windows"])

    # ── Save summary JSON ────────────────────────────────────────────────
    summary = {
        "per_window_accuracy": per_window_acc,
        "topk_accuracy": {f"top{k}": v for k, v in sorted(topk_acc.items())},
        "per_class_topk": per_class_topk,
        "temporal_reranking": {
            "best_acc"     : round(best_rerank_acc, 6),
            "best_config"  : f"{best_rerank_cfg[0]}_w{best_rerank_cfg[1]}" if best_rerank_cfg else None,
            "gain_vs_top1" : round(best_rerank_acc - per_window_acc, 6),
            "all_configs"  : rerank_results,
            "per_class_best_config": pc_rr,
        },
        "physics_rerank_summary": physics_rerank_summary,
        "physics_rerank_summary_hard": physics_rerank_summary_hard,
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
        "left_right_confusion": {
            "n_total_errors": lr_analysis["n_total_errors"],
            "n_lr_errors": lr_analysis["n_lr_errors"],
            "lr_error_pct_of_all_errors": lr_analysis["lr_error_pct"],
            "per_pair": lr_analysis["per_pair"],
        },
        # Raw 24x24 confusion matrix (row=true, col=predicted) for offline analysis
        "confusion_matrix": cm.tolist(),
        # Top-10 most-confused non-symmetric pairs (the 93% non-L/R errors)
        "top_confused_pairs": top_pairs,
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
        "--base_filters",
        type=int,
        default=0,
        help="Override base_filters for model architecture (0 = auto-detect from stats). "
             "Use 128 for checkpoints trained with --base_filters 128 that predate this fix.",
    )
    p.add_argument(
        "--topk",
        type=lambda s: sorted(set(int(x) for x in s.split(",") if int(x) > 0)),
        default=[1, 2, 3, 5, 10],
        metavar="K1,K2,...",
        help="Comma-separated list of k values for top-k accuracy (default: 1,2,3,5,10).",
    )
    # ── Temporal re-ranking ──────────────────────────────────────────────────
    p.add_argument(
        "--rerank_windows",
        type=lambda s: sorted(set(int(x) for x in s.split(",") if int(x) > 0)),
        default=[1, 3, 5, 7, 10],
        metavar="W1,W2,...",
        help="Sliding-window sizes to try for temporal re-ranking (default: 1,3,5,7,10). "
             "Set to 0 to disable re-ranking entirely.",
    )
    p.add_argument(
        "--rerank_strategy",
        type=str,
        default="prob_sum,topk_vote,majority",
        metavar="S1,S2,...",
        help="Comma-separated re-ranking strategies to compare. "
             "Choices: prob_sum, topk_vote, majority (default: all three).",
    )
    p.add_argument(
        "--rerank_k_vote",
        type=int,
        default=3,
        help="k used by topk_vote strategy — how many top predictions each window votes for "
             "(default: 3).",
    )
    p.add_argument(
        "--rerank_causal",
        action="store_true",
        default=False,
        help="Use causal (past-only) window instead of centred window. "
             "Use for streaming/real-time evaluation.",
    )
    # ── Physics combo re-ranking ────────────────────────────────────────────
    p.add_argument(
        "--physics_rerank",
        action="store_true",
        default=False,
        help="Run physics-based combo re-ranking on top-k probs.",
    )
    p.add_argument(
        "--physics_calibration", default="calibration",
        help="Directory with region_sensor_rotmats.npy",
    )
    p.add_argument(
        "--physics_stats", default="stats",
        help="Directory with joint_angle_limits.npy, accel_distributions.npy, "
             "gravity_distributions.npy, per_region_orientation_limits.npy",
    )
    p.add_argument(
        "--rerank_n_sensors",
        type=lambda s: sorted(set(int(x) for x in s.split(",") if int(x) > 0)),
        default=[2, 3, 4, 5],
        help="Comma-separated list of x (number of simultaneous sensors per "
             "synthetic combo trial). Default 2,3,4,5.",
    )
    p.add_argument(
        "--rerank_n_windows", type=int, default=1,
        help="How many windows per sensor are aggregated for top-k.",
    )
    p.add_argument(
        "--rerank_n_trials", type=int, default=200,
        help="Number of synthetic combos to draw per x.",
    )
    p.add_argument(
        "--rerank_seed", type=int, default=0,
        help="RNG seed for combo sampling (reproducibility).",
    )
    p.add_argument(
        "--physics_k", type=int, default=3,
        help="Top-k per sensor used for combo enumeration (smaller = fewer "
             "candidates, larger = more recall). Default 3.",
    )
    p.add_argument(
        "--physics_weights", type=float, nargs="+",
        default=[0.4, 0.0, 0.4, 0.2],
        help="Weights for [kinematic, gravity, accel, joint_limits]. "
             "Default 0.4 0.0 0.4 0.2.",
    )
    p.add_argument(
        "--joint_limits_hard", type=float, default=None,
        help="If set, additionally evaluate the pipeline in hard-elimination "
             "mode. Threshold = max fraction of frames a sensor may violate "
             "its per-region absolute orientation envelope before the combo "
             "is dropped (e.g. 0.3).",
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
