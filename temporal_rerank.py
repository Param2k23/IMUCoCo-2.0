"""
temporal_rerank.py
==================
Temporal re-ranking for IMU sensor-location classification.

Instead of accepting the per-window argmax as the final prediction, we
aggregate the softmax probability distributions across a short sliding
window of consecutive frames.  Because consecutive windows come from
the same physical sensor, their distributions should agree — and a class
that narrowly beats the true label in one frame is often out-voted once
neighbouring frames are included.

Three strategies are provided so they can be compared easily:

  prob_sum   — Sum raw softmax probabilities across the window (default).
               Works best when confidence is well-calibrated.

  topk_vote  — Each window casts a vote for every class that appears in
               its top-k predictions (weighted by softmax prob).
               Practically identical to prob_sum when k == num_classes,
               but more interpretable when k is small (e.g. k=3).

  majority   — Classic majority vote: each window casts one vote for its
               top-1 prediction.  Equivalent to the existing
               majority_vote_stream but always produces a prediction
               (no "locked / unlocked" concept).

Usage
-----
    from temporal_rerank import temporal_rerank, rerank_accuracy_table

    # topk_indices : (N, k_max)  from predict_all_topk
    # topk_probs   : (N, k_max)  from predict_all_topk
    results = rerank_accuracy_table(
        y_true, topk_indices, topk_probs,
        window_sizes=[1, 3, 5, 7, 10],
        strategies=["prob_sum", "topk_vote", "majority"],
        n_classes=24,
        causal=False,    # True = streaming (past-only), False = offline (centred)
    )
    # results: dict  (strategy, window_size) -> accuracy float
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _build_prob_matrix(
    topk_indices: np.ndarray,
    topk_probs: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Convert sparse top-k representation to a dense (N, n_classes) probability
    matrix.  Entries not in the top-k are left at zero.

    Parameters
    ----------
    topk_indices : (N, k)  int64
    topk_probs   : (N, k)  float32  (softmax probabilities, row-normalised)
    n_classes    : int

    Returns
    -------
    prob_matrix  : (N, n_classes)  float32
    """
    N, k = topk_indices.shape
    prob_matrix = np.zeros((N, n_classes), dtype=np.float32)
    # Vectorised scatter: for each sample, scatter probs into correct columns
    rows = np.arange(N).repeat(k)
    cols = topk_indices.reshape(-1)
    vals = topk_probs.reshape(-1)
    np.add.at(prob_matrix, (rows, cols), vals)
    return prob_matrix


def _sliding_sum(
    prob_matrix: np.ndarray,
    window_size: int,
    causal: bool,
) -> np.ndarray:
    """
    For every position i, sum prob_matrix rows over a sliding window and
    return the argmax class.

    causal=False  → centred window [i - half_w, i + half_w]  (offline)
    causal=True   → left-only window [i - window_size + 1, i]  (streaming)

    Returns
    -------
    y_reranked : (N,)  int64
    """
    N, C = prob_matrix.shape
    if window_size == 1:
        return prob_matrix.argmax(axis=1).astype(np.int64)

    half_w = window_size // 2
    y_reranked = np.empty(N, dtype=np.int64)

    for i in range(N):
        if causal:
            start = max(0, i - window_size + 1)
            end   = i + 1
        else:
            start = max(0, i - half_w)
            end   = min(N, i + half_w + 1)
        y_reranked[i] = prob_matrix[start:end].sum(axis=0).argmax()

    return y_reranked


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def rerank_prob_sum(
    topk_indices: np.ndarray,
    topk_probs: np.ndarray,
    window_size: int,
    n_classes: int,
    causal: bool = False,
) -> np.ndarray:
    """
    Sum softmax probabilities over a sliding window and pick argmax.

    Best strategy when the model's confidence is well-calibrated.
    """
    prob_matrix = _build_prob_matrix(topk_indices, topk_probs, n_classes)
    return _sliding_sum(prob_matrix, window_size, causal)


def rerank_topk_vote(
    topk_indices: np.ndarray,
    topk_probs: np.ndarray,
    window_size: int,
    n_classes: int,
    k_vote: int = 3,
    causal: bool = False,
) -> np.ndarray:
    """
    Each window contributes a weighted vote for every class in its top-k_vote
    predictions.  The weight is the softmax probability.

    When k_vote >= k_max this is identical to prob_sum.
    When k_vote == 1 this is a probability-weighted majority vote.
    """
    k_eff = min(k_vote, topk_indices.shape[1])
    prob_matrix = _build_prob_matrix(
        topk_indices[:, :k_eff], topk_probs[:, :k_eff], n_classes
    )
    return _sliding_sum(prob_matrix, window_size, causal)


def rerank_majority(
    topk_indices: np.ndarray,
    window_size: int,
    n_classes: int,
    causal: bool = False,
) -> np.ndarray:
    """
    Classic majority vote: each window casts one hard vote for its top-1
    prediction.  The class with the most votes in the window wins.
    """
    y_pred = topk_indices[:, 0].astype(np.int64)
    # Build one-hot matrix then use sliding_sum
    one_hot = np.zeros((len(y_pred), n_classes), dtype=np.float32)
    one_hot[np.arange(len(y_pred)), y_pred] = 1.0
    return _sliding_sum(one_hot, window_size, causal)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def temporal_rerank(
    topk_indices: np.ndarray,
    topk_probs: np.ndarray,
    window_size: int,
    n_classes: int,
    strategy: str = "prob_sum",
    k_vote: int = 3,
    causal: bool = False,
) -> np.ndarray:
    """
    Apply temporal re-ranking and return re-ranked predictions (N,).

    Parameters
    ----------
    topk_indices : (N, k_max)
    topk_probs   : (N, k_max)
    window_size  : int  — number of consecutive windows to aggregate
    n_classes    : int  — total number of output classes
    strategy     : "prob_sum" | "topk_vote" | "majority"
    k_vote       : int  — only used by "topk_vote" strategy
    causal       : bool — True = past-only window (streaming mode)

    Returns
    -------
    y_reranked : (N,)  int64
    """
    if strategy == "prob_sum":
        return rerank_prob_sum(topk_indices, topk_probs, window_size, n_classes, causal)
    elif strategy == "topk_vote":
        return rerank_topk_vote(topk_indices, topk_probs, window_size, n_classes, k_vote, causal)
    elif strategy == "majority":
        return rerank_majority(topk_indices, window_size, n_classes, causal)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from prob_sum, topk_vote, majority")


def rerank_accuracy_table(
    y_true: np.ndarray,
    topk_indices: np.ndarray,
    topk_probs: np.ndarray,
    window_sizes: list[int],
    strategies: list[str],
    n_classes: int,
    k_vote: int = 3,
    causal: bool = False,
) -> dict[tuple[str, int], float]:
    """
    Compute re-ranked accuracy for every (strategy, window_size) combination.

    Returns
    -------
    results : dict  (strategy, window_size) -> accuracy float
    """
    results = {}
    for strategy in strategies:
        for w in window_sizes:
            y_rr = temporal_rerank(
                topk_indices, topk_probs,
                window_size=w, n_classes=n_classes,
                strategy=strategy, k_vote=k_vote, causal=causal,
            )
            acc = float((y_rr == y_true).mean())
            results[(strategy, w)] = acc
    return results


def rerank_per_class_accuracy(
    y_true: np.ndarray,
    topk_indices: np.ndarray,
    topk_probs: np.ndarray,
    window_size: int,
    strategy: str,
    n_classes: int,
    class_names: list[str],
    k_vote: int = 3,
    causal: bool = False,
) -> list[dict]:
    """
    Per-class accuracy after re-ranking with the given (strategy, window_size).

    Returns list of {class, n_windows, top1_acc, reranked_acc, gain}
    """
    y_top1    = topk_indices[:, 0]
    y_reranked = temporal_rerank(
        topk_indices, topk_probs,
        window_size=window_size, n_classes=n_classes,
        strategy=strategy, k_vote=k_vote, causal=causal,
    )

    rows = []
    for cls_idx, name in enumerate(class_names):
        mask = y_true == cls_idx
        n = int(mask.sum())
        if n == 0:
            continue
        top1_acc = float((y_top1[mask] == cls_idx).mean())
        rr_acc   = float((y_reranked[mask] == cls_idx).mean())
        rows.append({
            "class"        : name,
            "n_windows"    : n,
            "top1_acc"     : round(top1_acc, 4),
            "reranked_acc" : round(rr_acc, 4),
            "gain"         : round(rr_acc - top1_acc, 4),
        })
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """Quick sanity check — run with:  python temporal_rerank.py"""
    rng = np.random.default_rng(42)
    N, C, k = 100, 24, 5
    # Simulate a sequence where class 3 is the true label, but model sometimes
    # puts it 2nd (class 7 beats it in individual windows)
    true_cls = np.full(N, 3, dtype=np.int64)
    indices  = rng.integers(0, C, size=(N, k))
    indices[:, 0] = rng.choice([3, 7], size=N, p=[0.6, 0.4])  # noisy top-1
    indices[:, 1] = np.where(indices[:, 0] == 3, 7, 3)         # 2nd is the other
    probs    = np.sort(rng.dirichlet(np.ones(k), size=N), axis=1)[:, ::-1].astype(np.float32)

    results = rerank_accuracy_table(
        true_cls, indices, probs,
        window_sizes=[1, 3, 5, 7],
        strategies=["prob_sum", "topk_vote", "majority"],
        n_classes=C,
    )
    print("Smoke test — reranking accuracy table:")
    print(f"  {'Strategy':<12} {'W':>4}  Acc")
    print("  " + "-" * 26)
    for (strat, w), acc in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"  {strat:<12} {w:>4}  {acc:.4f}")
    print("OK")
