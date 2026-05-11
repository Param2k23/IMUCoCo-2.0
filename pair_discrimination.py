"""
pair_discrimination.py
======================
Per-confusion-pair discrimination test for the kinematic biophysical scorer.

Question
--------
For each top confused pair (true=A, classifier-output=B) in
`results/penalty_sweep/eval_best_final/confusion_aggregate.json`, embed A
inside a multi-sensor combo. The classifier hands back the wrong combo
(B substituted for A, *same underlying sensor data*). Can the kinematic
scorer score the true combo strictly above the swapped one?

Two regimes per pair
--------------------
1. best_case  : context = parent of A + one child of A (kinematic neighbors).
                Combo size 2-3. Upper bound on what the scorer can do.
2. realistic  : context = random regions excluding {A, B}, several trials.
                Combo size 2-5. Estimates how often the scorer has anything
                to say in the wild.

Output
------
- stats/pair_discrimination_report.json : full numeric results
- stdout: table sorted by error count

This is read-only diagnostic. The kinematic scorer in verify_combo.py is
not modified; we measure what it already does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_sweep import _auc, _build_segment_index  # noqa: E402
from smpl_regions import REGION_NAMES, REGION_PARENTS  # noqa: E402
from verify_combo import (  # noqa: E402
    load_calibration,
    load_data,
    load_stats,
    score_kinematic_chain,
)

NAME_TO_ID = {name: i for i, name in enumerate(REGION_NAMES)}


# ---------------------------------------------------------------------------
# Pair list construction
# ---------------------------------------------------------------------------
def parse_confusion_pairs(confusion_path: str) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """Return (directed non-LR pairs, directed LR pairs).

    LR entries in the JSON are undirected; we expand each into two directed
    pairs that share the count (the actual direction split is unknown from
    the aggregate file).
    """
    with open(confusion_path) as f:
        agg = json.load(f)

    nonlr: List[Tuple[str, str, int]] = []
    for entry in agg["top_nonlr_pairs"]:
        a, b = entry["pair"].split(" -> ")
        nonlr.append((a, b, int(entry["total_count"])))

    lr: List[Tuple[str, str, int]] = []
    for key, count in agg["lr_pair_totals"].items():
        # Key uses U+2194 LEFT RIGHT ARROW
        a, b = [s.strip() for s in key.split("↔")]
        lr.append((a, b, int(count)))
        lr.append((b, a, int(count)))

    return nonlr, lr


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------
def _children_of(region_id: int) -> List[int]:
    return [r for r, p in REGION_PARENTS.items() if p == region_id]


def best_case_context(a: int, b: int) -> List[int]:
    """Return parent(A) + one child(A), excluding B. Falls back gracefully."""
    extras: List[int] = []
    parent = REGION_PARENTS.get(a, -1)
    if parent != -1 and parent != b:
        extras.append(parent)
    children = [c for c in _children_of(a) if c != b]
    if children:
        extras.append(children[0])
    # Last resort: include grandparent or sibling that isn't B
    if not extras:
        gp = REGION_PARENTS.get(parent, -1) if parent != -1 else -1
        if gp != -1 and gp != b:
            extras.append(gp)
    # Final fallback: any region != A, B
    if not extras:
        for r in range(24):
            if r != a and r != b:
                extras.append(r)
                break
    return extras


# ---------------------------------------------------------------------------
# Single-pair experiment
# ---------------------------------------------------------------------------
def _score_combo(combo: List[int], Xs: np.ndarray, calibration, jal) -> Tuple[float, int]:
    """Return (kinematic_score, n_valid_envelope_pairs)."""
    score, diag = score_kinematic_chain(combo, Xs, calibration, jal)
    n_valid = int(diag.get("valid_pairs", 0))
    return score, n_valid


def run_pair(
    a_name: str,
    b_name: str,
    X: np.ndarray,
    segments: List[Dict[int, int]],
    calibration: np.ndarray,
    jal: dict,
    realistic_sizes: Tuple[int, ...] = (2, 3, 4, 5),
    n_random_contexts: int = 8,
    rng: np.random.Generator = None,
) -> Dict:
    if rng is None:
        rng = np.random.default_rng(0)
    a = NAME_TO_ID[a_name]
    b = NAME_TO_ID[b_name]

    # ---- Best case ----
    extras = best_case_context(a, b)
    combo_true = [a] + extras
    combo_swap = [b] + extras
    true_scores, swap_scores = [], []
    n_valid_true, n_valid_swap = [], []
    for seg in segments:
        # Need every region in the combo to be present in this segment.
        if not all(r in seg for r in combo_true) or not all(r in seg for r in combo_swap):
            continue
        Xs = np.stack([X[seg[r]] for r in combo_true])  # data uses A
        st, vt = _score_combo(combo_true, Xs, calibration, jal)
        ss, vs = _score_combo(combo_swap, Xs, calibration, jal)
        true_scores.append(st)
        swap_scores.append(ss)
        n_valid_true.append(vt)
        n_valid_swap.append(vs)

    best = _summarize(true_scores, swap_scores, n_valid_true, n_valid_swap, extras=extras)

    # ---- Realistic case ----
    realistic = {}
    other_regions = [r for r in range(24) if r != a and r != b]
    for n_total in realistic_sizes:
        n_extra = n_total - 1
        if n_extra < 1:
            continue
        ts, ss_, vt_all, vs_all = [], [], [], []
        for seg in segments:
            for _ in range(n_random_contexts):
                # sample n_extra distinct extra regions, all of which must be
                # in this segment
                avail = [r for r in other_regions if r in seg]
                if len(avail) < n_extra:
                    continue
                extras_r = list(rng.choice(avail, size=n_extra, replace=False))
                combo_t = [a] + extras_r
                combo_s = [b] + extras_r
                Xs = np.stack([X[seg[r]] for r in combo_t])
                st, vt = _score_combo(combo_t, Xs, calibration, jal)
                ss, vs = _score_combo(combo_s, Xs, calibration, jal)
                ts.append(st)
                ss_.append(ss)
                vt_all.append(vt)
                vs_all.append(vs)
        realistic[f"n_ctx_{n_total}"] = _summarize(ts, ss_, vt_all, vs_all)

    return {"best_case": best, "realistic": realistic}


def _summarize(
    true_scores: List[float],
    swap_scores: List[float],
    n_valid_true: List[int],
    n_valid_swap: List[int],
    extras: Optional[List[int]] = None,
) -> Dict:
    n = len(true_scores)
    if n == 0:
        return {
            "n_trials": 0,
            "auc": float("nan"),
            "gap": float("nan"),
            "win_rate": float("nan"),
            "win_when_decided": float("nan"),
            "decided_rate": 0.0,
            "envelope_pct_true": 0.0,
            "envelope_pct_swap": 0.0,
            "both_envelopes_pct": 0.0,
            "win_when_both_envelopes": float("nan"),
            "extras": extras,
        }
    ts = np.asarray(true_scores)
    ss = np.asarray(swap_scores)
    valid_t = np.asarray(n_valid_true) > 0
    valid_s = np.asarray(n_valid_swap) > 0
    both = valid_t & valid_s
    decided = ts != ss
    n_decided = int(decided.sum())
    n_both = int(both.sum())

    # Win when both true and swap have at least one valid envelope pair —
    # this isolates *envelope-quality* discrimination from "structural" wins
    # where the swap topology has no valid pairs at all.
    win_when_both = float((ts[both] > ss[both]).mean()) if n_both > 0 else float("nan")

    out = {
        "n_trials": n,
        "auc": _auc(ts, ss),
        "gap": float(ts.mean() - ss.mean()),
        "win_rate": float((ts > ss).mean()),
        "tie_rate": float((ts == ss).mean()),
        "decided_rate": float(decided.mean()),
        "win_when_decided": float((ts[decided] > ss[decided]).mean()) if n_decided > 0 else float("nan"),
        "envelope_pct_true": float(valid_t.mean()),
        "envelope_pct_swap": float(valid_s.mean()),
        "both_envelopes_pct": float(both.mean()),
        "win_when_both_envelopes": win_when_both,
        "true_mean": float(ts.mean()),
        "swap_mean": float(ss.mean()),
    }
    if extras is not None:
        out["extras"] = [REGION_NAMES[r] for r in extras]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--calibration", default="calibration")
    p.add_argument("--stats_dir", default="stats")
    p.add_argument(
        "--confusion_json",
        default="results/penalty_sweep/eval_best_final/confusion_aggregate.json",
    )
    p.add_argument("--out_json", default="stats/pair_discrimination_report.json")
    p.add_argument("--n_random_contexts", type=int, default=8)
    p.add_argument("--realistic_sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 72)
    print("PER-PAIR KINEMATIC DISCRIMINATION")
    print("=" * 72)
    print(f"  data          : {args.data}")
    print(f"  confusion_json: {args.confusion_json}")
    print(f"  realistic_sz  : {args.realistic_sizes}  ({args.n_random_contexts} contexts each)")

    nonlr, lr = parse_confusion_pairs(args.confusion_json)
    print(f"  non-LR pairs  : {len(nonlr)}")
    print(f"  LR pairs (x2 dirs): {len(lr)}")

    X, y, _, _ = load_data(args.data)
    calibration = load_calibration(args.calibration)
    stats = load_stats(args.stats_dir)
    jal = stats.get("joint_angle_limits")
    if jal is None:
        print("FATAL: stats/joint_angle_limits.npy missing")
        sys.exit(1)

    segments, n_segs = _build_segment_index(y)
    print(f"  segments      : {n_segs} complete")

    rng = np.random.default_rng(args.seed)

    results: List[Dict] = []
    for kind, plist in (("nonlr", nonlr), ("lr", lr)):
        for a_name, b_name, count in plist:
            if a_name not in NAME_TO_ID or b_name not in NAME_TO_ID:
                print(f"  skip unknown region: {a_name} -> {b_name}")
                continue
            row = {
                "kind": kind,
                "true": a_name,
                "wrong": b_name,
                "error_count": count,
            }
            row.update(
                run_pair(
                    a_name,
                    b_name,
                    X,
                    segments,
                    calibration,
                    jal,
                    realistic_sizes=tuple(args.realistic_sizes),
                    n_random_contexts=args.n_random_contexts,
                    rng=rng,
                )
            )
            results.append(row)

    # ---- Print sorted table ----
    print("\n" + "=" * 130)
    print("PAIR DISCRIMINATION TABLE")
    print("=" * 130)
    # Columns:
    #   best_AUC       AUC of true vs swap in best-case context
    #   best_gap       mean(true) - mean(swap)
    #   bothEnv%       % of trials where both true and swap had envelope pairs
    #                  (pure-envelope-quality regime)
    #   winBothEnv     win rate within those bothEnv trials (0.5 = no signal)
    #   r3_dec%        % of n_ctx=3 random contexts where scorer fires (decides)
    #   r3_win|dec     win rate among decided trials at n_ctx=3
    #   reduce_n3      ceiling: count × dec × max(0, 2*win|dec - 1)
    hdr = (
        f"{'kind':5} {'true':14} -> {'wrong':14} {'count':>5} | "
        f"{'bestAUC':>7} {'bestGap':>7} {'bothEnv%':>9} {'winBothE':>9} {'extras':28} | "
        f"{'r3_dec%':>7} {'r3_win|d':>9} {'r3_AUC':>7} | {'reduce_n3':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    results_sorted = sorted(results, key=lambda r: -r["error_count"])
    for r in results_sorted:
        bc = r["best_case"]
        rl = r["realistic"]
        n3 = rl.get("n_ctx_3", {})
        ceil_n3 = r["error_count"] * float(n3.get("decided_rate", 0.0)) * max(
            0.0, 2.0 * float(n3.get("win_when_decided") or 0.5) - 1.0
        )
        extras_str = ",".join(bc.get("extras") or [])[:26]
        wbe = bc.get("win_when_both_envelopes")
        wbe_str = f"{wbe:.3f}" if isinstance(wbe, float) and wbe == wbe else "  n/a"
        wd3 = n3.get("win_when_decided")
        wd3_str = f"{wd3:.3f}" if isinstance(wd3, float) and wd3 == wd3 else "  n/a"
        print(
            f"{r['kind']:5} {r['true']:14} -> {r['wrong']:14} {r['error_count']:5d} | "
            f"{bc['auc']:7.3f} {bc['gap']:+7.3f} {bc['both_envelopes_pct']*100:8.0f}% {wbe_str:>9} {extras_str:28} | "
            f"{n3.get('decided_rate', 0.0)*100:6.0f}% {wd3_str:>9} {n3.get('auc', float('nan')):7.3f} | "
            f"{ceil_n3:9.1f}"
        )

    # ---- Aggregate ceilings ----
    nonlr_total = sum(r["error_count"] for r in results if r["kind"] == "nonlr")
    lr_total = sum(r["error_count"] for r in results if r["kind"] == "lr") // 2  # we doubled

    def _ceil_one(row, regime):
        """Per-row error reduction ceiling under a regime ('best_case' or
        a realistic key like 'n_ctx_3'). Uses decided_rate × win_when_decided
        so that ties (no signal) don't count against the scorer.

        ceiling = count × decided_rate × max(0, 2 * win_when_decided - 1)

        Interpretation: if a perfect reranker uses the kinematic scorer as the
        sole tiebreaker, we recover an error iff (a) the scorer fires (decides)
        and (b) it picks correctly. The ×2−1 maps win-when-decided=0.5 → 0%
        recovered, win=1 → 100% recovered.
        """
        if regime == "best_case":
            d = row["best_case"]
        else:
            d = row["realistic"].get(regime, {})
        decided = d.get("decided_rate", 0.0)
        wwd = d.get("win_when_decided", float("nan"))
        if not (isinstance(wwd, float) and wwd == wwd):
            return 0.0
        return row["error_count"] * float(decided) * max(0.0, 2.0 * float(wwd) - 1.0)

    nonlr_rows = [r for r in results if r["kind"] == "nonlr"]
    lr_rows = [r for r in results if r["kind"] == "lr"]

    summary = {
        "nonlr_total_errors": nonlr_total,
        "lr_total_errors": lr_total,
        "nonlr_reduction_ceiling": {
            "best_case": sum(_ceil_one(r, "best_case") for r in nonlr_rows),
            "realistic_n3": sum(_ceil_one(r, "n_ctx_3") for r in nonlr_rows),
        },
        # Each LR pair appears twice (both directions, same count). Halve to
        # avoid double-counting against the unique LR error total.
        "lr_reduction_ceiling": {
            "best_case": sum(_ceil_one(r, "best_case") for r in lr_rows) / 2.0,
            "realistic_n3": sum(_ceil_one(r, "n_ctx_3") for r in lr_rows) / 2.0,
        },
    }

    print("\n" + "=" * 72)
    print("AGGREGATE CEILING (errors recoverable if scorer is sole tiebreaker)")
    print("=" * 72)
    print(f"  Non-LR total errors        : {summary['nonlr_total_errors']}")
    print(f"  Non-LR ceiling, best case  : {summary['nonlr_reduction_ceiling']['best_case']:.1f}"
          f" ({100*summary['nonlr_reduction_ceiling']['best_case']/max(1,summary['nonlr_total_errors']):.1f}%)")
    print(f"  Non-LR ceiling, realistic_n3: {summary['nonlr_reduction_ceiling']['realistic_n3']:.1f}"
          f" ({100*summary['nonlr_reduction_ceiling']['realistic_n3']/max(1,summary['nonlr_total_errors']):.1f}%)")
    print(f"  L/R total errors           : {summary['lr_total_errors']}")
    print(f"  L/R ceiling, best case     : {summary['lr_reduction_ceiling']['best_case']:.1f}"
          f" ({100*summary['lr_reduction_ceiling']['best_case']/max(1,summary['lr_total_errors']):.1f}%)")
    print(f"  L/R ceiling, realistic_n3  : {summary['lr_reduction_ceiling']['realistic_n3']:.1f}"
          f" ({100*summary['lr_reduction_ceiling']['realistic_n3']/max(1,summary['lr_total_errors']):.1f}%)")

    out = {
        "data_path": args.data,
        "confusion_path": args.confusion_json,
        "n_segments": n_segs,
        "pairs": results_sorted,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nReport -> {args.out_json}")


if __name__ == "__main__":
    main()
