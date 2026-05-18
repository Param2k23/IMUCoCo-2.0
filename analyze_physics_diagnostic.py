#!/usr/bin/env python3
"""
analyze_physics_diagnostic.py
=============================
Post-hoc analyzer for run_physics_diagnostic.slurm output.

Reads the 5 per-trial JSONL files plus eval_summary.json from each config
subdirectory under <out_base>, computes per-scorer attribution against the
baseline (paired trial-by-trial), the ceiling (frac_gt_in_candidates), the
default-blend headline, per-region failure modes, and an easy/hard stratified
summary (by classifier top1). Writes:

  <out_base>/diagnostic_report.json     machine-readable
  <out_base>/diagnostic_report.md       printed markdown summary

Usage:
  python analyze_physics_diagnostic.py results/.../physics_diagnostic_YYYY...
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from smpl_regions import REGION_NAMES

# Tags (in display order). Baseline must be first — it's the reference for
# attribution and the source of ceiling / candidate sets.
CONFIG_TAGS = [
    "baseline_classifier_only",
    "pure_kin",
    "pure_acc",
    "pure_jointlim_soft",
    "default_blend",
]

EASY_TOP1_THRESHOLD = 0.95  # regions with classifier top1 >= this are "easy"


# ───────────────────────────── I/O helpers ──────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def index_by_trial(records: list[dict]) -> dict[tuple[int, int], dict]:
    """Key trials by (n_sensors, trial_idx) so configs can be paired exactly."""
    return {(r["n_sensors"], r["trial_idx"]): r for r in records}


# ──────────────────────────── core computations ─────────────────────────────

def compute_ceiling(baseline_records: list[dict]) -> dict:
    """
    Per n_sensors: frac of trials whose ground-truth tuple appears in the
    enumerated candidate set, and frac whose GT is achievable per-sensor
    (each true region is in the corresponding sensor's top-k). The first is
    the exact-match ceiling at this physics_k; the second is the per-sensor
    ceiling.
    """
    by_x: dict[int, list[dict]] = defaultdict(list)
    for r in baseline_records:
        by_x[r["n_sensors"]].append(r)

    out: dict[str, dict] = {}
    for x, recs in sorted(by_x.items()):
        n = len(recs)
        if n == 0:
            continue
        gt_in_cand = sum(1 for r in recs if r["gt_in_candidates"])
        # Per-sensor ceiling: classifier_only_top1 fields don't carry the
        # per-sensor top-k after enumeration, but `gt_rank_classifier == 0`
        # means the GT tuple itself was the highest-classifier-prob combo
        # (so every position had GT as top-1). For the marginal "was each
        # individual sensor recoverable" question we count from
        # gt_in_candidates: if the tuple is in the candidate set, each
        # position individually had GT in its per-sensor top-k.
        out[str(x)] = {
            "n_trials": n,
            "exact_match_ceiling": round(gt_in_cand / n, 4),
            "per_sensor_ceiling_estimate": round(gt_in_cand / n, 4),
        }
    return out


def per_sensor_outcomes(record: dict) -> dict:
    """For one trial, return classifier-only and physics per-sensor hit lists."""
    true = record["true_regions"]
    cls_top1 = record["classifier_only_top1"]
    phys_top1 = record["physics_top1"]
    return {
        "true": true,
        "cls_hit":  [int(c == t) for c, t in zip(cls_top1, true)],
        "phys_hit": [int(p == t) for p, t in zip(phys_top1, true)],
        "cls_pred":  cls_top1,
        "phys_pred": phys_top1,
    }


def compare_against_baseline(
    baseline: dict[tuple[int, int], dict],
    candidate: dict[tuple[int, int], dict],
    region_class: dict[int, str] | None = None,  # region_id -> "easy"/"hard"
) -> dict:
    """
    Paired per-trial comparison: per-scorer attribution against the baseline.
    Returns total / per-x / (optional) per-bucket breakdowns.
    """
    n_fixed_per_sensor = 0
    n_broken_per_sensor = 0
    cls_total_correct = 0
    phys_total_correct = 0
    total_sensors = 0

    exact_baseline = 0
    exact_candidate = 0
    total_trials = 0

    # Confusion pairs fixed/broken: dict {(true, pred_baseline) -> count}
    fixed_pairs: Counter = Counter()
    broken_pairs: Counter = Counter()

    # Bucket (easy/hard) breakdown keyed by region difficulty
    bucket = {"easy": defaultdict(int), "hard": defaultdict(int)}
    # Per-x breakdown
    per_x = defaultdict(lambda: {
        "n_fixed": 0, "n_broken": 0,
        "cls_correct": 0, "phys_correct": 0, "total_sensors": 0,
        "exact_baseline": 0, "exact_candidate": 0, "n_trials": 0,
    })

    for key, base_rec in baseline.items():
        cand_rec = candidate.get(key)
        if cand_rec is None:
            continue
        x = base_rec["n_sensors"]
        total_trials += 1
        per_x[x]["n_trials"] += 1

        base_out = per_sensor_outcomes(base_rec)
        cand_out = per_sensor_outcomes(cand_rec)

        # Per-sensor paired comparison: anchor on classifier_only_top1 from
        # the baseline record (deterministic from trial sampling, independent
        # of baseline's physics weights), compared to candidate physics_top1.
        for s, true_r in enumerate(base_out["true"]):
            cls_hit = base_out["cls_hit"][s]
            phys_hit = cand_out["phys_hit"][s]
            cls_total_correct  += cls_hit
            phys_total_correct += phys_hit
            total_sensors += 1
            per_x[x]["cls_correct"]  += cls_hit
            per_x[x]["phys_correct"] += phys_hit
            per_x[x]["total_sensors"] += 1

            base_pred = base_out["phys_pred"][s]
            cand_pred = cand_out["phys_pred"][s]
            if cls_hit == 0 and phys_hit == 1:
                n_fixed_per_sensor += 1
                per_x[x]["n_fixed"] += 1
                fixed_pairs[(true_r, base_pred)] += 1
                if region_class is not None:
                    bucket[region_class[true_r]]["n_fixed"] += 1
            elif cls_hit == 1 and phys_hit == 0:
                n_broken_per_sensor += 1
                per_x[x]["n_broken"] += 1
                broken_pairs[(true_r, cand_pred)] += 1
                if region_class is not None:
                    bucket[region_class[true_r]]["n_broken"] += 1
            if region_class is not None:
                bucket[region_class[true_r]]["cls_correct"]  += cls_hit
                bucket[region_class[true_r]]["phys_correct"] += phys_hit
                bucket[region_class[true_r]]["total_sensors"] += 1

        # Exact match (whole tuple)
        if base_out["cls_pred"] == base_out["true"]:
            exact_baseline += 1
            per_x[x]["exact_baseline"] += 1
        if cand_out["phys_pred"] == cand_out["true"]:
            exact_candidate += 1
            per_x[x]["exact_candidate"] += 1

    def pct(num, den):
        return round(num / den, 4) if den else 0.0

    summary = {
        "n_trials_paired": total_trials,
        "n_sensors_paired": total_sensors,
        "n_fixed": n_fixed_per_sensor,
        "n_broken": n_broken_per_sensor,
        "net_fix": n_fixed_per_sensor - n_broken_per_sensor,
        "per_sensor_acc_baseline":  pct(cls_total_correct, total_sensors),
        "per_sensor_acc_candidate": pct(phys_total_correct, total_sensors),
        "delta_per_sensor_acc": round(
            pct(phys_total_correct, total_sensors)
            - pct(cls_total_correct, total_sensors), 4),
        "exact_match_baseline":  pct(exact_baseline, total_trials),
        "exact_match_candidate": pct(exact_candidate, total_trials),
        "delta_exact_match": round(
            pct(exact_candidate, total_trials)
            - pct(exact_baseline, total_trials), 4),
        "per_x": {},
        "top_fixed_confusion_pairs":
            [{"true": REGION_NAMES[t], "predicted_by_baseline": REGION_NAMES[p],
              "count": c} for (t, p), c in fixed_pairs.most_common(10)],
        "top_broken_confusion_pairs":
            [{"true": REGION_NAMES[t], "newly_predicted": REGION_NAMES[p],
              "count": c} for (t, p), c in broken_pairs.most_common(10)],
    }
    for x, d in sorted(per_x.items()):
        summary["per_x"][str(x)] = {
            "n_trials": d["n_trials"],
            "n_fixed":  d["n_fixed"],
            "n_broken": d["n_broken"],
            "net_fix":  d["n_fixed"] - d["n_broken"],
            "per_sensor_acc_baseline":  pct(d["cls_correct"],  d["total_sensors"]),
            "per_sensor_acc_candidate": pct(d["phys_correct"], d["total_sensors"]),
            "exact_match_baseline":  pct(d["exact_baseline"],  d["n_trials"]),
            "exact_match_candidate": pct(d["exact_candidate"], d["n_trials"]),
        }
    if region_class is not None:
        summary["stratified"] = {}
        for bucket_name, d in bucket.items():
            summary["stratified"][bucket_name] = {
                "n_fixed":  d["n_fixed"],
                "n_broken": d["n_broken"],
                "net_fix":  d["n_fixed"] - d["n_broken"],
                "per_sensor_acc_baseline":  pct(d["cls_correct"],  d["total_sensors"]),
                "per_sensor_acc_candidate": pct(d["phys_correct"], d["total_sensors"]),
                "n_sensor_obs": d["total_sensors"],
            }
    return summary


def build_region_difficulty(per_class_topk: list[dict],
                            threshold: float = EASY_TOP1_THRESHOLD) -> dict[int, str]:
    """Map region_id -> 'easy' (top1 >= threshold) or 'hard'."""
    name_to_id = {n: i for i, n in enumerate(REGION_NAMES)}
    out: dict[int, str] = {}
    for row in per_class_topk:
        rid = name_to_id.get(row["class"])
        if rid is None:
            continue
        out[rid] = "easy" if row.get("top1", 0.0) >= threshold else "hard"
    # Any region with no entry: default to "hard" (safer — keep them in the
    # bucket we care about).
    for rid in range(len(REGION_NAMES)):
        out.setdefault(rid, "hard")
    return out


# ──────────────────────────────── render ────────────────────────────────────

def render_markdown(report: dict) -> str:
    lines: list[str] = []
    A = lines.append

    A("# Physics Diagnostic Report")
    A("")
    A(f"_Source: `{report['out_base']}`_  ")
    A(f"_Configs analyzed: {', '.join(report['configs_found'])}_")
    A("")

    # ─ Ceiling ─
    A("## 1. Ceiling at `physics_k = 3`")
    A("")
    A("Upper bound that any rescoring (physics or otherwise) can achieve, "
      "given the classifier's top-k candidate set per sensor.")
    A("")
    A("| n_sensors | exact-match ceiling | trials |")
    A("|---:|---:|---:|")
    for x, row in report["ceiling"].items():
        A(f"| {x} | {row['exact_match_ceiling']:.4f} | {row['n_trials']} |")
    A("")

    # ─ Per-scorer attribution ─
    A("## 2. Per-scorer attribution vs `baseline_classifier_only`")
    A("")
    A("Paired trial-by-trial; same seed, same trial sampling. `net_fix = "
      "fixed − broken` is the marginal correction count.")
    A("")
    A("| config | Δ per_sensor_acc | Δ exact_match | n_fixed | n_broken | net_fix |")
    A("|---|---:|---:|---:|---:|---:|")
    for tag in CONFIG_TAGS[1:]:
        attr = report["attribution"].get(tag)
        if attr is None:
            A(f"| {tag} | _missing_ | | | | |"); continue
        A(f"| {tag} | {attr['delta_per_sensor_acc']:+.4f} | "
          f"{attr['delta_exact_match']:+.4f} | "
          f"{attr['n_fixed']} | {attr['n_broken']} | {attr['net_fix']:+d} |")
    A("")

    # ─ Per-x breakdown for default_blend ─
    blend = report["attribution"].get("default_blend")
    if blend is not None:
        A("### 2a. `default_blend` breakdown by n_sensors")
        A("")
        A("| n_sensors | base per_sensor | blend per_sensor | net_fix | base exact | blend exact |")
        A("|---:|---:|---:|---:|---:|---:|")
        for x, row in blend["per_x"].items():
            A(f"| {x} | {row['per_sensor_acc_baseline']:.4f} | "
              f"{row['per_sensor_acc_candidate']:.4f} | {row['net_fix']:+d} | "
              f"{row['exact_match_baseline']:.4f} | {row['exact_match_candidate']:.4f} |")
        A("")

    # ─ Top confusion pairs fixed/broken by default_blend ─
    if blend is not None:
        A("### 2b. Top confusion pairs `default_blend` *fixed* (baseline was wrong, blend is right)")
        A("")
        if blend["top_fixed_confusion_pairs"]:
            A("| true region | wrongly predicted by baseline | count |")
            A("|---|---|---:|")
            for row in blend["top_fixed_confusion_pairs"]:
                A(f"| {row['true']} | {row['predicted_by_baseline']} | {row['count']} |")
        else:
            A("_(none)_")
        A("")
        A("### 2c. Top confusion pairs `default_blend` *broke* (baseline was right, blend is wrong)")
        A("")
        if blend["top_broken_confusion_pairs"]:
            A("| true region | newly predicted by blend | count |")
            A("|---|---|---:|")
            for row in blend["top_broken_confusion_pairs"]:
                A(f"| {row['true']} | {row['newly_predicted']} | {row['count']} |")
        else:
            A("_(none)_")
        A("")

    # ─ Headline ─
    A("## 3. Headline answer")
    A("")
    A(f"Region difficulty threshold: classifier top1 ≥ {EASY_TOP1_THRESHOLD} → 'easy', else 'hard'.")
    A("")
    if blend is not None:
        verdict = (
            "**Physics helps**" if blend["net_fix"] > 0
            else "**Physics hurts**" if blend["net_fix"] < 0
            else "**Physics is a wash**"
        )
        A(f"- `default_blend` vs baseline: Δ per_sensor_acc = "
          f"**{blend['delta_per_sensor_acc']:+.4f}**, "
          f"net_fix = **{blend['net_fix']:+d}** ({blend['n_fixed']} fixed − "
          f"{blend['n_broken']} broken over {blend['n_sensors_paired']} sensor obs). "
          f"{verdict}.")
        pures = [report["attribution"].get(t, {}).get("delta_per_sensor_acc", 0.0)
                 for t in ("pure_kin", "pure_acc", "pure_jointlim_soft")]
        if any(p is not None for p in pures):
            best_pure = max(pures)
            A(f"- Best single scorer Δ per_sensor_acc = **{best_pure:+.4f}**; "
              f"blend Δ = {blend['delta_per_sensor_acc']:+.4f} "
              f"({'additive' if blend['delta_per_sensor_acc'] > best_pure + 1e-4 else 'not additive'}).")
    A("")

    # ─ Stratified ─
    A("## 4. Stratified by region difficulty (appended, granular output above is preserved)")
    A("")
    A("For each candidate config, attribution split by whether the *true* sensor "
      "region is 'easy' (classifier already gets it right ≥95% of the time) or "
      "'hard'. Reveals where physics actually earns its keep.")
    A("")
    for tag in CONFIG_TAGS[1:]:
        attr = report["attribution"].get(tag)
        if attr is None or "stratified" not in attr:
            continue
        A(f"### {tag}")
        A("")
        A("| bucket | n_obs | base per_sensor | new per_sensor | n_fixed | n_broken | net_fix |")
        A("|---|---:|---:|---:|---:|---:|---:|")
        for bucket in ("easy", "hard"):
            s = attr["stratified"].get(bucket, {})
            A(f"| {bucket} | {s.get('n_sensor_obs', 0)} | "
              f"{s.get('per_sensor_acc_baseline', 0):.4f} | "
              f"{s.get('per_sensor_acc_candidate', 0):.4f} | "
              f"{s.get('n_fixed', 0)} | {s.get('n_broken', 0)} | "
              f"{s.get('net_fix', 0):+d} |")
        A("")

    return "\n".join(lines)


# ──────────────────────────────── main ──────────────────────────────────────

def main(out_base: Path) -> int:
    if not out_base.is_dir():
        print(f"ERROR: not a directory: {out_base}", file=sys.stderr)
        return 1

    baseline_dir = out_base / CONFIG_TAGS[0]
    if not (baseline_dir / "physics_rerank_predictions.jsonl").exists():
        print(f"ERROR: baseline JSONL missing at {baseline_dir}", file=sys.stderr)
        return 1

    baseline_records = load_jsonl(baseline_dir / "physics_rerank_predictions.jsonl")
    baseline_summary = load_summary(baseline_dir / "eval_summary.json")
    baseline_idx = index_by_trial(baseline_records)

    region_class = build_region_difficulty(
        baseline_summary.get("per_class_topk", []),
        threshold=EASY_TOP1_THRESHOLD,
    )

    report: dict = {
        "out_base": str(out_base),
        "configs_found": [],
        "ceiling": compute_ceiling(baseline_records),
        "attribution": {},
        "baseline_top1": baseline_summary.get("topk_accuracy", {}).get("top1"),
        "easy_hard_threshold": EASY_TOP1_THRESHOLD,
        "region_difficulty": {
            REGION_NAMES[r]: region_class[r] for r in range(len(REGION_NAMES))
        },
    }

    for tag in CONFIG_TAGS:
        d = out_base / tag
        if (d / "physics_rerank_predictions.jsonl").exists():
            report["configs_found"].append(tag)

    for tag in CONFIG_TAGS[1:]:
        d = out_base / tag
        jsonl = d / "physics_rerank_predictions.jsonl"
        if not jsonl.exists():
            print(f"  [warn] missing JSONL for {tag}, skipping attribution")
            continue
        cand_idx = index_by_trial(load_jsonl(jsonl))
        report["attribution"][tag] = compare_against_baseline(
            baseline_idx, cand_idx, region_class=region_class,
        )

    # Write outputs
    json_path = out_base / "diagnostic_report.json"
    md_path   = out_base / "diagnostic_report.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w") as f:
        f.write(render_markdown(report))

    # Print summary to stdout for quick inspection
    print(render_markdown(report))
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_physics_diagnostic.py <out_base_dir>",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
