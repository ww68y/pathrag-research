#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit router calibration and false-escalation control offline.

The development audit evaluates the frozen v31 router against its original
OOF severe-gap labels. The holdout audit separately measures whether higher
risk scores concentrate beneficial repairs and reduce harmful escalations.
The holdout section is diagnostic because repairs were not randomized.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_impact_aware_validation import metric_for


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "risk_only_stability"
OUT_JSON = OUT_DIR / "RISK_CALIBRATION_FALSE_ESCALATION_AUDIT.json"
OUT_MD = OUT_DIR / "RISK_CALIBRATION_FALSE_ESCALATION_AUDIT.md"

REFERENCE_BUDGET_RATE = 32 / 130
OOF_PATH = ROOT / "results" / "v31_full_rf_oof.jsonl"

HOLDOUTS = {
    "new48": {
        "base": ROOT / "results/impact_aware_validation/new48_never_escalate.jsonl",
        "repair": ROOT / "results/impact_aware_validation/new48_always_escalate.jsonl",
        "policy": ROOT / "results/impact_aware_validation/new48_risk_only.jsonl",
        "selected": ROOT / "results/impact_aware_validation/new48_risk_only_lines_v1_corrected.txt",
        "scored": ROOT / "results/impact_aware_validation/new48_policy_scored_v1_corrected.jsonl",
    },
    "new35": {
        "base": ROOT / "results/impact_aware_v2_validation/new35_never_escalate.jsonl",
        "repair": ROOT / "results/impact_aware_v2_validation/new35_always_escalate.jsonl",
        "policy": ROOT / "results/impact_aware_v2_validation/new35_risk_only.jsonl",
        "selected": ROOT / "results/impact_aware_v2_validation/new35_risk_lines_v2.txt",
        "scored": ROOT / "results/impact_aware_v2_validation/new35_policy_scored_v2.jsonl",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(rank_values(left), rank_values(right))[0, 1])


def oof_calibration() -> dict[str, Any]:
    rows = load_jsonl(OOF_PATH)
    probabilities = np.array([safe_float(row.get("oof_probability")) for row in rows])
    labels = np.array([int(row.get("label", 0)) for row in rows], dtype=float)
    brier = float(np.mean((probabilities - labels) ** 2))
    bins = np.array([0.0, 0.05, 0.10, 0.13, 0.20, 0.30, 0.50, 0.70, 1.000001])
    reliability = []
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        mean_probability = float(np.mean(probabilities[mask]))
        event_rate = float(np.mean(labels[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(mean_probability - event_rate)
        reliability.append({
            "lower": float(lower),
            "upper": float(upper),
            "n": int(mask.sum()),
            "mean_probability": mean_probability,
            "event_rate": event_rate,
            "absolute_gap": abs(mean_probability - event_rate),
        })

    threshold = 0.13
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum((~predicted) & (labels == 1)))
    tn = int(np.sum((~predicted) & (labels == 0)))
    return {
        "n": len(rows),
        "positive_label_rate": float(np.mean(labels)),
        "brier_score": brier,
        "ece": float(ece),
        "threshold": threshold,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "reliability_bins": reliability,
    }


def align_holdout(paths: dict[str, Path]) -> list[dict[str, Any]]:
    base_rows = {int(row["line_no"]): row for row in load_jsonl(paths["base"])}
    repair_rows = {int(row["line_no"]): row for row in load_jsonl(paths["repair"])}
    policy_rows = {int(row["line_no"]): row for row in load_jsonl(paths["policy"])}
    scored_rows = {int(row["line_no"]): row for row in load_jsonl(paths["scored"])}
    selected_lines = {
        int(value)
        for value in paths["selected"].read_text(encoding="utf-8").strip().split(",")
        if value.strip()
    }
    common = sorted(set(base_rows) & set(repair_rows) & set(policy_rows) & set(scored_rows))
    records = []
    for line_no in common:
        base = base_rows[line_no]
        repair = repair_rows[line_no]
        policy_row = policy_rows[line_no]
        scored = scored_rows[line_no]
        base_prediction = str(base.get("prediction") or "")
        repair_prediction = str(repair.get("prediction") or "")
        policy_prediction = str(policy_row.get("prediction") or "")
        base_metrics = metric_for(base, base_prediction)
        repair_metrics = metric_for(base, repair_prediction)
        policy_metrics = metric_for(base, policy_prediction)
        delta = float(repair_metrics[0] - base_metrics[0])
        policy_delta = float(policy_metrics[0] - base_metrics[0])
        policy = scored.get("impact_aware_policy") or {}
        records.append({
            "line_no": line_no,
            "dataset": str(base.get("dataset") or "unknown"),
            "risk_probability": safe_float(policy.get("risk_probability")),
            "f1_delta": delta,
            "positive_benefit": int(delta > 1e-12),
            "harmful_repair": int(delta < -1e-12),
            "frozen_policy_selected": line_no in selected_lines,
            "frozen_policy_f1_delta": policy_delta,
        })
    if len(records) != len(base_rows):
        raise RuntimeError(f"Alignment mismatch: {len(records)} / {len(base_rows)} rows")
    return records


def threshold_curve(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = np.array([row["risk_probability"] for row in records], dtype=float)
    deltas = np.array([row["f1_delta"] for row in records], dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    rates = [0.10, 0.15, 0.20, REFERENCE_BUDGET_RATE, 0.30, 0.40]
    curve = []
    for rate in rates:
        budget = max(1, min(len(records), int(round(rate * len(records)))))
        selected = order[:budget]
        selected_deltas = deltas[selected]
        curve.append({
            "target_rate": rate,
            "budget": budget,
            "score_cutoff": float(scores[selected[-1]]),
            "mean_delta_all_rows": float(np.sum(selected_deltas) / len(records)),
            "selected_mean_delta": float(np.mean(selected_deltas)),
            "positive_rate": float(np.mean(selected_deltas > 1e-12)),
            "harmful_rate": float(np.mean(selected_deltas < -1e-12)),
            "nonpositive_rate": float(np.mean(selected_deltas <= 1e-12)),
        })
    return curve


def holdout_audit(name: str, paths: dict[str, Path]) -> dict[str, Any]:
    records = align_holdout(paths)
    scores = np.array([row["risk_probability"] for row in records], dtype=float)
    deltas = np.array([row["f1_delta"] for row in records], dtype=float)
    positive = np.array([row["positive_benefit"] for row in records], dtype=float)
    harmful = np.array([row["harmful_repair"] for row in records], dtype=float)
    frozen_selected = np.array([row["frozen_policy_selected"] for row in records], dtype=bool)
    frozen_deltas = np.array([row["frozen_policy_f1_delta"] for row in records], dtype=float)
    selected_frozen_deltas = frozen_deltas[frozen_selected]
    budget = max(1, min(len(records), int(round(REFERENCE_BUDGET_RATE * len(records)))))
    top = np.argsort(-scores, kind="mergesort")[:budget]
    selected_deltas = deltas[top]
    random_positive_rate = float(np.mean(positive))
    bins = np.quantile(scores, [0.0, 0.25, 0.50, 0.75, 1.0])
    quartiles = []
    for index in range(4):
        lower = bins[index]
        upper = bins[index + 1]
        mask = (scores >= lower) & ((scores <= upper) if index == 3 else (scores < upper))
        if not np.any(mask):
            continue
        quartiles.append({
            "quartile": index + 1,
            "n": int(mask.sum()),
            "score_min": float(np.min(scores[mask])),
            "score_max": float(np.max(scores[mask])),
            "mean_score": float(np.mean(scores[mask])),
            "mean_delta": float(np.mean(deltas[mask])),
            "positive_rate": float(np.mean(positive[mask])),
            "harmful_rate": float(np.mean(harmful[mask])),
        })
    return {
        "holdout": name,
        "n": len(records),
        "budget": budget,
        "budget_rate": budget / len(records),
        "score_range": [float(np.min(scores)), float(np.max(scores))],
        "score_vs_delta_spearman": spearman(scores, deltas),
        "score_vs_benefit_spearman": spearman(scores, positive),
        "overall_positive_rate": random_positive_rate,
        "overall_harmful_rate": float(np.mean(harmful)),
        "frozen_policy_selected": int(np.sum(frozen_selected)),
        "frozen_policy_mean_delta_all_rows": float(np.mean(frozen_deltas)),
        "frozen_policy_selected_mean_delta": float(np.mean(selected_frozen_deltas)) if selected_frozen_deltas.size else 0.0,
        "frozen_policy_positive_rate": float(np.mean(selected_frozen_deltas > 1e-12)) if selected_frozen_deltas.size else 0.0,
        "frozen_policy_harmful_rate": float(np.mean(selected_frozen_deltas < -1e-12)) if selected_frozen_deltas.size else 0.0,
        "frozen_policy_nonpositive_rate": float(np.mean(selected_frozen_deltas <= 1e-12)) if selected_frozen_deltas.size else 0.0,
        "top_budget_cutoff": float(scores[top[-1]]),
        "top_budget_mean_delta_all_rows": float(np.sum(selected_deltas) / len(records)),
        "top_budget_selected_mean_delta": float(np.mean(selected_deltas)),
        "top_budget_positive_rate": float(np.mean(selected_deltas > 1e-12)),
        "top_budget_harmful_rate": float(np.mean(selected_deltas < -1e-12)),
        "top_budget_nonpositive_rate": float(np.mean(selected_deltas <= 1e-12)),
        "top_budget_positive_lift": float(np.mean(selected_deltas > 1e-12) - random_positive_rate),
        "threshold_curve": threshold_curve(records),
        "risk_quartiles": quartiles,
        "rows": records,
    }


def pooled_summary(holdouts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for holdout in holdouts for row in holdout["rows"]]
    scores = np.array([row["risk_probability"] for row in rows], dtype=float)
    deltas = np.array([row["f1_delta"] for row in rows], dtype=float)
    positive = np.array([row["positive_benefit"] for row in rows], dtype=float)
    harmful = np.array([row["harmful_repair"] for row in rows], dtype=float)
    frozen_rows = [row for row in rows if row["frozen_policy_selected"]]
    return {
        "n": len(rows),
        "score_vs_delta_spearman": spearman(scores, deltas),
        "score_vs_benefit_spearman": spearman(scores, positive),
        "overall_positive_rate": float(np.mean(positive)),
        "overall_harmful_rate": float(np.mean(harmful)),
        "frozen_policy_mean_delta_all_rows": float(sum(
            holdout["frozen_policy_mean_delta_all_rows"] * holdout["n"]
            for holdout in holdouts
        ) / sum(holdout["n"] for holdout in holdouts)),
        "frozen_policy_selected": len(frozen_rows),
        "frozen_policy_selected_mean_delta": float(np.mean([
            row["frozen_policy_f1_delta"] for row in frozen_rows
        ])),
        "frozen_policy_positive_rate": float(np.mean([
            row["frozen_policy_f1_delta"] > 1e-12 for row in frozen_rows
        ])),
        "frozen_policy_harmful_rate": float(np.mean([
            row["frozen_policy_f1_delta"] < -1e-12 for row in frozen_rows
        ])),
        "holdout_positive_count": sum(
            holdout["top_budget_positive_rate"] > holdout["overall_positive_rate"]
            for holdout in holdouts
        ),
        "holdout_harmful_rate_below_overall_count": sum(
            holdout["top_budget_harmful_rate"] <= holdout["overall_harmful_rate"]
            for holdout in holdouts
        ),
    }


def write_report(payload: dict[str, Any]) -> None:
    dev = payload["development_oof_calibration"]
    lines = [
        "# Risk Calibration and False-Escalation Audit",
        "",
        "> Offline audit of the frozen v31 risk router; no new API calls.",
        "> Holdout alignment uses completed always-repair candidates and is diagnostic, not causal.",
        "",
        "## Development OOF Calibration",
        "",
        "The development section evaluates the router against its original severe-gap labels.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| OOF rows | {dev['n']} |",
        f"| Positive label rate | {dev['positive_label_rate']:.3f} |",
        f"| Brier score | {dev['brier_score']:.4f} |",
        f"| Expected calibration error | {dev['ece']:.4f} |",
        f"| Threshold 0.13 precision | {dev['precision']:.3f} |",
        f"| Threshold 0.13 recall | {dev['recall']:.3f} |",
        "",
        "| Probability bin | N | Mean probability | Observed severe-gap rate | Absolute gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in dev["reliability_bins"]:
        lines.append(
            f"| [{row['lower']:.2f}, {row['upper']:.2f}) | {row['n']} | "
            f"{row['mean_probability']:.3f} | {row['event_rate']:.3f} | {row['absolute_gap']:.3f} |"
        )

    lines.extend([
        "",
        "## Holdout Decision Alignment",
        "",
        "Here, a positive benefit means the completed repair improved lexical F1 over never-escalate; a harmful repair reduced it.",
        "",
        "| Holdout | N | Spearman(score, delta) | Budget | Frozen policy delta | Frozen selected mean | Frozen harmful rate | Diagnostic top-k positive rate | Diagnostic top-k harmful rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for holdout in payload["holdouts"]:
        lines.append(
            f"| {holdout['holdout']} | {holdout['n']} | {holdout['score_vs_delta_spearman']:+.3f} | "
            f"{holdout['budget']} | {holdout['frozen_policy_mean_delta_all_rows']:+.4f} | "
            f"{holdout['frozen_policy_selected_mean_delta']:+.4f} | "
            f"{holdout['frozen_policy_harmful_rate']:.1%} | "
            f"{holdout['top_budget_positive_rate']:.1%} | {holdout['top_budget_harmful_rate']:.1%} |"
        )

    lines.extend([
        "",
        "## Risk Quartiles",
        "",
        "| Holdout | Quartile | N | Mean score | Mean delta | Positive rate | Harmful rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for holdout in payload["holdouts"]:
        for row in holdout["risk_quartiles"]:
            lines.append(
                f"| {holdout['holdout']} | Q{row['quartile']} | {row['n']} | {row['mean_score']:.3f} | "
                f"{row['mean_delta']:+.4f} | {row['positive_rate']:.1%} | {row['harmful_rate']:.1%} |"
            )

    lines.extend([
        "",
        "## Fixed-Budget Threshold Curve",
        "",
        "| Holdout | Target rate | Budget | Score cutoff | Mean delta / all rows | Selected mean delta | Positive rate | Harmful rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for holdout in payload["holdouts"]:
        for row in holdout["threshold_curve"]:
            lines.append(
                f"| {holdout['holdout']} | {row['target_rate']:.1%} | {row['budget']} | {row['score_cutoff']:.3f} | "
                f"{row['mean_delta_all_rows']:+.4f} | {row['selected_mean_delta']:+.4f} | "
                f"{row['positive_rate']:.1%} | {row['harmful_rate']:.1%} |"
            )

    pooled = payload["pooled"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- The frozen router's development OOF calibration is summarized by Brier `{dev['brier_score']:.4f}` and ECE `{dev['ece']:.4f}` against the original severe-gap labels.",
        f"- On the two primary holdouts, risk scores have pooled Spearman correlation `{pooled['score_vs_delta_spearman']:+.3f}` with observed repair deltas.",
        "- The frozen-policy columns report the actual risk-only outputs; the diagnostic top-k columns reuse completed always-repair candidates to test score ordering.",
        "- The diagnostic top-k rows should be read as concentration evidence, not as the realized guarded-repair outcome.",
        "- This audit does not establish a causal treatment effect and does not tune a new production threshold on New48 or New35.",
        "- The next paper-grade step is a precommitted calibration/guard policy on a fresh holdout, with abstention or retain-original decisions for low-confidence repairs.",
        "",
        "## Reproducibility",
        "",
        f"- Development OOF input: `{OOF_PATH.relative_to(ROOT)}`",
        "- Primary holdouts: New48 and New35 frozen direct-context outputs",
        f"- JSON output: `{OUT_JSON.relative_to(ROOT)}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holdouts = [holdout_audit(name, paths) for name, paths in HOLDOUTS.items()]
    payload = {
        "protocol": "frozen_v31_router_calibration_and_false_escalation_audit",
        "offline": True,
        "reference_budget_rate": REFERENCE_BUDGET_RATE,
        "development_oof_calibration": oof_calibration(),
        "holdouts": holdouts,
        "pooled": pooled_summary(holdouts),
        "causal_caveat": "Completed always-repair candidates are reused for holdout alignment; no randomized escalation treatment was run.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps({
        "development_oof_calibration": payload["development_oof_calibration"],
        "holdouts": [
            {
                "holdout": item["holdout"],
                "score_vs_delta_spearman": item["score_vs_delta_spearman"],
                "frozen_policy_mean_delta_all_rows": item["frozen_policy_mean_delta_all_rows"],
                "frozen_policy_harmful_rate": item["frozen_policy_harmful_rate"],
                "top_budget_positive_rate": item["top_budget_positive_rate"],
                "top_budget_harmful_rate": item["top_budget_harmful_rate"],
            }
            for item in holdouts
        ],
        "pooled": payload["pooled"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
