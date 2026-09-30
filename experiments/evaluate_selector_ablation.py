#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline fixed-budget selector ablation on completed repair candidates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_impact_aware_validation import metric_for

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_BUDGET_RATE = 32 / 130
OUT_DIR = ROOT / "results" / "risk_only_stability"
OUT_JSON = OUT_DIR / "SELECTOR_ABLATION_NEW48_NEW35.json"
OUT_MD = OUT_DIR / "SELECTOR_ABLATION_NEW48_NEW35.md"

HOLDOUTS = {
    "new48": {
        "base": ROOT / "results/impact_aware_validation/new48_never_escalate.jsonl",
        "always": ROOT / "results/impact_aware_validation/new48_always_escalate.jsonl",
        "dry": ROOT / "results/impact_aware_validation/new48_contract_completion_dry.jsonl",
        "scored": ROOT / "results/impact_aware_validation/new48_policy_scored_v1_corrected.jsonl",
    },
    "new35": {
        "base": ROOT / "results/impact_aware_v2_validation/new35_never_escalate.jsonl",
        "always": ROOT / "results/impact_aware_v2_validation/new35_always_escalate.jsonl",
        "dry": ROOT / "results/impact_aware_v2_validation/new35_contract_completion_dry.jsonl",
        "scored": ROOT / "results/impact_aware_v2_validation/new35_policy_scored_v2.jsonl",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selector_scores(dry_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    risk = np.array([
        float((row.get("impact_aware_policy") or {}).get("risk_probability", 0.0))
        for row in scored_rows
    ])
    low_support = np.array([
        1.0 - float(((row.get("contract_completion_agent") or {}).get("pre_claim_metrics") or {}).get("mean_support", 0.0))
        for row in dry_rows
    ])
    unsupported = np.array([
        float(((row.get("contract_completion_agent") or {}).get("pre_claim_metrics") or {}).get("unsupported_ratio", 0.0))
        for row in dry_rows
    ])
    uncertainty = np.array([
        int("uncertainty_or_hedging" in ((row.get("contract_completion_agent") or {}).get("risk_reasons") or []))
        for row in dry_rows
    ], dtype=float)
    shape = np.array([
        int("answer_shape_below_contract" in ((row.get("contract_completion_agent") or {}).get("risk_reasons") or []))
        for row in dry_rows
    ], dtype=float)
    return {
        "risk_probability": risk,
        "low_support_gap": low_support,
        "unsupported_ratio": unsupported,
        "uncertainty_flag": uncertainty,
        "shape_flag": shape,
    }


def evaluate_selector(
    name: str,
    base_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    scores: np.ndarray,
    budget: int,
) -> dict[str, Any]:
    selected = set(np.argsort(-scores)[:budget].tolist())
    base_f1: list[float] = []
    final_f1: list[float] = []
    base_rouge: list[float] = []
    final_rouge: list[float] = []
    selected_deltas: list[float] = []
    for index, base_row in enumerate(base_rows):
        base_prediction = str(base_row.get("prediction") or "")
        final_prediction = str(repair_rows[index].get("prediction") or "") if index in selected else base_prediction
        base_metrics = metric_for(base_row, base_prediction)
        final_metrics = metric_for(base_row, final_prediction)
        base_f1.append(base_metrics[0])
        final_f1.append(final_metrics[0])
        base_rouge.append(base_metrics[1])
        final_rouge.append(final_metrics[1])
        if index in selected:
            selected_deltas.append(final_metrics[0] - base_metrics[0])
    return {
        "selector": name,
        "budget": budget,
        "mean_f1": float(np.mean(final_f1)),
        "mean_rouge_l": float(np.mean(final_rouge)),
        "f1_delta": float(np.mean(final_f1) - np.mean(base_f1)),
        "rouge_l_delta": float(np.mean(final_rouge) - np.mean(base_rouge)),
        "selected_mean_f1_delta": float(np.mean(selected_deltas)) if selected_deltas else 0.0,
        "positive_selected": int(sum(delta > 1e-12 for delta in selected_deltas)),
        "negative_selected": int(sum(delta < -1e-12 for delta in selected_deltas)),
        "wrong_escalation_rate": float(sum(delta < -1e-12 for delta in selected_deltas) / max(1, len(selected_deltas))),
        "selected_indices": sorted(selected),
    }


def evaluate_holdout(name: str, paths: dict[str, Path]) -> dict[str, Any]:
    base_rows = load_jsonl(paths["base"])
    repair_rows = load_jsonl(paths["always"])
    dry_rows = load_jsonl(paths["dry"])
    scored_rows = load_jsonl(paths["scored"])
    if not (len(base_rows) == len(repair_rows) == len(dry_rows) == len(scored_rows)):
        raise RuntimeError(f"Row count mismatch for {name}")
    budget = max(1, int(round(REFERENCE_BUDGET_RATE * len(base_rows))))
    scores = selector_scores(dry_rows, scored_rows)
    results = [
        evaluate_selector(selector, base_rows, repair_rows, values, budget)
        for selector, values in scores.items()
    ]
    rng = np.random.default_rng(42)
    random_rows = []
    for _ in range(2000):
        random_scores = rng.random(len(base_rows))
        random_rows.append(evaluate_selector("random", base_rows, repair_rows, random_scores, budget))
    random_summary = {
        "selector": "random",
        "budget": budget,
        "mean_f1": float(np.mean([row["mean_f1"] for row in random_rows])),
        "mean_rouge_l": float(np.mean([row["mean_rouge_l"] for row in random_rows])),
        "f1_delta": float(np.mean([row["f1_delta"] for row in random_rows])),
        "rouge_l_delta": float(np.mean([row["rouge_l_delta"] for row in random_rows])),
        "selected_mean_f1_delta": float(np.mean([row["selected_mean_f1_delta"] for row in random_rows])),
        "wrong_escalation_rate": float(np.mean([row["wrong_escalation_rate"] for row in random_rows])),
        "f1_delta_std": float(np.std([row["f1_delta"] for row in random_rows])),
    }
    return {"holdout": name, "n": len(base_rows), "budget": budget, "selectors": [*results, random_summary]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [evaluate_holdout(name, paths) for name, paths in HOLDOUTS.items()]
    payload = {
        "protocol": "fixed_budget_selector_ablation_on_completed_repairs",
        "offline": True,
        "reference_budget_rate": REFERENCE_BUDGET_RATE,
        "holdouts": results,
        "caveat": "Selectors are compared using already-computed always-repair candidates; this is a diagnostic ablation, not a fresh causal experiment.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Fixed-Budget Selector Ablation",
        "",
        "> Offline diagnostic on completed repair candidates; no new API calls and no holdout tuning.",
        f"> Precommitted budget rate: `{REFERENCE_BUDGET_RATE:.4%}`.",
        "",
    ]
    for result in results:
        lines.extend([
            f"## {result['holdout']}",
            "",
            "| Selector | Budget | Mean F1 | F1 delta | Mean ROUGE-L | ROUGE-L delta | Selected mean delta | Wrong escalation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in result["selectors"]:
            lines.append(
                f"| {row['selector']} | {row['budget']} | {row['mean_f1']:.4f} | {row['f1_delta']:+.4f} | "
                f"{row['mean_rouge_l']:.4f} | {row['rouge_l_delta']:+.4f} | {row['selected_mean_f1_delta']:+.4f} | {row['wrong_escalation_rate']:.1%} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "- `risk_probability` is the frozen v31 router used by the primary risk-only policy.",
        "- Contract-gap, unsupported-ratio, uncertainty, and shape selectors are diagnostic runtime-only alternatives.",
        "- The random row is the mean of 2,000 fixed-budget draws and is an empirical lower baseline.",
        "- Results should guide mechanism interpretation, not be presented as a new causal estimate.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
