#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pool primary direct-context risk-only evidence with bootstrap intervals."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_impact_aware_validation import metric_for

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "risk_only_stability"
OUT_JSON = OUT_DIR / "RISK_ONLY_POOLED_EVIDENCE.json"
OUT_MD = OUT_DIR / "RISK_ONLY_POOLED_EVIDENCE.md"

PRIMARY = {
    "new48_direct_context": {
        "base": ROOT / "results/impact_aware_validation/new48_never_escalate.jsonl",
        "policy": ROOT / "results/impact_aware_validation/new48_risk_only.jsonl",
        "selected": ROOT / "results/impact_aware_validation/new48_risk_only_lines_v1_corrected.txt",
    },
    "new35_direct_context": {
        "base": ROOT / "results/impact_aware_v2_validation/new35_never_escalate.jsonl",
        "policy": ROOT / "results/impact_aware_v2_validation/new35_risk_only.jsonl",
        "selected": ROOT / "results/impact_aware_v2_validation/new35_risk_lines_v2.txt",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_selected(path: Path) -> set[int]:
    return {int(value) for value in path.read_text(encoding="utf-8").split(",") if value.strip()}


def bootstrap_mean(values: np.ndarray, repeats: int = 10000) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def evaluate_primary(name: str, paths: dict[str, Path]) -> dict[str, Any]:
    base_rows = load_jsonl(paths["base"])
    policy_rows = {int(row["line_no"]): row for row in load_jsonl(paths["policy"])}
    selected = load_selected(paths["selected"])
    deltas: list[float] = []
    rouge_deltas: list[float] = []
    selected_deltas: list[float] = []
    domain_deltas: dict[str, list[float]] = defaultdict(list)
    domain_selected: dict[str, int] = defaultdict(int)
    for row in base_rows:
        line_no = int(row["line_no"])
        base_prediction = str(row.get("prediction") or "")
        final_prediction = str(policy_rows[line_no].get("prediction") or "")
        base_f1, base_rouge = metric_for(row, base_prediction)
        final_f1, final_rouge = metric_for(row, final_prediction)
        delta = final_f1 - base_f1
        rouge_delta = final_rouge - base_rouge
        deltas.append(delta)
        rouge_deltas.append(rouge_delta)
        domain = str(row.get("dataset") or row.get("label") or "unknown")
        domain_deltas[domain].append(delta)
        if line_no in selected:
            selected_deltas.append(delta)
            domain_selected[domain] += 1
    f1_mean, f1_low, f1_high = bootstrap_mean(np.asarray(deltas, dtype=float))
    rouge_mean, rouge_low, rouge_high = bootstrap_mean(np.asarray(rouge_deltas, dtype=float))
    selected_array = np.asarray(selected_deltas, dtype=float)
    return {
        "holdout": name,
        "n": len(base_rows),
        "budget": len(selected),
        "budget_rate": len(selected) / len(base_rows),
        "mean_f1_delta": f1_mean,
        "f1_delta_ci95": [f1_low, f1_high],
        "mean_rouge_l_delta": rouge_mean,
        "rouge_l_delta_ci95": [rouge_low, rouge_high],
        "positive_rows": int(np.sum(np.asarray(deltas) > 1e-12)),
        "negative_rows": int(np.sum(np.asarray(deltas) < -1e-12)),
        "selected_mean_f1_delta": float(selected_array.mean()) if selected_array.size else 0.0,
        "selected_positive": int(np.sum(selected_array > 1e-12)) if selected_array.size else 0,
        "selected_negative": int(np.sum(selected_array < -1e-12)) if selected_array.size else 0,
        "domain": {
            domain: {
                "n": len(values),
                "mean_f1_delta": float(np.mean(values)),
                "selected": domain_selected[domain],
            }
            for domain, values in sorted(domain_deltas.items())
        },
        "row_deltas": [
            {"line_no": int(row["line_no"]), "dataset": row.get("dataset"), "f1_delta": float(delta)}
            for row, delta in zip(base_rows, deltas)
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = [evaluate_primary(name, paths) for name, paths in PRIMARY.items()]
    all_deltas = np.asarray([delta for result in primary for delta in (row["f1_delta"] for row in result["row_deltas"])], dtype=float)
    all_rouge = np.asarray([
        delta
        for name, paths in PRIMARY.items()
        for base, policy in zip(load_jsonl(paths["base"]), load_jsonl(paths["policy"]))
        for delta in [metric_for(base, str(policy.get("prediction") or ""))[1] - metric_for(base, str(base.get("prediction") or ""))[1]]
    ], dtype=float)
    f1_mean, f1_low, f1_high = bootstrap_mean(all_deltas)
    rouge_mean, rouge_low, rouge_high = bootstrap_mean(all_rouge)
    domain_values: dict[str, list[float]] = defaultdict(list)
    for result in primary:
        for row in result["row_deltas"]:
            domain_values[str(row.get("dataset") or "unknown")].append(float(row["f1_delta"]))
    pooled = {
        "n": int(len(all_deltas)),
        "mean_f1_delta": f1_mean,
        "f1_delta_ci95": [f1_low, f1_high],
        "mean_rouge_l_delta": rouge_mean,
        "rouge_l_delta_ci95": [rouge_low, rouge_high],
        "positive_rows": int(np.sum(all_deltas > 1e-12)),
        "negative_rows": int(np.sum(all_deltas < -1e-12)),
        "zero_rows": int(np.sum(np.abs(all_deltas) <= 1e-12)),
        "positive_holdouts": int(sum(result["mean_f1_delta"] > 1e-12 for result in primary)),
        "negative_holdouts": int(sum(result["mean_f1_delta"] < -1e-12 for result in primary)),
        "domain": {
            domain: {"n": len(values), "mean_f1_delta": float(np.mean(values))}
            for domain, values in sorted(domain_values.items())
        },
    }
    payload = {
        "protocol": "pooled_primary_risk_only_evidence",
        "bootstrap_repeats": 10000,
        "primary_holdouts": primary,
        "pooled": pooled,
        "supplementary_stability": "results/risk_only_stability/RISK_ONLY_REPEAT_STABILITY.json",
        "caveat": "Primary pooled evidence combines two direct-context holdouts; intervals are descriptive paired bootstrap intervals, not randomized treatment-effect confidence intervals.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Pooled Risk-Only Evidence",
        "",
        "> Descriptive paired bootstrap summary for the primary direct-context holdouts.",
        "> Bootstrap intervals are not randomized treatment-effect confidence intervals.",
        "",
        "## Primary Holdouts",
        "",
        "| Holdout | N | Budget | F1 delta | 95% CI | ROUGE-L delta | 95% CI | Positive rows | Negative rows |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for result in primary:
        lines.append(
            f"| {result['holdout']} | {result['n']} | {result['budget']} | {result['mean_f1_delta']:+.4f} | "
            f"[{result['f1_delta_ci95'][0]:+.4f}, {result['f1_delta_ci95'][1]:+.4f}] | {result['mean_rouge_l_delta']:+.4f} | "
            f"[{result['rouge_l_delta_ci95'][0]:+.4f}, {result['rouge_l_delta_ci95'][1]:+.4f}] | {result['positive_rows']} | {result['negative_rows']} |"
        )
    lines.extend([
        "",
        "## Pooled Result",
        "",
        "| N | Mean F1 delta | 95% CI | Mean ROUGE-L delta | 95% CI | Positive rows | Negative rows | Positive holdouts |",
        "|---:|---:|---|---:|---|---:|---:|---:|",
        f"| {pooled['n']} | {pooled['mean_f1_delta']:+.4f} | [{pooled['f1_delta_ci95'][0]:+.4f}, {pooled['f1_delta_ci95'][1]:+.4f}] | "
        f"{pooled['mean_rouge_l_delta']:+.4f} | [{pooled['rouge_l_delta_ci95'][0]:+.4f}, {pooled['rouge_l_delta_ci95'][1]:+.4f}] | "
        f"{pooled['positive_rows']} | {pooled['negative_rows']} | {pooled['positive_holdouts']} / {len(primary)} |",
        "",
        "## Domain Breakdown",
        "",
        "| Domain | N | Mean F1 delta |",
        "|---|---:|---:|",
    ])
    for domain, values in pooled["domain"].items():
        lines.append(f"| {domain} | {values['n']} | {values['mean_f1_delta']:+.4f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Risk-only is positive on both primary direct-context holdouts and positive on 3 of 4 supplementary HyDE repeats.",
        "- The pooled result supports risk-only as the current selective-routing baseline, but does not identify a causal effect because repairs were not randomized.",
        "- The next methodological work should focus on calibration and false-escalation control, not another utility ranker.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": pooled, "primary": primary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
