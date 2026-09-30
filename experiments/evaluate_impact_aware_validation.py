#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the four frozen escalation policies on the new locked holdout."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize(text: str) -> list[str]:
    return str(text or "").lower().split()


def f1(prediction: str, reference: str) -> float:
    pred = Counter(tokenize(prediction))
    gold = Counter(tokenize(reference))
    if not gold or not pred:
        return 0.0
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred = tokenize(prediction)
    gold = tokenize(reference)
    if not pred or not gold:
        return 0.0
    previous = [0] * (len(gold) + 1)
    for token in pred:
        current = [0]
        for index, gold_token in enumerate(gold, start=1):
            if token == gold_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def metric_for(row: dict[str, Any], prediction: str) -> tuple[float, float]:
    references = row.get("reference_answers") or row.get("answers") or []
    if not isinstance(references, list):
        references = [str(references)]
    references = [str(reference) for reference in references if str(reference).strip()]
    if not references:
        return 0.0, 0.0
    return (
        max(f1(prediction, reference) for reference in references),
        max(rouge_l(prediction, reference) for reference in references),
    )


def read_line_set(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(value) for value in path.read_text(encoding="utf-8").split(",") if value.strip()}


def prediction_map(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["line_no"]): row for row in load_jsonl(path)}


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def evaluate_policy(
    name: str,
    base_rows: dict[int, dict[str, Any]],
    final_rows: dict[int, dict[str, Any]],
    selected: set[int],
) -> dict[str, Any]:
    per_row: list[dict[str, Any]] = []
    for line_no in sorted(base_rows):
        base = base_rows[line_no]
        final = final_rows[line_no]
        base_prediction = str(base.get("prediction") or "")
        final_prediction = str(final.get("prediction") or "")
        base_f1, base_rouge = metric_for(base, base_prediction)
        final_f1, final_rouge = metric_for(base, final_prediction)
        completion = final.get("contract_completion_agent") or {}
        per_row.append({
            "line_no": line_no,
            "dataset": base.get("dataset") or base.get("label") or "unknown",
            "selected": line_no in selected,
            "base_f1": base_f1,
            "final_f1": final_f1,
            "f1_delta": final_f1 - base_f1,
            "base_rouge_l": base_rouge,
            "final_rouge_l": final_rouge,
            "rouge_l_delta": final_rouge - base_rouge,
            "prediction_changed": final_prediction != base_prediction,
            "verification_attempted": bool(completion.get("verification_attempted")),
            "repair_attempted": bool(completion.get("repair_attempted")),
            "repair_accepted": bool(completion.get("repair_accepted")),
        })

    selected_rows = [row for row in per_row if row["selected"]]
    changed_rows = [row for row in per_row if row["prediction_changed"]]
    accepted_rows = [row for row in per_row if row["repair_accepted"]]
    negative_selected = [row for row in selected_rows if row["f1_delta"] < -1e-12]
    positive_selected = [row for row in selected_rows if row["f1_delta"] > 1e-12]
    final_f1_values = [row["final_f1"] for row in per_row]
    final_rouge_values = [row["final_rouge_l"] for row in per_row]
    base_f1_values = [row["base_f1"] for row in per_row]
    base_rouge_values = [row["base_rouge_l"] for row in per_row]
    by_domain: dict[str, dict[str, float]] = {}
    for domain in sorted({row["dataset"] for row in per_row}):
        domain_rows = [row for row in per_row if row["dataset"] == domain]
        by_domain[domain] = {
            "n": len(domain_rows),
            "base_f1": safe_mean([row["base_f1"] for row in domain_rows]),
            "final_f1": safe_mean([row["final_f1"] for row in domain_rows]),
            "f1_delta": safe_mean([row["f1_delta"] for row in domain_rows]),
            "selected": sum(row["selected"] for row in domain_rows),
        }
    return {
        "policy": name,
        "n": len(per_row),
        "selected": len(selected_rows),
        "escalation_rate": len(selected_rows) / max(1, len(per_row)),
        "mean_f1": safe_mean(final_f1_values),
        "mean_rouge_l": safe_mean(final_rouge_values),
        "base_mean_f1": safe_mean(base_f1_values),
        "base_mean_rouge_l": safe_mean(base_rouge_values),
        "mean_f1_delta": safe_mean([row["f1_delta"] for row in per_row]),
        "mean_rouge_l_delta": safe_mean([row["rouge_l_delta"] for row in per_row]),
        "selected_mean_f1_delta": safe_mean([row["f1_delta"] for row in selected_rows]),
        "selected_mean_rouge_l_delta": safe_mean([row["rouge_l_delta"] for row in selected_rows]),
        "positive_selected": len(positive_selected),
        "negative_selected": len(negative_selected),
        "wrong_escalation_rate": len(negative_selected) / max(1, len(selected_rows)),
        "changed_rows": len(changed_rows),
        "verification_attempted": sum(row["verification_attempted"] for row in per_row),
        "repair_attempted": sum(row["repair_attempted"] for row in per_row),
        "repair_accepted": len(accepted_rows),
        "estimated_api_calls": len(per_row) + sum(row["verification_attempted"] for row in per_row) + sum(row["repair_attempted"] for row in per_row),
        "by_domain": by_domain,
        "rows": per_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--risk", required=True)
    parser.add_argument("--impact", required=True)
    parser.add_argument("--always", required=True)
    parser.add_argument("--risk-lines", required=True)
    parser.add_argument("--impact-lines", required=True)
    parser.add_argument("--impact-name", default="impact_aware_v1")
    parser.add_argument("--holdout-label", default="New48")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    base_rows = prediction_map(ROOT / args.base)
    risk_rows = prediction_map(ROOT / args.risk)
    impact_rows = prediction_map(ROOT / args.impact)
    always_rows = prediction_map(ROOT / args.always)
    expected = set(base_rows)
    if not expected <= set(risk_rows) or not expected <= set(impact_rows) or not expected <= set(always_rows):
        raise RuntimeError("Policy outputs do not cover the full locked holdout")

    risk_lines = read_line_set(ROOT / args.risk_lines)
    impact_lines = read_line_set(ROOT / args.impact_lines)
    all_lines = set(base_rows)
    policies = [
        evaluate_policy("never_escalate", base_rows, base_rows, set()),
        evaluate_policy("risk_only", base_rows, risk_rows, risk_lines),
        evaluate_policy(args.impact_name, base_rows, impact_rows, impact_lines),
        evaluate_policy("always_escalate", base_rows, always_rows, all_lines),
    ]
    payload = {
        "protocol": "impact_aware_frozen_four_policy_validation",
        "holdout_label": args.holdout_label,
        "date": "2026-09-18",
        "n": len(base_rows),
        "policy_results": policies,
        "selection": {
            "risk_lines": sorted(risk_lines),
            "impact_lines": sorted(impact_lines),
            "risk_only_budget": len(risk_lines),
            "impact_aware_budget": len(impact_lines),
        },
        "caveat": "Lexical metrics are primary automatic checks. Routing never uses reference answers; answer-matched context windows were frozen during preprocessing.",
    }
    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Impact-Aware {args.holdout_label} Four-Policy Validation",
        "",
        "- Locked source holdout uses one question per source context with strict overlap exclusion.",
        "- Routing and repair do not use reference answers or evaluation labels.",
        "- Lexical F1 and ROUGE-L are computed after all policy outputs are frozen.",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Selected | Escalation | Mean F1 | Mean ROUGE-L | F1 delta vs never | Selected mean delta | Wrong escalation | Verified | Repairs accepted | Est. calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policies:
        lines.append(
            f"| {row['policy']} | {row['selected']} | {row['escalation_rate']:.1%} | "
            f"{row['mean_f1']:.4f} | {row['mean_rouge_l']:.4f} | {row['mean_f1_delta']:+.4f} | "
            f"{row['selected_mean_f1_delta']:+.4f} | {row['wrong_escalation_rate']:.1%} | "
            f"{row['verification_attempted']} | {row['repair_accepted']} | {row['estimated_api_calls']} |"
        )
    lines.extend([
        "",
        "## Domain Breakdown",
        "",
        "| Policy | Domain | N | Base F1 | Final F1 | Delta | Selected |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for policy in policies:
        for domain, values in sorted(policy["by_domain"].items()):
            lines.append(
                f"| {policy['policy']} | {domain} | {values['n']} | {values['base_f1']:.4f} | "
                f"{values['final_f1']:.4f} | {values['f1_delta']:+.4f} | {values['selected']} |"
            )
    lines.extend(["", "## Interpretation", "", "- `selected_mean_f1_delta` measures realized lexical gain only on rows selected for escalation.", "- `wrong_escalation_rate` counts selected rows whose final F1 is no better than the never-escalate base.", "- These are observed validation outcomes, not randomized treatment effects."])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "n": len(base_rows),
        "policies": [
            {key: row[key] for key in ("policy", "selected", "mean_f1", "mean_rouge_l", "mean_f1_delta", "selected_mean_f1_delta", "wrong_escalation_rate", "repair_accepted")}
            for row in policies
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
