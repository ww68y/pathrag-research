#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline fixed-budget risk-only stability check on existing SAGE repeats."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from evaluate_impact_aware_validation import metric_for
from score_impact_aware_policy import RISK_MODEL, build_feature_rows

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_BUDGET_RATE = 32 / 130
REPEATS = (
    "sage_locked_unseen25_repeat1",
    "sage_locked_unseen40_repeat2",
    "sage_locked_unseen24_repeat3",
    "sage_locked_unseen24_repeat4",
)
OUT_DIR = ROOT / "results" / "risk_only_stability"
OUT_JSON = OUT_DIR / "RISK_ONLY_REPEAT_STABILITY.json"
OUT_MD = OUT_DIR / "RISK_ONLY_REPEAT_STABILITY.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_repeat(name: str, risk_model: Any) -> dict[str, Any]:
    directory = ROOT / "results" / name
    dry_rows = load_jsonl(directory / "hyde_contract_completion_text_only_dry.jsonl")
    executed_rows = {int(row["line_no"]): row for row in load_jsonl(directory / "hyde_contract_completion_text_only.jsonl")}
    risk_frame, _, _ = build_feature_rows(dry_rows)
    probabilities = risk_model.predict_proba(risk_frame)[:, 1]
    budget = max(1, int(round(REFERENCE_BUDGET_RATE * len(dry_rows))))
    selected_indices = set(np.argsort(-probabilities)[:budget].tolist())
    selected_lines = sorted(int(dry_rows[index]["line_no"]) for index in selected_indices)

    base_f1: list[float] = []
    final_f1: list[float] = []
    base_rouge: list[float] = []
    final_rouge: list[float] = []
    selected_deltas: list[float] = []
    accepted_selected = 0
    changed_selected = 0
    for index, base_row in enumerate(dry_rows):
        line_no = int(base_row["line_no"])
        executed = executed_rows[line_no]
        selected = index in selected_indices
        final_prediction = str(executed.get("prediction") or "") if selected else str(base_row.get("prediction") or "")
        base_metrics = metric_for(base_row, str(base_row.get("prediction") or ""))
        final_metrics = metric_for(base_row, final_prediction)
        base_f1.append(base_metrics[0])
        final_f1.append(final_metrics[0])
        base_rouge.append(base_metrics[1])
        final_rouge.append(final_metrics[1])
        if selected:
            selected_deltas.append(final_metrics[0] - base_metrics[0])
            agent = executed.get("contract_completion_agent") or {}
            accepted_selected += int(bool(agent.get("repair_accepted")))
            changed_selected += int(final_prediction != str(base_row.get("prediction") or ""))

    return {
        "repeat": name,
        "n": len(dry_rows),
        "budget": budget,
        "budget_rate": budget / len(dry_rows),
        "selected_lines": selected_lines,
        "base_f1": float(np.mean(base_f1)),
        "risk_only_f1": float(np.mean(final_f1)),
        "f1_delta": float(np.mean(final_f1) - np.mean(base_f1)),
        "base_rouge_l": float(np.mean(base_rouge)),
        "risk_only_rouge_l": float(np.mean(final_rouge)),
        "rouge_l_delta": float(np.mean(final_rouge) - np.mean(base_rouge)),
        "selected_mean_f1_delta": float(np.mean(selected_deltas)) if selected_deltas else 0.0,
        "positive_selected": int(sum(delta > 1e-12 for delta in selected_deltas)),
        "negative_selected": int(sum(delta < -1e-12 for delta in selected_deltas)),
        "wrong_escalation_rate": float(sum(delta < -1e-12 for delta in selected_deltas) / max(1, len(selected_deltas))),
        "accepted_selected": accepted_selected,
        "changed_selected": changed_selected,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_artifact = joblib.load(RISK_MODEL)
    risk_model = risk_artifact["model"]
    repeats = [evaluate_repeat(name, risk_model) for name in REPEATS]
    total_n = sum(row["n"] for row in repeats)
    aggregate = {
        "n": total_n,
        "budget_rate_reference": REFERENCE_BUDGET_RATE,
        "base_f1": sum(row["base_f1"] * row["n"] for row in repeats) / total_n,
        "risk_only_f1": sum(row["risk_only_f1"] * row["n"] for row in repeats) / total_n,
        "f1_delta": sum(row["f1_delta"] * row["n"] for row in repeats) / total_n,
        "base_rouge_l": sum(row["base_rouge_l"] * row["n"] for row in repeats) / total_n,
        "risk_only_rouge_l": sum(row["risk_only_rouge_l"] * row["n"] for row in repeats) / total_n,
        "rouge_l_delta": sum(row["rouge_l_delta"] * row["n"] for row in repeats) / total_n,
        "positive_repeats": sum(row["f1_delta"] > 1e-12 for row in repeats),
        "negative_repeats": sum(row["f1_delta"] < -1e-12 for row in repeats),
    }
    payload = {
        "protocol": "risk_only_fixed_budget_existing_sage_repeats",
        "offline": True,
        "base_method": "HyDE",
        "risk_model": str(RISK_MODEL.relative_to(ROOT)),
        "reference_budget_rate": REFERENCE_BUDGET_RATE,
        "repeats": repeats,
        "aggregate": aggregate,
        "caveat": "This is a supplementary lexical stability check on existing HyDE-based repeats, not a replacement for the direct-context New48/New35 validation.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Risk-Only Fixed-Budget Stability on Existing SAGE Repeats",
        "",
        "> Offline supplementary check; no new API calls and no threshold tuning.",
        "",
        f"- Frozen risk model: `{RISK_MODEL.relative_to(ROOT)}`",
        f"- Precommitted budget rate: `{REFERENCE_BUDGET_RATE:.4%}`",
        "- Base method: HyDE; selected rows use the already-computed contract-completion repair output.",
        "",
        "## Per-Repeat Results",
        "",
        "| Repeat | N | Budget | Base F1 | Risk-only F1 | Delta | Base ROUGE-L | Risk-only ROUGE-L | Delta | Wrong escalation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in repeats:
        lines.append(
            f"| {row['repeat']} | {row['n']} | {row['budget']} | {row['base_f1']:.4f} | {row['risk_only_f1']:.4f} | {row['f1_delta']:+.4f} | "
            f"{row['base_rouge_l']:.4f} | {row['risk_only_rouge_l']:.4f} | {row['rouge_l_delta']:+.4f} | {row['wrong_escalation_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Aggregate",
        "",
        "| N | Base F1 | Risk-only F1 | Delta | Base ROUGE-L | Risk-only ROUGE-L | Delta | Positive repeats | Negative repeats |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {aggregate['n']} | {aggregate['base_f1']:.4f} | {aggregate['risk_only_f1']:.4f} | {aggregate['f1_delta']:+.4f} | {aggregate['base_rouge_l']:.4f} | {aggregate['risk_only_rouge_l']:.4f} | {aggregate['rouge_l_delta']:+.4f} | {aggregate['positive_repeats']} | {aggregate['negative_repeats']} |",
        "",
        "## Interpretation",
        "",
        "- Selection is fixed before reading the repeat outcomes: the v31 risk router ranks rows and receives the same 24.6% budget on every repeat.",
        "- These results diagnose routing stability only; they do not establish a causal treatment effect and do not replace the direct-context New35 experiment.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "repeats": repeats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
