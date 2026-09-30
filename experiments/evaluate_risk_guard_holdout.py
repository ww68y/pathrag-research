#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the frozen risk-only guarded-repair policy on a fresh holdout."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_impact_aware_validation import metric_for


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "risk_guard_validation"
BASE_PATH = OUT_DIR / "legal20_direct_context.jsonl"
POLICY_PATH = OUT_DIR / "legal20_risk_only.jsonl"
SELECTED_PATH = OUT_DIR / "legal20_risk_lines.txt"
OUT_JSON = OUT_DIR / "RISK_GUARD_LEGAL20_RESULTS.json"
OUT_MD = OUT_DIR / "RISK_GUARD_LEGAL20_RESULTS.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_lines() -> set[int]:
    return {
        int(value)
        for value in SELECTED_PATH.read_text(encoding="utf-8").split(",")
        if value.strip()
    }


def row_metrics(row: dict[str, Any], prediction: str) -> tuple[float, float]:
    return metric_for(row, prediction)


def candidate_prediction(row: dict[str, Any]) -> str:
    agent = row.get("contract_completion_agent") or {}
    if agent.get("repair_accepted"):
        return str(row.get("prediction") or "")
    return str(agent.get("rejected_contract_completion_prediction") or "")


def main() -> None:
    base_rows = {int(row["line_no"]): row for row in load_jsonl(BASE_PATH)}
    policy_rows = {int(row["line_no"]): row for row in load_jsonl(POLICY_PATH)}
    selected = selected_lines()
    if set(base_rows) != set(policy_rows):
        raise RuntimeError("Base and policy line sets do not match")

    per_row = []
    for line_no in sorted(base_rows):
        base = base_rows[line_no]
        final = policy_rows[line_no]
        base_prediction = str(base.get("prediction") or "")
        final_prediction = str(final.get("prediction") or "")
        base_f1, base_rouge = row_metrics(base, base_prediction)
        final_f1, final_rouge = row_metrics(base, final_prediction)
        agent = final.get("contract_completion_agent") or {}
        selected_flag = line_no in selected
        candidate = candidate_prediction(final) if selected_flag else ""
        candidate_f1 = candidate_rouge = None
        if candidate:
            candidate_f1, candidate_rouge = row_metrics(base, candidate)
        per_row.append({
            "line_no": line_no,
            "selected": selected_flag,
            "base_f1": base_f1,
            "final_f1": final_f1,
            "f1_delta": final_f1 - base_f1,
            "base_rouge_l": base_rouge,
            "final_rouge_l": final_rouge,
            "rouge_l_delta": final_rouge - base_rouge,
            "repair_attempted": bool(agent.get("repair_attempted")),
            "repair_accepted": bool(agent.get("repair_accepted")),
            "candidate_f1": candidate_f1,
            "candidate_f1_delta": None if candidate_f1 is None else candidate_f1 - base_f1,
            "candidate_rouge_l": candidate_rouge,
            "guard_rejection_reasons": (agent.get("repair_guard") or {}).get("rejection_reasons", []),
        })

    all_rows = per_row
    selected_rows = [row for row in all_rows if row["selected"]]
    attempted_rows = [row for row in selected_rows if row["repair_attempted"]]
    accepted_rows = [row for row in selected_rows if row["repair_accepted"]]
    rejected_candidates = [
        row for row in selected_rows
        if row["candidate_f1_delta"] is not None and not row["repair_accepted"]
    ]
    guard_saved_harm = [row for row in rejected_candidates if row["candidate_f1_delta"] < 0 and abs(row["f1_delta"]) <= 1e-12]
    guard_rejected_benefit = [row for row in rejected_candidates if row["candidate_f1_delta"] > 0]

    summary = {
        "protocol": "fresh_legal20_frozen_v31_risk_only_guard",
        "holdout": "risk_guard_locked_legal20_windowed",
        "n": len(all_rows),
        "budget": len(selected_rows),
        "budget_rate": len(selected_rows) / len(all_rows),
        "base_f1": sum(row["base_f1"] for row in all_rows) / len(all_rows),
        "final_f1": sum(row["final_f1"] for row in all_rows) / len(all_rows),
        "f1_delta": sum(row["f1_delta"] for row in all_rows) / len(all_rows),
        "base_rouge_l": sum(row["base_rouge_l"] for row in all_rows) / len(all_rows),
        "final_rouge_l": sum(row["final_rouge_l"] for row in all_rows) / len(all_rows),
        "rouge_l_delta": sum(row["rouge_l_delta"] for row in all_rows) / len(all_rows),
        "selected_mean_f1_delta": sum(row["f1_delta"] for row in selected_rows) / max(1, len(selected_rows)),
        "selected_positive": sum(row["f1_delta"] > 1e-12 for row in selected_rows),
        "selected_negative": sum(row["f1_delta"] < -1e-12 for row in selected_rows),
        "selected_nonpositive": sum(row["f1_delta"] <= 1e-12 for row in selected_rows),
        "repairs_attempted": len(attempted_rows),
        "repairs_accepted": len(accepted_rows),
        "repairs_rejected": len(rejected_candidates),
        "guard_saved_harm": len(guard_saved_harm),
        "guard_rejected_benefit": len(guard_rejected_benefit),
        "rejection_reasons": dict(Counter(
            reason
            for row in rejected_candidates
            for reason in row["guard_rejection_reasons"]
        )),
        "rows": all_rows,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Risk Guard Legal20 Fresh Holdout Results",
        "",
        "> Fresh strict zero-overlap Legal-only holdout; frozen v31 risk router; 24.6% budget.",
        "> No threshold tuning or reference-answer input was used at runtime.",
        "",
        "## Automatic Results",
        "",
        "| Policy | Budget | Mean F1 | Mean ROUGE-L | F1 delta | ROUGE-L delta | Selected mean delta | Positive / negative selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| direct_context / never | 0/20 | {summary['base_f1']:.4f} | {summary['base_rouge_l']:.4f} | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |",
        f"| frozen risk-only guard | {summary['budget']}/20 | {summary['final_f1']:.4f} | {summary['final_rouge_l']:.4f} | {summary['f1_delta']:+.4f} | {summary['rouge_l_delta']:+.4f} | {summary['selected_mean_f1_delta']:+.4f} | {summary['selected_positive']} / {summary['selected_negative']} |",
        "",
        "## Guard Behavior",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Selected rows | {summary['budget']} |",
        f"| Repairs attempted | {summary['repairs_attempted']} |",
        f"| Repairs accepted | {summary['repairs_accepted']} |",
        f"| Repair candidates rejected | {summary['repairs_rejected']} |",
        f"| Harmful candidates prevented by guard | {summary['guard_saved_harm']} |",
        f"| Beneficial candidates rejected by guard | {summary['guard_rejected_benefit']} |",
        "",
        "## Interpretation",
        "",
        "- This is the first fresh holdout for the calibration/guard policy and is intentionally Legal-only because no clean Biology, History, or Physics contexts remained under the strict repository-wide exclusion protocol.",
        "- Results support or weaken the guard claim only for this fresh Legal slice; they do not establish cross-domain generalization by themselves.",
        "- The guard is evaluated by comparing the accepted final answer with the rejected repair candidate on selected rows, while the main F1 delta compares final output against direct-context baseline.",
        "",
        "## Reproducibility",
        "",
        "- Dataset: `datasets/UltraDomain/risk_guard_locked_legal20_windowed.jsonl`",
        "- Baseline: `results/risk_guard_validation/legal20_direct_context.jsonl`",
        "- Policy: `results/risk_guard_validation/legal20_risk_only.jsonl`",
        f"- JSON output: `{OUT_JSON.relative_to(ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
