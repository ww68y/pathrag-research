import argparse
import json
from pathlib import Path

from build_evidence_calibrated_lazy_agent import (
    by_line,
    clean_text,
    load_jsonl,
    sanitize_evidence_labels,
    supported_text_repair,
)
from build_text_first_contract_agent import route_text_answer


ROOT_DIR = Path(__file__).resolve().parents[1]


SHAPE_ONLY_REASONS = {
    "answer_too_short_for_reliable_use",
    "comparison_contract_incomplete",
    "definition_purpose_contract_incomplete",
    "literal_metaphorical_contract_incomplete",
    "mechanism_contract_too_thin",
    "multi_item_contract_incomplete",
}


def as_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def verification_vetoes_shape_trigger(
    completion_row: dict,
    signals: dict,
    reasons: list[str],
) -> tuple[bool, dict]:
    agent = completion_row.get("contract_completion_agent") or {}
    verification = agent.get("verification") or {}
    pre_claims = agent.get("pre_claim_metrics") or {}
    normalized_reasons = [str(reason).split(":", 1)[0] for reason in reasons]
    shape_only = bool(normalized_reasons) and all(reason in SHAPE_ONLY_REASONS for reason in normalized_reasons)
    answer = clean_text(completion_row.get("prediction_before_contract_completion", completion_row.get("prediction", "")))
    uncertainty_patterns = signals.get("uncertainty_patterns") or []
    covered_facts = verification.get("covered_facts") or []
    missing_facts = verification.get("missing_supported_facts") or []
    unsupported_facts = verification.get("unsupported_or_off_topic_claims") or []
    coverage = as_float(verification.get("contract_coverage"))
    grounding = as_float(verification.get("evidence_grounding"))
    estimated_points = int(signals.get("estimated_distinct_points") or 0)
    reason_set = set(normalized_reasons)

    enough_evidence_for_short_answer = (
        reason_set == {"answer_too_short_for_reliable_use"}
        and coverage >= 0.85
        and grounding >= 0.85
    )
    enough_evidence_for_cardinality = (
        "multi_item_contract_incomplete" in reason_set
        and coverage >= 0.70
        and grounding >= 0.75
        and len(covered_facts) >= max(2, estimated_points)
    )
    enough_evidence_for_other_shape = coverage >= 0.85 and grounding >= 0.85

    checks = {
        "shape_only_trigger": shape_only,
        "text_only_evidence": agent.get("evidence_mode") == "text-only",
        "verification_attempted": bool(agent.get("verification_attempted")),
        "top_level_repair_not_needed": not bool(agent.get("repair_needed")),
        "repair_not_accepted": not bool(agent.get("repair_accepted")),
        "answer_present": bool(answer),
        "no_explicit_uncertainty": not bool(uncertainty_patterns),
        "no_missing_supported_facts": len(missing_facts) == 0,
        "pre_answer_has_no_unsupported_claims": int(pre_claims.get("unsupported_claims", 0) or 0) == 0,
        "moderate_grounding": grounding >= 0.75,
        "shape_confidence_satisfied": (
            enough_evidence_for_short_answer
            or enough_evidence_for_cardinality
            or enough_evidence_for_other_shape
        ),
        "unsupported_verifier_items_limited": len(unsupported_facts) <= 1,
    }
    details = {
        "enabled": True,
        "version": "v41",
        "checks": checks,
        "route_reasons": reasons,
        "normalized_reasons": normalized_reasons,
        "contract_coverage": coverage,
        "evidence_grounding": grounding,
        "covered_fact_count": len(covered_facts),
        "missing_fact_count": len(missing_facts),
        "unsupported_verifier_item_count": len(unsupported_facts),
    }
    return all(checks.values()), details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    text_rows = by_line(load_jsonl(ROOT_DIR / args.text))
    completion_rows = by_line(load_jsonl(ROOT_DIR / args.completion))
    common_lines = sorted(set(text_rows) & set(completion_rows))
    output_rows = []
    trigger_lines = []

    for line_no in common_lines:
        text_row = text_rows[line_no]
        completion_row = completion_rows[line_no]
        dominance, dominance_checks = supported_text_repair(completion_row)
        route_action, contract, signals, reasons = route_text_answer(text_row)
        vetoed, veto = verification_vetoes_shape_trigger(completion_row, signals, reasons)
        row = dict(text_row)

        if dominance:
            action = "use_text_repair"
            requires_graph = False
            prediction = sanitize_evidence_labels(str(completion_row.get("prediction", "")))
            final_reasons = ["supported_text_repair_dominates_explicit_uncertainty"]
        elif route_action == "graph_rescue" and vetoed:
            action = "keep_text_verifier_veto"
            requires_graph = False
            prediction = str(text_row.get("prediction", ""))
            final_reasons = ["verifier_calibrated_shape_trigger_veto"] + reasons
        elif route_action == "graph_rescue":
            action = "request_graph"
            requires_graph = True
            prediction = str(text_row.get("prediction", ""))
            trigger_lines.append(line_no)
            final_reasons = reasons
        else:
            action = "keep_text"
            requires_graph = False
            prediction = str(text_row.get("prediction", ""))
            final_reasons = reasons

        row["prediction_before_v41_pregraph_plan"] = row.get("prediction", "")
        row["prediction"] = prediction
        row["method"] = f"sage_v41_pregraph_{action}"
        row["v41_pregraph_plan"] = {
            "enabled": True,
            "version": "v41",
            "action": action,
            "requires_graph": requires_graph,
            "contract": contract,
            "signals": signals,
            "repair_dominance": dominance,
            "repair_dominance_checks": dominance_checks,
            "verification_veto": veto,
            "reasons": final_reasons,
            "reference_answer_used": False,
        }
        output_rows.append(row)

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    action_counts: dict[str, int] = {}
    for row in output_rows:
        action = row["v41_pregraph_plan"]["action"]
        action_counts[action] = action_counts.get(action, 0) + 1

    lines = [
        "# SAGE v41 Verifier-Calibrated Pre-Graph Plan\n\n",
        f"- Samples: {len(output_rows)}\n",
        f"- Text repairs accepted before graph: {action_counts.get('use_text_repair', 0)}\n",
        f"- Verification vetoes before graph: {action_counts.get('keep_text_verifier_veto', 0)}\n",
        f"- Graph trigger lines: {trigger_lines}\n",
        f"- Graph attempts: {len(trigger_lines)} / {len(output_rows)}\n",
        f"- Planned graph-query reduction: {(1 - len(trigger_lines) / max(1, len(output_rows))) * 100:.2f}%\n",
        "- Reference answers and evaluation scores are not routing inputs.\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(action_counts.items()):
        lines.append(f"| {action} | {count} |\n")
    lines.extend([
        "\n## Per-Sample Routing\n\n",
        "| Line | Dataset | Action | Requires Graph | Reasons |\n",
        "|---:|---|---|---|---|\n",
    ])
    for row in output_rows:
        plan = row["v41_pregraph_plan"]
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {plan.get('action')} | "
            f"{plan.get('requires_graph')} | {'; '.join(plan.get('reasons', [])).replace('|', '/')} |\n"
        )
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved v41 pre-graph plan to {output_path}")
    print(f"Graph trigger lines: {','.join(str(line) for line in trigger_lines) or 'none'}")


if __name__ == "__main__":
    main()
