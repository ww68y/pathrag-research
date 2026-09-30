import argparse
import json
from pathlib import Path

from build_answer_contract_agent import load_jsonl
from build_claim_evidence_agent import write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]
DIMENSIONS = ("contract_coverage", "evidence_grounding", "relevance", "conciseness")


def utility(answer: dict) -> float:
    return sum(float(answer.get(dimension, 0)) for dimension in DIMENSIONS) / len(DIMENSIONS)


def primary_view(line_no: int, consensus: dict) -> dict | None:
    repair_is_a = line_no % 2 == 0
    return next(
        (view for view in consensus.get("views", []) if bool(view.get("repair_is_a")) == repair_is_a),
        None,
    )


def route(line_no: int, consensus: dict, threshold: float) -> dict:
    view = primary_view(line_no, consensus)
    if not view:
        return {
            "escalated": True,
            "risk_reasons": ["primary_view_missing"],
            "primary_utility_gain": 0.0,
            "primary_repair_is_a": line_no % 2 == 0,
        }
    repaired = view.get("repaired") or {}
    original = view.get("original") or {}
    gain = utility(repaired) - utility(original)
    reasons = []
    if view.get("winner") != "repair":
        reasons.append("primary_view_does_not_prefer_repair")
    if repaired.get("missing_required_facts"):
        reasons.append("primary_view_reports_missing_facts")
    if repaired.get("unsupported_claims"):
        reasons.append("primary_view_reports_unsupported_claims")
    if gain <= threshold:
        reasons.append("primary_utility_gain_not_positive")
    return {
        "escalated": bool(reasons),
        "risk_reasons": reasons,
        "primary_winner": view.get("winner"),
        "primary_utility_gain": round(gain, 6),
        "primary_repair_is_a": bool(view.get("repair_is_a")),
        "primary_missing_facts": repaired.get("missing_required_facts") or [],
        "primary_unsupported_claims": repaired.get("unsupported_claims") or [],
    }


def validation_metrics(rows: list[dict], threshold: float) -> dict:
    candidates = [row for row in rows if (row.get("post_repair_consensus_agent") or {}).get("enabled")]
    escalated = 0
    matched = 0
    for row in candidates:
        consensus = row["post_repair_consensus_agent"]
        decision = route(int(row.get("line_no", 0)), consensus, threshold)
        escalated += int(decision["escalated"])
        cascaded_action = consensus.get("final_action") if decision["escalated"] else "use_repair"
        matched += int(cascaded_action == consensus.get("final_action"))
    full_calls = 2 * len(candidates)
    cascade_calls = len(candidates) + escalated
    return {
        "samples": len(candidates),
        "full_consensus_calls": full_calls,
        "cascade_calls": cascade_calls,
        "second_view_calls": escalated,
        "call_reduction": round(1 - cascade_calls / max(1, full_calls), 6),
        "action_match": round(matched / max(1, len(candidates)), 6),
    }


def write_summary(path: Path, rows: list[dict], validation: dict, source_path: Path) -> None:
    agents = [row.get("selective_consensus_router") or {} for row in rows]
    checked = [agent for agent in agents if agent.get("enabled")]
    escalated = sum(bool(agent.get("second_view_used")) for agent in checked)
    retained = sum(agent.get("final_action") == "use_repair" for agent in checked)
    reverted = sum(agent.get("final_action") == "retain_original" for agent in checked)
    full_calls = 2 * len(checked)
    cascade_calls = len(checked) + escalated
    lines = [
        "# SAGE v29 Selective Consensus Router Summary\n\n",
        f"- Source: `{source_path.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Router inputs use only first-view runtime signals; references and evaluation scores are not used.\n\n",
        "## Expanded 24-Repair Validation\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Repair samples | {validation['samples']} |\n",
        f"| Full two-view calls | {validation['full_consensus_calls']} |\n",
        f"| Cascaded calls | {validation['cascade_calls']} |\n",
        f"| Call reduction | {validation['call_reduction'] * 100:.2f}% |\n",
        f"| Full-consensus action match | {validation['action_match'] * 100:.2f}% |\n\n",
        "## Conservative v27 Deployment Replay\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Repairs checked | {len(checked)} |\n",
        f"| First-view calls | {len(checked)} |\n",
        f"| Second-view calls | {escalated} |\n",
        f"| Total cascaded calls | {cascade_calls} |\n",
        f"| Full two-view calls | {full_calls} |\n",
        f"| Call reduction | {(1 - cascade_calls / max(1, full_calls)) * 100:.2f}% |\n",
        f"| Repairs retained | {retained} |\n",
        f"| Repairs reverted | {reverted} |\n\n",
        "## Per-Sample Routing\n\n",
        "| Line | First gain | Escalated | Reasons | Final action |\n",
        "|---:|---:|---|---|---|\n",
    ]
    for row in rows:
        agent = row.get("selective_consensus_router") or {}
        if not agent.get("enabled"):
            continue
        lines.append(
            f"| {row.get('line_no')} | {agent.get('primary_utility_gain', 0)} | "
            f"{agent.get('second_view_used')} | {', '.join(agent.get('risk_reasons', [])) or 'none'} | "
            f"{agent.get('final_action')} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--expanded-consensus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    args = parser.parse_args()

    target_path = ROOT_DIR / args.target
    rows = load_jsonl(target_path)
    expanded_rows = load_jsonl(ROOT_DIR / args.expanded_consensus)
    expanded_by_line = {int(row.get("line_no", 0)): row for row in expanded_rows}
    validation = validation_metrics(expanded_rows, args.utility_threshold)

    for row in rows:
        line_no = int(row.get("line_no", 0))
        contract_agent = row.get("contract_completion_agent") or {}
        enabled = bool(contract_agent.get("repair_accepted") and row.get("prediction_before_contract_completion"))
        router = {
            "enabled": enabled,
            "utility_threshold": args.utility_threshold,
            "second_view_used": False,
            "final_action": "not_checked",
        }
        row["selective_consensus_router"] = router
        if not enabled:
            continue
        expanded = expanded_by_line.get(line_no, {})
        consensus = expanded.get("post_repair_consensus_agent") or {}
        decision = route(line_no, consensus, args.utility_threshold)
        router.update(decision)
        router["second_view_used"] = decision["escalated"]
        router["full_consensus_action"] = consensus.get("final_action")
        final_action = consensus.get("final_action") if decision["escalated"] else "use_repair"
        router["final_action"] = final_action
        if final_action == "retain_original":
            row["prediction_before_selective_consensus"] = row.get("prediction", "")
            row["prediction"] = row.get("prediction_before_contract_completion", "")
            row["method"] = "sage_v29_selective_consensus_original"
        else:
            row["method"] = "sage_v29_selective_consensus_repair"

    output_path = ROOT_DIR / args.output
    write_jsonl(rows, output_path)
    write_summary(ROOT_DIR / args.summary, rows, validation, target_path)
    print(f"Saved SAGE v29 results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
