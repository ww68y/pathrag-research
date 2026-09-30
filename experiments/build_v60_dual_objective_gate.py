import argparse
import json
from collections import Counter
from pathlib import Path

from build_v59_evidence_risk_gate import load_jsonl, risk_gate, write_jsonl


def dual_objective_gate(agent: dict) -> tuple[bool, dict]:
    accepted, gate = risk_gate(agent)
    guard = agent.get("guard") or {}
    current_len = int(guard.get("current_chars") or 0)
    candidate_len = int(guard.get("candidate_chars") or 0)
    facet_stats = guard.get("supported_facet_stats") or {}
    supported_before_terms = int(facet_stats.get("supported_before_terms") or 0)
    retention = float(facet_stats.get("supported_facet_retention") or 1.0)
    compression_ratio = candidate_len / max(1, current_len)

    high_compression_facet_retention = True
    if compression_ratio < 0.70 and supported_before_terms >= 35:
        high_compression_facet_retention = retention >= 0.65

    gate["version"] = "v60"
    gate["compression_ratio"] = round(compression_ratio, 6)
    gate["checks"]["high_compression_facet_retention"] = high_compression_facet_retention
    accepted = accepted and high_compression_facet_retention
    gate["accepted"] = accepted
    gate["action"] = "keep_residual_hardcase_edit" if accepted else "revert_residual_hardcase_edit"
    gate["rejection_reasons"] = [name for name, passed in gate["checks"].items() if not passed]
    return accepted, gate


def build_summary(rows: list[dict], source: Path) -> str:
    enabled = [r for r in rows if (r.get("dual_objective_gate") or {}).get("enabled")]
    kept = [r for r in enabled if (r.get("dual_objective_gate") or {}).get("accepted")]
    actions = Counter((r.get("dual_objective_gate") or {}).get("action", "not_applicable") for r in rows)
    lines = [
        "# SAGE v60 Dual-Objective Evidence Gate Summary\n\n",
        f"- Source: `{source}`\n",
        "- Purpose: preserve v58 hard-case improvements while blocking edits that sacrifice evidence facets under heavy compression.\n",
        "- Reference answers, judge scores, and pairwise labels are not runtime inputs.\n\n",
        "| Metric | Count |\n|---|---:|\n",
        f"| Candidate edits checked | {len(enabled)} |\n",
        f"| Candidate edits kept | {len(kept)} |\n",
        f"| Candidate edits reverted | {len(enabled) - len(kept)} |\n\n",
        "## Actions\n\n",
        "| Action | Count |\n|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")
    lines.extend(
        [
            "\n## Per-Sample Decisions\n\n",
            "| Line | Dataset | Action | Compression | QTerm Cov | Facet Retention | Reasons |\n",
            "|---:|---|---|---:|---:|---:|---|\n",
        ]
    )
    for row in enabled:
        gate = row.get("dual_objective_gate") or {}
        reasons = ", ".join(gate.get("rejection_reasons") or []) or "none"
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {gate.get('action')} | "
            f"{gate.get('compression_ratio'):.3f} | "
            f"{gate.get('current_question_term_coverage'):.3f} -> {gate.get('candidate_question_term_coverage'):.3f} | "
            f"{gate.get('supported_facet_retention'):.3f} | {reasons} |\n"
        )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    for row in rows:
        agent = row.get("residual_hardcase_editor") or {}
        if not agent.get("accepted"):
            row["dual_objective_gate"] = {"enabled": False, "version": "v60", "action": "not_applicable"}
            continue
        row["prediction_before_dual_objective_gate"] = row.get("prediction", "")
        accepted, gate = dual_objective_gate(agent)
        gate["enabled"] = True
        row["dual_objective_gate"] = gate
        if not accepted:
            row["rejected_dual_objective_gate_prediction"] = row.get("prediction", "")
            row["prediction"] = row.get("prediction_before_residual_hardcase_editor", row.get("prediction", ""))
        row["method"] = "mse-pathrag-sage-v60-dual-objective-gate"

    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(build_summary(rows, args.input), encoding="utf-8")
    print(f"Saved v60 results to {args.output}")
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
