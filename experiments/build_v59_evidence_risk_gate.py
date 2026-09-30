import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def risk_gate(agent: dict) -> tuple[bool, dict]:
    guard = agent.get("guard") or {}
    current_len = int(guard.get("current_chars") or 0)
    candidate_len = int(guard.get("candidate_chars") or 0)
    current_qcov = float(guard.get("current_question_term_coverage") or 0.0)
    candidate_qcov = float(guard.get("candidate_question_term_coverage") or 0.0)
    current_claim = guard.get("current_claim_metrics") or {}
    candidate_claim = guard.get("candidate_claim_metrics") or {}
    facet_stats = guard.get("supported_facet_stats") or {}

    current_support = float(current_claim.get("mean_support") or 0.0)
    candidate_support = float(candidate_claim.get("mean_support") or 0.0)
    current_unsupported = float(current_claim.get("unsupported_ratio") or 0.0)
    candidate_unsupported = float(candidate_claim.get("unsupported_ratio") or 0.0)
    supported_before_terms = int(facet_stats.get("supported_before_terms") or 0)
    retention = float(facet_stats.get("supported_facet_retention") or 1.0)

    is_compression = candidate_len < current_len
    severe_compression = candidate_len < current_len * 0.90
    any_compression = candidate_len < current_len * 0.98

    compression_guard = True
    if any_compression and supported_before_terms >= 35:
        compression_guard = retention >= 0.35
    if severe_compression and supported_before_terms >= 25:
        compression_guard = compression_guard and retention >= 0.50

    checks = {
        "source_editor_accepted": bool(agent.get("accepted")),
        "question_terms_preserved": candidate_qcov + 0.25 >= current_qcov,
        "support_not_degraded": candidate_support >= max(0.25, current_support - 0.10),
        "unsupported_ratio_not_degraded": candidate_unsupported <= max(0.45, current_unsupported + 0.12),
        "supported_facets_preserved": compression_guard,
    }
    accepted = all(checks.values())
    return accepted, {
        "version": "v59",
        "accepted": accepted,
        "action": "keep_residual_hardcase_edit" if accepted else "revert_residual_hardcase_edit",
        "checks": checks,
        "current_chars": current_len,
        "candidate_chars": candidate_len,
        "current_question_term_coverage": current_qcov,
        "candidate_question_term_coverage": candidate_qcov,
        "current_mean_support": current_support,
        "candidate_mean_support": candidate_support,
        "current_unsupported_ratio": current_unsupported,
        "candidate_unsupported_ratio": candidate_unsupported,
        "supported_before_terms": supported_before_terms,
        "supported_facet_retention": retention,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }


def build_summary(rows: list[dict], source: Path) -> str:
    enabled = [r for r in rows if (r.get("evidence_risk_gate") or {}).get("enabled")]
    kept = [r for r in enabled if (r.get("evidence_risk_gate") or {}).get("accepted")]
    actions = Counter((r.get("evidence_risk_gate") or {}).get("action", "not_applicable") for r in rows)

    lines = [
        "# SAGE v59 Evidence Risk Gate Summary\n\n",
        f"- Source: `{source}`\n",
        "- Purpose: post-filter residual hard-case edits with runtime-only evidence risk checks.\n",
        "- Reference answers and judge scores are not runtime inputs.\n\n",
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
            "| Line | Dataset | Action | QTerm Cov | Facet Retention | Support | Reasons |\n",
            "|---:|---|---|---:|---:|---:|---|\n",
        ]
    )
    for row in enabled:
        gate = row.get("evidence_risk_gate") or {}
        reasons = ", ".join(gate.get("rejection_reasons") or []) or "none"
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {gate.get('action')} | "
            f"{gate.get('current_question_term_coverage'):.3f} -> {gate.get('candidate_question_term_coverage'):.3f} | "
            f"{gate.get('supported_facet_retention'):.3f} | "
            f"{gate.get('current_mean_support'):.3f} -> {gate.get('candidate_mean_support'):.3f} | "
            f"{reasons} |\n"
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
            row["evidence_risk_gate"] = {"enabled": False, "version": "v59", "action": "not_applicable"}
            continue

        row["prediction_before_evidence_risk_gate"] = row.get("prediction", "")
        accepted, gate = risk_gate(agent)
        gate["enabled"] = True
        row["evidence_risk_gate"] = gate
        if not accepted:
            row["rejected_evidence_risk_gate_prediction"] = row.get("prediction", "")
            row["prediction"] = row.get("prediction_before_residual_hardcase_editor", row.get("prediction", ""))
        row["method"] = "mse-pathrag-sage-v59-evidence-risk-gate"

    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(build_summary(rows, args.input), encoding="utf-8")
    print(f"Saved v59 results to {args.output}")
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
