from pathlib import Path
import sys
from collections import Counter


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v6_claim_guard_suite as suite


suite.OUT_JSONL = ROOT_DIR / "results" / "interactive_agent_v8_facet_completion_suite_18.jsonl"
suite.OUT_SUMMARY = ROOT_DIR / "results" / "interactive_agent_v8_facet_completion_suite_18_summary.md"


def summarize(rows: list[dict]) -> str:
    ok_rows = [row for row in rows if row.get("ok")]
    profile_counts = Counter(row.get("selected_profile", "unknown") for row in ok_rows)
    policy_counts = Counter(row.get("final_policy", "unknown") for row in ok_rows)
    type_counts = Counter(row.get("question_type", "unknown") for row in ok_rows)
    risk_counts = Counter(flag for row in ok_rows for flag in row.get("risk_flags", []))
    repair_triggered = sum(1 for row in ok_rows if row.get("repair_triggered"))
    repair_accepted = sum(1 for row in ok_rows if row.get("repair_accepted"))
    claim_prune_triggered = sum(1 for row in ok_rows if row.get("claim_prune_triggered"))
    claim_prune_accepted = sum(1 for row in ok_rows if row.get("claim_prune_accepted"))
    facet_completion_triggered = sum(1 for row in ok_rows if (row.get("trace") or {}).get("facet_completion_triggered"))
    facet_completion_accepted = sum(1 for row in ok_rows if (row.get("trace") or {}).get("facet_completion_accepted"))
    avg_runtime = sum(row.get("runtime_seconds", 0) for row in ok_rows) / max(1, len(ok_rows))
    avg_candidates = sum(row.get("candidate_count", 0) for row in ok_rows) / max(1, len(ok_rows))
    avg_evidence_edges = sum(row.get("evidence_edges", 0) for row in ok_rows) / max(1, len(ok_rows))
    avg_claim_support = sum(row.get("claim_support_ratio", 0) for row in ok_rows) / max(1, len(ok_rows))
    total_claims = sum(row.get("claim_count", 0) for row in ok_rows)
    total_unsupported = sum(row.get("unsupported_claim_count", 0) for row in ok_rows)
    facet_coverages = [
        ((row.get("trace") or {}).get("facet_coverage") or {}).get("coverage_ratio", 0)
        for row in ok_rows
    ]
    avg_facet_coverage = sum(facet_coverages) / max(1, len(facet_coverages))

    lines = [
        "# Interactive Agent v8 Supported-Facet Completion Suite",
        "",
        "Dataset: current uploaded Aurora Battery Pilot Project demo document.",
        "",
        "## Aggregate",
        "",
        f"- Samples: {len(rows)}",
        f"- Successful: {len(ok_rows)}",
        f"- Avg runtime seconds: {avg_runtime:.2f}",
        f"- Avg candidate count: {avg_candidates:.2f}",
        f"- Avg evidence edges: {avg_evidence_edges:.2f}",
        f"- Avg claim support ratio: {avg_claim_support:.3f}",
        f"- Total checked claims: {total_claims}",
        f"- Total unsupported claims: {total_unsupported}",
        f"- Unsupported claim rate: {total_unsupported / max(1, total_claims):.3f}",
        f"- Avg supported-facet coverage: {avg_facet_coverage:.3f}",
        f"- Question types: {dict(type_counts)}",
        f"- Selected profiles: {dict(profile_counts)}",
        f"- Final policies: {dict(policy_counts)}",
        f"- Evidence repair triggered: {repair_triggered}",
        f"- Evidence repair accepted: {repair_accepted}",
        f"- Claim prune triggered: {claim_prune_triggered}",
        f"- Claim prune accepted: {claim_prune_accepted}",
        f"- Facet completion triggered: {facet_completion_triggered}",
        f"- Facet completion accepted: {facet_completion_accepted}",
        f"- Risk flags: {dict(risk_counts)}",
        "",
        "## Per Sample",
        "",
        "| # | Question | Type | Profile | Candidates | Repair | Claim prune | Facet completion | Claims | Unsupported | Facets | Policy | Runtime | Risks |",
        "|---:|---|---|---|---:|---|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        trace = row.get("trace") or {}
        facet_coverage = trace.get("facet_coverage") or {}
        risks = ", ".join(row.get("risk_flags", [])) or "none"
        question = row["question"].replace("|", "/")
        repair = "accepted" if row.get("repair_accepted") else ("triggered" if row.get("repair_triggered") else "skipped")
        claim_prune = (
            "accepted"
            if row.get("claim_prune_accepted")
            else ("triggered" if row.get("claim_prune_triggered") else "skipped")
        )
        facet_completion = (
            "accepted"
            if trace.get("facet_completion_accepted")
            else ("triggered" if trace.get("facet_completion_triggered") else "skipped")
        )
        lines.append(
            f"| {row['index']} | {question} | {row.get('question_type', 'error')} | "
            f"{row.get('selected_profile', 'error')} | {row.get('candidate_count', 0)} | "
            f"{repair} | {claim_prune} | {facet_completion} | "
            f"{row.get('claim_supported_count', 0)}/{row.get('claim_count', 0)} | "
            f"{row.get('unsupported_claim_count', 0)} | "
            f"{facet_coverage.get('covered_count', 0)}/{facet_coverage.get('facet_count', 0)} | "
            f"{row.get('final_policy', 'error')} | {row.get('runtime_seconds', 0):.2f} | {risks} |"
        )
    lines.append("")
    return "\n".join(lines)


suite.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(suite.main())
