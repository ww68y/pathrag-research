from collections import Counter
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_vsr_pathrag_v4_crossdomain_suite as parent


parent.base.OUT_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v5_crossdomain_suite_12.jsonl"
parent.base.OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v5_crossdomain_suite_12_summary.md"
parent.base.SUMMARY_TITLE = "# VSR-PathRAG v5 Active Evidence Cross-Domain Suite"
_v4_summarize = parent.base.summarize


def summarize(rows: list[dict]) -> str:
    text = _v4_summarize(rows)
    ok_rows = [row for row in rows if row.get("ok")]
    triggered = 0
    expanded = 0
    queries = 0
    admitted = 0
    rejected = 0
    resolved = 0
    obligations = 0
    gains = []
    initial_gaps = []
    final_gaps = []
    stop_reasons = Counter()
    claim_safety_gates = Counter()
    claim_safety_triggers = 0
    claim_safety_overrides = 0
    repair_triggers = 0
    for row in ok_rows:
        trace = row.get("trace") or {}
        active = trace.get("active_evidence_acquisition") or {}
        triggered += int(bool(active.get("triggered")))
        expanded += int(bool(active.get("evidence_view_expanded")))
        queries += int(active.get("query_count", 0) or 0)
        admitted += int(active.get("admitted_relation_count", 0) or 0)
        rejected += int(active.get("rejected_relation_count", 0) or 0)
        resolved += int(active.get("resolved_obligation_count", 0) or 0)
        obligations += len(active.get("obligations", []))
        gains.append(float(active.get("marginal_information_gain", 0.0) or 0.0))
        initial_gaps.append(float(active.get("initial_gap_score", 0.0) or 0.0))
        final_gaps.append(float(active.get("final_gap_score", 0.0) or 0.0))
        stop_reasons[str(active.get("stop_reason", "unknown"))] += 1
        claim_safety = trace.get("signed_candidate_safety") or {}
        claim_safety_triggers += int(bool(claim_safety.get("triggered")))
        claim_safety_overrides += int(bool(claim_safety.get("override")))
        claim_safety_gates[str(claim_safety.get("safety_gate", "unknown"))] += 1
        repair_triggers += int(bool(trace.get("graph_repair", {}).get("triggered")))
    denominator = max(1, len(ok_rows))
    return text + "\n" + "\n".join(
        [
            "## VSR-PathRAG v5 Active Evidence Metrics",
            "",
            f"- Active acquisition triggered: {triggered}/{len(ok_rows)}",
            f"- Evidence views expanded: {expanded}/{len(ok_rows)}",
            f"- Targeted retrieval queries: {queries}",
            f"- Provenance-admitted relations: {admitted}",
            f"- Rejected or duplicate relations: {rejected}",
            f"- Resolved evidence obligations: {resolved}/{obligations}",
            f"- Avg marginal information gain: {sum(gains) / denominator:.4f}",
            f"- Avg graph gap: {sum(initial_gaps) / denominator:.3f} -> {sum(final_gaps) / denominator:.3f}",
            f"- Residual graph-repair triggers: {repair_triggers}/{len(ok_rows)}",
            f"- Stop reasons: {dict(stop_reasons)}",
            f"- Signed-claim safety triggers: {claim_safety_triggers}/{len(ok_rows)}",
            f"- Contradicted candidates replaced: {claim_safety_overrides}",
            f"- Claim-safety gates: {dict(claim_safety_gates)}",
            "- Base graph mutation: 0",
            "",
        ]
    )


parent.base.summarize = summarize
base = parent.base


if __name__ == "__main__":
    import asyncio

    asyncio.run(base.main())
