from collections import Counter
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_vsr_pathrag_v5_crossdomain_suite as parent


parent.base.OUT_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v6_crossdomain_suite_12.jsonl"
parent.base.OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v6_crossdomain_suite_12_summary.md"
parent.base.SUMMARY_TITLE = "# VSR-PathRAG v6 Claim-Verifying Cross-Domain Suite"
_v5_summarize = parent.base.summarize


def summarize(rows: list[dict]) -> str:
    text = _v5_summarize(rows)
    ok_rows = [row for row in rows if row.get("ok")]
    triggered = 0
    expanded = 0
    targets = 0
    plans = 0
    queries = 0
    provenance_admitted = 0
    committed = 0
    rejected = 0
    resolved = 0
    initial_statuses = Counter()
    final_statuses = Counter()
    stop_reasons = Counter()
    for row in ok_rows:
        verification = (row.get("trace") or {}).get("claim_active_verification") or {}
        triggered += int(bool(verification.get("triggered")))
        expanded += int(bool(verification.get("evidence_view_expanded")))
        targets += int(verification.get("target_claim_count", 0) or 0)
        plans += int(verification.get("planned_claim_count", 0) or 0)
        queries += int(verification.get("query_count", 0) or 0)
        provenance_admitted += int(verification.get("provenance_admitted_count", 0) or 0)
        committed += int(verification.get("counterfactual_committed_count", 0) or 0)
        rejected += int(verification.get("counterfactual_rejected_count", 0) or 0)
        resolved += int(verification.get("resolved_claim_count", 0) or 0)
        initial_statuses.update(verification.get("initial_status_counts") or {})
        final_statuses.update(verification.get("final_status_counts") or {})
        stop_reasons[str(verification.get("stop_reason", "unknown"))] += 1
    return text + "\n" + "\n".join(
        [
            "## VSR-PathRAG v6 Claim-Level Verification Metrics",
            "",
            f"- Claim verification triggered: {triggered}/{len(ok_rows)}",
            f"- Claim evidence views expanded: {expanded}/{len(ok_rows)}",
            f"- Unverified claim targets: {targets}",
            f"- Claims selected under budget: {plans}",
            f"- Targeted claim queries: {queries}",
            f"- Provenance-admitted relations: {provenance_admitted}",
            f"- Counterfactually committed relations: {committed}",
            f"- Rejected for no safe claim gain: {rejected}",
            f"- Claims resolved to supported: {resolved}",
            f"- Draft signed statuses: {dict(initial_statuses)}",
            f"- Post-verification signed statuses: {dict(final_statuses)}",
            f"- Stop reasons: {dict(stop_reasons)}",
            "- Base graph mutation: 0",
            "",
        ]
    )


parent.base.summarize = summarize
base = parent.base


if __name__ == "__main__":
    import asyncio

    asyncio.run(base.main())

