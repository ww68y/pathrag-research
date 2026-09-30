from collections import Counter
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v20_crossdomain_cached_suite as base


base.OUT_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v1_crossdomain_suite_12.jsonl"
base.OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v1_crossdomain_suite_12_summary.md"
base.REQUESTED_MODE = "agent_vsr_fast"
base.SAMPLE_TIMEOUT_SECONDS = 90
base.SUMMARY_TITLE = "# VSR-PathRAG v1 Cross-Domain Suite"
_base_summarize = base.summarize


def summarize(rows: list[dict]) -> str:
    text = _base_summarize(rows)
    ok_rows = [row for row in rows if row.get("ok")]
    gap_required = 0
    repair_triggered = 0
    accepted_repairs = 0
    rejected_repairs = 0
    rollback_cases = 0
    accepted_paths = 0
    extraction_counts = Counter()
    claim_status_counts = Counter()
    for row in ok_rows:
        trace = row.get("trace") or {}
        gap = trace.get("knowledge_gap") or {}
        repair = trace.get("graph_repair") or {}
        validation = repair.get("validation") or {}
        overlay = repair.get("overlay") or {}
        claim_graph = trace.get("claim_evidence_graph") or {}
        if gap.get("repair_required"):
            gap_required += 1
        if repair.get("triggered"):
            repair_triggered += 1
        accepted_repairs += int(validation.get("accepted_count", 0) or 0)
        rejected_repairs += int(validation.get("rejected_count", 0) or 0)
        accepted_paths += int(overlay.get("path_count", 0) or 0)
        if (trace.get("counterfactual_rollback") or {}).get("triggered"):
            rollback_cases += 1
        extraction_counts["llm"] += int(bool((repair.get("proposal") or {}).get("llm_used")))
        extraction_counts["deterministic"] += int(
            bool((repair.get("proposal") or {}).get("candidates"))
            and not (repair.get("proposal") or {}).get("llm_used")
        )
        claim_status_counts.update(claim_graph.get("status_counts") or {})

    return text + "\n" + "\n".join(
        [
            "## VSR-PathRAG Innovation Metrics",
            "",
            f"- Knowledge-gap cases: {gap_required}/{len(ok_rows)}",
            f"- Query-local repair triggered: {repair_triggered}/{len(ok_rows)}",
            f"- Accepted temporary relations: {accepted_repairs}",
            f"- Rejected candidate relations: {rejected_repairs}",
            f"- Ranked overlay paths: {accepted_paths}",
            f"- Counterfactual rollback cases: {rollback_cases}",
            f"- Extraction routes: {dict(extraction_counts)}",
            f"- Signed claim statuses: {dict(claim_status_counts)}",
            "- Base graph mutation: 0 (all repairs are query-local overlays)",
            "",
        ]
    )


base.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(base.main())
