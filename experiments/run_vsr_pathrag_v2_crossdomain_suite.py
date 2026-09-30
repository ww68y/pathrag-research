from collections import Counter
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v20_crossdomain_cached_suite as base


base.OUT_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v2_crossdomain_suite_12.jsonl"
base.OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v2_crossdomain_suite_12_summary.md"
base.REQUESTED_MODE = "agent_vsr_fast"
base.SAMPLE_TIMEOUT_SECONDS = 90
base.SUMMARY_TITLE = "# VSR-PathRAG v2 Risk-Aware Portfolio Cross-Domain Suite"
_base_summarize = base.summarize


def summarize(rows: list[dict]) -> str:
    text = _base_summarize(rows)
    ok_rows = [row for row in rows if row.get("ok")]
    gap_required = 0
    repair_triggered = 0
    accepted_repairs = 0
    rollback_cases = 0
    claim_status_counts = Counter()
    portfolio_risks = Counter()
    selected_paths = 0
    candidate_paths = 0
    estimated_tokens = 0
    token_budgets = 0
    goal_coverage = []
    repair_dependency = []
    redundancies = []
    unique_sources = 0

    for row in ok_rows:
        trace = row.get("trace") or {}
        gap = trace.get("knowledge_gap") or {}
        repair = trace.get("graph_repair") or {}
        validation = repair.get("validation") or {}
        portfolio = trace.get("evidence_portfolio") or {}
        claim_graph = trace.get("claim_evidence_graph") or {}
        if gap.get("repair_required"):
            gap_required += 1
        if repair.get("triggered"):
            repair_triggered += 1
        accepted_repairs += int(validation.get("accepted_count", 0) or 0)
        if (trace.get("counterfactual_rollback") or {}).get("triggered"):
            rollback_cases += 1
        claim_status_counts.update(claim_graph.get("status_counts") or {})
        portfolio_risks[str(portfolio.get("risk_level", "unknown"))] += 1
        selected_paths += int(portfolio.get("selected_count", 0) or 0)
        candidate_paths += int(portfolio.get("candidate_count", 0) or 0)
        estimated_tokens += int(portfolio.get("estimated_tokens", 0) or 0)
        token_budgets += int(portfolio.get("token_budget", 0) or 0)
        unique_sources += int(portfolio.get("unique_source_count", 0) or 0)
        goal_coverage.append(float(portfolio.get("goal_coverage_ratio", 0) or 0))
        repair_dependency.append(float(portfolio.get("repair_dependency_ratio", 0) or 0))
        redundancies.append(float(portfolio.get("mean_pairwise_redundancy", 0) or 0))

    denominator = max(1, len(ok_rows))
    return text + "\n" + "\n".join(
        [
            "## VSR-PathRAG v2 Innovation Metrics",
            "",
            f"- Knowledge-gap cases: {gap_required}/{len(ok_rows)}",
            f"- Query-local repair triggered: {repair_triggered}/{len(ok_rows)}",
            f"- Accepted temporary relations: {accepted_repairs}",
            f"- Counterfactual rollback cases: {rollback_cases}",
            f"- Signed claim statuses: {dict(claim_status_counts)}",
            f"- Portfolio candidates: {candidate_paths}",
            f"- Portfolio selected paths: {selected_paths}",
            f"- Portfolio compression ratio: {selected_paths / max(1, candidate_paths):.3f}",
            f"- Avg selected paths: {selected_paths / denominator:.2f}",
            f"- Avg unique sources: {unique_sources / denominator:.2f}",
            f"- Avg goal coverage: {sum(goal_coverage) / denominator:.3f}",
            f"- Avg repair dependency: {sum(repair_dependency) / denominator:.3f}",
            f"- Avg pairwise redundancy: {sum(redundancies) / denominator:.3f}",
            f"- Token budget utilization: {estimated_tokens}/{token_budgets} ({estimated_tokens / max(1, token_budgets):.3f})",
            f"- Portfolio risk levels: {dict(portfolio_risks)}",
            "- Base graph mutation: 0",
            "",
        ]
    )


base.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(base.main())
