from collections import Counter
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v20_crossdomain_cached_suite as base


base.OUT_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v3_crossdomain_suite_12.jsonl"
base.OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v3_crossdomain_suite_12_summary.md"
base.REQUESTED_MODE = "agent_vsr_fast"
base.SAMPLE_TIMEOUT_SECONDS = 90
base.SUMMARY_TITLE = "# VSR-PathRAG v3 Conflict-Aware Budgeted Cross-Domain Suite"
_base_summarize = base.summarize


def summarize(rows: list[dict]) -> str:
    text = _base_summarize(rows)
    ok_rows = [row for row in rows if row.get("ok")]
    conflict_cases = 0
    unresolved_conflicts = 0
    suppressed_evidence = 0
    accepted_repairs = 0
    rollback_cases = 0
    portfolio_candidates = 0
    selected_paths = 0
    estimated_tokens = 0
    token_budgets = 0
    unique_sources = 0
    uncertainty_scores = []
    goal_coverage = []
    repair_dependency = []
    redundancies = []
    budget_tiers = Counter()
    verification_depths = Counter()
    portfolio_risks = Counter()
    claim_status_counts = Counter()
    escalations = 0

    for row in ok_rows:
        trace = row.get("trace") or {}
        conflict = trace.get("source_conflict_graph") or {}
        arbitration = trace.get("source_conflict_arbitration") or {}
        budget = trace.get("compute_budget") or {}
        portfolio = trace.get("evidence_portfolio") or {}
        repair = trace.get("graph_repair") or {}
        claim_graph = trace.get("claim_evidence_graph") or {}
        if int(conflict.get("conflict_count", 0) or 0):
            conflict_cases += 1
        unresolved_conflicts += int(conflict.get("unresolved_count", 0) or 0)
        suppressed_evidence += int(arbitration.get("suppressed_count", 0) or 0)
        accepted_repairs += int((repair.get("validation") or {}).get("accepted_count", 0) or 0)
        if (trace.get("counterfactual_rollback") or {}).get("triggered"):
            rollback_cases += 1
        portfolio_candidates += int(portfolio.get("candidate_count", 0) or 0)
        selected_paths += int(portfolio.get("selected_count", 0) or 0)
        estimated_tokens += int(portfolio.get("estimated_tokens", 0) or 0)
        token_budgets += int(portfolio.get("token_budget", 0) or 0)
        unique_sources += int(portfolio.get("unique_source_count", 0) or 0)
        uncertainty_scores.append(float(budget.get("uncertainty_score", 0) or 0))
        goal_coverage.append(float(portfolio.get("goal_coverage_ratio", 0) or 0))
        repair_dependency.append(float(portfolio.get("repair_dependency_ratio", 0) or 0))
        redundancies.append(float(portfolio.get("mean_pairwise_redundancy", 0) or 0))
        budget_tiers[str(budget.get("tier", "unknown"))] += 1
        verification_depths[str(budget.get("verification_depth", "unknown"))] += 1
        portfolio_risks[str(portfolio.get("risk_level", "unknown"))] += 1
        claim_status_counts.update(claim_graph.get("status_counts") or {})
        escalations += int(bool(budget.get("escalated_after_portfolio")))

    denominator = max(1, len(ok_rows))
    return text + "\n" + "\n".join(
        [
            "## VSR-PathRAG v3 Innovation Metrics",
            "",
            f"- Cross-source conflict cases: {conflict_cases}/{len(ok_rows)}",
            f"- Unresolved conflicts: {unresolved_conflicts}",
            f"- Quarantined evidence items: {suppressed_evidence}",
            f"- Accepted temporary relations: {accepted_repairs}",
            f"- Counterfactual rollback cases: {rollback_cases}",
            f"- Signed claim statuses: {dict(claim_status_counts)}",
            f"- Budget tiers: {dict(budget_tiers)}",
            f"- Verification depths: {dict(verification_depths)}",
            f"- Post-portfolio escalations: {escalations}",
            f"- Avg uncertainty score: {sum(uncertainty_scores) / denominator:.3f}",
            f"- Portfolio candidates: {portfolio_candidates}",
            f"- Portfolio selected paths: {selected_paths}",
            f"- Portfolio compression ratio: {selected_paths / max(1, portfolio_candidates):.3f}",
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
