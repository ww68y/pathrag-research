import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.vsr_pathrag import (
    apply_source_conflict_arbitration,
    build_source_conflict_graph,
    finalize_uncertainty_compute_budget,
    plan_uncertainty_compute_budget,
)


OUT_JSON = ROOT_DIR / "results" / "vsr_pathrag_v3_conflict_budget_benchmark.json"
OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v3_conflict_budget_benchmark_summary.md"


def numeric_case(index: int) -> dict:
    expected_value = 280 + index
    conflicting_value = 320 + index
    evidence = {
        "entities": [],
        "relationships": [
            {
                "src": f"AURORA {index}",
                "tgt": "OPERATING RANGE",
                "keywords": "operating range",
                "description": f"The system has an operating range of {expected_value} kilometers.",
                "source_ids": [f"numeric-{index}-a", f"numeric-{index}-b"],
            },
            {
                "src": f"AURORA {index}",
                "tgt": "OPERATING RANGE",
                "keywords": "operating range",
                "description": f"The system has an operating range of {conflicting_value} kilometers.",
                "source_ids": [f"numeric-{index}-c"],
            },
        ],
    }
    graph = build_source_conflict_graph("What is the operating range?", evidence)
    arbitrated, decision = apply_source_conflict_arbitration(evidence, graph)
    active_text = " ".join(row.get("description", "") for row in arbitrated["relationships"])
    return {
        "case_id": f"numeric-{index}",
        "expected": "resolved_majority",
        "graph": graph,
        "arbitration": decision,
        "passed": bool(
            graph["conflict_count"] >= 1
            and graph["conflicts"][0]["resolution"] == "independent_source_majority"
            and str(expected_value) in active_text
            and str(conflicting_value) not in active_text
        ),
    }


def polarity_case(index: int) -> dict:
    evidence = {
        "entities": [],
        "relationships": [
            {
                "src": f"AURORA BOARD {index}",
                "tgt": "PUBLIC RELEASE",
                "keywords": "approval status",
                "description": f"The Aurora board {index} approved the public release.",
                "source_ids": [f"polarity-{index}-positive"],
            },
            {
                "src": f"AURORA BOARD {index}",
                "tgt": "PUBLIC RELEASE",
                "keywords": "approval status",
                "description": f"The Aurora board {index} did not approve the public release.",
                "source_ids": [f"polarity-{index}-negative"],
            },
        ],
    }
    graph = build_source_conflict_graph("Did the board approve the release?", evidence)
    arbitrated, decision = apply_source_conflict_arbitration(evidence, graph)
    return {
        "case_id": f"polarity-{index}",
        "expected": "unresolved_quarantine",
        "graph": graph,
        "arbitration": decision,
        "passed": bool(
            graph["unresolved_count"] >= 1
            and not arbitrated["relationships"]
            and decision["suppressed_count"] == 2
        ),
    }


def clean_case(index: int) -> dict:
    evidence = {
        "entities": [],
        "relationships": [
            {
                "src": f"AURORA {index}",
                "tgt": "OPERATING RANGE",
                "keywords": "operating range",
                "description": f"Aurora {index} has a range of {280 + index} kilometers.",
                "source_ids": [f"clean-{index}-a"],
            },
            {
                "src": f"BOREALIS {index}",
                "tgt": "BATTERY CAPACITY",
                "keywords": "battery capacity",
                "description": f"Borealis {index} has a capacity of {320 + index} kilowatt-hours.",
                "source_ids": [f"clean-{index}-b"],
            },
        ],
    }
    graph = build_source_conflict_graph("Compare Aurora and Borealis.", evidence)
    arbitrated, decision = apply_source_conflict_arbitration(evidence, graph)
    return {
        "case_id": f"clean-{index}",
        "expected": "clean",
        "graph": graph,
        "arbitration": decision,
        "passed": bool(
            graph["clean"]
            and len(arbitrated["relationships"]) == 2
            and decision["suppressed_count"] == 0
        ),
    }


def budget_case(expected_tier: str, index: int) -> dict:
    if expected_tier == "low":
        query_graph = {"requires_multi_hop": False}
        gap = {"gap_score": 0.08 + index * 0.02, "anchor_coverage": 1.0, "repair_required": False}
        conflict = {"risk_score": 0.0, "unresolved_count": 0}
    elif expected_tier == "medium":
        query_graph = {"requires_multi_hop": True}
        gap = {"gap_score": 0.48 + index * 0.04, "anchor_coverage": 0.8, "repair_required": True}
        conflict = {"risk_score": 0.15, "unresolved_count": 0}
    else:
        query_graph = {"requires_multi_hop": True}
        gap = {"gap_score": 0.82 + index * 0.03, "anchor_coverage": 0.0, "repair_required": True}
        conflict = {"risk_score": 0.75, "unresolved_count": 1}
    plan = plan_uncertainty_compute_budget(
        query_graph,
        gap,
        conflict,
        requested_fast=False,
        contract_type="balanced",
    )
    finalized = finalize_uncertainty_compute_budget(
        plan,
        {
            "repair_dependency_ratio": 0.0 if expected_tier != "high" else 0.6,
            "risk_level": "low" if expected_tier != "high" else "high",
            "budget_utilization": 0.5,
            "estimated_tokens": plan["token_budget"] // 2,
        },
    )
    return {
        "case_id": f"budget-{expected_tier}-{index}",
        "expected_tier": expected_tier,
        "plan": finalized,
        "passed": plan["tier"] == expected_tier and finalized["within_token_budget"],
    }


def main() -> None:
    conflict_rows = []
    conflict_rows.extend(numeric_case(index) for index in range(10))
    conflict_rows.extend(polarity_case(index) for index in range(10))
    conflict_rows.extend(clean_case(index) for index in range(10))
    budget_rows = [
        budget_case(tier, index)
        for tier in ["low", "medium", "high"]
        for index in range(3)
    ]
    payload = {
        "benchmark": "vsr_pathrag_v3_conflict_budget",
        "conflict_cases": conflict_rows,
        "budget_cases": budget_rows,
        "metrics": {
            "conflict_passed": sum(row["passed"] for row in conflict_rows),
            "conflict_total": len(conflict_rows),
            "numeric_majority_passed": sum(row["passed"] for row in conflict_rows[:10]),
            "polarity_quarantine_passed": sum(row["passed"] for row in conflict_rows[10:20]),
            "clean_no_false_positive": sum(row["passed"] for row in conflict_rows[20:]),
            "budget_passed": sum(row["passed"] for row in budget_rows),
            "budget_total": len(budget_rows),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = payload["metrics"]
    lines = [
        "# VSR-PathRAG v3 Conflict and Budget Benchmark",
        "",
        "## Controlled Conflict Cases",
        "",
        f"- Total passed: `{metrics['conflict_passed']}/{metrics['conflict_total']}`",
        f"- Numeric majority resolution: `{metrics['numeric_majority_passed']}/10`",
        f"- Equal-provenance polarity quarantine: `{metrics['polarity_quarantine_passed']}/10`",
        f"- Clean cases without false positives: `{metrics['clean_no_false_positive']}/10`",
        "",
        "## Uncertainty Budget Cases",
        "",
        f"- Correct tier and budget compliance: `{metrics['budget_passed']}/{metrics['budget_total']}`",
        "- Tiers: `3 low + 3 medium + 3 high`",
        "- High-risk repair dependency forces counterfactual-depth verification.",
        "",
    ]
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_SUMMARY}")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
