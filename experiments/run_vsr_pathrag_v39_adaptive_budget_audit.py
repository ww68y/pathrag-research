import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    build_correction_persistence_constraints,
    build_counterevidence_cost_context,
    plan_adaptive_counterevidence_query_budget,
    realize_answer,
    run_active_counterevidence_challenge,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
    load_committed_source,
    load_jsonl,
)


def build_inputs(
    claims: list[str],
    risk_scores: list[float],
) -> tuple[dict, dict, dict]:
    transaction = build_transaction(
        claims,
        [
            {
                "risk_level": (
                    "high" if score >= 0.75 else "medium"
                ),
                "risk_score": score,
            }
            for score in risk_scores
        ],
    )
    nodes = {
        row["node_id"]: row
        for row in transaction["dependency_graph"]["nodes"]
    }
    certificates = {
        row["node_id"]: row
        for row in transaction[
            "counterfactual_stability_certificate"
        ]["provenance_certificates"]
    }
    return transaction, nodes, certificates


async def execute_challenge(
    claims: list[str],
    risk_scores: list[float],
    hard_budget: int,
    priors: list[dict] | None = None,
) -> tuple[list[str], dict]:
    transaction, _nodes, _certificates = build_inputs(
        claims,
        risk_scores,
    )
    queries = []

    async def local_search(query, _max_results):
        queries.append(query)
        return {
            "rows": [
                {
                    "source_id": f"irrelevant-{len(queries)}",
                    "document_id": "weather-record",
                    "text": "A weather archive recorded rainfall.",
                    "source_reliability": 0.9,
                }
            ],
            "latency_ms": 10,
            "usage": {"call_count": 0, "total_tokens": 0},
        }

    _answer, challenge = await run_active_counterevidence_challenge(
        " ".join(claims),
        transaction,
        local_search,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_risk_coverage_guard=True,
        enable_adaptive_query_budget=True,
        historical_cost_priors=priors,
        max_queries_per_claim=3,
        max_global_queries=hard_budget,
    )
    return queries, challenge


async def main_async(args) -> dict:
    medium_claims = [
        "The campaign drew wider attendance.",
        "The program produced higher revenue.",
    ]
    medium_queries, medium = await execute_challenge(
        medium_claims,
        [0.46, 0.46],
        4,
    )
    medium_case = {
        "case": "medium_risk_portfolio_stops_at_coverage_floor",
        "observed": {
            "query_count": medium["query_count"],
            "effective_budget": medium[
                "effective_global_query_budget"
            ],
            "saved_budget": medium[
                "adaptive_query_budget_saved_count"
            ],
            "covered": medium["covered_risky_claim_count"],
            "queries": medium_queries,
        },
        "expectation_met": bool(
            medium["query_count"] == 2
            and medium["effective_global_query_budget"] == 2
            and medium["adaptive_query_budget_saved_count"] == 2
            and medium["covered_risky_claim_count"] == 2
        ),
        "challenge": medium,
    }

    high_claims = [
        "The program caused higher sales.",
        "The program caused greater revenue.",
    ]
    _transaction, high_nodes, high_certificates = build_inputs(
        high_claims,
        [0.9, 0.9],
    )
    unseen_plan = plan_adaptive_counterevidence_query_budget(
        list(high_nodes),
        high_nodes,
        high_certificates,
        hard_query_budget=4,
    )
    high_risk_case = {
        "case": "high_risk_unseen_routes_earn_full_extension",
        "observed": unseen_plan,
        "expectation_met": bool(
            unseen_plan["coverage_floor"] == 2
            and unseen_plan["risk_extension_query_count"] == 2
            and unseen_plan["effective_query_budget"] == 4
            and unseen_plan["budget_saved_count"] == 0
        ),
    }

    low_yield_priors = [
        {
            "context": build_counterevidence_cost_context(
                "claim",
                [claim],
            ),
            "sample_count": 10,
            "mean_normalized_cost": 2.0,
            "mean_yield_score": 0.0,
        }
        for claim in high_claims
    ]
    low_yield_plan = (
        plan_adaptive_counterevidence_query_budget(
            list(high_nodes),
            high_nodes,
            high_certificates,
            hard_query_budget=4,
            historical_cost_priors=low_yield_priors,
        )
    )
    low_yield_case = {
        "case": "conservative_low_yield_reduces_extension",
        "observed": low_yield_plan,
        "expectation_met": bool(
            low_yield_plan["risk_extension_query_count"] == 1
            and low_yield_plan["effective_query_budget"] == 3
            and low_yield_plan["budget_saved_count"] == 1
            and all(
                row["yield_source"]
                == "historical_lower_confidence_bound"
                for row in low_yield_plan["claim_plans"]
            )
        ),
    }

    cap_queries, capped = await execute_challenge(
        high_claims,
        [0.9, 0.9],
        1,
    )
    hard_cap_case = {
        "case": "hard_budget_remains_nonnegotiable",
        "observed": {
            "query_count": capped["query_count"],
            "effective_budget": capped[
                "effective_global_query_budget"
            ],
            "covered": capped["covered_risky_claim_count"],
            "uncovered": capped["uncovered_risky_claim_count"],
            "queries": cap_queries,
        },
        "expectation_met": bool(
            capped["query_count"] == 1
            and capped["effective_global_query_budget"] == 1
            and capped["covered_risky_claim_count"] == 1
            and capped["uncovered_risky_claim_count"] == 1
        ),
        "challenge": capped,
    }

    replay_rows = [
        row
        for row in load_jsonl(ROOT_DIR / args.input)
        if (row.get("diagnosis_answer_correction") or {}).get(
            "accepted"
        )
    ]
    replay_row = replay_rows[0]
    constraints = build_correction_persistence_constraints(
        replay_row["diagnosis_answer_correction"]
    )
    correction_source_id = constraints[0]["source_ids"][0]
    committed_source = load_committed_source(
        replay_row,
        correction_source_id,
    )
    question = "How did Adebanwi move from journalism into academia?"
    fallback = (
        "Adebanwi worked as a freelance journalist before academia."
    )
    unsafe = (
        "Adebanwi worked as a television journalist during a Mars "
        "expedition."
    )
    marginal = (
        "Newspapers and magazines accompanied zephyr quuxium vortex "
        "lunar cobalt amber prism nebula."
    )

    async def execute_runtime(
        name: str,
        generated: str,
        expected: dict,
    ) -> dict:
        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": generated,
                    "used_source_ids": [correction_source_id],
                    "insufficient": False,
                }
            )

        async def empty_search(_query, _max_results):
            return []

        final_answer, result = await realize_answer(
            llm_func,
            question,
            fallback,
            {"entities": [], "relationships": []},
            [
                {
                    "source_id": correction_source_id,
                    "document_id": correction_source_id,
                    "text": committed_source,
                    "score": 0.9,
                }
            ],
            prohibited_assertions=constraints,
            counterevidence_search_func=empty_search,
            enable_counterevidence_value_control=True,
            enable_joint_counterevidence_portfolio=True,
            enable_fused_counterevidence_queries=True,
            enable_residual_counterevidence_decomposition=True,
            enable_observed_counterevidence_cost_control=True,
            counterevidence_cost_priors=[],
            enable_counterevidence_risk_coverage_guard=True,
            enable_adaptive_counterevidence_query_budget=True,
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "controller": transaction["controller"],
            "stability_pruned": transaction[
                "stability_pruned_claim_count"
            ],
        }
        return {
            "case": name,
            "expected": expected,
            "observed": observed,
            "expectation_met": observed == expected,
            "final_answer": final_answer,
            "result": result,
        }

    runtime_cases = [
        await execute_runtime(
            "real_source_marginal_claim_preserves_transaction",
            f"{unsafe} {marginal}",
            {
                "triggered": True,
                "transaction_committed": True,
                "overall_committed": True,
                "controller": (
                    "adaptive_budget_transactional_rollback_v17"
                ),
                "stability_pruned": 1,
            },
        ),
        await execute_runtime(
            "certified_generation_needs_no_transaction",
            (
                "Adebanwi later joined the University of Ibadan as a "
                f"lecturer and researcher. {fallback}"
            ),
            {
                "triggered": False,
                "transaction_committed": False,
                "overall_committed": True,
                "controller": (
                    "adaptive_budget_transactional_rollback_v17"
                ),
                "stability_pruned": 0,
            },
        ),
    ]
    cases = [
        medium_case,
        high_risk_case,
        low_yield_case,
        hard_cap_case,
        *runtime_cases,
    ]
    return {
        "agent_version": "vsr-pathrag-v39",
        "protocol": "adaptive_risk_yield_query_budget",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "medium_effective_budget": medium[
            "effective_global_query_budget"
        ],
        "medium_saved_budget": medium[
            "adaptive_query_budget_saved_count"
        ],
        "high_risk_effective_budget": unseen_plan[
            "effective_query_budget"
        ],
        "low_yield_effective_budget": low_yield_plan[
            "effective_query_budget"
        ],
        "hard_cap_query_count": capped["query_count"],
        "reference_answer_used": False,
        "generator_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "results/"
            "vsr_pathrag_v19_diagnosis_correction_local_index.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "vsr_pathrag_v39_adaptive_budget_audit.json"
        ),
    )
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "cases={case_count} met={expectation_met_count} "
        "medium_budget={medium_effective_budget} "
        "saved={medium_saved_budget} "
        "high_budget={high_risk_effective_budget} "
        "low_yield_budget={low_yield_effective_budget} "
        "hard_cap_queries={hard_cap_query_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
