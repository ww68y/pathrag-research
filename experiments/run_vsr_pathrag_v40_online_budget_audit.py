import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    build_correction_persistence_constraints,
    realize_answer,
    run_active_counterevidence_challenge,
    update_online_counterevidence_query_budget,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
    load_committed_source,
    load_jsonl,
)


async def execute_online_case(
    claims: list[str],
    risk_scores: list[float],
    relevant: bool,
) -> dict:
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
    queries = []

    async def local_search(query, _max_results):
        queries.append(query)
        if relevant:
            claim = next(
                claim for claim in claims if claim.rstrip(".") in query
            )
            return [
                {
                    "source_id": f"relevant-{len(queries)}",
                    "document_id": f"document-{len(queries)}",
                    "text": claim,
                    "source_reliability": 0.5,
                }
            ]
        return [
            {
                "source_id": f"irrelevant-{len(queries)}",
                "document_id": "weather-record",
                "text": "A weather archive recorded rainfall.",
                "source_reliability": 0.9,
            }
        ]

    _answer, challenge = await run_active_counterevidence_challenge(
        " ".join(claims),
        transaction,
        local_search,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_risk_coverage_guard=True,
        enable_adaptive_query_budget=True,
        enable_online_budget_reallocation=True,
        max_queries_per_claim=3,
        max_global_queries=4,
    )
    return {
        "queries": queries,
        "challenge": challenge,
    }


async def main_async(args) -> dict:
    released = update_online_counterevidence_query_budget(
        current_query_budget=2,
        hard_query_budget=4,
        executed_query_count=1,
        unresolved_unqueried_count=1,
        realized_information_value=0.8,
        selected_task_remains_unresolved=True,
    )
    release_case = {
        "case": "high_realized_value_releases_extension",
        "observed": released,
        "expectation_met": bool(
            released["action"] == "release"
            and released["updated_query_budget"] == 3
            and released["coverage_required_budget"] == 2
        ),
    }

    floor_held = update_online_counterevidence_query_budget(
        current_query_budget=2,
        hard_query_budget=4,
        executed_query_count=1,
        unresolved_unqueried_count=1,
        realized_information_value=0.1,
        selected_task_remains_unresolved=True,
    )
    floor_case = {
        "case": "low_value_cannot_revoke_coverage_floor",
        "observed": floor_held,
        "expectation_met": bool(
            floor_held["action"] == "hold"
            and floor_held["updated_query_budget"] == 2
            and floor_held["coverage_required_budget"] == 2
        ),
    }

    relevant_run = await execute_online_case(
        [
            "The program caused higher sales.",
            "The program caused greater revenue.",
        ],
        [0.46, 0.46],
        relevant=True,
    )
    relevant = relevant_run["challenge"]
    expansion_case = {
        "case": "relevant_unresolved_routes_expand_live_budget",
        "observed": {
            "initial_budget": relevant[
                "initial_effective_global_query_budget"
            ],
            "final_budget": relevant[
                "effective_global_query_budget"
            ],
            "releases": relevant[
                "online_budget_release_count"
            ],
            "revocations": relevant[
                "online_budget_revocation_count"
            ],
            "covered": relevant["covered_risky_claim_count"],
        },
        "expectation_met": bool(
            relevant["initial_effective_global_query_budget"] == 2
            and relevant["effective_global_query_budget"] > 2
            and relevant["online_budget_release_count"] >= 1
            and relevant["covered_risky_claim_count"] == 2
        ),
        "challenge": relevant,
    }

    low_value_run = await execute_online_case(
        [
            "The campaign always drew wider attendance.",
            "The program caused 20 percent higher revenue.",
        ],
        [0.9, 0.9],
        relevant=False,
    )
    low_value = low_value_run["challenge"]
    revocation_case = {
        "case": "low_value_extensions_are_revoked_after_coverage",
        "observed": {
            "initial_budget": low_value[
                "initial_effective_global_query_budget"
            ],
            "final_budget": low_value[
                "effective_global_query_budget"
            ],
            "releases": low_value[
                "online_budget_release_count"
            ],
            "revocations": low_value[
                "online_budget_revocation_count"
            ],
            "saved": low_value["online_budget_saved_count"],
            "covered": low_value["covered_risky_claim_count"],
        },
        "expectation_met": bool(
            low_value["initial_effective_global_query_budget"] == 4
            and low_value["effective_global_query_budget"] == 2
            and low_value["online_budget_revocation_count"] == 2
            and low_value["online_budget_saved_count"] == 2
            and low_value["covered_risky_claim_count"] == 2
        ),
        "challenge": low_value,
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
            enable_online_counterevidence_budget_reallocation=True,
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
                    "online_budget_transactional_rollback_v18"
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
                    "online_budget_transactional_rollback_v18"
                ),
                "stability_pruned": 0,
            },
        ),
    ]
    cases = [
        release_case,
        floor_case,
        expansion_case,
        revocation_case,
        *runtime_cases,
    ]
    return {
        "agent_version": "vsr-pathrag-v40",
        "protocol": "online_information_budget_reallocation",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "relevant_initial_budget": relevant[
            "initial_effective_global_query_budget"
        ],
        "relevant_final_budget": relevant[
            "effective_global_query_budget"
        ],
        "online_release_count": relevant[
            "online_budget_release_count"
        ],
        "low_value_initial_budget": low_value[
            "initial_effective_global_query_budget"
        ],
        "low_value_final_budget": low_value[
            "effective_global_query_budget"
        ],
        "online_revocation_count": low_value[
            "online_budget_revocation_count"
        ],
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
            "vsr_pathrag_v40_online_budget_audit.json"
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
        "relevant_budget={relevant_initial_budget}->"
        "{relevant_final_budget} releases={online_release_count} "
        "low_value_budget={low_value_initial_budget}->"
        "{low_value_final_budget} "
        "revocations={online_revocation_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
