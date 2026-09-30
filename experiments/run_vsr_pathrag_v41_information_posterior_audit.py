import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    apply_counterevidence_information_profile,
    build_correction_persistence_constraints,
    realize_answer,
    run_active_counterevidence_challenge,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
    load_committed_source,
    load_jsonl,
)


def profile_task(member_node_ids: list[str]) -> dict:
    return {
        "task_type": "claim",
        "node_id": "claim-1",
        "member_node_ids": member_node_ids,
        "adjusted_expected_value": 1.0,
        "value_per_cost": 1.0,
    }


async def main_async(args) -> dict:
    low_profile = {
        "claim-1": {
            "sample_count": 1,
            "mean_realized_information_value": 0.3,
            "low_value_streak": 1,
        }
    }
    penalized = apply_counterevidence_information_profile(
        profile_task(["claim-1"]),
        low_profile,
        True,
    )
    low_value_case = {
        "case": "repeated_low_value_claim_receives_posterior_penalty",
        "observed": penalized,
        "expectation_met": bool(
            penalized["information_value_multiplier"] < 0.6
            and penalized["low_value_streak"] == 1
            and penalized["information_profile_source"]
            == "claim_information_posterior"
        ),
    }

    fused = apply_counterevidence_information_profile(
        profile_task(["claim-1", "claim-2"]),
        low_profile,
        True,
    )
    fused_case = {
        "case": "unseen_fused_member_attenuates_existing_penalty",
        "observed": fused,
        "expectation_met": bool(
            fused["information_value_multiplier"]
            > penalized["information_value_multiplier"]
            and fused["information_profile_sample_count"] == 1
        ),
    }

    productive = apply_counterevidence_information_profile(
        profile_task(["claim-1"]),
        {
            "claim-1": {
                "sample_count": 2,
                "mean_realized_information_value": 0.85,
                "low_value_streak": 0,
            }
        },
        True,
    )
    productive_case = {
        "case": "productive_claim_retains_most_of_its_value",
        "observed": productive,
        "expectation_met": bool(
            productive["information_value_multiplier"] >= 0.93
            and productive["low_value_streak"] == 0
        ),
    }

    claims = [
        "The campaign always drew wider attendance.",
        "The program caused 20 percent higher revenue.",
    ]
    transaction = build_transaction(
        claims,
        [
            {"risk_level": "medium", "risk_score": 0.0},
            {"risk_level": "high", "risk_score": 0.9},
        ],
    )
    queries = []

    async def irrelevant_search(query, _max_results):
        queries.append(query)
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
        irrelevant_search,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_risk_coverage_guard=True,
        enable_adaptive_query_budget=True,
        enable_online_budget_reallocation=True,
        enable_information_posterior_control=True,
        max_queries_per_claim=3,
        max_global_queries=3,
    )
    second_selection = challenge[
        "information_selection_audit"
    ][1]
    reprioritization_case = {
        "case": "posterior_switches_route_before_coverage_override",
        "observed": {
            "queries": queries,
            "information_reprioritizations": challenge[
                "information_reprioritization_count"
            ],
            "coverage_interventions": challenge[
                "coverage_reserve_intervention_count"
            ],
            "second_selection": second_selection,
            "profiles": challenge["information_profiles"],
        },
        "expectation_met": bool(
            "20 percent" in queries[0]
            and "campaign" in queries[1].lower()
            and challenge["information_reprioritization_count"] >= 1
            and challenge["coverage_reserve_intervention_count"] == 0
            and second_selection["cost_selected_node_id"]
            == "claim-2"
            and second_selection["information_selected_node_id"]
            == "claim-1"
        ),
        "challenge": challenge,
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
            enable_counterevidence_information_posterior_control=True,
        )
        transaction_result = result[
            "transactional_correction_rollback"
        ]
        observed = {
            "triggered": transaction_result["triggered"],
            "transaction_committed": transaction_result["committed"],
            "overall_committed": result["committed"],
            "controller": transaction_result["controller"],
            "stability_pruned": transaction_result[
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
                    "information_posterior_transactional_rollback_v19"
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
                    "information_posterior_transactional_rollback_v19"
                ),
                "stability_pruned": 0,
            },
        ),
    ]
    cases = [
        low_value_case,
        fused_case,
        productive_case,
        reprioritization_case,
        *runtime_cases,
    ]
    return {
        "agent_version": "vsr-pathrag-v41",
        "protocol": "claim_information_posterior_control",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "low_value_multiplier": penalized[
            "information_value_multiplier"
        ],
        "fused_multiplier": fused[
            "information_value_multiplier"
        ],
        "productive_multiplier": productive[
            "information_value_multiplier"
        ],
        "information_reprioritization_count": challenge[
            "information_reprioritization_count"
        ],
        "coverage_intervention_count": challenge[
            "coverage_reserve_intervention_count"
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
            "vsr_pathrag_v41_information_posterior_audit.json"
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
        "low_multiplier={low_value_multiplier} "
        "fused_multiplier={fused_multiplier} "
        "productive_multiplier={productive_multiplier} "
        "reprioritized={information_reprioritization_count} "
        "coverage_interventions={coverage_intervention_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
