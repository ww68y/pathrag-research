import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    build_correction_persistence_constraints,
    build_counterevidence_cost_context,
    realize_answer,
    run_active_counterevidence_challenge,
)
from api.features.chats.counterevidence_cost_memory import (
    append_counterevidence_cost_updates,
    load_counterevidence_cost_memory,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
    load_committed_source,
    load_jsonl,
)


async def execute_prior_case(
    name: str,
    prior: dict,
    expected_first_term: str,
    expected_applications: int,
    expected_reprioritizations: int,
    audit_validator,
) -> dict:
    claims = [
        "The campaign always drew wider attendance.",
        "The program caused 20 percent higher revenue.",
    ]
    risks = [
        {
            "risk_level": "medium",
            "risk_score": 0.46,
        },
        {
            "risk_level": "high",
            "risk_score": 0.9,
        },
    ]
    query_log = []

    async def local_search(query, _max_results):
        query_log.append(query)
        return {
            "rows": [],
            "latency_ms": 10,
            "usage": {
                "call_count": 0,
                "total_tokens": 0,
            },
            "retrieval_adapter": "local_text_chunk_scan",
            "usage_status": "complete_local_retrieval",
        }

    _answer, challenge = (
        await run_active_counterevidence_challenge(
            " ".join(claims),
            build_transaction(claims, risks),
            local_search,
            enable_value_control=True,
            enable_joint_portfolio=True,
            enable_observed_cost_control=True,
            historical_cost_priors=[prior],
            max_queries_per_claim=3,
            max_global_queries=1,
        )
    )
    selection = challenge["observed_selection_audit"][0]
    observed = {
        "first_query": query_log[0],
        "prior_applications": challenge[
            "historical_cost_prior_task_application_count"
        ],
        "reprioritizations": challenge[
            "observed_cost_reprioritization_count"
        ],
        "proxy_cost_upper_bound": selection[
            "proxy_historical_prior_cost_upper_bound"
        ],
        "proxy_yield_lower_bound": selection[
            "proxy_historical_prior_yield_lower_bound"
        ],
        "proxy_effective_samples": selection[
            "proxy_historical_prior_effective_samples"
        ],
    }
    return {
        "case": name,
        "observed": observed,
        "expectation_met": bool(
            expected_first_term.lower()
            in observed["first_query"].lower()
            and observed["prior_applications"]
            == expected_applications
            and observed["reprioritizations"]
            == expected_reprioritizations
            and audit_validator(selection)
        ),
        "challenge": challenge,
    }


async def main_async(args) -> dict:
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
    priority_claim = (
        "The program caused 20 percent higher revenue."
    )
    context = build_counterevidence_cost_context(
        "claim",
        [priority_claim],
    )
    uncertain_prior = {
        "context": context,
        "sample_count": 10,
        "mean_normalized_cost": 5.0,
        "mean_yield_score": 0.8,
        "cost_m2": 90.0,
        "yield_m2": 0.9,
        "updated_at": int(time.time()),
    }
    stale_prior = {
        **uncertain_prior,
        "updated_at": int(time.time()) - 365 * 86400,
    }
    scheduler_cases = [
        await execute_prior_case(
            "uncertain_prior_uses_conservative_bounds",
            uncertain_prior,
            "campaign",
            1,
            1,
            lambda row: bool(
                row["proxy_historical_prior_cost_upper_bound"]
                > uncertain_prior["mean_normalized_cost"]
                and row[
                    "proxy_historical_prior_yield_lower_bound"
                ]
                < uncertain_prior["mean_yield_score"]
            ),
        ),
        await execute_prior_case(
            "stale_prior_is_rejected_after_recency_decay",
            stale_prior,
            "20 percent",
            0,
            0,
            lambda row: bool(
                row["proxy_selected_cost_source"]
                == "query_length_proxy"
                and row[
                    "proxy_historical_prior_effective_samples"
                ]
                == 0.0
            ),
        ),
    ]

    single_claim = "The program caused higher sales."
    low_sample_prior = {
        "context": build_counterevidence_cost_context(
            "claim",
            [single_claim],
        ),
        "sample_count": 1,
        "mean_normalized_cost": 10.0,
        "mean_yield_score": 0.0,
        "updated_at": int(time.time()),
    }

    async def empty_local_search(_query, _max_results):
        return {
            "rows": [],
            "latency_ms": 10,
            "usage": {
                "call_count": 0,
                "total_tokens": 0,
            },
        }

    _answer, low_sample_challenge = (
        await run_active_counterevidence_challenge(
            single_claim,
            build_transaction(
                [single_claim],
                [
                    {
                        "risk_level": "medium",
                        "risk_score": 0.46,
                    }
                ],
            ),
            empty_local_search,
            max_claims=1,
            enable_value_control=True,
            enable_joint_portfolio=True,
            enable_observed_cost_control=True,
            historical_cost_priors=[low_sample_prior],
            max_queries_per_claim=3,
            max_global_queries=1,
        )
    )
    low_sample_selection = (
        low_sample_challenge["observed_selection_audit"][0]
    )
    low_sample_case = {
        "case": "low_sample_prior_has_bounded_transfer_weight",
        "observed": {
            "cost_source": low_sample_selection[
                "selected_cost_source"
            ],
            "transfer_weight": low_sample_selection[
                "historical_prior_transfer_weight"
            ],
            "effective_samples": low_sample_selection[
                "historical_prior_effective_samples"
            ],
        },
        "expectation_met": bool(
            low_sample_selection["selected_cost_source"]
            == "historical_similarity_prior"
            and 0.0
            < low_sample_selection[
                "historical_prior_transfer_weight"
            ]
            <= 0.334
            and low_sample_selection[
                "historical_prior_effective_samples"
            ]
            == 1.0
        ),
        "challenge": low_sample_challenge,
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "robust-cost-memory.jsonl"
        append_counterevidence_cost_updates(
            [
                {
                    "context": context,
                    "sample_count": 4,
                    "mean_normalized_cost": 2.0,
                    "mean_yield_score": 0.7,
                    "cost_m2": 0.12,
                    "yield_m2": 0.01,
                }
            ],
            path,
        )
        outlier_feedback = append_counterevidence_cost_updates(
            [
                {
                    "context": context,
                    "sample_count": 1,
                    "mean_normalized_cost": 100.0,
                    "mean_yield_score": 0.0,
                }
            ],
            path,
        )
        robust_row = load_counterevidence_cost_memory(path)[0]
    outlier_case = {
        "case": "extreme_cost_batch_is_clipped_before_merge",
        "observed": {
            "clipped_updates": outlier_feedback[
                "clipped_update_count"
            ],
            "sample_count": robust_row["sample_count"],
            "mean_cost": robust_row["mean_normalized_cost"],
            "cost_variance": robust_row["cost_variance"],
        },
        "expectation_met": bool(
            outlier_feedback["clipped_update_count"] == 1
            and robust_row["sample_count"] == 5
            and robust_row["mean_normalized_cost"] < 3.0
            and robust_row["cost_variance"] > 0.0
        ),
    }

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
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "cost_updates": len(
                transaction[
                    "counterevidence_cost_memory_updates"
                ]
            ),
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
            "real_source_marginal_claim_pruned_before_robust_memory",
            f"{unsafe} {marginal}",
            {
                "triggered": True,
                "transaction_committed": True,
                "overall_committed": True,
                "cost_updates": 0,
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
                "cost_updates": 0,
                "stability_pruned": 0,
            },
        ),
    ]
    cases = (
        scheduler_cases
        + [low_sample_case, outlier_case]
        + runtime_cases
    )
    result = {
        "agent_version": "vsr-pathrag-v37",
        "protocol": "controlled_robust_cost_memory",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "uncertain_prior_application_count": scheduler_cases[0][
            "observed"
        ]["prior_applications"],
        "stale_prior_application_count": scheduler_cases[1][
            "observed"
        ]["prior_applications"],
        "robust_reprioritization_count": scheduler_cases[0][
            "observed"
        ]["reprioritizations"],
        "low_sample_transfer_weight": low_sample_case[
            "observed"
        ]["transfer_weight"],
        "clipped_outlier_update_count": outlier_case[
            "observed"
        ]["clipped_updates"],
        "reference_answer_used": False,
        "generator_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    return result


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
            "vsr_pathrag_v37_robust_cost_memory_audit.json"
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
        "uncertain_applied={uncertain_prior_application_count} "
        "stale_applied={stale_prior_application_count} "
        "reprioritized={robust_reprioritization_count} "
        "low_sample_weight={low_sample_transfer_weight} "
        "clipped={clipped_outlier_update_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
