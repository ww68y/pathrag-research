import argparse
import asyncio
import json
import sys
import tempfile
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


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_committed_source(row: dict, source_id: str) -> str:
    store = json.loads(
        (
            ROOT_DIR
            / row["working_dir"]
            / "kv_store_text_chunks.json"
        ).read_text(encoding="utf-8")
    )
    return str((store.get(source_id) or {}).get("content", ""))


def build_transaction(
    claims: list[str],
    risks: list[dict],
) -> dict:
    nodes = [
        {
            "node_id": f"claim-{index}",
            "claim_index": index - 1,
            "text": claim,
            "evidence_support": [
                {
                    "evidence_id": f"S{index}",
                    "family_ids": [
                        f"support-document-{index}"
                    ],
                }
            ],
        }
        for index, claim in enumerate(claims, start=1)
    ]
    node_ids = [row["node_id"] for row in nodes]
    return {
        "dependency_graph": {
            "nodes": nodes,
            "edges": [],
            "kept_node_ids": node_ids,
        },
        "claim_bundle_selection": {
            "final_selected_node_ids": node_ids,
        },
        "counterfactual_stability_certificate": {
            "semantic_family_quorum_node_ids": node_ids,
            "final_retained_node_ids": node_ids,
            "provenance_certificates": [
                {
                    "node_id": node_id,
                    "claim_relation_risk": risks[index - 1],
                    "support_families": [
                        {
                            "family_ids": [
                                f"support-document-{index}"
                            ]
                        }
                    ],
                }
                for index, node_id in enumerate(
                    node_ids,
                    start=1,
                )
            ],
        },
        "rollback_source_ids": [],
        "actions": [],
    }


async def execute_prior_case(
    name: str,
    prior_claim: str,
    expected_first_term: str,
    expected_applications: int,
    expected_reprioritizations: int,
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
    prior = {
        "context": build_counterevidence_cost_context(
            "claim",
            [prior_claim],
        ),
        "sample_count": 10,
        "mean_normalized_cost": 20.0,
        "mean_yield_score": 0.0,
    }
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
    observed = {
        "first_query": query_log[0],
        "prior_candidates": challenge[
            "historical_cost_prior_candidate_count"
        ],
        "prior_applications": challenge[
            "historical_cost_prior_task_application_count"
        ],
        "reprioritizations": challenge[
            "observed_cost_reprioritization_count"
        ],
        "memory_updates": len(
            challenge["counterevidence_cost_memory_updates"]
        ),
    }
    return {
        "case": name,
        "observed": observed,
        "expectation_met": bool(
            expected_first_term.lower()
            in observed["first_query"].lower()
            and observed["prior_candidates"] == 1
            and observed["prior_applications"]
            == expected_applications
            and observed["reprioritizations"]
            == expected_reprioritizations
            and observed["memory_updates"] == 1
            and prior_claim.lower()
            not in json.dumps(
                challenge[
                    "counterevidence_cost_memory_updates"
                ]
            ).lower()
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
    scheduler_cases = [
        await execute_prior_case(
            "similar_expensive_prior_changes_initial_priority",
            priority_claim,
            "campaign",
            1,
            1,
        ),
        await execute_prior_case(
            "unrelated_prior_is_rejected",
            "The orchard harvest increased apple exports.",
            "20 percent",
            0,
            0,
        ),
    ]

    local_claim = "The program caused higher sales."

    async def local_zero_usage_search(_query, _max_results):
        return {
            "rows": [
                {
                    "source_id": "local-confirmation",
                    "document_id": "local-document",
                    "text": local_claim,
                    "source_reliability": 0.9,
                }
            ],
            "latency_ms": 25,
            "usage": {
                "call_count": 0,
                "total_tokens": 0,
            },
            "retrieval_adapter": "local_text_chunk_scan",
            "usage_status": "complete_local_retrieval",
        }

    _answer, local_challenge = (
        await run_active_counterevidence_challenge(
            local_claim,
            build_transaction(
                [local_claim],
                [
                    {
                        "risk_level": "medium",
                        "risk_score": 0.46,
                    }
                ],
            ),
            local_zero_usage_search,
            max_claims=1,
            enable_value_control=True,
            enable_joint_portfolio=True,
            enable_observed_cost_control=True,
            historical_cost_priors=[],
            max_queries_per_claim=3,
        )
    )
    local_row = local_challenge["observed_cost_rows"][0]
    local_case = {
        "case": "local_retrieval_propagates_zero_provider_usage",
        "observed": {
            "calls": local_challenge[
                "observed_provider_call_count"
            ],
            "tokens": local_challenge[
                "observed_provider_token_count"
            ],
            "adapter": local_row["retrieval_adapter"],
            "usage_status": local_row["usage_status"],
            "passed": len(local_challenge["passed_node_ids"]),
        },
        "expectation_met": bool(
            local_challenge["observed_provider_call_count"] == 0
            and local_challenge[
                "observed_provider_token_count"
            ]
            == 0
            and local_row["retrieval_adapter"]
            == "local_text_chunk_scan"
            and local_row["usage_status"]
            == "complete_local_retrieval"
            and local_challenge["passed_node_ids"] == ["claim-1"]
        ),
        "challenge": local_challenge,
    }

    context = build_counterevidence_cost_context(
        "claim",
        [priority_claim],
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cost-memory.jsonl"
        first_write = append_counterevidence_cost_updates(
            [
                {
                    "context": context,
                    "sample_count": 1,
                    "mean_normalized_cost": 2.0,
                    "mean_yield_score": 0.5,
                }
            ],
            path,
        )
        second_write = append_counterevidence_cost_updates(
            [
                {
                    "context": context,
                    "sample_count": 2,
                    "mean_normalized_cost": 5.0,
                    "mean_yield_score": 0.8,
                }
            ],
            path,
        )
        persisted = load_counterevidence_cost_memory(path)
    persistence_case = {
        "case": "cost_memory_persists_weighted_hashed_profile",
        "observed": {
            "first_written": first_write["written"],
            "second_written": second_write["written"],
            "memory_size": len(persisted),
            "sample_count": persisted[0]["sample_count"],
            "mean_cost": persisted[0]["mean_normalized_cost"],
            "mean_yield": persisted[0]["mean_yield_score"],
        },
        "expectation_met": bool(
            len(persisted) == 1
            and persisted[0]["sample_count"] == 3
            and persisted[0]["mean_normalized_cost"] == 4.0
            and persisted[0]["mean_yield_score"] == 0.7
            and "program" not in json.dumps(persisted).lower()
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
            "real_source_marginal_claim_pruned_before_memory_stage",
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
        + [local_case, persistence_case]
        + runtime_cases
    )
    result = {
        "agent_version": "vsr-pathrag-v36",
        "protocol": "controlled_similarity_gated_cost_memory",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "similar_prior_application_count": scheduler_cases[0][
            "observed"
        ]["prior_applications"],
        "unrelated_prior_application_count": scheduler_cases[1][
            "observed"
        ]["prior_applications"],
        "cost_memory_reprioritization_count": scheduler_cases[0][
            "observed"
        ]["reprioritizations"],
        "local_provider_call_count": local_case["observed"][
            "calls"
        ],
        "local_provider_token_count": local_case["observed"][
            "tokens"
        ],
        "persisted_profile_sample_count": persistence_case[
            "observed"
        ]["sample_count"],
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
            "vsr_pathrag_v36_cost_memory_audit.json"
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
        "similar_applied={similar_prior_application_count} "
        "unrelated_applied={unrelated_prior_application_count} "
        "reprioritized={cost_memory_reprioritization_count} "
        "local_calls={local_provider_call_count} "
        "local_tokens={local_provider_token_count} "
        "persisted_samples={persisted_profile_sample_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
