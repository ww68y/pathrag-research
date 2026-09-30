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


async def run_scheduler_case(
    name: str,
    claims: list[str],
    risks: list[dict],
    search_func,
    max_global_queries: int,
    max_claims: int,
    validator,
    enable_fusion: bool = False,
    enable_residual: bool = False,
) -> dict:
    query_log = []

    async def logged_search(query, max_results):
        query_log.append(query)
        return await search_func(query, max_results)

    answer, challenge = await run_active_counterevidence_challenge(
        "Safe fallback. " + " ".join(claims),
        build_transaction(claims, risks),
        logged_search,
        max_claims=max_claims,
        max_queries_per_claim=3,
        max_global_queries=max_global_queries,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_query_fusion=enable_fusion,
        enable_residual_decomposition=enable_residual,
        enable_observed_cost_control=True,
    )
    observed = {
        "queries": challenge["query_count"],
        "saved_queries": challenge["saved_query_count"],
        "profile_updates": challenge[
            "observed_cost_profile_update_count"
        ],
        "reprioritizations": challenge[
            "observed_cost_reprioritization_count"
        ],
        "provider_calls": challenge[
            "observed_provider_call_count"
        ],
        "provider_tokens": challenge[
            "observed_provider_token_count"
        ],
        "unique_evidence": challenge[
            "observed_unique_evidence_count"
        ],
        "information_yield": challenge[
            "observed_information_yield"
        ],
        "residual_queries": challenge["residual_query_count"],
        "passed": len(challenge["passed_node_ids"]),
        "pruned": len(challenge["pruned_node_ids"]),
    }
    return {
        "case": name,
        "observed": observed,
        "expectation_met": bool(
            validator(observed, challenge, query_log)
        ),
        "query_log": query_log,
        "final_answer": answer,
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
    medium_risk = {
        "risk_level": "medium",
        "risk_score": 0.46,
    }
    low_value_medium_risk = {
        "risk_level": "medium",
        "risk_score": 0.0,
    }
    high_risk = {
        "risk_level": "high",
        "risk_score": 0.9,
    }

    priority_claims = [
        "The campaign always drew wider attendance.",
        "The program caused 20 percent higher revenue.",
    ]

    async def expensive_search(query, _max_results):
        if "20 percent" in query:
            return {
                "rows": [
                    {
                        "source_id": "expensive-irrelevant",
                        "document_id": "weather-document",
                        "text": "A weather archive recorded rainfall.",
                        "source_reliability": 0.9,
                    }
                ],
                "latency_ms": 8000,
                "usage": {
                    "call_count": 2,
                    "total_tokens": 4000,
                },
            }
        return {
            "rows": [],
            "latency_ms": 100,
            "usage": {
                "call_count": 1,
                "total_tokens": 100,
            },
        }

    def validate_reprioritization(observed, _challenge, queries):
        return (
            observed
            == {
                "queries": 2,
                "saved_queries": 4,
                "profile_updates": 2,
                "reprioritizations": 1,
                "provider_calls": 3,
                "provider_tokens": 4100,
                "unique_evidence": 1,
                "information_yield": 1.0,
                "residual_queries": 0,
                "passed": 0,
                "pruned": 2,
            }
            and "20 percent" in queries[0]
            and "campaign" in queries[1].lower()
        )

    single_claim = ["The program caused higher sales."]

    async def productive_structured_search(_query, _max_results):
        return {
            "rows": [
                {
                    "source_id": "productive-confirmation",
                    "document_id": "independent-document",
                    "text": single_claim[0],
                    "source_reliability": 0.9,
                }
            ],
            "latency_ms": 500,
            "usage": {
                "call_count": 2,
                "prompt_tokens": 800,
                "completion_tokens": 400,
            },
        }

    def validate_structured(observed, challenge, _queries):
        row = challenge["observed_cost_rows"][0]
        return (
            observed
            == {
                "queries": 1,
                "saved_queries": 2,
                "profile_updates": 1,
                "reprioritizations": 0,
                "provider_calls": 2,
                "provider_tokens": 1200,
                "unique_evidence": 1,
                "information_yield": 2.0,
                "residual_queries": 0,
                "passed": 1,
                "pruned": 0,
            }
            and row["metadata_source"] == "structured_response"
            and row["latency_ms"] == 500.0
        )

    async def productive_legacy_search(_query, _max_results):
        return [
            {
                "source_id": "legacy-confirmation",
                "document_id": "legacy-document",
                "text": single_claim[0],
                "source_reliability": 0.9,
            }
        ]

    def validate_legacy(observed, challenge, _queries):
        row = challenge["observed_cost_rows"][0]
        return (
            observed["queries"] == 1
            and observed["profile_updates"] == 1
            and observed["provider_calls"] == 1
            and observed["provider_tokens"] == 0
            and observed["passed"] == 1
            and row["metadata_source"]
            == "measured_legacy_response"
            and row["measured_latency_ms"] >= 0.0
        )

    related_claims = [
        "The program caused higher sales.",
        "The program caused greater revenue.",
    ]

    async def observed_residual_search(query, _max_results):
        if "previous multi-claim search" in query.lower():
            return {
                "rows": [
                    {
                        "source_id": "residual-revenue",
                        "document_id": "revenue-document",
                        "text": related_claims[1],
                        "source_reliability": 0.9,
                    }
                ],
                "latency_ms": 250,
                "usage": {
                    "call_count": 1,
                    "total_tokens": 500,
                },
            }
        return {
            "rows": [
                {
                    "source_id": "partial-sales",
                    "document_id": "sales-document",
                    "text": related_claims[0],
                    "source_reliability": 0.9,
                }
            ],
            "latency_ms": 400,
            "usage": {
                "call_count": 1,
                "total_tokens": 1000,
            },
        }

    def validate_residual(observed, challenge, _queries):
        return (
            observed["queries"] == 2
            and observed["profile_updates"] == 2
            and observed["provider_calls"] == 2
            and observed["provider_tokens"] == 1500
            and observed["residual_queries"] == 1
            and observed["passed"] == 2
            and challenge["partial_coverage_cluster_count"] == 1
        )

    scheduler_cases = [
        await run_scheduler_case(
            "observed_cost_reprioritizes_expensive_high_risk_claim",
            priority_claims,
            [low_value_medium_risk, high_risk],
            expensive_search,
            2,
            2,
            validate_reprioritization,
        ),
        await run_scheduler_case(
            "structured_usage_metadata_is_accounted",
            single_claim,
            [medium_risk],
            productive_structured_search,
            2,
            1,
            validate_structured,
        ),
        await run_scheduler_case(
            "legacy_list_response_remains_compatible",
            single_claim,
            [medium_risk],
            productive_legacy_search,
            2,
            1,
            validate_legacy,
        ),
        await run_scheduler_case(
            "observed_cost_integrates_with_residual_query",
            related_claims,
            [medium_risk, medium_risk],
            observed_residual_search,
            4,
            2,
            validate_residual,
            enable_fusion=True,
            enable_residual=True,
        ),
    ]

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
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "observed_queries": transaction[
                "counterevidence_observed_cost_profile_update_count"
            ],
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
            "real_source_marginal_claim_pruned_before_cost_stage",
            f"{unsafe} {marginal}",
            {
                "triggered": True,
                "transaction_committed": True,
                "overall_committed": True,
                "observed_queries": 0,
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
                "observed_queries": 0,
                "stability_pruned": 0,
            },
        ),
    ]
    cases = scheduler_cases + runtime_cases
    result = {
        "agent_version": "vsr-pathrag-v35",
        "protocol": "controlled_observed_cost_counterevidence",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "scheduler_query_count": sum(
            int(row["observed"].get("queries", 0))
            for row in scheduler_cases
        ),
        "scheduler_saved_query_count": sum(
            int(row["observed"].get("saved_queries", 0))
            for row in scheduler_cases
        ),
        "cost_profile_update_count": sum(
            int(row["observed"].get("profile_updates", 0))
            for row in scheduler_cases
        ),
        "cost_reprioritization_count": sum(
            int(row["observed"].get("reprioritizations", 0))
            for row in scheduler_cases
        ),
        "provider_call_count": sum(
            int(row["observed"].get("provider_calls", 0))
            for row in scheduler_cases
        ),
        "provider_token_count": sum(
            int(row["observed"].get("provider_tokens", 0))
            for row in scheduler_cases
        ),
        "unique_evidence_count": sum(
            int(row["observed"].get("unique_evidence", 0))
            for row in scheduler_cases
        ),
        "information_yield": round(
            sum(
                float(
                    row["observed"].get(
                        "information_yield",
                        0.0,
                    )
                )
                for row in scheduler_cases
            ),
            3,
        ),
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
            "vsr_pathrag_v35_observed_cost_audit.json"
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
        "queries={scheduler_query_count} "
        "saved={scheduler_saved_query_count} "
        "profiles={cost_profile_update_count} "
        "reprioritized={cost_reprioritization_count} "
        "calls={provider_call_count} "
        "tokens={provider_token_count} "
        "unique={unique_evidence_count} "
        "yield={information_yield} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
