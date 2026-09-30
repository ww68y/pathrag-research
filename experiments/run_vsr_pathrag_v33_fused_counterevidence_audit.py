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
                    "node_id": f"claim-{index}",
                    "claim_relation_risk": risks[index - 1],
                    "support_families": [
                        {
                            "family_ids": [
                                f"support-document-{index}"
                            ]
                        }
                    ],
                }
                for index in range(1, len(claims) + 1)
            ],
        },
        "rollback_source_ids": [],
        "actions": [],
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
    high_risk = {
        "risk_level": "high",
        "risk_score": 0.66,
    }

    async def execute_scheduler(
        name: str,
        claims: list[str],
        risks: list[dict],
        search_func,
        expected: dict,
        max_global_queries: int = 4,
    ) -> dict:
        query_log = []

        async def logged_search(query, max_results):
            query_log.append(query)
            return await search_func(query, max_results)

        answer, challenge = (
            await run_active_counterevidence_challenge(
                "Safe fallback. " + " ".join(claims),
                build_transaction(claims, risks),
                logged_search,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_query_fusion=True,
                max_queries_per_claim=3,
                max_global_queries=max_global_queries,
            )
        )
        observed = {
            "queries": challenge["query_count"],
            "saved_queries": challenge["saved_query_count"],
            "fused_queries": challenge["fused_query_count"],
            "fused_clusters": challenge[
                "fused_cluster_count"
            ],
            "fused_claim_coverage": challenge[
                "fused_claim_coverage_count"
            ],
            "generic_reuse_rejected": challenge[
                "generic_reuse_rejected_count"
            ],
            "cross_claim_reuse": challenge[
                "cross_claim_reuse_count"
            ],
            "shared_evidence": challenge[
                "shared_evidence_count"
            ],
            "passed": len(challenge["passed_node_ids"]),
            "contradicted": len(
                challenge["contradicted_node_ids"]
            ),
            "inconclusive": len(
                challenge["inconclusive_node_ids"]
            ),
            "pruned": len(challenge["pruned_node_ids"]),
        }
        return {
            "case": name,
            "expected": expected,
            "observed": observed,
            "expectation_met": observed == expected,
            "query_log": query_log,
            "final_answer": answer,
            "challenge": challenge,
        }

    related_claims = [
        "The program caused higher sales.",
        "The program caused greater revenue.",
    ]

    async def fused_confirmation(_query, _max_results):
        return [
            {
                "source_id": "fused-confirmation",
                "document_id": "independent-document",
                "text": (
                    "The program caused higher sales and greater "
                    "revenue."
                ),
                "source_reliability": 0.9,
            }
        ]

    async def fused_contradiction(_query, _max_results):
        return [
            {
                "source_id": "fused-contradiction",
                "document_id": "independent-document",
                "text": (
                    "The program did not cause higher sales or "
                    "greater revenue."
                ),
                "source_reliability": 0.9,
            }
        ]

    async def generic_result(_query, _max_results):
        return [
            {
                "source_id": "generic-result",
                "document_id": "independent-document",
                "text": "The program caused results.",
                "source_reliability": 0.9,
            }
        ]

    unrelated_claims = [
        "The campaign always drew wider attendance.",
        "The program caused 20 percent higher revenue.",
    ]

    async def targeted_confirmation(query, _max_results):
        if "20 percent" in query:
            text = unrelated_claims[1]
            source_id = "high-risk-confirmation"
        else:
            text = unrelated_claims[0]
            source_id = "medium-risk-confirmation"
        return [
            {
                "source_id": source_id,
                "document_id": f"{source_id}-document",
                "text": text,
                "source_reliability": 0.9,
            }
        ]

    scheduler_cases = [
        await execute_scheduler(
            "fused_confirmation_covers_two_claims",
            related_claims,
            [medium_risk, medium_risk],
            fused_confirmation,
            {
                "queries": 1,
                "saved_queries": 5,
                "fused_queries": 1,
                "fused_clusters": 1,
                "fused_claim_coverage": 2,
                "generic_reuse_rejected": 0,
                "cross_claim_reuse": 1,
                "shared_evidence": 1,
                "passed": 2,
                "contradicted": 0,
                "inconclusive": 0,
                "pruned": 0,
            },
        ),
        await execute_scheduler(
            "fused_contradiction_revokes_two_claims",
            related_claims,
            [medium_risk, medium_risk],
            fused_contradiction,
            {
                "queries": 1,
                "saved_queries": 5,
                "fused_queries": 1,
                "fused_clusters": 1,
                "fused_claim_coverage": 2,
                "generic_reuse_rejected": 0,
                "cross_claim_reuse": 1,
                "shared_evidence": 1,
                "passed": 0,
                "contradicted": 2,
                "inconclusive": 0,
                "pruned": 2,
            },
        ),
        await execute_scheduler(
            "generic_shared_predicate_is_not_reused",
            related_claims,
            [medium_risk, medium_risk],
            generic_result,
            {
                "queries": 1,
                "saved_queries": 5,
                "fused_queries": 1,
                "fused_clusters": 1,
                "fused_claim_coverage": 0,
                "generic_reuse_rejected": 2,
                "cross_claim_reuse": 0,
                "shared_evidence": 0,
                "passed": 0,
                "contradicted": 0,
                "inconclusive": 2,
                "pruned": 2,
            },
            max_global_queries=1,
        ),
        await execute_scheduler(
            "unrelated_claims_remain_separate",
            unrelated_claims,
            [medium_risk, high_risk],
            targeted_confirmation,
            {
                "queries": 2,
                "saved_queries": 4,
                "fused_queries": 0,
                "fused_clusters": 0,
                "fused_claim_coverage": 0,
                "generic_reuse_rejected": 0,
                "cross_claim_reuse": 0,
                "shared_evidence": 0,
                "passed": 2,
                "contradicted": 0,
                "inconclusive": 0,
                "pruned": 0,
            },
        ),
    ]
    scheduler_cases[3]["priority_order_correct"] = bool(
        scheduler_cases[3]["query_log"]
        and "20 percent" in scheduler_cases[3]["query_log"][0]
    )
    scheduler_cases[3]["expectation_met"] = bool(
        scheduler_cases[3]["expectation_met"]
        and scheduler_cases[3]["priority_order_correct"]
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
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "fused_queries": transaction[
                "counterevidence_fused_query_count"
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
            "real_source_marginal_claim_pruned_before_fusion",
            f"{unsafe} {marginal}",
            {
                "triggered": True,
                "transaction_committed": True,
                "overall_committed": True,
                "fused_queries": 0,
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
                "fused_queries": 0,
                "stability_pruned": 0,
            },
        ),
    ]
    cases = scheduler_cases + runtime_cases
    result = {
        "agent_version": "vsr-pathrag-v33",
        "protocol": "controlled_fused_multi_claim_counterevidence",
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
        "fused_query_count": sum(
            int(row["observed"].get("fused_queries", 0))
            for row in scheduler_cases
        ),
        "fused_claim_coverage_count": sum(
            int(
                row["observed"].get(
                    "fused_claim_coverage",
                    0,
                )
            )
            for row in scheduler_cases
        ),
        "generic_reuse_rejected_count": sum(
            int(
                row["observed"].get(
                    "generic_reuse_rejected",
                    0,
                )
            )
            for row in scheduler_cases
        ),
        "portfolio_passed_claim_count": sum(
            int(row["observed"].get("passed", 0))
            for row in scheduler_cases
        ),
        "portfolio_pruned_claim_count": sum(
            int(row["observed"].get("pruned", 0))
            for row in scheduler_cases
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
            "vsr_pathrag_v33_fused_counterevidence_audit.json"
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
        "fused={fused_query_count} "
        "coverage={fused_claim_coverage_count} "
        "generic_rejected={generic_reuse_rejected_count} "
        "passed={portfolio_passed_claim_count} "
        "pruned={portfolio_pruned_claim_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
