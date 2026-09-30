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


def build_transaction(claims: list[str]) -> dict:
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
                    "claim_relation_risk": {
                        "risk_level": "medium",
                        "risk_score": 0.46,
                    },
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
    claims = [
        "The program caused higher sales.",
        "The program caused greater revenue.",
    ]

    async def execute_scheduler(
        name: str,
        residual_text: str,
        expected: dict,
        generic_first_result: bool = False,
        max_global_queries: int = 4,
    ) -> dict:
        query_log = []

        async def search_func(query, _max_results):
            query_log.append(query)
            if generic_first_result:
                return [
                    {
                        "source_id": "generic-result",
                        "document_id": "generic-document",
                        "text": "The program caused results.",
                        "source_reliability": 0.9,
                    }
                ]
            if "previous multi-claim search" in query.lower():
                return [
                    {
                        "source_id": "residual-result",
                        "document_id": "residual-document",
                        "text": residual_text,
                        "source_reliability": 0.9,
                    }
                ]
            return [
                {
                    "source_id": "partial-sales",
                    "document_id": "sales-document",
                    "text": claims[0],
                    "source_reliability": 0.9,
                }
            ]

        answer, challenge = (
            await run_active_counterevidence_challenge(
                "Safe fallback. " + " ".join(claims),
                build_transaction(claims),
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_query_fusion=True,
                enable_residual_decomposition=True,
                max_queries_per_claim=3,
                max_global_queries=max_global_queries,
            )
        )
        observed = {
            "queries": challenge["query_count"],
            "saved_queries": challenge["saved_query_count"],
            "fused_queries": challenge["fused_query_count"],
            "partial_clusters": challenge[
                "partial_coverage_cluster_count"
            ],
            "residual_queries": challenge[
                "residual_query_count"
            ],
            "residual_claims": challenge[
                "residual_claim_count"
            ],
            "residual_resolutions": challenge[
                "residual_resolution_count"
            ],
            "alias_coverage": challenge[
                "alias_coverage_count"
            ],
            "generic_reuse_rejected": challenge[
                "generic_reuse_rejected_count"
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

    scheduler_cases = [
        await execute_scheduler(
            "partial_confirmation_resolves_with_residual_query",
            claims[1],
            {
                "queries": 2,
                "saved_queries": 4,
                "fused_queries": 1,
                "partial_clusters": 1,
                "residual_queries": 1,
                "residual_claims": 1,
                "residual_resolutions": 1,
                "alias_coverage": 0,
                "generic_reuse_rejected": 2,
                "passed": 2,
                "contradicted": 0,
                "inconclusive": 0,
                "pruned": 0,
            },
        ),
        await execute_scheduler(
            "partial_contradiction_revokes_residual_claim",
            "The program did not cause greater revenue.",
            {
                "queries": 2,
                "saved_queries": 4,
                "fused_queries": 1,
                "partial_clusters": 1,
                "residual_queries": 1,
                "residual_claims": 1,
                "residual_resolutions": 1,
                "alias_coverage": 0,
                "generic_reuse_rejected": 1,
                "passed": 1,
                "contradicted": 1,
                "inconclusive": 0,
                "pruned": 1,
            },
        ),
        await execute_scheduler(
            "evidence_grounded_alias_resolves_residual_claim",
            "The program caused greater earnings.",
            {
                "queries": 2,
                "saved_queries": 4,
                "fused_queries": 1,
                "partial_clusters": 1,
                "residual_queries": 1,
                "residual_claims": 1,
                "residual_resolutions": 1,
                "alias_coverage": 1,
                "generic_reuse_rejected": 2,
                "passed": 2,
                "contradicted": 0,
                "inconclusive": 0,
                "pruned": 0,
            },
        ),
        await execute_scheduler(
            "generic_result_cannot_trigger_false_partial_coverage",
            "",
            {
                "queries": 1,
                "saved_queries": 5,
                "fused_queries": 1,
                "partial_clusters": 0,
                "residual_queries": 0,
                "residual_claims": 0,
                "residual_resolutions": 0,
                "alias_coverage": 0,
                "generic_reuse_rejected": 2,
                "passed": 0,
                "contradicted": 0,
                "inconclusive": 2,
                "pruned": 2,
            },
            generic_first_result=True,
            max_global_queries=1,
        ),
    ]
    scheduler_cases[0]["residual_query_is_specialized"] = bool(
        len(scheduler_cases[0]["query_log"]) == 2
        and "[R1]" in scheduler_cases[0]["query_log"][1]
        and "revenue" in scheduler_cases[0]["query_log"][1]
    )
    scheduler_cases[0]["expectation_met"] = bool(
        scheduler_cases[0]["expectation_met"]
        and scheduler_cases[0]["residual_query_is_specialized"]
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
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "residual_queries": transaction[
                "counterevidence_residual_query_count"
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
            "real_source_marginal_claim_pruned_before_residual_stage",
            f"{unsafe} {marginal}",
            {
                "triggered": True,
                "transaction_committed": True,
                "overall_committed": True,
                "residual_queries": 0,
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
                "residual_queries": 0,
                "stability_pruned": 0,
            },
        ),
    ]
    cases = scheduler_cases + runtime_cases
    result = {
        "agent_version": "vsr-pathrag-v34",
        "protocol": "controlled_partial_cluster_decomposition",
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
        "partial_coverage_cluster_count": sum(
            int(row["observed"].get("partial_clusters", 0))
            for row in scheduler_cases
        ),
        "residual_query_count": sum(
            int(row["observed"].get("residual_queries", 0))
            for row in scheduler_cases
        ),
        "residual_resolution_count": sum(
            int(
                row["observed"].get(
                    "residual_resolutions",
                    0,
                )
            )
            for row in scheduler_cases
        ),
        "alias_coverage_count": sum(
            int(row["observed"].get("alias_coverage", 0))
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
            "vsr_pathrag_v34_residual_cluster_audit.json"
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
        "partial={partial_coverage_cluster_count} "
        "residual={residual_query_count} "
        "resolved={residual_resolution_count} "
        "aliases={alias_coverage_count} "
        "generic_rejected={generic_reuse_rejected_count} "
        "passed={portfolio_passed_claim_count} "
        "pruned={portfolio_pruned_claim_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
