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
                    "family_ids": [f"support-document-{index}"],
                }
            ],
        }
        for index, claim in enumerate(claims, start=1)
    ]
    node_ids = [node["node_id"] for node in nodes]
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

    async def execute_joint(
        name: str,
        claims: list[str],
        risks: list[dict],
        search_func,
        expected: dict,
        max_global_queries: int = 4,
    ) -> dict:
        transaction = build_transaction(claims, risks)
        query_log = []

        async def logged_search(query, max_results):
            query_log.append(query)
            return await search_func(query, max_results)

        answer, challenge = (
            await run_active_counterevidence_challenge(
                "Safe fallback. " + " ".join(claims),
                transaction,
                logged_search,
                enable_value_control=True,
                enable_joint_portfolio=True,
                max_queries_per_claim=3,
                max_global_queries=max_global_queries,
            )
        )
        observed = {
            "queries": challenge["query_count"],
            "saved_queries": challenge["saved_query_count"],
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
            "decisive_stops": challenge[
                "decisive_stop_count"
            ],
            "diminishing_stops": challenge[
                "diminishing_return_stop_count"
            ],
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

    shared_claims = [
        "The program caused higher sales.",
        "The program caused greater revenue.",
    ]

    async def shared_confirmation(_query, _max_results):
        return [
            {
                "source_id": "shared-confirmation",
                "document_id": "independent-document",
                "text": (
                    "The program caused higher sales and greater "
                    "revenue."
                ),
                "source_reliability": 0.9,
            }
        ]

    async def shared_contradiction(_query, _max_results):
        return [
            {
                "source_id": "shared-contradiction",
                "document_id": "independent-document",
                "text": (
                    "The program did not cause higher sales or "
                    "greater revenue."
                ),
                "source_reliability": 0.9,
            }
        ]

    priority_claims = [
        "The campaign always drew wider attendance.",
        "The program caused 20 percent higher revenue.",
    ]

    async def targeted_confirmation(query, _max_results):
        if "20 percent" in query:
            text = priority_claims[1]
            source_id = "high-risk-confirmation"
        else:
            text = priority_claims[0]
            source_id = "medium-risk-confirmation"
        return [
            {
                "source_id": source_id,
                "document_id": f"{source_id}-document",
                "text": text,
                "source_reliability": 0.9,
            }
        ]

    async def empty_search(_query, _max_results):
        return []

    scheduler_cases = [
        await execute_joint(
            "one_shared_result_resolves_two_claims",
            shared_claims,
            [medium_risk, medium_risk],
            shared_confirmation,
            {
                "queries": 1,
                "saved_queries": 5,
                "cross_claim_reuse": 1,
                "shared_evidence": 1,
                "passed": 2,
                "contradicted": 0,
                "inconclusive": 0,
                "pruned": 0,
                "decisive_stops": 0,
                "diminishing_stops": 2,
            },
        ),
        await execute_joint(
            "one_shared_contradiction_revokes_two_claims",
            shared_claims,
            [medium_risk, medium_risk],
            shared_contradiction,
            {
                "queries": 1,
                "saved_queries": 5,
                "cross_claim_reuse": 1,
                "shared_evidence": 1,
                "passed": 0,
                "contradicted": 2,
                "inconclusive": 0,
                "pruned": 2,
                "decisive_stops": 2,
                "diminishing_stops": 0,
            },
        ),
        await execute_joint(
            "high_risk_query_wins_global_priority",
            priority_claims,
            [medium_risk, high_risk],
            targeted_confirmation,
            {
                "queries": 2,
                "saved_queries": 4,
                "cross_claim_reuse": 0,
                "shared_evidence": 0,
                "passed": 2,
                "contradicted": 0,
                "inconclusive": 0,
                "pruned": 0,
                "decisive_stops": 0,
                "diminishing_stops": 2,
            },
        ),
        await execute_joint(
            "global_budget_caps_unproductive_queries",
            shared_claims,
            [medium_risk, medium_risk],
            empty_search,
            {
                "queries": 2,
                "saved_queries": 4,
                "cross_claim_reuse": 0,
                "shared_evidence": 0,
                "passed": 0,
                "contradicted": 0,
                "inconclusive": 2,
                "pruned": 2,
                "decisive_stops": 0,
                "diminishing_stops": 0,
            },
            max_global_queries=2,
        ),
    ]
    scheduler_cases[2]["priority_order_correct"] = bool(
        scheduler_cases[2]["query_log"]
        and "20 percent" in scheduler_cases[2]["query_log"][0]
    )
    scheduler_cases[2]["expectation_met"] = bool(
        scheduler_cases[2]["expectation_met"]
        and scheduler_cases[2]["priority_order_correct"]
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
        source_rows = [
            {
                "source_id": correction_source_id,
                "document_id": correction_source_id,
                "text": committed_source,
                "score": 0.9,
            }
        ]

        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": generated,
                    "used_source_ids": [correction_source_id],
                    "insufficient": False,
                }
            )

        async def search_func(_query, _max_results):
            return []

        final_answer, result = await realize_answer(
            llm_func,
            question,
            fallback,
            {"entities": [], "relationships": []},
            source_rows,
            prohibited_assertions=constraints,
            counterevidence_search_func=search_func,
            enable_counterevidence_value_control=True,
            enable_joint_counterevidence_portfolio=True,
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "portfolio_queries": transaction[
                "counterevidence_query_count"
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
            "real_source_marginal_claim_pruned_before_portfolio",
            f"{unsafe} {marginal}",
            {
                "triggered": True,
                "transaction_committed": True,
                "overall_committed": True,
                "portfolio_queries": 0,
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
                "portfolio_queries": 0,
                "stability_pruned": 0,
            },
        ),
    ]
    cases = scheduler_cases + runtime_cases
    result = {
        "agent_version": "vsr-pathrag-v32",
        "protocol": "controlled_joint_counterevidence_portfolio",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(case["expectation_met"]) for case in cases
        ),
        "all_expectations_met": all(
            case["expectation_met"] for case in cases
        ),
        "scheduler_query_count": sum(
            int(case["observed"].get("queries", 0))
            for case in scheduler_cases
        ),
        "scheduler_saved_query_count": sum(
            int(case["observed"].get("saved_queries", 0))
            for case in scheduler_cases
        ),
        "cross_claim_reuse_count": sum(
            int(case["observed"].get("cross_claim_reuse", 0))
            for case in scheduler_cases
        ),
        "shared_evidence_count": sum(
            int(case["observed"].get("shared_evidence", 0))
            for case in scheduler_cases
        ),
        "portfolio_passed_claim_count": sum(
            int(case["observed"].get("passed", 0))
            for case in scheduler_cases
        ),
        "portfolio_pruned_claim_count": sum(
            int(case["observed"].get("pruned", 0))
            for case in scheduler_cases
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
            "vsr_pathrag_v32_joint_counterevidence_portfolio_audit.json"
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
        "reuse={cross_claim_reuse_count} "
        "shared={shared_evidence_count} "
        "passed={portfolio_passed_claim_count} "
        "pruned={portfolio_pruned_claim_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
