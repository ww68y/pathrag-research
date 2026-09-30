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
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
    load_committed_source,
    load_jsonl,
)


CLAIMS = [
    "The campaign always drew wider attendance.",
    "The program caused 20 percent higher revenue.",
]
RISKS = [
    {"risk_level": "medium", "risk_score": 0.0},
    {"risk_level": "high", "risk_score": 0.9},
]


async def execute_schedule(
    *,
    max_global_queries: int,
    enable_fusion: bool = False,
    related_claims: bool = False,
) -> tuple[list[str], dict]:
    claims = (
        [
            "The program caused higher sales.",
            "The program caused greater revenue.",
        ]
        if related_claims
        else CLAIMS
    )
    risks = (
        [
            {"risk_level": "medium", "risk_score": 0.46},
            {"risk_level": "medium", "risk_score": 0.46},
        ]
        if related_claims
        else RISKS
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
            "retrieval_adapter": "local_text_chunk_scan",
            "usage_status": "complete_local_retrieval",
        }

    _answer, challenge = await run_active_counterevidence_challenge(
        " ".join(claims),
        build_transaction(claims, risks),
        local_search,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_query_fusion=enable_fusion,
        enable_risk_coverage_guard=True,
        max_queries_per_claim=3,
        max_global_queries=max_global_queries,
    )
    return queries, challenge


async def main_async(args) -> dict:
    queries, guarded = await execute_schedule(
        max_global_queries=2,
    )
    audit = guarded["coverage_reserve_selection_audit"]
    starvation_case = {
        "case": "critical_budget_prevents_claim_starvation",
        "observed": {
            "queries": queries,
            "interventions": guarded[
                "coverage_reserve_intervention_count"
            ],
            "covered": guarded["covered_risky_claim_count"],
            "uncovered": guarded["uncovered_risky_claim_count"],
        },
        "expectation_met": bool(
            "20 percent" in queries[0]
            and "campaign" in queries[1].lower()
            and guarded[
                "coverage_reserve_intervention_count"
            ]
            == 1
            and guarded["covered_risky_claim_count"] == 2
            and guarded["uncovered_risky_claim_count"] == 0
        ),
        "challenge": guarded,
    }
    risk_priority_case = {
        "case": "first_round_preserves_high_risk_priority",
        "observed": audit[0],
        "expectation_met": bool(
            audit[0]["normal_selected_node_id"] == "claim-2"
            and audit[0]["selected_node_id"] == "claim-2"
            and not audit[0]["intervened"]
        ),
    }

    _fused_queries, fused = await execute_schedule(
        max_global_queries=1,
        enable_fusion=True,
        related_claims=True,
    )
    fused_case = {
        "case": "fused_query_covers_all_member_claims",
        "observed": {
            "fused_queries": fused["fused_query_count"],
            "covered_node_ids": fused[
                "covered_risky_node_ids"
            ],
            "uncovered": fused["uncovered_risky_claim_count"],
            "interventions": fused[
                "coverage_reserve_intervention_count"
            ],
        },
        "expectation_met": bool(
            fused["fused_query_count"] == 1
            and fused["covered_risky_node_ids"]
            == ["claim-1", "claim-2"]
            and fused["uncovered_risky_claim_count"] == 0
            and fused[
                "coverage_reserve_intervention_count"
            ]
            == 0
        ),
        "challenge": fused,
    }

    ample_queries, ample = await execute_schedule(
        max_global_queries=3,
    )
    ample_audit = ample["coverage_reserve_selection_audit"]
    ample_budget_case = {
        "case": "reserve_waits_until_budget_is_critical",
        "observed": {
            "queries": ample_queries,
            "audit": ample_audit,
            "interventions": ample[
                "coverage_reserve_intervention_count"
            ],
        },
        "expectation_met": bool(
            len(ample_audit) == 3
            and not ample_audit[0]["intervened"]
            and not ample_audit[1]["reserve_active"]
            and ample_audit[2]["reserve_active"]
            and not ample_audit[2]["intervened"]
            and ample["covered_risky_claim_count"] == 2
        ),
        "challenge": ample,
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
            "real_source_marginal_claim_pruned_with_coverage_guard",
            f"{unsafe} {marginal}",
            {
                "triggered": True,
                "transaction_committed": True,
                "overall_committed": True,
                "controller": (
                    "risk_coverage_transactional_rollback_v16"
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
                    "risk_coverage_transactional_rollback_v16"
                ),
                "stability_pruned": 0,
            },
        ),
    ]
    cases = [
        starvation_case,
        risk_priority_case,
        fused_case,
        ample_budget_case,
        *runtime_cases,
    ]
    return {
        "agent_version": "vsr-pathrag-v38",
        "protocol": "risk_coverage_budget_guard",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "coverage_reserve_intervention_count": guarded[
            "coverage_reserve_intervention_count"
        ],
        "covered_risky_claim_count": guarded[
            "covered_risky_claim_count"
        ],
        "uncovered_risky_claim_count": guarded[
            "uncovered_risky_claim_count"
        ],
        "fused_covered_risky_claim_count": fused[
            "covered_risky_claim_count"
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
            "vsr_pathrag_v38_risk_coverage_audit.json"
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
        "interventions={coverage_reserve_intervention_count} "
        "covered={covered_risky_claim_count} "
        "uncovered={uncovered_risky_claim_count} "
        "fused_covered={fused_covered_risky_claim_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
