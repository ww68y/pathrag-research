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


async def main_async(args) -> dict:
    rows = [
        row
        for row in load_jsonl(ROOT_DIR / args.input)
        if (row.get("diagnosis_answer_correction") or {}).get(
            "accepted"
        )
    ]
    row = rows[0]
    constraints = build_correction_persistence_constraints(
        row["diagnosis_answer_correction"]
    )
    correction_source_id = constraints[0]["source_ids"][0]
    committed_source = load_committed_source(
        row,
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
    distractor = (
        "A comet made him internationally famous near Nemerax amid "
        "zephyr quuxium cobalt amber prism lunar vortex nebula "
        "solstice titanium orchard cipher delta quartz aurora harbor "
        "matrix kinetic falcon granite meadow plasma radian summit."
    )
    claim = "This experience made him internationally famous."
    contradiction = (
        "This experience did not make him internationally famous."
    )
    neutral = "This experience involved him and became famous."
    low_risk_claim = (
        "This experience included an academic lecture."
    )
    marginal = (
        "Newspapers and magazines accompanied zephyr quuxium vortex "
        "lunar cobalt amber prism nebula."
    )

    async def execute(
        name: str,
        answer: str,
        source_rows: list[dict],
        challenge_rows: list[dict],
        expected: dict,
    ) -> dict:
        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": answer,
                    "used_source_ids": [
                        source["source_id"]
                        for source in source_rows
                    ],
                    "insufficient": False,
                }
            )

        async def search_func(_query, max_results):
            return challenge_rows[:max_results]

        final_answer, result = await realize_answer(
            llm_func,
            question,
            fallback,
            {"entities": [], "relationships": []},
            source_rows,
            prohibited_assertions=constraints,
            counterevidence_search_func=search_func,
            enable_counterevidence_value_control=True,
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "challenge_triggered": transaction[
                "counterevidence_challenge_triggered_count"
            ],
            "queries": transaction["counterevidence_query_count"],
            "saved_queries": transaction[
                "counterevidence_saved_query_count"
            ],
            "early_stops": transaction[
                "counterevidence_early_stop_count"
            ],
            "decisive_stops": transaction[
                "counterevidence_decisive_stop_count"
            ],
            "diminishing_stops": transaction[
                "counterevidence_diminishing_return_stop_count"
            ],
            "skipped_low_value": transaction[
                "counterevidence_skipped_low_value_query_count"
            ],
            "contradictions": transaction[
                "counterevidence_contradiction_count"
            ],
            "passed": transaction[
                "counterevidence_passed_claim_count"
            ],
            "inconclusive": transaction[
                "counterevidence_inconclusive_claim_count"
            ],
            "challenge_pruned": transaction[
                "active_counterevidence_pruned_claim_count"
            ],
            "stability_pruned": transaction[
                "stability_pruned_claim_count"
            ],
            "stale_dropped": transaction[
                "dropped_stale_evidence_count"
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

    correction_source = {
        "source_id": correction_source_id,
        "document_id": "career-document",
        "text": fallback,
        "score": 0.9,
    }

    def source(
        source_id: str,
        document_id: str,
        text: str,
        reliability: float = 0.72,
    ) -> dict:
        return {
            "source_id": source_id,
            "document_id": document_id,
            "text": text,
            "score": 0.9,
            "source_reliability": reliability,
        }

    def support_sources(target_claim: str) -> list[dict]:
        return [
            correction_source,
            source("support-1", "document-1", target_claim),
            source("support-2", "document-2", target_claim),
        ]

    committed = {
        "triggered": True,
        "transaction_committed": True,
        "overall_committed": True,
    }

    def challenge_expected(
        *,
        queries: int,
        saved: int,
        early: int,
        decisive: int,
        diminishing: int,
        skipped: int,
        contradictions: int,
        passed: int,
        inconclusive: int,
        pruned: int,
        stale: int,
    ) -> dict:
        return {
            **committed,
            "challenge_triggered": 1,
            "queries": queries,
            "saved_queries": saved,
            "early_stops": early,
            "decisive_stops": decisive,
            "diminishing_stops": diminishing,
            "skipped_low_value": skipped,
            "contradictions": contradictions,
            "passed": passed,
            "inconclusive": inconclusive,
            "challenge_pruned": pruned,
            "stability_pruned": 0,
            "stale_dropped": stale,
        }

    cases = [
        await execute(
            "decisive_contradiction_stops_after_one_query",
            f"{unsafe} {distractor} {claim}",
            support_sources(claim),
            [
                source(
                    "counter-1",
                    "counter-document",
                    contradiction,
                    0.9,
                )
            ],
            challenge_expected(
                queries=1,
                saved=2,
                early=1,
                decisive=1,
                diminishing=0,
                skipped=0,
                contradictions=1,
                passed=0,
                inconclusive=0,
                pruned=1,
                stale=2,
            ),
        ),
        await execute(
            "strong_independent_result_stops_after_one_query",
            f"{unsafe} {distractor} {claim}",
            support_sources(claim),
            [
                source(
                    "confirm-1",
                    "confirm-document",
                    claim,
                    0.9,
                )
            ],
            challenge_expected(
                queries=1,
                saved=2,
                early=1,
                decisive=0,
                diminishing=1,
                skipped=0,
                contradictions=0,
                passed=1,
                inconclusive=0,
                pruned=0,
                stale=0,
            ),
        ),
        await execute(
            "neutral_result_gets_second_query_then_stops",
            f"{unsafe} {distractor} {claim}",
            support_sources(claim),
            [
                source(
                    "neutral-1",
                    "neutral-document",
                    neutral,
                    0.9,
                )
            ],
            challenge_expected(
                queries=2,
                saved=1,
                early=1,
                decisive=0,
                diminishing=1,
                skipped=1,
                contradictions=0,
                passed=1,
                inconclusive=0,
                pruned=0,
                stale=0,
            ),
        ),
        await execute(
            "same_family_echo_stops_at_diminishing_return",
            f"{unsafe} {distractor} {claim}",
            support_sources(claim),
            [
                source(
                    "support-1",
                    "document-1",
                    claim,
                    0.9,
                )
            ],
            challenge_expected(
                queries=2,
                saved=1,
                early=1,
                decisive=0,
                diminishing=1,
                skipped=1,
                contradictions=0,
                passed=0,
                inconclusive=1,
                pruned=1,
                stale=2,
            ),
        ),
        await execute(
            "empty_search_stops_at_diminishing_return",
            f"{unsafe} {distractor} {claim}",
            support_sources(claim),
            [],
            challenge_expected(
                queries=2,
                saved=1,
                early=1,
                decisive=0,
                diminishing=1,
                skipped=1,
                contradictions=0,
                passed=0,
                inconclusive=1,
                pruned=1,
                stale=2,
            ),
        ),
        await execute(
            "stable_low_risk_claim_uses_no_challenge_queries",
            f"{unsafe} {distractor} {low_risk_claim}",
            support_sources(low_risk_claim),
            [],
            {
                **committed,
                "challenge_triggered": 0,
                "queries": 0,
                "saved_queries": 0,
                "early_stops": 0,
                "decisive_stops": 0,
                "diminishing_stops": 0,
                "skipped_low_value": 0,
                "contradictions": 0,
                "passed": 0,
                "inconclusive": 0,
                "challenge_pruned": 0,
                "stability_pruned": 0,
                "stale_dropped": 0,
            },
        ),
        await execute(
            "real_source_marginal_claim_pruned_before_challenge",
            f"{unsafe} {marginal}",
            [
                {
                    "source_id": correction_source_id,
                    "document_id": correction_source_id,
                    "text": committed_source,
                    "score": 0.9,
                }
            ],
            [],
            {
                **committed,
                "challenge_triggered": 0,
                "queries": 0,
                "saved_queries": 0,
                "early_stops": 0,
                "decisive_stops": 0,
                "diminishing_stops": 0,
                "skipped_low_value": 0,
                "contradictions": 0,
                "passed": 0,
                "inconclusive": 0,
                "challenge_pruned": 0,
                "stability_pruned": 1,
                "stale_dropped": 0,
            },
        ),
        await execute(
            "certified_generation_needs_no_transaction",
            (
                "Adebanwi later joined the University of Ibadan as a "
                f"lecturer and researcher. {fallback}"
            ),
            [
                {
                    "source_id": correction_source_id,
                    "document_id": correction_source_id,
                    "text": committed_source,
                    "score": 0.9,
                }
            ],
            [],
            {
                "triggered": False,
                "transaction_committed": False,
                "overall_committed": True,
                "challenge_triggered": 0,
                "queries": 0,
                "saved_queries": 0,
                "early_stops": 0,
                "decisive_stops": 0,
                "diminishing_stops": 0,
                "skipped_low_value": 0,
                "contradictions": 0,
                "passed": 0,
                "inconclusive": 0,
                "challenge_pruned": 0,
                "stability_pruned": 0,
                "stale_dropped": 0,
            },
        ),
    ]
    result = {
        "agent_version": "vsr-pathrag-v31",
        "protocol": "controlled_voi_counterevidence_search",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(case["expectation_met"]) for case in cases
        ),
        "all_expectations_met": all(
            case["expectation_met"] for case in cases
        ),
        "reference_answer_used": False,
        "generator_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    for key in (
        "challenge_triggered",
        "queries",
        "saved_queries",
        "early_stops",
        "decisive_stops",
        "diminishing_stops",
        "skipped_low_value",
        "contradictions",
        "passed",
        "inconclusive",
        "challenge_pruned",
        "stability_pruned",
        "stale_dropped",
    ):
        result[f"{key}_count"] = sum(
            int(case["observed"][key]) for case in cases
        )
    result["transaction_triggered_count"] = sum(
        bool(case["observed"]["triggered"]) for case in cases
    )
    result["transaction_accepted_count"] = sum(
        bool(case["observed"]["transaction_committed"])
        for case in cases
    )
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
            "vsr_pathrag_v31_voi_counterevidence_audit.json"
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
        "triggered={transaction_triggered_count} "
        "accepted={transaction_accepted_count} "
        "challenges={challenge_triggered_count} "
        "queries={queries_count} saved={saved_queries_count} "
        "early={early_stops_count} "
        "decisive={decisive_stops_count} "
        "diminishing={diminishing_stops_count} "
        "contradictions={contradictions_count} "
        "passed={passed_count} inconclusive={inconclusive_count} "
        "challenge_pruned={challenge_pruned_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
