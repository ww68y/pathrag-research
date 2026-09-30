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
    causal_claim = "This experience made him internationally famous."
    causal_counterevidence = (
        "This experience did not make him internationally famous."
    )
    neutral_challenge = (
        "This experience involved him and became famous."
    )
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
        selected_ids = [
            source["source_id"] for source in source_rows
        ]

        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": answer,
                    "used_source_ids": selected_ids,
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
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "semantic_protected": transaction[
                "semantic_family_protected_count"
            ],
            "challenge_triggered": transaction[
                "counterevidence_challenge_triggered_count"
            ],
            "challenge_queries": transaction[
                "counterevidence_query_count"
            ],
            "challenge_contradictions": transaction[
                "counterevidence_contradiction_count"
            ],
            "challenge_passed": transaction[
                "counterevidence_passed_claim_count"
            ],
            "challenge_inconclusive": transaction[
                "counterevidence_inconclusive_claim_count"
            ],
            "challenge_pruned": transaction[
                "active_counterevidence_pruned_claim_count"
            ],
            "stability_pruned": transaction[
                "stability_pruned_claim_count"
            ],
            "stale_evidence_dropped": transaction[
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

    def support(
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

    def supported_sources(claim: str) -> list[dict]:
        return [
            correction_source,
            support("support-1", "document-1", claim),
            support("support-2", "document-2", claim),
        ]

    committed = {
        "triggered": True,
        "transaction_committed": True,
        "overall_committed": True,
    }
    cases = [
        await execute(
            "independent_counterevidence_revokes_rescue",
            f"{unsafe} {distractor} {causal_claim}",
            supported_sources(causal_claim),
            [
                support(
                    "challenge-1",
                    "counter-document",
                    causal_counterevidence,
                    0.9,
                )
            ],
            {
                **committed,
                "semantic_protected": 1,
                "challenge_triggered": 1,
                "challenge_queries": 2,
                "challenge_contradictions": 1,
                "challenge_passed": 0,
                "challenge_inconclusive": 0,
                "challenge_pruned": 1,
                "stability_pruned": 0,
                "stale_evidence_dropped": 2,
            },
        ),
        await execute(
            "independent_support_survives_challenge",
            f"{unsafe} {distractor} {causal_claim}",
            supported_sources(causal_claim),
            [
                support(
                    "challenge-2",
                    "independent-document",
                    causal_claim,
                    0.9,
                )
            ],
            {
                **committed,
                "semantic_protected": 1,
                "challenge_triggered": 1,
                "challenge_queries": 2,
                "challenge_contradictions": 0,
                "challenge_passed": 1,
                "challenge_inconclusive": 0,
                "challenge_pruned": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
        await execute(
            "independent_neutral_evidence_survives_challenge",
            f"{unsafe} {distractor} {causal_claim}",
            supported_sources(causal_claim),
            [
                support(
                    "challenge-3",
                    "neutral-document",
                    neutral_challenge,
                    0.9,
                )
            ],
            {
                **committed,
                "semantic_protected": 1,
                "challenge_triggered": 1,
                "challenge_queries": 2,
                "challenge_contradictions": 0,
                "challenge_passed": 1,
                "challenge_inconclusive": 0,
                "challenge_pruned": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
        await execute(
            "same_family_echo_is_inconclusive",
            f"{unsafe} {distractor} {causal_claim}",
            supported_sources(causal_claim),
            [
                support(
                    "support-1",
                    "document-1",
                    causal_claim,
                    0.9,
                )
            ],
            {
                **committed,
                "semantic_protected": 1,
                "challenge_triggered": 1,
                "challenge_queries": 2,
                "challenge_contradictions": 0,
                "challenge_passed": 0,
                "challenge_inconclusive": 1,
                "challenge_pruned": 1,
                "stability_pruned": 0,
                "stale_evidence_dropped": 2,
            },
        ),
        await execute(
            "empty_search_is_inconclusive",
            f"{unsafe} {distractor} {causal_claim}",
            supported_sources(causal_claim),
            [],
            {
                **committed,
                "semantic_protected": 1,
                "challenge_triggered": 1,
                "challenge_queries": 2,
                "challenge_contradictions": 0,
                "challenge_passed": 0,
                "challenge_inconclusive": 1,
                "challenge_pruned": 1,
                "stability_pruned": 0,
                "stale_evidence_dropped": 2,
            },
        ),
        await execute(
            "low_risk_stable_claim_does_not_spend_challenge_budget",
            f"{unsafe} {distractor} {low_risk_claim}",
            supported_sources(low_risk_claim),
            [
                support(
                    "challenge-4",
                    "counter-document",
                    (
                        "This experience did not include an academic "
                        "lecture."
                    ),
                    0.9,
                )
            ],
            {
                **committed,
                "semantic_protected": 0,
                "challenge_triggered": 0,
                "challenge_queries": 0,
                "challenge_contradictions": 0,
                "challenge_passed": 0,
                "challenge_inconclusive": 0,
                "challenge_pruned": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
        await execute(
            "real_source_marginal_claim_is_pruned_before_challenge",
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
                "semantic_protected": 0,
                "challenge_triggered": 0,
                "challenge_queries": 0,
                "challenge_contradictions": 0,
                "challenge_passed": 0,
                "challenge_inconclusive": 0,
                "challenge_pruned": 0,
                "stability_pruned": 1,
                "stale_evidence_dropped": 0,
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
                "semantic_protected": 0,
                "challenge_triggered": 0,
                "challenge_queries": 0,
                "challenge_contradictions": 0,
                "challenge_passed": 0,
                "challenge_inconclusive": 0,
                "challenge_pruned": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
    ]
    return {
        "agent_version": "vsr-pathrag-v30",
        "protocol": "controlled_bounded_counterevidence_challenge",
        "input": args.input,
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(case["expectation_met"]) for case in cases
        ),
        "all_expectations_met": all(
            case["expectation_met"] for case in cases
        ),
        "transaction_triggered_count": sum(
            bool(case["observed"]["triggered"]) for case in cases
        ),
        "transaction_accepted_count": sum(
            bool(case["observed"]["transaction_committed"])
            for case in cases
        ),
        "counterevidence_challenge_triggered_count": sum(
            int(case["observed"]["challenge_triggered"])
            for case in cases
        ),
        "counterevidence_query_count": sum(
            int(case["observed"]["challenge_queries"])
            for case in cases
        ),
        "counterevidence_contradiction_count": sum(
            int(case["observed"]["challenge_contradictions"])
            for case in cases
        ),
        "counterevidence_passed_claim_count": sum(
            int(case["observed"]["challenge_passed"])
            for case in cases
        ),
        "counterevidence_inconclusive_claim_count": sum(
            int(case["observed"]["challenge_inconclusive"])
            for case in cases
        ),
        "active_counterevidence_pruned_claim_count": sum(
            int(case["observed"]["challenge_pruned"])
            for case in cases
        ),
        "stability_pruned_claim_count": sum(
            int(case["observed"]["stability_pruned"])
            for case in cases
        ),
        "stale_evidence_dropped_count": sum(
            int(case["observed"]["stale_evidence_dropped"])
            for case in cases
        ),
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
            "vsr_pathrag_v30_counterevidence_challenge_audit.json"
        ),
    )
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    output = ROOT_DIR / args.output
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "cases={case_count} met={expectation_met_count} "
        "triggered={transaction_triggered_count} "
        "accepted={transaction_accepted_count} "
        "challenges={counterevidence_challenge_triggered_count} "
        "queries={counterevidence_query_count} "
        "contradictions={counterevidence_contradiction_count} "
        "passed={counterevidence_passed_claim_count} "
        "inconclusive={counterevidence_inconclusive_claim_count} "
        "challenge_pruned={active_counterevidence_pruned_claim_count} "
        "stability_pruned={stability_pruned_claim_count} "
        "stale_dropped={stale_evidence_dropped_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
