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
    neutral_causal = (
        "This experience involved him and became famous."
    )
    negative_causal = (
        "This experience did not make him internationally famous."
    )
    numeric_claim = (
        "This experience made him internationally famous in 2009."
    )
    wrong_numeric = (
        "This experience made him internationally famous in 2019."
    )
    predicate_claim = (
        "This experience allowed him to join academia."
    )
    opposite_predicate = (
        "This experience prevented him from joining academia."
    )
    marginal = (
        "Newspapers and magazines accompanied zephyr quuxium vortex "
        "lunar cobalt amber prism nebula."
    )

    async def execute(
        name: str,
        answer: str,
        source_rows: list[dict],
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

        final_answer, result = await realize_answer(
            llm_func,
            question,
            fallback,
            {"entities": [], "relationships": []},
            source_rows,
            prohibited_assertions=constraints,
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "semantic_protected": transaction[
                "semantic_family_protected_count"
            ],
            "neutral_rejected": transaction[
                "neutral_relation_rejected_count"
            ],
            "reliability_rejected": transaction[
                "reliability_quorum_rejected_count"
            ],
            "high_risk_rejected": transaction[
                "high_risk_quorum_rejected_count"
            ],
            "conflict_rejected": transaction[
                "conflicting_support_rejected_count"
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
        reliability: float | None = None,
    ) -> dict:
        result = {
            "source_id": source_id,
            "document_id": document_id,
            "text": text,
            "score": 0.9,
        }
        if reliability is not None:
            result["source_reliability"] = reliability
        return result

    committed = {
        "triggered": True,
        "transaction_committed": True,
        "overall_committed": True,
    }
    cases = [
        await execute(
            "neutral_overlap_does_not_form_support_quorum",
            f"{unsafe} {distractor} {causal_claim}",
            [
                correction_source,
                support("support-1", "document-1", causal_claim),
                support("support-2", "document-2", neutral_causal),
            ],
            {
                **committed,
                "semantic_protected": 0,
                "neutral_rejected": 1,
                "reliability_rejected": 0,
                "high_risk_rejected": 0,
                "conflict_rejected": 0,
                "stability_pruned": 1,
                "stale_evidence_dropped": 2,
            },
        ),
        await execute(
            "polarity_conflict_blocks_support_quorum",
            f"{unsafe} {distractor} {causal_claim}",
            [
                correction_source,
                support("support-1", "document-1", causal_claim),
                support("support-2", "document-2", causal_claim),
                support("support-3", "document-3", negative_causal),
            ],
            {
                **committed,
                "semantic_protected": 0,
                "neutral_rejected": 0,
                "reliability_rejected": 0,
                "high_risk_rejected": 0,
                "conflict_rejected": 1,
                "stability_pruned": 1,
                "stale_evidence_dropped": 3,
            },
        ),
        await execute(
            "numeric_conflict_blocks_support_quorum",
            f"{unsafe} {distractor} {numeric_claim}",
            [
                correction_source,
                support("support-1", "document-1", numeric_claim),
                support("support-2", "document-2", numeric_claim),
                support("support-3", "document-3", wrong_numeric),
            ],
            {
                **committed,
                "semantic_protected": 0,
                "neutral_rejected": 0,
                "reliability_rejected": 0,
                "high_risk_rejected": 0,
                "conflict_rejected": 1,
                "stability_pruned": 1,
                "stale_evidence_dropped": 3,
            },
        ),
        await execute(
            "predicate_conflict_blocks_support_quorum",
            f"{unsafe} {distractor} {predicate_claim}",
            [
                correction_source,
                support("support-1", "document-1", predicate_claim),
                support("support-2", "document-2", predicate_claim),
                support(
                    "support-3",
                    "document-3",
                    opposite_predicate,
                ),
            ],
            {
                **committed,
                "semantic_protected": 0,
                "neutral_rejected": 0,
                "reliability_rejected": 0,
                "high_risk_rejected": 0,
                "conflict_rejected": 1,
                "stability_pruned": 1,
                "stale_evidence_dropped": 3,
            },
        ),
        await execute(
            "high_risk_default_reliability_is_insufficient",
            f"{unsafe} {distractor} {numeric_claim}",
            [
                correction_source,
                support("support-1", "document-1", numeric_claim),
                support("support-2", "document-2", numeric_claim),
            ],
            {
                **committed,
                "semantic_protected": 0,
                "neutral_rejected": 0,
                "reliability_rejected": 1,
                "high_risk_rejected": 1,
                "conflict_rejected": 0,
                "stability_pruned": 1,
                "stale_evidence_dropped": 2,
            },
        ),
        await execute(
            "authoritative_sources_rescue_high_risk_claim",
            f"{unsafe} {distractor} {numeric_claim}",
            [
                correction_source,
                support(
                    "support-1",
                    "document-1",
                    numeric_claim,
                    0.9,
                ),
                support(
                    "support-2",
                    "document-2",
                    numeric_claim,
                    0.9,
                ),
            ],
            {
                **committed,
                "semantic_protected": 1,
                "neutral_rejected": 0,
                "reliability_rejected": 0,
                "high_risk_rejected": 0,
                "conflict_rejected": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
        await execute(
            "default_sources_rescue_medium_risk_claim",
            f"{unsafe} {distractor} {causal_claim}",
            [
                correction_source,
                support("support-1", "document-1", causal_claim),
                support("support-2", "document-2", causal_claim),
            ],
            {
                **committed,
                "semantic_protected": 1,
                "neutral_rejected": 0,
                "reliability_rejected": 0,
                "high_risk_rejected": 0,
                "conflict_rejected": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
        await execute(
            "real_source_marginal_claim_is_pruned",
            f"{unsafe} {marginal}",
            [
                {
                    "source_id": correction_source_id,
                    "document_id": correction_source_id,
                    "text": committed_source,
                    "score": 0.9,
                }
            ],
            {
                **committed,
                "semantic_protected": 0,
                "neutral_rejected": 0,
                "reliability_rejected": 0,
                "high_risk_rejected": 0,
                "conflict_rejected": 0,
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
            {
                "triggered": False,
                "transaction_committed": False,
                "overall_committed": True,
                "semantic_protected": 0,
                "neutral_rejected": 0,
                "reliability_rejected": 0,
                "high_risk_rejected": 0,
                "conflict_rejected": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
    ]
    return {
        "agent_version": "vsr-pathrag-v29",
        "protocol": "controlled_semantic_calibrated_stability_quorum",
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
        "semantic_family_protected_count": sum(
            int(case["observed"]["semantic_protected"])
            for case in cases
        ),
        "neutral_relation_rejected_count": sum(
            int(case["observed"]["neutral_rejected"])
            for case in cases
        ),
        "reliability_quorum_rejected_count": sum(
            int(case["observed"]["reliability_rejected"])
            for case in cases
        ),
        "high_risk_quorum_rejected_count": sum(
            int(case["observed"]["high_risk_rejected"])
            for case in cases
        ),
        "conflicting_support_rejected_count": sum(
            int(case["observed"]["conflict_rejected"])
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
            "vsr_pathrag_v29_semantic_quorum_audit.json"
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
        "semantic_protected={semantic_family_protected_count} "
        "neutral_rejected={neutral_relation_rejected_count} "
        "reliability_rejected={reliability_quorum_rejected_count} "
        "high_risk_rejected={high_risk_quorum_rejected_count} "
        "conflict_rejected={conflicting_support_rejected_count} "
        "pruned={stability_pruned_claim_count} "
        "stale_dropped={stale_evidence_dropped_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
