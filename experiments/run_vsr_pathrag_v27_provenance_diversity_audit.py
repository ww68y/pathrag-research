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
        if (row.get("diagnosis_answer_correction") or {}).get("accepted")
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
    safe_improvement = (
        "Adebanwi later joined the University of Ibadan as a lecturer "
        "and researcher."
    )
    unsafe = (
        "Adebanwi worked as a television journalist during a Mars "
        "expedition."
    )
    dependent = "This experience made him internationally famous."
    distractor = (
        "A comet made him internationally famous near Nemerax amid "
        "zephyr quuxium cobalt amber prism lunar vortex nebula "
        "solstice titanium orchard cipher delta quartz aurora harbor "
        "matrix kinetic falcon granite meadow plasma radian summit."
    )
    unstable_answer = f"{unsafe} {distractor} {dependent}"
    marginal = (
        "Newspapers and magazines accompanied zephyr quuxium vortex "
        "lunar cobalt amber prism nebula."
    )
    numeric = "Adebanwi joined the university in 2099."

    async def execute(
        name: str,
        answer: str,
        source_rows: list[dict],
        expected: dict,
        evidence_summary: dict | None = None,
        used_ids: list[str] | None = None,
    ) -> dict:
        selected_ids = used_ids or [
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
            evidence_summary
            or {"entities": [], "relationships": []},
            source_rows,
            prohibited_assertions=constraints,
        )
        transaction = result["transactional_correction_rollback"]
        observed = {
            "triggered": transaction["triggered"],
            "transaction_committed": transaction["committed"],
            "overall_committed": result["committed"],
            "single_source_rejected": transaction[
                "single_source_stability_rejected_count"
            ],
            "diverse_provenance_protected": transaction[
                "diverse_provenance_protected_count"
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
        "origin_ids": ["career-document"],
        "text": fallback,
        "score": 0.9,
    }
    single_support = {
        "source_id": "support-1",
        "origin_ids": ["independent-1"],
        "text": dependent,
        "score": 0.9,
    }
    independent_support = {
        "source_id": "support-2",
        "origin_ids": ["independent-2"],
        "text": dependent,
        "score": 0.9,
    }
    duplicate_support = {
        "source_id": "support-2",
        "origin_ids": ["independent-1"],
        "text": dependent,
        "score": 0.9,
    }
    transaction_expected = {
        "triggered": True,
        "transaction_committed": True,
        "overall_committed": True,
    }
    cases = [
        await execute(
            "single_source_cannot_rescue_unstable_claim",
            unstable_answer,
            [correction_source, single_support],
            {
                **transaction_expected,
                "single_source_rejected": 1,
                "diverse_provenance_protected": 0,
                "stability_pruned": 1,
                "stale_evidence_dropped": 1,
            },
        ),
        await execute(
            "independent_text_quorum_rescues_unstable_claim",
            unstable_answer,
            [
                correction_source,
                single_support,
                independent_support,
            ],
            {
                **transaction_expected,
                "single_source_rejected": 0,
                "diverse_provenance_protected": 1,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
        await execute(
            "duplicate_ids_same_origin_do_not_form_quorum",
            unstable_answer,
            [
                correction_source,
                single_support,
                duplicate_support,
            ],
            {
                **transaction_expected,
                "single_source_rejected": 1,
                "diverse_provenance_protected": 0,
                "stability_pruned": 1,
                "stale_evidence_dropped": 2,
            },
        ),
        await execute(
            "graph_text_agreement_rescues_unstable_claim",
            unstable_answer,
            [correction_source, single_support],
            {
                **transaction_expected,
                "single_source_rejected": 0,
                "diverse_provenance_protected": 1,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
            evidence_summary={
                "entities": [
                    {
                        "name": "EXPERIENCE",
                        "description": dependent,
                        "source_ids": ["independent-1"],
                    }
                ],
                "relationships": [],
            },
            used_ids=[
                correction_source_id,
                "support-1",
                "G1",
            ],
        ),
        await execute(
            "real_source_marginal_selection_is_pruned",
            f"{unsafe} {marginal}",
            [
                {
                    "source_id": correction_source_id,
                    "origin_ids": [correction_source_id],
                    "text": committed_source,
                    "score": 0.9,
                }
            ],
            {
                **transaction_expected,
                "single_source_rejected": 0,
                "diverse_provenance_protected": 0,
                "stability_pruned": 1,
                "stale_evidence_dropped": 0,
            },
        ),
        await execute(
            "complete_numeric_audit_rejects_stable_bundle",
            f"{unsafe} {numeric}",
            [
                {
                    "source_id": correction_source_id,
                    "origin_ids": [correction_source_id],
                    "text": committed_source,
                    "score": 0.9,
                }
            ],
            {
                "triggered": True,
                "transaction_committed": False,
                "overall_committed": False,
                "single_source_rejected": 0,
                "diverse_provenance_protected": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
        await execute(
            "certified_generation_needs_no_transaction",
            f"{safe_improvement} {fallback}",
            [
                {
                    "source_id": correction_source_id,
                    "origin_ids": [correction_source_id],
                    "text": committed_source,
                    "score": 0.9,
                }
            ],
            {
                "triggered": False,
                "transaction_committed": False,
                "overall_committed": True,
                "single_source_rejected": 0,
                "diverse_provenance_protected": 0,
                "stability_pruned": 0,
                "stale_evidence_dropped": 0,
            },
        ),
    ]
    return {
        "agent_version": "vsr-pathrag-v27",
        "protocol": "controlled_provenance_diverse_stability",
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
        "single_source_rejected_count": sum(
            int(case["observed"]["single_source_rejected"])
            for case in cases
        ),
        "diverse_provenance_protected_count": sum(
            int(
                case["observed"][
                    "diverse_provenance_protected"
                ]
            )
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
            "vsr_pathrag_v27_provenance_diversity_audit.json"
        ),
    )
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"cases={result['case_count']} "
        f"met={result['expectation_met_count']} "
        f"triggered={result['transaction_triggered_count']} "
        f"accepted={result['transaction_accepted_count']} "
        f"single_rejected={result['single_source_rejected_count']} "
        f"diverse_protected="
        f"{result['diverse_provenance_protected_count']} "
        f"stability_pruned={result['stability_pruned_claim_count']} "
        f"stale_dropped={result['stale_evidence_dropped_count']} "
        f"all={result['all_expectations_met']}"
    )


if __name__ == "__main__":
    main()
