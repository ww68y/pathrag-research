import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    apply_transactional_correction_rollback,
    audit_realized_answer,
    build_correction_persistence_constraints,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_committed_source(row: dict, source_id: str) -> str:
    store_path = (
        ROOT_DIR / row["working_dir"] / "kv_store_text_chunks.json"
    )
    chunk_store = json.loads(store_path.read_text(encoding="utf-8"))
    return str((chunk_store.get(source_id) or {}).get("content", ""))


def audit_answer(
    question: str,
    answer: str,
    source_id: str,
    source_text: str,
    constraints: list[dict],
) -> dict:
    return audit_realized_answer(
        question,
        answer,
        {"entities": [], "relationships": []},
        [{"source_id": source_id, "text": source_text, "score": 0.9}],
        used_evidence_ids=[source_id],
        supported_facets=[],
        facet_gate_mode="skip",
        prohibited_assertions=constraints,
    )


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
            "vsr_pathrag_v23_transactional_rollback_audit.json"
        ),
    )
    args = parser.parse_args()

    accepted_rows = [
        row
        for row in load_jsonl(ROOT_DIR / args.input)
        if (row.get("diagnosis_answer_correction") or {}).get("accepted")
    ]
    cases = []
    for row in accepted_rows:
        constraints = build_correction_persistence_constraints(
            row["diagnosis_answer_correction"]
        )
        source_id = constraints[0]["source_ids"][0]
        source_text = load_committed_source(row, source_id)
        question = (
            "How did Adebanwi move from journalism into academia?"
        )
        fallback = (
            "Adebanwi worked as a freelance journalist before entering "
            "academia."
        )
        safe_improvement = (
            "Adebanwi later joined the University of Ibadan as a lecturer "
            "and researcher."
        )
        unsupported_inversion = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        same_polarity_recurrence = (
            "His reporting career remains undocumented."
        )
        unsupported_other_claim = (
            "Zyphron calibrated quuxium vortices beneath Nemerax."
        )
        candidate_rows = [
            {
                "case": "uncertified_inversion_with_safe_improvement",
                "answer": (
                    f"{safe_improvement} {unsupported_inversion}"
                ),
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 1,
                "expected_replaced": 1,
                "expected_removed": 0,
            },
            {
                "case": "same_polarity_relapse_with_safe_improvement",
                "answer": (
                    f"{safe_improvement} {same_polarity_recurrence}"
                ),
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 1,
                "expected_replaced": 1,
                "expected_removed": 0,
            },
            {
                "case": "mixed_certified_and_uncertified_inversions",
                "answer": f"{fallback} {unsupported_inversion}",
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 1,
                "expected_replaced": 0,
                "expected_removed": 1,
            },
            {
                "case": "unsafe_only_reverts_to_corrected_claim",
                "answer": unsupported_inversion,
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 0,
                "expected_replaced": 1,
                "expected_removed": 0,
            },
            {
                "case": "patch_fails_full_answer_audit",
                "answer": (
                    f"{unsupported_other_claim} "
                    f"{unsupported_inversion}"
                ),
                "expected_triggered": True,
                "expected_patch_accepted": False,
                "expected_preserved_claims": 1,
                "expected_replaced": 1,
                "expected_removed": 0,
            },
            {
                "case": "certified_generation_needs_no_transaction",
                "answer": f"{safe_improvement} {fallback}",
                "expected_triggered": False,
                "expected_patch_accepted": False,
                "expected_preserved_claims": 0,
                "expected_replaced": 0,
                "expected_removed": 0,
            },
        ]
        for candidate in candidate_rows:
            raw_audit = audit_answer(
                question,
                candidate["answer"],
                source_id,
                source_text,
                constraints,
            )
            patched_answer, transaction = (
                apply_transactional_correction_rollback(
                    candidate["answer"],
                    fallback,
                    constraints,
                    raw_audit,
                )
            )
            patched_audit = (
                audit_answer(
                    question,
                    patched_answer,
                    source_id,
                    source_text,
                    constraints,
                )
                if transaction["generated_answer_changed"]
                else None
            )
            patch_accepted = bool(
                patched_audit and patched_audit["accepted"]
            )
            expectation_met = bool(
                transaction["triggered"]
                == candidate["expected_triggered"]
                and patch_accepted
                == candidate["expected_patch_accepted"]
                and transaction["preserved_generated_claim_count"]
                == candidate["expected_preserved_claims"]
                and transaction["replaced_span_count"]
                == candidate["expected_replaced"]
                and transaction["removed_span_count"]
                == candidate["expected_removed"]
            )
            cases.append(
                {
                    "line_no": row["line_no"],
                    "case": candidate["case"],
                    "expected_triggered": candidate[
                        "expected_triggered"
                    ],
                    "observed_triggered": transaction["triggered"],
                    "expected_patch_accepted": candidate[
                        "expected_patch_accepted"
                    ],
                    "observed_patch_accepted": patch_accepted,
                    "expected_preserved_claims": candidate[
                        "expected_preserved_claims"
                    ],
                    "observed_preserved_claims": transaction[
                        "preserved_generated_claim_count"
                    ],
                    "expected_replaced": candidate["expected_replaced"],
                    "observed_replaced": transaction[
                        "replaced_span_count"
                    ],
                    "expected_removed": candidate["expected_removed"],
                    "observed_removed": transaction[
                        "removed_span_count"
                    ],
                    "expectation_met": expectation_met,
                    "raw_accepted": raw_audit["accepted"],
                    "patched_answer": patched_answer,
                    "transaction": transaction,
                    "patched_audit": patched_audit,
                }
            )

    result = {
        "agent_version": "vsr-pathrag-v23",
        "protocol": "controlled_transactional_correction_rollback",
        "input": args.input,
        "accepted_correction_row_count": len(accepted_rows),
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": bool(cases)
        and all(row["expectation_met"] for row in cases),
        "transaction_triggered_count": sum(
            bool(row["observed_triggered"]) for row in cases
        ),
        "transaction_accepted_count": sum(
            bool(row["observed_patch_accepted"]) for row in cases
        ),
        "safe_generated_claims_preserved": sum(
            int(row["observed_preserved_claims"]) for row in cases
            if row["observed_patch_accepted"]
        ),
        "full_audit_rejected_patch_count": sum(
            bool(
                row["observed_triggered"]
                and not row["observed_patch_accepted"]
            )
            for row in cases
        ),
        "reference_answer_used": False,
        "generator_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    output_path = ROOT_DIR / args.output
    output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"rows={len(accepted_rows)} cases={len(cases)} "
        f"met={result['expectation_met_count']} "
        f"triggered={result['transaction_triggered_count']} "
        f"accepted={result['transaction_accepted_count']} "
        f"preserved={result['safe_generated_claims_preserved']} "
        f"rejected={result['full_audit_rejected_patch_count']} "
        f"all={result['all_expectations_met']}"
    )


if __name__ == "__main__":
    main()
