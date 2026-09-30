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
            "vsr_pathrag_v24_dependency_transaction_audit.json"
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
        supported_dependency = (
            "This experience came before he joined the University of "
            "Ibadan's Department of Political Science."
        )
        unsupported_inversion = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        unsupported_dependency = (
            "This achievement made him internationally famous."
        )
        unsupported_causal_dependency = (
            "Therefore, he received an interplanetary reporting award."
        )
        unsupported_other_claim = (
            "Zyphron calibrated quuxium vortices beneath Nemerax."
        )
        candidate_rows = [
            {
                "case": "duplicate_removal_cascades_anaphoric_claim",
                "answer": (
                    f"{fallback} {unsupported_inversion} "
                    f"{unsupported_dependency} {safe_improvement}"
                ),
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 2,
                "expected_dependency_removed": 1,
                "expected_dependency_rebound": 0,
            },
            {
                "case": "replacement_rebinds_supported_dependency",
                "answer": (
                    f"{unsupported_inversion} {supported_dependency} "
                    f"{safe_improvement}"
                ),
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 2,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 1,
            },
            {
                "case": "removal_cascades_multihop_causal_chain",
                "answer": (
                    f"{fallback} {unsupported_inversion} "
                    f"{unsupported_dependency} "
                    f"{unsupported_causal_dependency} "
                    f"{safe_improvement}"
                ),
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 2,
                "expected_dependency_removed": 2,
                "expected_dependency_rebound": 0,
            },
            {
                "case": "independent_claim_survives_parent_removal",
                "answer": (
                    f"{fallback} {unsupported_inversion} "
                    f"{safe_improvement}"
                ),
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 2,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 0,
            },
            {
                "case": "certified_generation_needs_no_transaction",
                "answer": f"{safe_improvement} {fallback}",
                "expected_triggered": False,
                "expected_patch_accepted": False,
                "expected_preserved_claims": 0,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 0,
            },
            {
                "case": "unsupported_independent_claim_fails_full_audit",
                "answer": (
                    f"{unsupported_inversion} {unsupported_other_claim}"
                ),
                "expected_triggered": True,
                "expected_patch_accepted": False,
                "expected_preserved_claims": 1,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 0,
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
                    question=question,
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
                and transaction["dependency_removed_span_count"]
                == candidate["expected_dependency_removed"]
                and transaction["dependency_rebound_count"]
                == candidate["expected_dependency_rebound"]
                and transaction["dependency_graph"][
                    "coherence_passed"
                ]
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
                    "expected_dependency_removed": candidate[
                        "expected_dependency_removed"
                    ],
                    "observed_dependency_removed": transaction[
                        "dependency_removed_span_count"
                    ],
                    "expected_dependency_rebound": candidate[
                        "expected_dependency_rebound"
                    ],
                    "observed_dependency_rebound": transaction[
                        "dependency_rebound_count"
                    ],
                    "expectation_met": expectation_met,
                    "raw_accepted": raw_audit["accepted"],
                    "patched_answer": patched_answer,
                    "transaction": transaction,
                    "patched_audit": patched_audit,
                }
            )

    result = {
        "agent_version": "vsr-pathrag-v24",
        "protocol": "controlled_dependency_aware_transaction",
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
        "dependency_cascade_removed_count": sum(
            int(row["observed_dependency_removed"])
            for row in cases
        ),
        "dependency_rebound_count": sum(
            int(row["observed_dependency_rebound"])
            for row in cases
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
        "all_transactions_coherent": all(
            row["transaction"]["dependency_graph"][
                "coherence_passed"
            ]
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
        f"cascade={result['dependency_cascade_removed_count']} "
        f"rebound={result['dependency_rebound_count']} "
        f"preserved={result['safe_generated_claims_preserved']} "
        f"rejected={result['full_audit_rejected_patch_count']} "
        f"coherent={result['all_transactions_coherent']} "
        f"all={result['all_expectations_met']}"
    )


if __name__ == "__main__":
    main()
