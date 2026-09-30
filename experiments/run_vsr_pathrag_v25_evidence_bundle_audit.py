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
    obligations: list[dict] | None = None,
) -> dict:
    return audit_realized_answer(
        question,
        answer,
        {"entities": [], "relationships": []},
        [{"source_id": source_id, "text": source_text, "score": 0.9}],
        used_evidence_ids=[source_id],
        obligations=obligations,
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
            "vsr_pathrag_v25_evidence_bundle_audit.json"
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
            "This reporting experience included work for newspapers and "
            "magazines."
        )
        unsupported_inversion = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        unsupported_dependency = (
            "This reporting experience made him internationally famous."
        )
        unsupported_other_claim = (
            "Zyphron calibrated quuxium vortices beneath Nemerax."
        )
        unsupported_number_claim = (
            "Adebanwi joined the university in 2099."
        )
        obligation = {
            "obligation_id": "O-career-transition",
            "evidence_id": source_id,
            "text": safe_improvement,
            "required_terms": ["lecturer", "researcher"],
            "score": 1.0,
        }
        candidate_rows = [
            {
                "case": "distant_unsafe_parent_cascades_dependency",
                "answer": (
                    f"{fallback} {unsupported_inversion} "
                    f"{safe_improvement} {unsupported_dependency}"
                ),
                "obligations": [],
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 2,
                "expected_dependency_removed": 1,
                "expected_dependency_rebound": 0,
                "expected_distant_edges": 1,
                "expected_bundle_pruned": 0,
                "expected_obligation_protected": 0,
            },
            {
                "case": "distant_replacement_rebinds_supported_claim",
                "answer": (
                    f"{unsupported_inversion} {safe_improvement} "
                    f"{supported_dependency}"
                ),
                "obligations": [],
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 2,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 1,
                "expected_distant_edges": 1,
                "expected_bundle_pruned": 0,
                "expected_obligation_protected": 0,
            },
            {
                "case": "global_bundle_prunes_independent_hallucination",
                "answer": (
                    f"{unsupported_inversion} {unsupported_other_claim}"
                ),
                "obligations": [],
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 0,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 0,
                "expected_distant_edges": 0,
                "expected_bundle_pruned": 1,
                "expected_obligation_protected": 0,
            },
            {
                "case": "source_bound_obligation_is_protected",
                "answer": (
                    f"{fallback} {unsupported_inversion} "
                    f"{safe_improvement}"
                ),
                "obligations": [obligation],
                "expected_triggered": True,
                "expected_patch_accepted": True,
                "expected_preserved_claims": 2,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 0,
                "expected_distant_edges": 0,
                "expected_bundle_pruned": 0,
                "expected_obligation_protected": 1,
            },
            {
                "case": "numeric_fidelity_still_rejects_selected_bundle",
                "answer": (
                    f"{unsupported_inversion} "
                    f"{unsupported_number_claim}"
                ),
                "obligations": [],
                "expected_triggered": True,
                "expected_patch_accepted": False,
                "expected_preserved_claims": 1,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 0,
                "expected_distant_edges": 0,
                "expected_bundle_pruned": 0,
                "expected_obligation_protected": 0,
            },
            {
                "case": "certified_generation_needs_no_transaction",
                "answer": f"{safe_improvement} {fallback}",
                "obligations": [],
                "expected_triggered": False,
                "expected_patch_accepted": False,
                "expected_preserved_claims": 0,
                "expected_dependency_removed": 0,
                "expected_dependency_rebound": 0,
                "expected_distant_edges": 0,
                "expected_bundle_pruned": 0,
                "expected_obligation_protected": 0,
            },
        ]
        for candidate in candidate_rows:
            obligations = candidate["obligations"]
            raw_audit = audit_answer(
                question,
                candidate["answer"],
                source_id,
                source_text,
                constraints,
                obligations=obligations,
            )
            patched_answer, transaction = (
                apply_transactional_correction_rollback(
                    candidate["answer"],
                    fallback,
                    constraints,
                    raw_audit,
                    question=question,
                    obligations=obligations,
                    supported_facets=[],
                    enable_global_bundle_selection=True,
                )
            )
            patched_audit = (
                audit_answer(
                    question,
                    patched_answer,
                    source_id,
                    source_text,
                    constraints,
                    obligations=obligations,
                )
                if transaction["generated_answer_changed"]
                else None
            )
            patch_accepted = bool(
                patched_audit and patched_audit["accepted"]
            )
            observed = {
                "triggered": transaction["triggered"],
                "patch_accepted": patch_accepted,
                "preserved_claims": transaction[
                    "preserved_generated_claim_count"
                ],
                "dependency_removed": transaction[
                    "dependency_removed_span_count"
                ],
                "dependency_rebound": transaction[
                    "dependency_rebound_count"
                ],
                "distant_edges": transaction["dependency_graph"][
                    "distant_edge_count"
                ],
                "bundle_pruned": transaction[
                    "bundle_pruned_claim_count"
                ],
                "obligation_protected": transaction[
                    "obligation_protected_claim_count"
                ],
            }
            expected = {
                key.removeprefix("expected_"): value
                for key, value in candidate.items()
                if key.startswith("expected_")
            }
            expectation_met = bool(
                observed == expected
                and transaction["dependency_graph"][
                    "coherence_passed"
                ]
                and transaction["claim_bundle_selection"][
                    "coherence_passed"
                ]
            )
            cases.append(
                {
                    "line_no": row["line_no"],
                    "case": candidate["case"],
                    "expected": expected,
                    "observed": observed,
                    "expectation_met": expectation_met,
                    "raw_accepted": raw_audit["accepted"],
                    "patched_answer": patched_answer,
                    "transaction": transaction,
                    "patched_audit": patched_audit,
                }
            )

    result = {
        "agent_version": "vsr-pathrag-v25",
        "protocol": "controlled_evidence_conditioned_bundle",
        "input": args.input,
        "accepted_correction_row_count": len(accepted_rows),
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": bool(cases)
        and all(row["expectation_met"] for row in cases),
        "transaction_triggered_count": sum(
            bool(row["observed"]["triggered"]) for row in cases
        ),
        "transaction_accepted_count": sum(
            bool(row["observed"]["patch_accepted"])
            for row in cases
        ),
        "distant_dependency_count": sum(
            int(row["observed"]["distant_edges"])
            for row in cases
        ),
        "dependency_cascade_removed_count": sum(
            int(row["observed"]["dependency_removed"])
            for row in cases
        ),
        "dependency_rebound_count": sum(
            int(row["observed"]["dependency_rebound"])
            for row in cases
        ),
        "bundle_pruned_claim_count": sum(
            int(row["observed"]["bundle_pruned"])
            for row in cases
        ),
        "obligation_protected_claim_count": sum(
            int(row["observed"]["obligation_protected"])
            for row in cases
        ),
        "full_audit_rejected_patch_count": sum(
            bool(
                row["observed"]["triggered"]
                and not row["observed"]["patch_accepted"]
            )
            for row in cases
        ),
        "all_transactions_coherent": all(
            row["transaction"]["dependency_graph"][
                "coherence_passed"
            ]
            and row["transaction"]["claim_bundle_selection"][
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
        f"distant={result['distant_dependency_count']} "
        f"cascade={result['dependency_cascade_removed_count']} "
        f"rebound={result['dependency_rebound_count']} "
        f"pruned={result['bundle_pruned_claim_count']} "
        f"protected={result['obligation_protected_claim_count']} "
        f"rejected={result['full_audit_rejected_patch_count']} "
        f"coherent={result['all_transactions_coherent']} "
        f"all={result['all_expectations_met']}"
    )


if __name__ == "__main__":
    main()
