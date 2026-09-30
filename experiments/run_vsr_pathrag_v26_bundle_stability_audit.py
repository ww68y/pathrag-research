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
    store = json.loads(
        (
            ROOT_DIR
            / row["working_dir"]
            / "kv_store_text_chunks.json"
        ).read_text(encoding="utf-8")
    )
    return str((store.get(source_id) or {}).get("content", ""))


def audit_answer(
    question: str,
    answer: str,
    source_id: str,
    source_text: str,
    constraints: list[dict],
    obligations: list[dict],
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
            "vsr_pathrag_v26_bundle_stability_audit.json"
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
        unsupported_other = (
            "Zyphron calibrated quuxium vortices beneath Nemerax."
        )
        marginal_supported = (
            "Newspapers and magazines accompanied zephyr quuxium vortex "
            "lunar cobalt amber prism nebula."
        )
        unsupported_number = (
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
                "case": "stable_distant_cascade",
                "answer": (
                    f"{fallback} {unsupported_inversion} "
                    f"{safe_improvement} {unsupported_dependency}"
                ),
                "obligations": [],
                "accepted": True,
                "distant": 1,
                "cascade": 1,
                "rebound": 0,
                "bundle_pruned": 0,
                "stability_pruned": 0,
                "protected": 0,
            },
            {
                "case": "stable_distant_rebind",
                "answer": (
                    f"{unsupported_inversion} {safe_improvement} "
                    f"{supported_dependency}"
                ),
                "obligations": [],
                "accepted": True,
                "distant": 1,
                "cascade": 0,
                "rebound": 1,
                "bundle_pruned": 0,
                "stability_pruned": 0,
                "protected": 0,
            },
            {
                "case": "marginal_selection_fails_perturbation",
                "answer": (
                    f"{unsupported_inversion} {marginal_supported}"
                ),
                "obligations": [],
                "accepted": True,
                "distant": 0,
                "cascade": 0,
                "rebound": 0,
                "bundle_pruned": 0,
                "stability_pruned": 1,
                "protected": 0,
            },
            {
                "case": "independent_hallucination_fails_baseline_bundle",
                "answer": (
                    f"{unsupported_inversion} {unsupported_other}"
                ),
                "obligations": [],
                "accepted": True,
                "distant": 0,
                "cascade": 0,
                "rebound": 0,
                "bundle_pruned": 1,
                "stability_pruned": 0,
                "protected": 0,
            },
            {
                "case": "source_bound_obligation_remains_stable",
                "answer": (
                    f"{fallback} {unsupported_inversion} "
                    f"{safe_improvement}"
                ),
                "obligations": [obligation],
                "accepted": True,
                "distant": 0,
                "cascade": 0,
                "rebound": 0,
                "bundle_pruned": 0,
                "stability_pruned": 0,
                "protected": 0,
            },
            {
                "case": "numeric_audit_rejects_stable_bundle",
                "answer": (
                    f"{unsupported_inversion} {unsupported_number}"
                ),
                "obligations": [],
                "accepted": False,
                "distant": 0,
                "cascade": 0,
                "rebound": 0,
                "bundle_pruned": 0,
                "stability_pruned": 0,
                "protected": 0,
            },
            {
                "case": "certified_generation_needs_no_transaction",
                "answer": f"{safe_improvement} {fallback}",
                "obligations": [],
                "accepted": False,
                "distant": 0,
                "cascade": 0,
                "rebound": 0,
                "bundle_pruned": 0,
                "stability_pruned": 0,
                "protected": 0,
                "triggered": False,
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
                obligations,
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
                    enable_counterfactual_stability=True,
                )
            )
            patched_audit = (
                audit_answer(
                    question,
                    patched_answer,
                    source_id,
                    source_text,
                    constraints,
                    obligations,
                )
                if transaction["generated_answer_changed"]
                else None
            )
            observed = {
                "triggered": transaction["triggered"],
                "accepted": bool(
                    patched_audit and patched_audit["accepted"]
                ),
                "distant": transaction["dependency_graph"][
                    "distant_edge_count"
                ],
                "cascade": transaction[
                    "dependency_removed_span_count"
                ],
                "rebound": transaction[
                    "dependency_rebound_count"
                ],
                "bundle_pruned": transaction[
                    "bundle_pruned_claim_count"
                ],
                "stability_pruned": transaction[
                    "stability_pruned_claim_count"
                ],
                "protected": transaction[
                    "stability_fallback_protected_count"
                ],
            }
            expected = {
                "triggered": candidate.get("triggered", True),
                **{
                    key: candidate[key]
                    for key in [
                        "accepted",
                        "distant",
                        "cascade",
                        "rebound",
                        "bundle_pruned",
                        "stability_pruned",
                        "protected",
                    ]
                },
            }
            certificate = transaction[
                "counterfactual_stability_certificate"
            ]
            expectation_met = bool(
                observed == expected
                and transaction["dependency_graph"][
                    "coherence_passed"
                ]
                and certificate["passed"]
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
        "agent_version": "vsr-pathrag-v26",
        "protocol": "controlled_counterfactual_bundle_stability",
        "input": args.input,
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
            bool(row["observed"]["accepted"]) for row in cases
        ),
        "distant_dependency_count": sum(
            int(row["observed"]["distant"]) for row in cases
        ),
        "bundle_pruned_claim_count": sum(
            int(row["observed"]["bundle_pruned"])
            for row in cases
        ),
        "stability_pruned_claim_count": sum(
            int(row["observed"]["stability_pruned"])
            for row in cases
        ),
        "stability_fallback_protected_count": sum(
            int(row["observed"]["protected"]) for row in cases
        ),
        "full_audit_rejected_patch_count": sum(
            bool(
                row["observed"]["triggered"]
                and not row["observed"]["accepted"]
            )
            for row in cases
        ),
        "all_transactions_coherent": all(
            row["transaction"]["dependency_graph"][
                "coherence_passed"
            ]
            and row["transaction"][
                "counterfactual_stability_certificate"
            ]["passed"]
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
        f"cases={result['case_count']} "
        f"met={result['expectation_met_count']} "
        f"triggered={result['transaction_triggered_count']} "
        f"accepted={result['transaction_accepted_count']} "
        f"distant={result['distant_dependency_count']} "
        f"bundle_pruned={result['bundle_pruned_claim_count']} "
        f"stability_pruned={result['stability_pruned_claim_count']} "
        f"protected={result['stability_fallback_protected_count']} "
        f"rejected={result['full_audit_rejected_patch_count']} "
        f"coherent={result['all_transactions_coherent']} "
        f"all={result['all_expectations_met']}"
    )


if __name__ == "__main__":
    main()
