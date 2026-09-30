import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    audit_correction_source_certificates,
    audit_prohibited_claim_relapse,
    build_correction_persistence_constraints,
    skipped_correction_source_certificate_audit,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_committed_sources(
    row: dict,
    constraints: list[dict],
) -> dict[str, str]:
    store_path = (
        ROOT_DIR / row["working_dir"] / "kv_store_text_chunks.json"
    )
    chunk_store = json.loads(store_path.read_text(encoding="utf-8"))
    source_ids = {
        source_id
        for constraint in constraints
        for source_id in constraint.get("source_ids", [])
    }
    return {
        source_id: str(
            (chunk_store.get(source_id) or {}).get("content", "")
        )
        for source_id in source_ids
        if (chunk_store.get(source_id) or {}).get("content")
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
            "vsr_pathrag_v22_source_certificate_audit.json"
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
        committed_sources = load_committed_sources(row, constraints)
        correction_source_id = constraints[0]["source_ids"][0]
        supported_opposite = (
            "Adebanwi worked as a freelance reporter, writer, journalist "
            "and editor for many newspapers and magazines before he "
            "joined academia."
        )
        candidates = [
            {
                "case": "corrected_answer_bound_source",
                "answer": row["corrected_answer"],
                "evidence": committed_sources,
                "used_ids": [correction_source_id],
                "expected_pass": True,
                "expected_failure_gate": None,
                "expected_certificate_reason": (
                    "source_bound_correction_certified"
                ),
            },
            {
                "case": "pre_correction_draft",
                "answer": row["draft_answer"],
                "evidence": committed_sources,
                "used_ids": [correction_source_id],
                "expected_pass": False,
                "expected_failure_gate": "correction_persistence",
                "expected_certificate_reason": None,
            },
            {
                "case": "supported_opposite_bound_source",
                "answer": supported_opposite,
                "evidence": committed_sources,
                "used_ids": [correction_source_id],
                "expected_pass": True,
                "expected_failure_gate": None,
                "expected_certificate_reason": (
                    "source_bound_correction_certified"
                ),
            },
            {
                "case": "supported_opposite_unbound_citation",
                "answer": supported_opposite,
                "evidence": {
                    **committed_sources,
                    "ordinary-source": supported_opposite,
                },
                "used_ids": ["ordinary-source"],
                "expected_pass": False,
                "expected_failure_gate": "source_certificate",
                "expected_certificate_reason": (
                    "correction_source_not_cited"
                ),
            },
            {
                "case": "unsupported_opposite_bound_source",
                "answer": (
                    "Adebanwi worked as a television journalist during "
                    "a Mars expedition."
                ),
                "evidence": committed_sources,
                "used_ids": [correction_source_id],
                "expected_pass": False,
                "expected_failure_gate": "source_certificate",
                "expected_certificate_reason": (
                    "correction_source_support_below_threshold"
                ),
            },
            {
                "case": "distant_negative_recurrence",
                "answer": "His reporting career remains undocumented.",
                "evidence": committed_sources,
                "used_ids": [correction_source_id],
                "expected_pass": False,
                "expected_failure_gate": "correction_persistence",
                "expected_certificate_reason": None,
            },
            {
                "case": "unrelated_positive_claim",
                "answer": (
                    "Her medical career was extensively documented."
                ),
                "evidence": committed_sources,
                "used_ids": [correction_source_id],
                "expected_pass": True,
                "expected_failure_gate": None,
                "expected_certificate_reason": None,
            },
            {
                "case": "supported_opposite_source_unavailable",
                "answer": supported_opposite,
                "evidence": {},
                "used_ids": [correction_source_id],
                "expected_pass": False,
                "expected_failure_gate": "source_certificate",
                "expected_certificate_reason": (
                    "correction_source_unavailable"
                ),
            },
        ]
        for candidate in candidates:
            persistence = audit_prohibited_claim_relapse(
                candidate["answer"],
                constraints,
            )
            certificates = (
                audit_correction_source_certificates(
                    candidate["answer"],
                    constraints,
                    candidate["evidence"],
                    candidate["used_ids"],
                )
                if persistence["passed"]
                else skipped_correction_source_certificate_audit(
                    constraints,
                    "same_polarity_correction_relapse_already_rejected",
                )
            )
            observed_pass = bool(
                persistence["passed"] and certificates["passed"]
            )
            observed_failure_gate = (
                "correction_persistence"
                if not persistence["passed"]
                else "source_certificate"
                if not certificates["passed"]
                else None
            )
            observed_certificate_reason = (
                certificates["certificates"][0]["reason"]
                if certificates["certificates"]
                else None
            )
            expectation_met = bool(
                observed_pass == candidate["expected_pass"]
                and observed_failure_gate
                == candidate["expected_failure_gate"]
                and observed_certificate_reason
                == candidate["expected_certificate_reason"]
            )
            cases.append(
                {
                    "line_no": row["line_no"],
                    "case": candidate["case"],
                    "expected_pass": candidate["expected_pass"],
                    "observed_pass": observed_pass,
                    "expected_failure_gate": candidate[
                        "expected_failure_gate"
                    ],
                    "observed_failure_gate": observed_failure_gate,
                    "expected_certificate_reason": candidate[
                        "expected_certificate_reason"
                    ],
                    "observed_certificate_reason": (
                        observed_certificate_reason
                    ),
                    "expectation_met": expectation_met,
                    "correction_persistence": persistence,
                    "source_certificates": certificates,
                }
            )

    result = {
        "agent_version": "vsr-pathrag-v22",
        "protocol": "controlled_source_bound_correction_certificate",
        "input": args.input,
        "accepted_correction_row_count": len(accepted_rows),
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": bool(cases)
        and all(row["expectation_met"] for row in cases),
        "certificate_required_case_count": sum(
            bool(
                row["source_certificates"][
                    "certificate_required_count"
                ]
            )
            for row in cases
        ),
        "certified_case_count": sum(
            bool(row["source_certificates"]["certified_count"])
            for row in cases
        ),
        "uncertified_case_count": sum(
            bool(row["source_certificates"]["uncertified_count"])
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
        f"required={result['certificate_required_case_count']} "
        f"certified={result['certified_case_count']} "
        f"uncertified={result['uncertified_case_count']} "
        f"all={result['all_expectations_met']}"
    )


if __name__ == "__main__":
    main()
