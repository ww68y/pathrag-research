import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    audit_prohibited_claim_relapse,
    build_correction_persistence_constraints,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
            "vsr_pathrag_v20_correction_persistence_audit.json"
        ),
    )
    args = parser.parse_args()

    input_path = ROOT_DIR / args.input
    accepted_rows = [
        row
        for row in load_jsonl(input_path)
        if (row.get("diagnosis_answer_correction") or {}).get("accepted")
    ]
    cases = []
    for row in accepted_rows:
        constraints = build_correction_persistence_constraints(
            row["diagnosis_answer_correction"]
        )
        candidates = [
            (
                "corrected_answer",
                row["corrected_answer"],
                True,
            ),
            (
                "pre_correction_draft",
                row["draft_answer"],
                False,
            ),
            (
                "same_polarity_semantic_recurrence",
                (
                    "Available information does not provide details about "
                    "his work as a journalist."
                ),
                False,
            ),
            (
                "opposite_polarity_form",
                (
                    "The source provides specific details about his "
                    "professional journalism work."
                ),
                True,
            ),
        ]
        for label, candidate, expected_pass in candidates:
            audit = audit_prohibited_claim_relapse(
                candidate,
                constraints,
            )
            cases.append(
                {
                    "line_no": row["line_no"],
                    "case": label,
                    "expected_pass": expected_pass,
                    "observed_pass": audit["passed"],
                    "expectation_met": audit["passed"] == expected_pass,
                    "audit": audit,
                }
            )

    result = {
        "agent_version": "vsr-pathrag-v20",
        "protocol": "controlled_correction_persistence_audit",
        "input": args.input,
        "accepted_correction_row_count": len(accepted_rows),
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": bool(cases)
        and all(row["expectation_met"] for row in cases),
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
        f"all={result['all_expectations_met']}"
    )


if __name__ == "__main__":
    main()
