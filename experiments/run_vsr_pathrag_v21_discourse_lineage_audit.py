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
            "vsr_pathrag_v21_discourse_lineage_audit.json"
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
        candidates = [
            {
                "case": "corrected_answer",
                "answer": row["corrected_answer"],
                "expected_pass": True,
                "expected_match_mode": None,
            },
            {
                "case": "pre_correction_draft",
                "answer": row["draft_answer"],
                "expected_pass": False,
                "expected_match_mode": "lexical_overlap",
            },
            {
                "case": "direct_same_polarity_recurrence",
                "answer": (
                    "Available information does not provide details about "
                    "his work as a journalist."
                ),
                "expected_pass": False,
                "expected_match_mode": "lexical_overlap",
            },
            {
                "case": "distant_implicit_absence_rewrite",
                "answer": "His reporting career remains undocumented.",
                "expected_pass": False,
                "expected_match_mode": "semantic_family",
            },
            {
                "case": "split_discourse_recurrence",
                "answer": (
                    "His career began in reporting. "
                    "The records remain silent on the specifics."
                ),
                "expected_pass": False,
                "expected_match_mode": "discourse_window",
            },
            {
                "case": "unrelated_absence_assertion",
                "answer": (
                    "The records remain silent on her medical history."
                ),
                "expected_pass": True,
                "expected_match_mode": None,
            },
            {
                "case": "opposite_polarity_form",
                "answer": (
                    "The source provides specific details about his "
                    "professional journalism work."
                ),
                "expected_pass": True,
                "expected_match_mode": None,
            },
        ]
        for candidate in candidates:
            audit = audit_prohibited_claim_relapse(
                candidate["answer"],
                constraints,
            )
            observed_mode = (
                audit["relapses"][0]["match_mode"]
                if audit["relapses"]
                else None
            )
            pass_met = audit["passed"] == candidate["expected_pass"]
            mode_met = (
                observed_mode == candidate["expected_match_mode"]
            )
            cases.append(
                {
                    "line_no": row["line_no"],
                    "case": candidate["case"],
                    "expected_pass": candidate["expected_pass"],
                    "observed_pass": audit["passed"],
                    "expected_match_mode": candidate[
                        "expected_match_mode"
                    ],
                    "observed_match_mode": observed_mode,
                    "expectation_met": pass_met and mode_met,
                    "audit": audit,
                }
            )

    result = {
        "agent_version": "vsr-pathrag-v21",
        "protocol": "controlled_discourse_lineage_audit",
        "input": args.input,
        "accepted_correction_row_count": len(accepted_rows),
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": bool(cases)
        and all(row["expectation_met"] for row in cases),
        "match_mode_counts": {
            mode: sum(
                row["observed_match_mode"] == mode for row in cases
            )
            for mode in [
                "lexical_overlap",
                "semantic_family",
                "discourse_window",
            ]
        },
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
        f"modes={result['match_mode_counts']} "
        f"all={result['all_expectations_met']}"
    )


if __name__ == "__main__":
    main()
