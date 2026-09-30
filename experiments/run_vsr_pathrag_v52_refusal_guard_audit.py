import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    assess_question_obligation_alignment,
    build_question_obligation_focused_answer,
    verify_claims_against_evidence,
)


def main() -> None:
    input_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v51_anchor_preserving_30.jsonl"
    )
    output_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v52_refusal_guard_audit.json"
    )
    rows = [
        json.loads(line)
        for line in input_path.open(encoding="utf-8")
        if line.strip()
    ]
    cases = []
    for row in rows:
        trace = row.get("trace") or {}
        evidence_summary = trace.get("evidence_summary") or {}
        answer = str(row.get("answer", ""))
        question = str(row.get("question", ""))
        alignment = assess_question_obligation_alignment(
            question,
            answer,
            evidence_summary,
        )
        focused_answer = None
        focused_verification = None
        eligible = False
        if alignment.get("anchor_evidence_conflict"):
            focused_answer = build_question_obligation_focused_answer(
                question,
                evidence_summary,
            )
            if focused_answer:
                focused_verification = verify_claims_against_evidence(
                    question,
                    focused_answer,
                    evidence_summary,
                )
                eligible = bool(
                    int(
                        focused_verification.get(
                            "unsupported_count",
                            0,
                        )
                        or 0
                    )
                    == 0
                    and len(focused_answer) < len(answer)
                )
        cases.append(
            {
                "line_no": int(row["line_no"]),
                "dataset": row.get("dataset"),
                "alignment": alignment,
                "focused_answer": focused_answer,
                "focused_verification": focused_verification,
                "eligible": eligible,
            }
        )

    conflicts = [
        case
        for case in cases
        if case["alignment"].get("anchor_evidence_conflict")
    ]
    result = {
        "agent_version": "vsr-pathrag-v52",
        "protocol": "evidence_available_refusal_guard_audit",
        "sample_count": len(cases),
        "refusal_count": sum(
            bool(case["alignment"].get("refusal_like"))
            for case in cases
        ),
        "anchor_evidence_conflict_count": len(conflicts),
        "eligible_focused_repair_count": sum(
            bool(case["eligible"]) for case in conflicts
        ),
        "conflict_line_nos": [
            case["line_no"] for case in conflicts
        ],
        "reference_answer_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"samples={result['sample_count']} "
        f"refusals={result['refusal_count']} "
        f"conflicts={result['anchor_evidence_conflict_count']} "
        f"eligible={result['eligible_focused_repair_count']} "
        f"lines={result['conflict_line_nos']}"
    )


if __name__ == "__main__":
    main()
