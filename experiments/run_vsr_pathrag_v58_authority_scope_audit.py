import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    assess_question_obligation_alignment,
    build_secured_default_action_nucleus,
    verify_claims_against_evidence,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    rows = load_jsonl(
        ROOT_DIR / "results" / "vsr_pathrag_v57_selective_replay_30.jsonl"
    )
    cases = []
    for row in rows:
        evidence = dict(
            (row.get("trace") or {}).get("evidence_summary") or {}
        )
        question = str(row.get("question", ""))
        alignment = assess_question_obligation_alignment(
            question,
            str(row.get("answer", "")),
            evidence,
        )
        nucleus = build_secured_default_action_nucleus(
            question,
            evidence,
        )
        candidate = str(nucleus["answer"]) if nucleus else None
        verification = (
            verify_claims_against_evidence(
                question,
                candidate,
                evidence,
            )
            if candidate
            else None
        )
        eligible = bool(
            alignment.get("secured_default_action_repair_required")
            and candidate
            and int(
                (verification or {}).get("unsupported_count", 0)
                or 0
            )
            == 0
        )
        cases.append(
            {
                "line_no": int(row["line_no"]),
                "dataset": row.get("dataset"),
                "alignment": alignment,
                "nucleus": nucleus,
                "verification": verification,
                "eligible": eligible,
            }
        )
    triggers = [
        case
        for case in cases
        if case["alignment"].get(
            "secured_default_action_repair_required"
        )
    ]
    output = {
        "agent_version": "vsr-pathrag-v58",
        "protocol": "secured_default_authority_scope_audit",
        "sample_count": len(cases),
        "trigger_count": len(triggers),
        "eligible_count": sum(case["eligible"] for case in triggers),
        "triggered_line_nos": [
            case["line_no"] for case in triggers
        ],
        "reference_answer_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v58_authority_scope_audit.json"
    )
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"samples={len(cases)} triggers={len(triggers)} "
        f"eligible={output['eligible_count']} "
        f"lines={output['triggered_line_nos']}"
    )


if __name__ == "__main__":
    main()
