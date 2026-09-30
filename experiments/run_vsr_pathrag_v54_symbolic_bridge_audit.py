import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    assess_question_obligation_alignment,
    build_symbolic_theme_bridge,
    verify_claims_against_evidence,
)


def main() -> None:
    input_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v53_selective_replay_30.jsonl"
    )
    output_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v54_symbolic_bridge_audit.json"
    )
    with input_path.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]

    cases = []
    for row in rows:
        trace = row.get("trace") or {}
        evidence = trace.get("evidence_summary") or {}
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        alignment = assess_question_obligation_alignment(
            question,
            answer,
            evidence,
        )
        bridge = build_symbolic_theme_bridge(question, evidence)
        candidate = str(bridge["answer"]) if bridge else None
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
            alignment.get("symbolic_theme_bridge_repair_required")
            and candidate
            and int(
                (verification or {}).get("unsupported_count", 0)
                or 0
            )
            == 0
            and len(candidate) < len(answer)
        )
        cases.append(
            {
                "line_no": int(row["line_no"]),
                "dataset": row.get("dataset"),
                "alignment": alignment,
                "bridge": bridge,
                "candidate": candidate,
                "verification": verification,
                "eligible": eligible,
            }
        )
    triggers = [
        case
        for case in cases
        if case["alignment"].get(
            "symbolic_theme_bridge_repair_required"
        )
    ]
    result = {
        "agent_version": "vsr-pathrag-v54",
        "protocol": "symbolic_theme_bridge_audit",
        "sample_count": len(cases),
        "trigger_count": len(triggers),
        "eligible_count": sum(
            bool(case["eligible"]) for case in triggers
        ),
        "triggered_line_nos": [
            case["line_no"] for case in triggers
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
        f"samples={len(cases)} triggers={len(triggers)} "
        f"eligible={result['eligible_count']} "
        f"lines={result['triggered_line_nos']}"
    )


if __name__ == "__main__":
    main()
