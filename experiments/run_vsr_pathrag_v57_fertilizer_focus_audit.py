import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    assess_question_obligation_alignment,
    build_fertilizer_influence_nucleus,
    verify_claims_against_evidence,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    results = load_jsonl(
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v56_selective_replay_30.jsonl"
    )
    cases = []
    for row in results:
        line_no = int(row["line_no"])
        evidence = dict(
            (row.get("trace") or {}).get("evidence_summary") or {}
        )
        question = str(row.get("question", ""))
        alignment = assess_question_obligation_alignment(
            question,
            str(row.get("answer", "")),
            evidence,
        )
        nucleus = build_fertilizer_influence_nucleus(
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
            alignment.get("fertilizer_influence_repair_required")
            and candidate
            and int(
                (verification or {}).get("unsupported_count", 0)
                or 0
            )
            == 0
            and len(candidate) <= 900
        )
        cases.append(
            {
                "line_no": line_no,
                "dataset": row.get("dataset"),
                "alignment": alignment,
                "nucleus": nucleus,
                "candidate": candidate,
                "verification": verification,
                "eligible": eligible,
            }
        )

    triggers = [
        case
        for case in cases
        if case["alignment"].get(
            "fertilizer_influence_repair_required"
        )
    ]
    output = {
        "agent_version": "vsr-pathrag-v57",
        "protocol": "fertilizer_main_effect_focus_audit",
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
    output_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v57_fertilizer_focus_audit.json"
    )
    output_path.write_text(
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
