import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    assess_question_obligation_alignment,
    build_ordered_proceeds_nucleus,
    verify_claims_against_evidence,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    results = load_jsonl(
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v54_selective_replay_30.jsonl"
    )
    dataset = load_jsonl(
        ROOT_DIR
        / "datasets"
        / "UltraDomain"
        / "vsr_v55_order_preserving_validation_30.jsonl"
    )
    if len(results) != len(dataset):
        raise ValueError("result and dataset row counts differ")

    cases = []
    for row, sample in zip(results, dataset):
        line_no = int(row["line_no"])
        if line_no != int(sample["_subset_line_no"]):
            raise ValueError(f"line mismatch at {line_no}")
        trace = row.get("trace") or {}
        evidence = dict(trace.get("evidence_summary") or {})
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        alignment = assess_question_obligation_alignment(
            question,
            answer,
            evidence,
        )
        candidate_evidence = dict(evidence)
        candidate_evidence["source_excerpts"] = [
            {
                "source_id": (
                    f"v55-context-line-{line_no}"
                ),
                "text": str(sample["context"]),
            }
        ]
        nucleus = build_ordered_proceeds_nucleus(
            question,
            candidate_evidence,
        )
        candidate = str(nucleus["answer"]) if nucleus else None
        verification = (
            verify_claims_against_evidence(
                question,
                candidate,
                candidate_evidence,
            )
            if candidate
            else None
        )
        eligible = bool(
            alignment.get("ordered_proceeds_incomplete")
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
        if case["alignment"].get("ordered_proceeds_incomplete")
    ]
    output = {
        "agent_version": "vsr-pathrag-v55",
        "protocol": "ordered_proceeds_completeness_audit",
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
        / "vsr_pathrag_v55_ordered_proceeds_audit.json"
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
