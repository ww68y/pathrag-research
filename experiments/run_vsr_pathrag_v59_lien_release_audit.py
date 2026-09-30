import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    assess_question_obligation_alignment,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    base_rows = load_jsonl(
        ROOT_DIR / "results" / "vsr_pathrag_v58_selective_replay_30.jsonl"
    )
    real_case = load_jsonl(
        ROOT_DIR / "results" / "vsr_pathrag_v59_lien_release_case.jsonl"
    )[0]
    cases = []
    for row in base_rows:
        evidence = dict(
            (row.get("trace") or {}).get("evidence_summary") or {}
        )
        alignment = assess_question_obligation_alignment(
            str(row.get("question", "")),
            str(row.get("answer", "")),
            evidence,
        )
        cases.append(
            {
                "line_no": int(row["line_no"]),
                "dataset": row.get("dataset"),
                "triggered": bool(
                    alignment.get(
                        "lien_release_conditions_repair_required"
                    )
                ),
                "missing": alignment.get(
                    "lien_release_conditions_missing", []
                ),
            }
        )
    real_trace = real_case["trace"]
    real_alignment = real_trace["question_obligation_alignment"]
    output = {
        "agent_version": "vsr-pathrag-v59",
        "protocol": "lien_release_condition_matrix_audit",
        "sample_count": len(cases),
        "trigger_count": sum(case["triggered"] for case in cases),
        "triggered_line_nos": [
            case["line_no"] for case in cases if case["triggered"]
        ],
        "real_case": {
            "line_no": int(real_case["line_no"]),
            "candidate_selected": real_alignment["candidate_selected"],
            "candidate_source_count": real_alignment[
                "candidate_source_count"
            ],
            "repair": real_alignment["lien_release_conditions_repair"],
            "supported_claims": real_trace["claim_verification"][
                "supported_count"
            ],
            "unsupported_claims": real_trace["claim_verification"][
                "unsupported_count"
            ],
            "contradicted_claims": real_trace["claim_evidence_graph"][
                "status_counts"
            ].get("contradicted", 0),
        },
        "reference_answer_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v59_lien_release_condition_matrix_audit.json"
    )
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"samples={len(cases)} triggers={output['trigger_count']} "
        f"lines={output['triggered_line_nos']} "
        f"selected={output['real_case']['candidate_selected']} "
        f"claims={output['real_case']['supported_claims']}/"
        f"{output['real_case']['supported_claims'] + output['real_case']['unsupported_claims']}"
    )


if __name__ == "__main__":
    main()
