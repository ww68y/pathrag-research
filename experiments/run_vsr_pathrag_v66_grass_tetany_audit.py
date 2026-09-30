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
    rows = load_jsonl(
        ROOT_DIR / "results" / "vsr_pathrag_v65_selective_replay_30.jsonl"
    )
    cases = []
    for row in rows:
        trace = row.get("trace") or {}
        audit = assess_question_obligation_alignment(
            str(row.get("question", "")),
            str(row.get("answer", "")),
            dict(trace.get("evidence_summary") or {}),
        )
        cases.append(
            {
                "line_no": int(row["line_no"]),
                "triggered": bool(
                    audit.get("grass_tetany_repair_required")
                ),
                "missing": audit.get("grass_tetany_missing", []),
            }
        )
    real = load_jsonl(
        ROOT_DIR / "results" / "vsr_pathrag_v66_grass_tetany_case.jsonl"
    )[0]
    trace = real["trace"]
    alignment = trace["question_obligation_alignment"]
    final_audit = assess_question_obligation_alignment(
        str(real.get("question", "")),
        str(real.get("answer", "")),
        dict(trace.get("evidence_summary") or {}),
    )
    output = {
        "agent_version": "vsr-pathrag-v66",
        "protocol": "observation_to_management_causality_audit",
        "sample_count": len(cases),
        "trigger_count": sum(case["triggered"] for case in cases),
        "triggered_line_nos": [
            case["line_no"] for case in cases if case["triggered"]
        ],
        "real_case": {
            "line_no": int(real["line_no"]),
            "candidate_selected": alignment["candidate_selected"],
            "candidate_source_count": alignment[
                "candidate_source_count"
            ],
            "repair": alignment["grass_tetany_repair"],
            "supported_claims": trace["claim_verification"][
                "supported_count"
            ],
            "unsupported_claims": trace["claim_verification"][
                "unsupported_count"
            ],
            "signed_statuses": trace["claim_evidence_graph"][
                "status_counts"
            ],
            "detected_pre_repair_missing": alignment[
                "grass_tetany_missing"
            ],
            "post_repair_missing": final_audit[
                "grass_tetany_missing"
            ],
        },
        "reference_answer_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    path = (
        ROOT_DIR / "results" / "vsr_pathrag_v66_grass_tetany_audit.json"
    )
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"triggers={output['trigger_count']} "
        f"lines={output['triggered_line_nos']} "
        f"selected={output['real_case']['candidate_selected']}"
    )


if __name__ == "__main__":
    main()
