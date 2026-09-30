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
        ROOT_DIR / "results" / "vsr_pathrag_v60_selective_replay_30.jsonl"
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
                    audit.get("sexuality_bridge_repair_required")
                ),
                "missing": audit.get("sexuality_bridge_missing", []),
                "unrelated_branch": audit.get(
                    "sexuality_bridge_unrelated_branch", False
                ),
            }
        )
    real = load_jsonl(
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v61_sexuality_bridge_case.jsonl"
    )[0]
    trace = real["trace"]
    alignment = trace["question_obligation_alignment"]
    output = {
        "agent_version": "vsr-pathrag-v61",
        "protocol": "sexuality_culture_art_bridge_audit",
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
            "repair": alignment["sexuality_bridge_repair"],
            "supported_claims": trace["claim_verification"][
                "supported_count"
            ],
            "unsupported_claims": trace["claim_verification"][
                "unsupported_count"
            ],
            "contradicted_claims": trace["claim_evidence_graph"][
                "status_counts"
            ].get("contradicted", 0),
            "contains_unrelated_opera_branch": (
                "sydney opera house" in real["answer"].lower()
            ),
        },
        "claim_polarity_scope": "claim_local_sentence_window",
        "reference_answer_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v61_sexuality_bridge_audit.json"
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
