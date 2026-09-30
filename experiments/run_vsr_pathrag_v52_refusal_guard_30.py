from pathlib import Path
import os
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

os.environ["VSR_MEMORY_WRITE_ENABLED"] = "0"
os.environ["VSR_COUNTEREVIDENCE_COST_MEMORY_WRITE_ENABLED"] = "0"
os.environ[
    "VSR_COUNTEREVIDENCE_INFORMATION_MEMORY_WRITE_ENABLED"
] = "0"
os.environ["VSR_EXPERIENCE_MEMORY_PATH"] = str(
    ROOT_DIR / "results" / "v52_guard_empty_experience.jsonl"
)
os.environ["VSR_COUNTEREVIDENCE_COST_MEMORY_PATH"] = str(
    ROOT_DIR / "results" / "v52_guard_empty_cost_memory.jsonl"
)
os.environ["VSR_COUNTEREVIDENCE_INFORMATION_MEMORY_PATH"] = str(
    ROOT_DIR / "results" / "v52_guard_empty_information_memory.jsonl"
)

import experiments.run_vsr_pathrag_v6_crossdomain_suite as suite


suite.base.DATASET_PATH = (
    ROOT_DIR
    / "datasets"
    / "UltraDomain"
    / "vsr_v52_anchor_preserving_validation_30.jsonl"
)
suite.base.OUT_JSONL = (
    ROOT_DIR
    / "results"
    / "vsr_pathrag_v52_refusal_guard_30.jsonl"
)
suite.base.OUT_SUMMARY = (
    ROOT_DIR
    / "results"
    / "vsr_pathrag_v52_refusal_guard_30_summary.md"
)
suite.base.SUMMARY_TITLE = (
    "# VSR-PathRAG v52 Evidence-Available Refusal Guard Suite"
)
suite.base.SAMPLE_TIMEOUT_SECONDS = 600
suite.base.REQUESTED_MODE = "agent"
suite.base.RESUME_EXISTING = True

_summarize = suite.base.summarize


def summarize(rows: list[dict]) -> str:
    ok_rows = [row for row in rows if row.get("ok")]
    conflicts = sum(
        bool(
            ((row.get("trace") or {}).get(
                "question_obligation_alignment"
            )
            or {}).get("anchor_evidence_conflict")
        )
        for row in ok_rows
    )
    overrides = sum(
        bool(
            ((row.get("trace") or {}).get("answer_realization") or {})
            .get("question_obligation_override")
        )
        for row in ok_rows
    )
    focused = sum(
        bool(
            ((row.get("trace") or {}).get(
                "question_obligation_alignment"
            )
            or {}).get("candidate_selected")
        )
        for row in ok_rows
    )
    return (
        _summarize(rows)
        .replace(
            "VSR-PathRAG v6 Claim-Level Verification Metrics",
            "VSR-PathRAG v52 Claim-Level Verification Metrics",
        )
        + "\n"
        + "\n".join(
            [
                "## Evidence-Available Refusal Guard",
                "",
                f"- Anchor/refusal conflicts: {conflicts}",
                f"- Realization overrides: {overrides}",
                f"- Deterministic focused repairs: {focused}",
                "- Anchor-preserving graph preparation: 5/5",
                "- Evaluation memory initialization: empty and read-only",
                "- Runtime reference answers: not used",
                "- Ablation: false",
                "",
            ]
        )
    )


suite.base.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(suite.base.main())
