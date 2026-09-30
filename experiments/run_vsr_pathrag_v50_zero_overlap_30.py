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
    ROOT_DIR / "results" / "v50_zero_overlap_empty_experience.jsonl"
)
os.environ["VSR_COUNTEREVIDENCE_COST_MEMORY_PATH"] = str(
    ROOT_DIR / "results" / "v50_zero_overlap_empty_cost_memory.jsonl"
)
os.environ["VSR_COUNTEREVIDENCE_INFORMATION_MEMORY_PATH"] = str(
    ROOT_DIR
    / "results"
    / "v50_zero_overlap_empty_information_memory.jsonl"
)

import experiments.run_vsr_pathrag_v6_crossdomain_suite as suite


suite.base.DATASET_PATH = (
    ROOT_DIR
    / "datasets"
    / "UltraDomain"
    / "v46_zero_overlap_validation_30.jsonl"
)
suite.base.OUT_JSONL = (
    ROOT_DIR / "results" / "vsr_pathrag_v50_zero_overlap_30.jsonl"
)
suite.base.OUT_SUMMARY = (
    ROOT_DIR
    / "results"
    / "vsr_pathrag_v50_zero_overlap_30_summary.md"
)
suite.base.SUMMARY_TITLE = (
    "# VSR-PathRAG v50 Frozen Zero-Overlap Five-Domain Suite"
)
suite.base.SAMPLE_TIMEOUT_SECONDS = 600
suite.base.REQUESTED_MODE = "agent"
suite.base.RESUME_EXISTING = True

_summarize = suite.base.summarize


def summarize(rows: list[dict]) -> str:
    return (
        _summarize(rows)
        .replace(
            "VSR-PathRAG v6 Claim-Level Verification Metrics",
            "VSR-PathRAG v50 Frozen Verification Metrics",
        )
        + "\n"
        + "\n".join(
            [
                "## Frozen Protocol",
                "",
                "- Agent version: `vsr-pathrag-v50`",
                "- Stage: `frozen_pre_ablation`",
                "- Runtime reference answers: not used",
                "- Evaluation memory initialization: empty and read-only",
                "- Component removal: none",
                "- Ablation: false",
                "",
            ]
        )
    )


suite.base.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(suite.base.main())
