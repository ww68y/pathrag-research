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
    ROOT_DIR / "results" / "v64_oyster_tech_empty_experience.jsonl"
)
os.environ["VSR_COUNTEREVIDENCE_COST_MEMORY_PATH"] = str(
    ROOT_DIR / "results" / "v64_oyster_tech_empty_cost_memory.jsonl"
)
os.environ["VSR_COUNTEREVIDENCE_INFORMATION_MEMORY_PATH"] = str(
    ROOT_DIR / "results" / "v64_oyster_tech_empty_information_memory.jsonl"
)

import experiments.run_vsr_pathrag_v6_crossdomain_suite as suite


suite.base.DATASET_PATH = (
    ROOT_DIR
    / "datasets"
    / "UltraDomain"
    / "vsr_v55_order_preserving_validation_30.jsonl"
)
suite.base.OUT_JSONL = (
    ROOT_DIR / "results" / "vsr_pathrag_v64_oyster_technology_case.jsonl"
)
suite.base.OUT_SUMMARY = (
    ROOT_DIR
    / "results"
    / "vsr_pathrag_v64_oyster_technology_case_summary.md"
)
suite.base.SUMMARY_TITLE = (
    "# VSR-PathRAG v64 Technology-to-Market Impact Case"
)
suite.base.SAMPLE_TIMEOUT_SECONDS = 600
suite.base.REQUESTED_MODE = "agent"
suite.base.RESUME_EXISTING = False

_load_jsonl = suite.base.load_jsonl


def load_selected(path: Path) -> list[dict]:
    return [
        row
        for row in _load_jsonl(path)
        if int(row["_line_no"]) == 3
    ]


suite.base.load_jsonl = load_selected


if __name__ == "__main__":
    import asyncio

    asyncio.run(suite.base.main())
