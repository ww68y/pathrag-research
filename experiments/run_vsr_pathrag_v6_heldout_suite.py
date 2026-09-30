from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_vsr_pathrag_v6_crossdomain_suite as suite


suite.base.DATASET_PATH = (
    ROOT_DIR / "datasets" / "UltraDomain" / "crossdomain_multiquery_heldout_12.jsonl"
)
suite.base.OUT_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v6_heldout_suite_12.jsonl"
suite.base.OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v6_heldout_suite_12_summary.md"
suite.base.SUMMARY_TITLE = "# VSR-PathRAG v6 Claim-Verifying Held-Out Suite"


if __name__ == "__main__":
    import asyncio

    asyncio.run(suite.base.main())

