from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v20_crossdomain_cached_suite as v20


v20.OUT_JSONL = ROOT_DIR / "results" / "interactive_agent_v24_facet_balanced_rescue_crossdomain_suite_12.jsonl"
v20.OUT_SUMMARY = ROOT_DIR / "results" / "interactive_agent_v24_facet_balanced_rescue_crossdomain_suite_12_summary.md"
v20.REQUESTED_MODE = "agent_fast"
v20.SAMPLE_TIMEOUT_SECONDS = 75
v20.SUMMARY_TITLE = "# Interactive Agent v24 Facet-Balanced Rescue Cross-Domain Suite"


if __name__ == "__main__":
    import asyncio

    asyncio.run(v20.main())
