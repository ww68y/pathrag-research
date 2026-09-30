from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v6_claim_guard_suite as suite

suite.OUT_JSONL = ROOT_DIR / "results" / "interactive_agent_v7_evidence_gap_claim_guard_suite_18.jsonl"
suite.OUT_SUMMARY = ROOT_DIR / "results" / "interactive_agent_v7_evidence_gap_claim_guard_suite_18_summary.md"

_summarize_v6 = suite.summarize


def summarize(rows: list[dict]) -> str:
    return _summarize_v6(rows).replace(
        "# Interactive Agent v6 Claim Guard Suite",
        "# Interactive Agent v7 Evidence-Gap Claim Guard Suite",
    )


suite.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(suite.main())
