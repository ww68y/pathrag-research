from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v12_quality_budget_suite as v12


v12.base.suite.OUT_JSONL = ROOT_DIR / "results" / "interactive_agent_v14_structured_control_suite_18.jsonl"
v12.base.suite.OUT_SUMMARY = ROOT_DIR / "results" / "interactive_agent_v14_structured_control_suite_18_summary.md"


def summarize(rows: list[dict]) -> str:
    return (
        v12.summarize(rows)
        .replace("# Interactive Agent v12 Quality-Budget Suite", "# Interactive Agent v14 Structured-Control Suite")
        .replace("v12", "v14")
    )


v12.base.suite.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(v12.base.suite.main())
