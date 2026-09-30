from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v9_cost_aware_contract_suite as base


base.suite.QUESTIONS = [
    "How did the Aurora battery system improve bus operating range and charging time?",
    "Which organizations were involved in the battery supply chain, and what role did each one play?",
    "Compare the roles of BlueLake Materials and SilverPeak Cells in the Aurora project.",
    "What software architecture did the Aurora system use for predictive maintenance?",
    "What percentage carbon-emission reduction did Sunvale University report?",
    "Which organization supplied the charging stations, and how many did it provide?",
    "What exact percentage reduction in carbon emissions and electricity consumption did the report provide?",
    "Which facts about Mark Chen, BlueLake Materials, and Sunvale Transit Authority are directly supported?",
]

base.suite.OUT_JSONL = ROOT_DIR / "results" / "interactive_agent_v10_relevance_filtered_target_suite_8.jsonl"
base.suite.OUT_SUMMARY = ROOT_DIR / "results" / "interactive_agent_v10_relevance_filtered_target_suite_8_summary.md"


_summarize_v9 = base.suite.summarize


def summarize(rows: list[dict]) -> str:
    return _summarize_v9(rows).replace(
        "# Interactive Agent v9 Cost-Aware Contract Router Suite",
        "# Interactive Agent v10 Relevance-Filtered Cost Router Target Suite",
    )


base.suite.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(base.suite.main())
