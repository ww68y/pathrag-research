from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import experiments.run_interactive_agent_v12_quality_budget_suite as v12


v12.base.suite.QUESTIONS = [
    "Did Northwind Logistics supply the lithium iron phosphate battery cells? Answer only with supported facts.",
    "Did Sunvale University report a 25 percent carbon-emission reduction for the Aurora system?",
    "Which company built the predictive-maintenance neural network used by the Aurora system?",
    "What was the annual budget of the Aurora Battery Pilot Project?",
    "How many buses failed after GreenVolt Energy replaced the faulty cooling valve?",
    "Did Mark Chen and Dr. Lin Mei both work for GreenVolt Energy? Answer only with supported facts.",
    "Which organization became the primary battery-cell supplier after August 2025?",
    "Did BlueLake Materials provide the charging stations for the pilot?",
    "What exact charging time reduction did the Aurora system achieve?",
    "Which facts about Northwind Logistics, Riverstone Components, and BlueLake Materials are directly supported?",
    "Compare the roles of Riverstone Components and SilverPeak Cells in the project.",
    "What evidence supports the claim that the Aurora system reduced hallucinations in large language models?",
]

v12.base.suite.OUT_JSONL = ROOT_DIR / "results" / "interactive_agent_v19_counterfactual_gap_stress_suite_12.jsonl"
v12.base.suite.OUT_SUMMARY = ROOT_DIR / "results" / "interactive_agent_v19_counterfactual_gap_stress_suite_12_summary.md"


def summarize(rows: list[dict]) -> str:
    return (
        v12.summarize(rows)
        .replace("# Interactive Agent v12 Quality-Budget Suite", "# Interactive Agent v19 Counterfactual-Gap Stress Suite")
        .replace("v12", "v19")
    )


v12.base.suite.summarize = summarize


if __name__ == "__main__":
    import asyncio

    asyncio.run(v12.base.suite.main())
