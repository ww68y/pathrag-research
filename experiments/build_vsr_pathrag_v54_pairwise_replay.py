import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DIMENSIONS = (
    "comprehensiveness",
    "diversity",
    "logicality",
    "relevance",
    "coherence",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    results_dir = ROOT_DIR / "results"
    base_path = (
        results_dir
        / "vsr_pathrag_v52_vs_hyde_anchor_pairwise.jsonl"
    )
    changed_path = (
        results_dir
        / "vsr_pathrag_v54_vs_hyde_changed_pairwise.jsonl"
    )
    output_path = (
        results_dir
        / "vsr_pathrag_v54_vs_hyde_pairwise_replay.jsonl"
    )
    summary_path = (
        results_dir
        / "vsr_pathrag_v54_vs_hyde_pairwise_replay_summary.md"
    )
    manifest_path = (
        results_dir
        / "vsr_pathrag_v54_vs_hyde_pairwise_replay_manifest.json"
    )

    base = {int(row["line_no"]): row for row in load_jsonl(base_path)}
    changed = {
        int(row["line_no"]): row for row in load_jsonl(changed_path)
    }
    if set(base) != set(range(30)):
        raise ValueError("base pairwise evaluation is incomplete")
    if set(changed) != {19, 29}:
        raise ValueError("changed pairwise rows must be 19 and 29")
    merged = [changed.get(line_no, base[line_no]) for line_no in range(30)]
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in merged
        ),
        encoding="utf-8",
    )

    counts = {}
    rates = {}
    for dimension in DIMENSIONS:
        counter = Counter(
            row["winners"][dimension] for row in merged
        )
        counts[dimension] = {
            "pathrag": counter["pathrag"],
            "baseline": counter["baseline"],
            "tie": counter["tie"],
        }
        rates[dimension] = round(
            (
                counter["pathrag"]
                + counter["tie"] * 0.5
            )
            / len(merged)
            * 100,
            2,
        )
    average_rate = round(statistics.mean(rates.values()), 2)
    manifest = {
        "protocol": "changed_answer_pairwise_replay",
        "agent_version": "vsr-pathrag-v54",
        "sample_count": 30,
        "rejudged_line_nos": sorted(changed),
        "reused_v52_pairwise_count": 28,
        "counts": counts,
        "win_rates": rates,
        "average_pathrag_win_rate": average_rate,
        "runtime_reference_answer_used": False,
        "judge_reference_used": True,
        "full_pairwise_rerun": False,
        "ablation": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# VSR-PathRAG v54 vs HyDE Pairwise Replay",
        "",
        "Rows 19 and 29 were independently rejudged because their "
        "answers changed after v52. The other 28 completed pairwise "
        "judgments were reused.",
        "",
        "| Dimension | PathRAG wins | HyDE wins | Ties | Win rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for dimension in DIMENSIONS:
        row = counts[dimension]
        lines.append(
            f"| {dimension} | {row['pathrag']} | "
            f"{row['baseline']} | {row['tie']} | "
            f"{rates[dimension]:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"- Average PathRAG win rate: {average_rate:.2f}%",
            "- Rejudged lines: [19, 29]",
            "- Reused pairwise judgments: 28",
            "- Runtime reference answers: not used",
            "- Formal ablation: false",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"samples=30 rejudged=2 average_win_rate={average_rate:.2f}%"
    )


if __name__ == "__main__":
    main()
