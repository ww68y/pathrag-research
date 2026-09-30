import json
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.build_vsr_pathrag_v54_pairwise_replay import (
    DIMENSIONS,
    load_jsonl,
)


def main() -> None:
    out = ROOT_DIR / "results"
    base_path = out / "vsr_pathrag_v56_vs_hyde_pairwise_replay.jsonl"
    changed_path = out / "vsr_pathrag_v57_vs_hyde_changed_pairwise.jsonl"
    result_path = out / "vsr_pathrag_v57_vs_hyde_pairwise_replay.jsonl"
    manifest_path = out / "vsr_pathrag_v57_vs_hyde_pairwise_replay_manifest.json"
    summary_path = out / "vsr_pathrag_v57_vs_hyde_pairwise_replay_summary.md"

    base = {int(row["line_no"]): row for row in load_jsonl(base_path)}
    changed = {int(row["line_no"]): row for row in load_jsonl(changed_path)}
    if set(base) != set(range(30)) or set(changed) != {26}:
        raise ValueError("v57 pairwise replay inputs are inconsistent")
    rows = [changed.get(line_no, base[line_no]) for line_no in range(30)]
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    counts = {}
    rates = {}
    for dimension in DIMENSIONS:
        counter = Counter(row["winners"][dimension] for row in rows)
        counts[dimension] = {
            "pathrag": counter["pathrag"],
            "baseline": counter["baseline"],
            "tie": counter["tie"],
        }
        rates[dimension] = round(
            (counter["pathrag"] + counter["tie"] * 0.5)
            / len(rows)
            * 100,
            2,
        )
    average = round(statistics.mean(rates.values()), 2)
    manifest = {
        "protocol": "changed_answer_pairwise_replay",
        "agent_version": "vsr-pathrag-v57",
        "sample_count": 30,
        "rejudged_line_nos": [26],
        "reused_v56_pairwise_count": 29,
        "counts": counts,
        "win_rates": rates,
        "average_pathrag_win_rate": average,
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
        "# VSR-PathRAG v57 vs HyDE Pairwise Replay",
        "",
        "| Dimension | PathRAG wins | HyDE wins | Ties | Win rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for dimension in DIMENSIONS:
        row = counts[dimension]
        lines.append(
            f"| {dimension} | {row['pathrag']} | {row['baseline']} | "
            f"{row['tie']} | {rates[dimension]:.2f}% |"
        )
    lines += [
        "",
        f"- Average PathRAG win rate: {average:.2f}%",
        "- Rejudged lines: [26]",
        "- Runtime reference answers: not used",
        "- Formal ablation: false",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"samples=30 rejudged=1 average_win_rate={average:.2f}%")


if __name__ == "__main__":
    main()
