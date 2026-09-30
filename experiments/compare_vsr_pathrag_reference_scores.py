import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "correctness",
    "coverage",
    "relevance",
    "conciseness",
    "overall",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(rows, key=lambda row: int(row["line_no"]))


def validate_alignment(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    baseline_name: str,
) -> None:
    if len(candidate) != len(baseline):
        raise ValueError(
            f"row count mismatch for {baseline_name}: "
            f"{len(candidate)} != {len(baseline)}"
        )
    for candidate_row, baseline_row in zip(candidate, baseline):
        identity = ("line_no", "dataset", "question")
        mismatches = [
            key
            for key in identity
            if candidate_row.get(key) != baseline_row.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"unaligned {baseline_name} row "
                f"{candidate_row.get('line_no')}: {mismatches}"
            )


def metric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: round(
            statistics.mean(
                float(row["scores"][metric]) for row in rows
            ),
            3,
        )
        for metric in METRICS
    }


def paired_comparison(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    comparison = {}
    for metric in METRICS:
        deltas = [
            float(candidate_row["scores"][metric])
            - float(baseline_row["scores"][metric])
            for candidate_row, baseline_row in zip(
                candidate,
                baseline,
            )
        ]
        comparison[metric] = {
            "mean_delta": round(statistics.mean(deltas), 3),
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
        }
    return comparison


def domain_means(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(
            float(row["scores"]["overall"])
        )
    return {
        domain: round(statistics.mean(values), 3)
        for domain, values in grouped.items()
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# VSR-PathRAG v51 Reference-Score Comparison",
        "",
        "This is a paired offline comparison of completed, independently "
        "judged reference scores. It is not an LLM pairwise preference test "
        "and it is not an ablation.",
        "",
        "## Mean Scores",
        "",
        "| Method | Correctness | Coverage | Relevance | Conciseness | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("v51", "v50", "hyde"):
        means = result["methods"][name]["means"]
        lines.append(
            f"| {name} | {means['correctness']:.3f} | "
            f"{means['coverage']:.3f} | {means['relevance']:.3f} | "
            f"{means['conciseness']:.3f} | {means['overall']:.3f} |"
        )
    for baseline in ("v50", "hyde"):
        lines.extend(
            [
                "",
                f"## v51 vs {baseline}",
                "",
                "| Metric | Mean delta | Wins | Ties | Losses |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in METRICS:
            item = result["comparisons"][baseline][metric]
            lines.append(
                f"| {metric} | {item['mean_delta']:+.3f} | "
                f"{item['wins']} | {item['ties']} | "
                f"{item['losses']} |"
            )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Samples: {result['sample_count']}",
            "- Runtime reference answers: not used.",
            "- Pairwise LLM judge: false.",
            "- Formal ablation: false.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v51",
        default="results/historical_vsr/vsr_pathrag_v51_zero_overlap_30_eval.jsonl",
    )
    parser.add_argument(
        "--v50",
        default="results/historical_vsr/vsr_pathrag_v50_zero_overlap_30_eval.jsonl",
    )
    parser.add_argument(
        "--hyde",
        default="results/historical_baselines/v46_heldout30_hyde_eval.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "vsr_pathrag_v51_reference_score_comparison.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        default=(
            "results/"
            "vsr_pathrag_v51_reference_score_comparison.md"
        ),
    )
    args = parser.parse_args()

    paths = {
        "v51": Path(args.v51),
        "v50": Path(args.v50),
        "hyde": Path(args.hyde),
    }
    rows = {name: load_rows(path) for name, path in paths.items()}
    for baseline_name in ("v50", "hyde"):
        validate_alignment(
            rows["v51"],
            rows[baseline_name],
            baseline_name,
        )

    result = {
        "protocol": "paired_reference_score_comparison",
        "sample_count": len(rows["v51"]),
        "methods": {
            name: {
                "source": str(paths[name]),
                "means": metric_means(method_rows),
                "domain_overall_means": domain_means(method_rows),
            }
            for name, method_rows in rows.items()
        },
        "comparisons": {
            baseline_name: paired_comparison(
                rows["v51"],
                rows[baseline_name],
            )
            for baseline_name in ("v50", "hyde")
        },
        "runtime_reference_answer_used": False,
        "pairwise_llm_judge": False,
        "ablation": False,
    }

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    markdown_path = Path(args.markdown_output)
    markdown_path.write_text(
        markdown_report(result),
        encoding="utf-8",
    )
    print(
        f"samples={result['sample_count']} "
        f"v51={result['methods']['v51']['means']['overall']:.3f} "
        f"v50={result['methods']['v50']['means']['overall']:.3f} "
        f"hyde={result['methods']['hyde']['means']['overall']:.3f}"
    )


if __name__ == "__main__":
    main()
