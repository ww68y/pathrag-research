import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS = ["correctness", "coverage", "relevance", "conciseness", "overall"]


def clean_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_source(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--source must use name:results_jsonl:eval_jsonl")
    name, result_path, eval_path = parts
    return name, ROOT_DIR / result_path, ROOT_DIR / eval_path


def load_sources(specs: list[str]) -> dict[tuple[int, str], dict]:
    by_prediction = {}
    for spec in specs:
        name, result_path, eval_path = parse_source(spec)
        result_rows = {row.get("line_no"): row for row in load_jsonl(result_path)}
        eval_rows = {row.get("line_no"): row for row in load_jsonl(eval_path)}
        for line_no, result in result_rows.items():
            scores = eval_rows.get(line_no, {}).get("scores")
            if not scores:
                continue
            key = (line_no, clean_text(result.get("prediction", "")))
            by_prediction[key] = {
                "source": name,
                "scores": scores,
            }
    return by_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--source", action="append", required=True)
    args = parser.parse_args()

    source_by_prediction = load_sources(args.source)
    routed_rows = load_jsonl(ROOT_DIR / args.routed)

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluated = []
    missing = []
    with output_path.open("w", encoding="utf-8") as out:
        for row in routed_rows:
            key = (row.get("line_no"), clean_text(row.get("prediction", "")))
            source_eval = source_by_prediction.get(key)
            if not source_eval:
                missing.append(row.get("line_no"))
                continue
            item = {
                "line_no": row.get("line_no"),
                "dataset": row.get("dataset"),
                "question": row.get("question"),
                "scores": source_eval["scores"],
                "derived_from": source_eval["source"],
            }
            evaluated.append(item)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    metric_sums = defaultdict(float)
    verdict_counts = Counter()
    source_counts = Counter()
    for item in evaluated:
        scores = item["scores"]
        for metric in METRICS:
            metric_sums[metric] += float(scores.get(metric, 0))
        verdict_counts[scores.get("verdict", "unknown")] += 1
        source_counts[item["derived_from"]] += 1

    n = max(1, len(evaluated))
    lines = [
        "# Routed Evaluation Summary\n\n",
        f"- Source results: `{args.routed}`\n",
        f"- Evaluated samples: {len(evaluated)}\n",
        f"- Missing unmatched samples: {len(missing)}\n",
        "- Judge: reused existing per-sample judgments only when the routed prediction exactly matched an evaluated source prediction\n",
        "- Scale: 1-5, higher is better\n\n",
        "## Average Scores\n\n",
        "| Metric | Score |\n",
        "|---|---:|\n",
    ]
    for metric in METRICS:
        lines.append(f"| {metric} | {metric_sums[metric] / n:.4g} |\n")

    lines.extend([
        "\n## Derived Source Counts\n\n",
        "| Source | Count |\n",
        "|---|---:|\n",
    ])
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |\n")

    lines.extend([
        "\n## Verdict Counts\n\n",
        "| Verdict | Count |\n",
        "|---|---:|\n",
    ])
    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f"| {verdict} | {count} |\n")

    if missing:
        lines.extend(["\n## Missing Lines\n\n", ", ".join(str(line) for line in missing), "\n"])

    lines.extend([
        "\n## Per-Sample Scores\n\n",
        "| Line | Dataset | Source | Correctness | Coverage | Relevance | Conciseness | Overall | Verdict | Reason |\n",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|\n",
    ])
    for item in evaluated:
        scores = item["scores"]
        lines.append(
            f"| {item.get('line_no')} | {item.get('dataset')} | {item.get('derived_from')} | "
            f"{scores.get('correctness')} | {scores.get('coverage')} | {scores.get('relevance')} | "
            f"{scores.get('conciseness')} | {scores.get('overall')} | {scores.get('verdict')} | "
            f"{clean_text(scores.get('short_reason', ''))} |\n"
        )

    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved routed eval to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
