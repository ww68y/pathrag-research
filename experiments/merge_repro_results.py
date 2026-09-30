import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def status_of(item: dict) -> str:
    prediction = item.get("prediction", "")
    if prediction and not prediction.startswith("Sorry, I'm not able"):
        return "ok"
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prefix", default="ultradomain_mix")
    parser.add_argument("--output", default="results/historical_datasets/ultradomain_mix_repro_current.jsonl")
    parser.add_argument("--summary", default="results/historical_reports/repro_current_summary.md")
    args = parser.parse_args()

    results_dir = ROOT_DIR / args.results_dir
    output_path = ROOT_DIR / args.output
    summary_path = ROOT_DIR / args.summary

    by_line: dict[int, dict] = {}
    candidates = []
    for path in results_dir.glob(f"{args.prefix}*.jsonl"):
        if path.resolve() == output_path.resolve():
            continue
        if "eval" in path.name:
            continue
        candidates.append(path)

    for path in sorted(candidates, key=lambda p: p.stat().st_mtime):
        for item in load_jsonl(path):
            if "line_no" in item and "prediction" in item:
                by_line[int(item["line_no"])] = item

    items = [by_line[line_no] for line_no in sorted(by_line)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )

    unique_contexts = {}
    for item in items:
        key = item.get("context_hash") or item.get("working_dir")
        unique_contexts[key] = item.get("insert_seconds", 0)

    lines = [
        "# Current PathRAG UltraDomain Mix Reproduction Summary\n\n",
        "| Line | Dataset | Nodes | Edges | Insert s | Query s | Status |\n",
        "|---:|---|---:|---:|---:|---:|---|\n",
    ]
    for item in items:
        lines.append(
            f"| {item['line_no']} | {item.get('dataset')} | "
            f"{item.get('graph_nodes')} | {item.get('graph_edges')} | "
            f"{item.get('insert_seconds')} | {item.get('query_seconds')} | "
            f"{status_of(item)} |\n"
        )

    total_query = sum(float(item.get("query_seconds", 0)) for item in items)
    total_index = sum(float(value) for value in unique_contexts.values())
    lines.extend(
        [
            "\n",
            f"- Samples: {len(items)}\n",
            f"- Unique indexed contexts: {len(unique_contexts)}\n",
            f"- Approx total indexing seconds: {round(total_index, 3)}\n",
            f"- Total query seconds: {round(total_query, 3)}\n",
            f"- Average query seconds: {round(total_query / len(items), 3) if items else 0}\n",
            f"- Average nodes: {round(sum(float(item.get('graph_nodes', 0)) for item in items) / len(items), 2) if items else 0}\n",
            f"- Average edges: {round(sum(float(item.get('graph_edges', 0)) for item in items) / len(items), 2) if items else 0}\n",
        ]
    )
    summary_path.write_text("".join(lines), encoding="utf-8")

    print(f"Merged {len(items)} samples from {len(candidates)} files")
    print(f"Saved results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
