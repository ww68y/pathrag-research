import argparse
import hashlib
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def context_group(dataset_name: str, source: dict) -> str:
    identity = clean_text(source.get("source_context_hash")) or clean_text(source.get("context_id"))
    if not identity:
        context = clean_text(source.get("context", ""))
        identity = hashlib.sha1(context.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_name}:{identity}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--used-oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    candidates = load_jsonl(ROOT_DIR / args.candidates)
    sources = load_jsonl(ROOT_DIR / args.source)
    source_by_line = {
        int(source.get("_subset_line_no", index)): source for index, source in enumerate(sources)
    }
    used_groups = {
        row.get("group")
        for row in load_jsonl(ROOT_DIR / args.used_oof)
        if row.get("source_dataset") == args.dataset_name
    }

    selected = []
    for row in candidates:
        agent = row.get("contract_completion_agent") or {}
        if not agent.get("risk_candidate"):
            continue
        line_no = int(row.get("line_no", -1))
        group = context_group(args.dataset_name, source_by_line[line_no])
        if group in used_groups:
            continue
        selected.append({
            "line_no": line_no,
            "task_dataset": row.get("dataset", "unknown"),
            "context_group": group,
            "question": row.get("question", ""),
        })

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_counts: dict[str, int] = {}
    for row in selected:
        task = str(row["task_dataset"])
        task_counts[task] = task_counts.get(task, 0) + 1
    lines = [
        "# Frozen v32 Held-Out Contract Candidate Manifest\n\n",
        f"- Dataset family: `{args.dataset_name}`\n",
        f"- Risk candidates: {len(selected)}\n",
        f"- Unseen context groups: {len({row['context_group'] for row in selected})}\n",
        f"- Excluded training groups: {len(used_groups)}\n\n",
        "| Task | Candidates |\n",
        "|---|---:|\n",
    ]
    for task, count in sorted(task_counts.items()):
        lines.append(f"| {task} | {count} |\n")
    lines.extend([
        "\n## Selected Lines\n\n",
        ", ".join(str(row["line_no"]) for row in selected),
        "\n",
    ])
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(",".join(str(row["line_no"]) for row in selected))


if __name__ == "__main__":
    main()
