import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT_DIR / "datasets" / "UltraDomain" / "mix.jsonl"
OUTPUT_PATH = ROOT_DIR / "datasets" / "UltraDomain" / "vsr_v6_cached_scale_30.jsonl"
MANIFEST_PATH = ROOT_DIR / "results" / "vsr_pathrag_v6_cached_scale_30_manifest.md"
TARGET_COUNTS = {
    "2wikimqa": 6,
    "hotpotqa": 5,
    "multifieldqa_en": 5,
    "musique": 6,
    "narrativeqa": 3,
    "qasper": 5,
}


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def has_cached_graph(value: str) -> bool:
    return any(
        (candidate / "graph_chunk_entity_relation.graphml").exists()
        for candidate in (ROOT_DIR / "data_repro").glob(f"ctx_*_{value}")
    )


def main() -> None:
    rows = []
    with SOURCE_PATH.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_no"] = line_no
            rows.append(row)

    datasets = sorted({str(row.get("dataset") or row.get("label")) for row in rows})
    if set(datasets) != set(TARGET_COUNTS):
        raise RuntimeError(f"Unexpected datasets: {datasets}")
    selected = []
    counts = Counter()
    seen_contexts: dict[str, set[str]] = {dataset: set() for dataset in datasets}
    for row in rows:
        dataset = str(row.get("dataset") or row.get("label"))
        if counts[dataset] >= TARGET_COUNTS[dataset]:
            continue
        value = context_hash(row.get("context", ""))
        if value in seen_contexts[dataset] or not has_cached_graph(value):
            continue
        item = dict(row)
        item["source_context_hash"] = value
        item["scale_subset_policy"] = "first_five_unique_cached_contexts_per_dataset"
        selected.append(item)
        seen_contexts[dataset].add(value)
        counts[dataset] += 1

    missing = {
        dataset: TARGET_COUNTS[dataset] - counts[dataset]
        for dataset in datasets
        if counts[dataset] < TARGET_COUNTS[dataset]
    }
    if missing:
        raise RuntimeError(f"Insufficient cached contexts: {missing}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = [
        "# VSR-PathRAG v6 Cached Scale-30 Manifest",
        "",
        f"- Source: `{SOURCE_PATH.relative_to(ROOT_DIR)}`",
        f"- Output: `{OUTPUT_PATH.relative_to(ROOT_DIR)}`",
        f"- Samples: {len(selected)}",
        f"- Dataset counts: {dict(counts)}",
        f"- Target counts: {TARGET_COUNTS}",
        "- Selection: first rows per target dataset with distinct cached graph contexts.",
        "- Reference answers are not used for selection.",
        "- Purpose: cross-task scale and stability evaluation, not a zero-overlap claim.",
        "",
        "| Subset line | Source line | Dataset | Context hash | Question |",
        "|---:|---:|---|---|---|",
    ]
    for index, row in enumerate(selected):
        question = str(row.get("input") or row.get("question") or "").replace("|", "/")
        manifest.append(
            f"| {index} | {row['_source_line_no']} | {row.get('dataset') or row.get('label')} | "
            f"`{row['source_context_hash']}` | {question} |"
        )
    MANIFEST_PATH.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} samples to {OUTPUT_PATH}")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
