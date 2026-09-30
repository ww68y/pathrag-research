import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "datasets" / "UltraDomain"


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_source_line_no"] = line_no
            rows.append(item)
    return rows


def build_subset(domain: str, sample_count: int) -> tuple[list[dict], list[dict]]:
    rows = load_jsonl(SOURCE_DIR / f"{domain}.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[context_hash(row.get("context", ""))].append(row)

    group_items = []
    for ctx_hash, samples in groups.items():
        context = samples[0].get("context", "")
        source_lines = [sample["_source_line_no"] for sample in samples]
        group_items.append((len(context), ctx_hash, samples, source_lines))
    group_items.sort(key=lambda item: (item[0], item[1]))

    selected = []
    manifest_rows = []
    for context_len, ctx_hash, samples, source_lines in group_items:
        if len(selected) >= sample_count:
            break
        remaining = sample_count - len(selected)
        chosen = samples[:remaining]
        for sample in chosen:
            item = dict(sample)
            item["dataset"] = domain
            item["source_dataset"] = domain
            item["source_line_no"] = sample["_source_line_no"]
            item["source_context_hash"] = ctx_hash
            item["source_context_chars"] = context_len
            item["window_start"] = 0
            item["window_end"] = context_len
            item["window_chars"] = context_len
            item["window_reason"] = "full_context_shortest_source_contexts"
            item.pop("_source_line_no", None)
            selected.append(item)
        manifest_rows.append(
            {
                "domain": domain,
                "context_hash": ctx_hash,
                "source_context_chars": context_len,
                "selected_source_lines": [sample["_source_line_no"] for sample in chosen],
                "available_source_lines": source_lines,
            }
        )
    if len(selected) != sample_count:
        raise RuntimeError(f"Expected {sample_count} samples, got {len(selected)}")
    for i, sample in enumerate(selected):
        sample["_subset_line_no"] = i
    return selected, manifest_rows


def write_manifest(path: Path, dataset_path: Path, domain: str, sample_count: int, manifest_rows: list[dict]) -> None:
    lines = [
        f"# {domain.title()} Full-Context {sample_count} Manifest\n\n",
        f"- Output dataset: `{dataset_path.relative_to(ROOT_DIR).as_posix()}`\n",
        f"- Domain: `{domain}`\n",
        f"- Samples: {sample_count}\n",
        "- Selection: shortest unique source contexts first, preserving full context\n",
        "\n",
        "This subset keeps the original questions, answers, and full source contexts. It is more expensive than the windowed subset but avoids answer loss from context truncation.\n\n",
        "| Context Hash | Source Context Chars | Selected Lines | Available Lines |\n",
        "|---|---:|---|---|\n",
    ]
    for row in manifest_rows:
        lines.append(
            f"| {row['context_hash']} | {row['source_context_chars']} | "
            f"{row['selected_source_lines']} | {row['available_source_lines']} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    output = ROOT_DIR / (
        args.output
        or f"datasets/UltraDomain/{args.domain}_fullcontext_{args.sample_count}.jsonl"
    )
    manifest = ROOT_DIR / (
        args.manifest
        or f"results/{args.domain}_fullcontext_{args.sample_count}_manifest.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    selected, manifest_rows = build_subset(args.domain, args.sample_count)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    write_manifest(manifest, output, args.domain, args.sample_count, manifest_rows)
    unique_contexts = len({row["context_hash"] for row in manifest_rows})
    total_context_chars = sum(row["source_context_chars"] for row in manifest_rows)
    print(f"Saved {len(selected)} samples to {output}")
    print(f"Saved manifest to {manifest}")
    print(f"Unique contexts: {unique_contexts}")
    print(f"Total unique context chars: {total_context_chars}")


if __name__ == "__main__":
    main()
