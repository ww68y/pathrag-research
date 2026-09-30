import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "datasets" / "UltraDomain"
OUTPUT_DATASET = SOURCE_DIR / "domain_expansion_windowed_50.jsonl"
OUTPUT_MANIFEST = ROOT_DIR / "results" / "domain_expansion_windowed_50_manifest.md"

DOMAINS = ["legal", "history", "biology", "agriculture", "physics"]
SAMPLES_PER_DOMAIN = 10
MAX_SAMPLES_PER_CONTEXT = 5
MAX_WINDOW_CHARS = 80_000
WINDOW_MARGIN_CHARS = 12_000


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


def answer_positions(context: str, samples: list[dict]) -> list[int]:
    positions = []
    lower_context = context.lower()
    for sample in samples:
        for answer in sample.get("answers", []):
            text = str(answer).strip()
            if not text:
                continue
            idx = lower_context.find(text.lower())
            if idx >= 0:
                positions.append(idx)
                break
    return positions


def make_window(context: str, samples: list[dict]) -> tuple[str, int, int, str]:
    positions = answer_positions(context, samples)
    if positions:
        start = max(0, min(positions) - WINDOW_MARGIN_CHARS)
        end = min(len(context), max(positions) + WINDOW_MARGIN_CHARS)
        if end - start > MAX_WINDOW_CHARS:
            center = (min(positions) + max(positions)) // 2
            start = max(0, center - MAX_WINDOW_CHARS // 2)
            end = min(len(context), start + MAX_WINDOW_CHARS)
            start = max(0, end - MAX_WINDOW_CHARS)
        reason = "answer_matched_window"
    else:
        start = 0
        end = min(len(context), MAX_WINDOW_CHARS)
        reason = "front_window_no_exact_answer_match"
    return context[start:end], start, end, reason


def select_domain_samples(domain: str) -> tuple[list[dict], list[dict]]:
    rows = load_jsonl(SOURCE_DIR / f"{domain}.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[context_hash(row.get("context", ""))].append(row)

    group_items = []
    for ctx_hash, samples in groups.items():
        context = samples[0].get("context", "")
        group_items.append((len(context), ctx_hash, samples))
    group_items.sort(key=lambda item: item[0])

    selected = []
    manifest_rows = []
    for context_len, ctx_hash, samples in group_items:
        if len(selected) >= SAMPLES_PER_DOMAIN:
            break
        take = min(MAX_SAMPLES_PER_CONTEXT, SAMPLES_PER_DOMAIN - len(selected), len(samples))
        chosen = samples[:take]
        window, start, end, reason = make_window(chosen[0].get("context", ""), chosen)
        for sample in chosen:
            item = dict(sample)
            item["context"] = window
            item["dataset"] = domain
            item["source_dataset"] = domain
            item["source_line_no"] = sample["_source_line_no"]
            item["source_context_hash"] = ctx_hash
            item["source_context_chars"] = context_len
            item["window_start"] = start
            item["window_end"] = end
            item["window_chars"] = len(window)
            item["window_reason"] = reason
            item.pop("_source_line_no", None)
            selected.append(item)
        manifest_rows.append(
            {
                "domain": domain,
                "context_hash": ctx_hash,
                "source_context_chars": context_len,
                "window_start": start,
                "window_end": end,
                "window_chars": len(window),
                "selected_source_lines": [sample["_source_line_no"] for sample in chosen],
                "reason": reason,
            }
        )
    return selected, manifest_rows


def main() -> None:
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    all_manifest = []
    for domain in DOMAINS:
        selected, manifest_rows = select_domain_samples(domain)
        if len(selected) != SAMPLES_PER_DOMAIN:
            raise RuntimeError(f"{domain}: expected {SAMPLES_PER_DOMAIN}, got {len(selected)}")
        all_samples.extend(selected)
        all_manifest.extend(manifest_rows)

    for i, sample in enumerate(all_samples):
        sample["_subset_line_no"] = i

    OUTPUT_DATASET.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in all_samples),
        encoding="utf-8",
    )

    lines = [
        "# Domain Expansion Windowed 50 Manifest\n\n",
        f"- Output dataset: `{OUTPUT_DATASET.relative_to(ROOT_DIR).as_posix()}`\n",
        f"- Domains: {', '.join(DOMAINS)}\n",
        f"- Samples per domain: {SAMPLES_PER_DOMAIN}\n",
        f"- Max samples per source context: {MAX_SAMPLES_PER_CONTEXT}\n",
        f"- Max window chars: {MAX_WINDOW_CHARS}\n",
        "\n",
        "This subset is a cost-controlled multi-domain expansion set. It keeps the original questions and answers, but uses an evidence-oriented context window instead of the full long document.\n\n",
        "| Domain | Context Hash | Source Context Chars | Window | Window Chars | Source Lines | Reason |\n",
        "|---|---|---:|---|---:|---|---|\n",
    ]
    for row in all_manifest:
        lines.append(
            f"| {row['domain']} | {row['context_hash']} | {row['source_context_chars']} | "
            f"{row['window_start']}-{row['window_end']} | {row['window_chars']} | "
            f"{row['selected_source_lines']} | {row['reason']} |\n"
        )
    OUTPUT_MANIFEST.write_text("".join(lines), encoding="utf-8")

    print(f"Saved {len(all_samples)} samples to {OUTPUT_DATASET}")
    print(f"Saved manifest to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
