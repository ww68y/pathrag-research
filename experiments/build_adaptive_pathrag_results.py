import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


UNCERTAINTY_PATTERNS = [
    "not explicitly stated",
    "not explicitly provided",
    "not provided",
    "not mentioned",
    "no mention",
    "no evidence",
    "insufficient information",
    "insufficient evidence",
    "cannot be determined",
    "cannot determine",
    "cannot answer",
    "can't answer",
    "i cannot answer",
    "i can't answer",
    "not possible to determine",
    "without additional",
    "without further",
    "does not contain enough",
    "not enough evidence",
    "not enough information",
    "sorry, i'm not able",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def should_fallback(prediction: str) -> tuple[bool, str]:
    lower = prediction.lower()
    for pattern in UNCERTAINTY_PATTERNS:
        if pattern in lower:
            return True, pattern
    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathrag", default="results/historical_datasets/ultradomain_mix_repro_current.jsonl")
    parser.add_argument("--baseline", default="results/historical_baselines/direct_context_baseline_current.jsonl")
    parser.add_argument("--fallback-source", default=None)
    parser.add_argument("--output", default="results/historical_baselines/adaptive_pathrag_current.jsonl")
    parser.add_argument("--summary", default="results/historical_baselines/adaptive_pathrag_summary.md")
    args = parser.parse_args()

    pathrag_items = load_jsonl(ROOT_DIR / args.pathrag)
    fallback_source = args.fallback_source or Path(args.baseline).stem
    baseline_by_line = {
        int(item["line_no"]): item for item in load_jsonl(ROOT_DIR / args.baseline)
    }

    output_items = []
    fallback_count = 0
    fallback_lines = []
    for item in pathrag_items:
        line_no = int(item["line_no"])
        fallback, reason = should_fallback(item.get("prediction", ""))
        merged = dict(item)
        merged["method"] = "adaptive_pathrag"
        merged["original_prediction"] = item.get("prediction", "")
        merged["fallback_used"] = False
        merged["fallback_reason"] = ""

        if fallback and line_no in baseline_by_line:
            baseline_item = baseline_by_line[line_no]
            merged["prediction"] = baseline_item.get("prediction", "")
            merged["fallback_used"] = True
            merged["fallback_reason"] = reason
            merged["fallback_source"] = fallback_source
            fallback_count += 1
            fallback_lines.append(line_no)

        output_items.append(merged)

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output_items),
        encoding="utf-8",
    )

    summary_lines = [
        "# Adaptive PathRAG Summary\n\n",
        f"- Source PathRAG samples: {len(pathrag_items)}\n",
        f"- Output samples: {len(output_items)}\n",
        f"- Fallback samples: {fallback_count}\n",
        f"- Fallback rate: {fallback_count / len(output_items):.2%}\n",
        f"- Fallback lines: {fallback_lines}\n",
        "\n",
        "Rule: keep the PathRAG answer unless it contains explicit uncertainty phrases; ",
        f"then replace it with the `{fallback_source}` answer for the same question.\n",
    ]
    (ROOT_DIR / args.summary).write_text("".join(summary_lines), encoding="utf-8")

    print(f"Saved adaptive results to {output_path}")
    print(f"Fallback samples: {fallback_count}/{len(output_items)}")


if __name__ == "__main__":
    main()
