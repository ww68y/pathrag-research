import argparse
import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


UNCERTAINTY_PATTERNS = [
    r"\bcannot be determined\b",
    r"\bcan not be determined\b",
    r"\bcannot determine\b",
    r"\bnot specified\b",
    r"\bdoes not specify\b",
    r"\bdoes not contain\b",
    r"\bdoes not mention\b",
    r"\bno specific information\b",
    r"\bno information\b",
    r"\bnot provide\b",
    r"\bnot provided\b",
    r"\blacks information\b",
]


def clean_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_uncertainty(answer: str) -> tuple[bool, list[str]]:
    lowered = clean_text(answer).lower()
    matched = [pattern for pattern in UNCERTAINTY_PATTERNS if re.search(pattern, lowered)]
    return bool(matched), matched


def should_fallback(base: dict, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons = []
    uncertain, patterns = has_uncertainty(base.get("prediction", ""))
    if uncertain:
        reasons.append("uncertainty_phrase:" + ",".join(patterns[:3]))

    stats = base.get("mse_stats") or {}
    if args.use_low_coverage and float(stats.get("term_coverage", 1.0)) < args.low_coverage_threshold:
        reasons.append(f"low_term_coverage:{stats.get('term_coverage')}")

    if args.use_sufficiency and stats.get("sufficiency_status") == "repair_needed":
        reasons.append("repair_needed_after_selection")

    return bool(reasons), reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--fallback-name", default="fallback")
    parser.add_argument("--use-low-coverage", action="store_true")
    parser.add_argument("--low-coverage-threshold", type=float, default=0.50)
    parser.add_argument("--use-sufficiency", action="store_true")
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT_DIR / args.base)
    fallback_by_line = {item.get("line_no"): item for item in load_jsonl(ROOT_DIR / args.fallback)}

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_rows = []
    reports = []
    with output_path.open("w", encoding="utf-8") as out:
        for base in base_rows:
            use_fallback, reasons = should_fallback(base, args)
            fallback = fallback_by_line.get(base.get("line_no"))
            if use_fallback and fallback is not None and clean_text(fallback.get("prediction", "")):
                merged = dict(base)
                merged["original_prediction"] = base.get("prediction", "")
                merged["prediction"] = fallback.get("prediction", "")
                merged["method"] = f"{base.get('method', 'base')}_uncertainty_fallback"
                action = f"use_{args.fallback_name}"
            else:
                merged = dict(base)
                action = "keep_base"
                if use_fallback and fallback is None:
                    reasons = reasons + ["missing_fallback_row"]
            merged["uncertainty_fallback"] = {
                "enabled": True,
                "action": action,
                "fallback_name": args.fallback_name,
                "reasons": reasons,
                "base_answer_chars": len(base.get("prediction", "")),
                "final_answer_chars": len(merged.get("prediction", "")),
            }
            merged_rows.append(merged)
            reports.append((base.get("line_no"), merged["uncertainty_fallback"]))
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    used = sum(1 for _, report in reports if report.get("action", "").startswith("use_"))
    avg_chars = sum(len(item.get("prediction", "")) for item in merged_rows) / max(1, len(merged_rows))
    lines = [
        "# Uncertainty Fallback Summary\n\n",
        f"- Base: `{args.base}`\n",
        f"- Fallback: `{args.fallback}`\n",
        f"- Output: `{args.output}`\n",
        f"- Samples: {len(merged_rows)}\n",
        f"- Fallback-used samples: {used}\n",
        f"- Avg answer chars: {avg_chars:.1f}\n\n",
        "| Line | Action | Reasons | Base chars | Final chars |\n",
        "|---:|---|---|---:|---:|\n",
    ]
    for line_no, report in reports:
        lines.append(
            f"| {line_no} | {report.get('action')} | "
            f"{', '.join(report.get('reasons', [])) or 'none'} | "
            f"{report.get('base_answer_chars')} | {report.get('final_answer_chars')} |\n"
        )
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved fallback results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
