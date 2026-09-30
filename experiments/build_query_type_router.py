import argparse
import json
import re
from collections import Counter
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

LEGAL_CUES = {
    "agreement",
    "contract",
    "clause",
    "section",
    "provision",
    "provisions",
    "party",
    "parties",
    "purchaser",
    "seller",
    "guarantor",
    "guarantors",
    "indemnification",
    "indemnified",
    "indemnifying",
    "termination",
    "confidentiality",
    "securities",
    "notes",
    "warrants",
    "conversion",
    "interest",
    "maturity",
    "liquidity",
    "conditions",
    "precedent",
    "controlled",
    "equity",
    "rights",
    "obligations",
}

LOOKUP_CUES = {
    "what",
    "who",
    "when",
    "which",
    "where",
    "list",
    "identify",
    "state",
    "describe",
    "how much",
}

GRAPH_REASONING_CUES = {
    "relationship",
    "relationships",
    "path",
    "paths",
    "connect",
    "connected",
    "between",
    "cause",
    "causal",
    "impact",
    "affect",
    "compare",
    "contrast",
    "multi-hop",
    "network",
}


def clean_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_question(item: dict) -> str:
    return clean_text(item.get("question") or item.get("input") or item.get("query") or "")


def is_uncertain(answer: str) -> tuple[bool, list[str]]:
    lowered = clean_text(answer).lower()
    matched = [pattern for pattern in UNCERTAINTY_PATTERNS if re.search(pattern, lowered)]
    return bool(matched), matched


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower()))


def classify_question(item: dict) -> tuple[str, list[str]]:
    question = get_question(item)
    lowered = question.lower()
    tokens = token_set(question)
    domain = clean_text(item.get("dataset") or item.get("label") or "").lower()

    reasons = []
    legal_hits = sorted(tokens & LEGAL_CUES)
    if domain == "legal":
        reasons.append("domain:legal")
    if legal_hits:
        reasons.append("legal_cues:" + ",".join(legal_hits[:5]))

    lookup_hit = False
    for cue in LOOKUP_CUES:
        if cue in lowered:
            lookup_hit = True
            reasons.append("lookup_cue:" + cue)
            break

    graph_hits = sorted(tokens & GRAPH_REASONING_CUES)
    if graph_hits:
        reasons.append("graph_reasoning_cues:" + ",".join(graph_hits[:5]))

    is_legal = domain == "legal" or bool(legal_hits)
    if is_legal and lookup_hit and not graph_hits:
        return "legal_clause_lookup", reasons
    if is_legal and lookup_hit:
        return "legal_lookup_with_relation_terms", reasons
    if graph_hits:
        return "graph_reasoning", reasons
    if lookup_hit:
        return "general_lookup", reasons
    return "general_reasoning", reasons


def should_route_to_fallback(base: dict, args: argparse.Namespace) -> tuple[bool, str, list[str]]:
    question_type, reasons = classify_question(base)

    if args.route_legal and question_type in {"legal_clause_lookup", "legal_lookup_with_relation_terms"}:
        return True, question_type, reasons + ["route:legal_lookup_to_fallback"]

    uncertain, patterns = is_uncertain(base.get("prediction", ""))
    if args.route_uncertain and uncertain:
        return True, question_type, reasons + ["route:uncertainty_to_fallback", "uncertainty:" + ",".join(patterns[:3])]

    stats = base.get("mse_stats") or {}
    term_coverage = float(stats.get("term_coverage", 1.0) or 1.0)
    if args.route_low_coverage and term_coverage < args.low_coverage_threshold:
        return True, question_type, reasons + [f"route:low_coverage:{term_coverage:.4f}"]

    return False, question_type, reasons + ["route:keep_graph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--rescue")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--fallback-name", default="fallback")
    parser.add_argument("--rescue-name", default="rescue")
    parser.add_argument("--route-legal", action="store_true")
    parser.add_argument("--route-uncertain", action="store_true")
    parser.add_argument("--route-low-coverage", action="store_true")
    parser.add_argument("--low-coverage-threshold", type=float, default=0.50)
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT_DIR / args.base)
    fallback_by_line = {item.get("line_no"): item for item in load_jsonl(ROOT_DIR / args.fallback)}
    rescue_by_line = {}
    if args.rescue:
        rescue_by_line = {item.get("line_no"): item for item in load_jsonl(ROOT_DIR / args.rescue)}

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_rows = []
    reports = []
    with output_path.open("w", encoding="utf-8") as out:
        for base in base_rows:
            use_fallback, question_type, reasons = should_route_to_fallback(base, args)
            fallback = fallback_by_line.get(base.get("line_no"))
            fallback_prediction = clean_text(fallback.get("prediction", "")) if fallback else ""
            rescue = rescue_by_line.get(base.get("line_no"))
            rescue_prediction = clean_text(rescue.get("prediction", "")) if rescue else ""

            if use_fallback and fallback_prediction:
                merged = dict(base)
                merged["graph_prediction_before_router"] = base.get("prediction", "")
                merged.setdefault("original_prediction", base.get("prediction", ""))
                merged["prediction"] = fallback.get("prediction", "")
                merged["method"] = f"{base.get('method', 'base')}_query_router"
                action = f"use_{args.fallback_name}"
            else:
                merged = dict(base)
                action = "keep_base"
                if use_fallback and not fallback_prediction:
                    reasons = reasons + ["missing_or_empty_fallback"]

            final_uncertain, final_patterns = is_uncertain(merged.get("prediction", ""))
            rescue_uncertain, _ = is_uncertain(rescue_prediction)
            rescue_used = False
            if args.rescue and final_uncertain and rescue_prediction and not rescue_uncertain:
                merged["prediction_before_rescue"] = merged.get("prediction", "")
                merged["prediction"] = rescue.get("prediction", "")
                merged["method"] = f"{base.get('method', 'base')}_query_router_{args.rescue_name}_rescue"
                action = f"use_{args.rescue_name}_rescue"
                rescue_used = True
                reasons = reasons + [
                    "rescue:final_answer_uncertain",
                    "uncertainty:" + ",".join(final_patterns[:3]),
                ]

            router = {
                "enabled": True,
                "action": action,
                "fallback_name": args.fallback_name,
                "rescue_name": args.rescue_name if args.rescue else "",
                "rescue_used": rescue_used,
                "question_type": question_type,
                "reasons": reasons,
                "base_answer_chars": len(base.get("prediction", "")),
                "final_answer_chars": len(merged.get("prediction", "")),
            }
            merged["query_router"] = router
            merged_rows.append(merged)
            reports.append((base.get("line_no"), router))
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    actions = Counter(report["action"] for _, report in reports)
    question_types = Counter(report["question_type"] for _, report in reports)
    avg_chars = sum(len(item.get("prediction", "")) for item in merged_rows) / max(1, len(merged_rows))
    routed = sum(1 for _, report in reports if report.get("action", "").startswith("use_"))

    lines = [
        "# Query-Type Router Summary\n\n",
        f"- Base: `{args.base}`\n",
        f"- Fallback: `{args.fallback}`\n",
        f"- Rescue: `{args.rescue or 'none'}`\n",
        f"- Output: `{args.output}`\n",
        f"- Samples: {len(merged_rows)}\n",
        f"- Routed-to-fallback samples: {routed}\n",
        f"- Avg answer chars: {avg_chars:.1f}\n\n",
        "## Action Counts\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")

    lines.extend([
        "\n## Question-Type Counts\n\n",
        "| Question type | Count |\n",
        "|---|---:|\n",
    ])
    for question_type, count in sorted(question_types.items()):
        lines.append(f"| {question_type} | {count} |\n")

    lines.extend([
        "\n## Per-Sample Routing\n\n",
        "| Line | Action | Question type | Reasons | Base chars | Final chars |\n",
        "|---:|---|---|---|---:|---:|\n",
    ])
    for line_no, report in reports:
        lines.append(
            f"| {line_no} | {report.get('action')} | {report.get('question_type')} | "
            f"{', '.join(report.get('reasons', [])) or 'none'} | "
            f"{report.get('base_answer_chars')} | {report.get('final_answer_chars')} |\n"
        )

    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved routed results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
