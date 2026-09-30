import argparse
import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "through",
    "about",
    "which",
    "what",
    "when",
    "where",
    "how",
    "does",
    "are",
    "can",
    "could",
    "would",
    "should",
    "their",
    "there",
    "these",
    "those",
    "such",
    "also",
    "more",
    "most",
    "some",
    "than",
    "then",
    "been",
    "being",
    "have",
    "has",
    "had",
    "not",
    "but",
    "its",
    "our",
    "your",
    "use",
    "used",
    "using",
}


EFFECT_INTENT_TERMS = {
    "affect",
    "affecting",
    "affected",
    "effect",
    "effects",
    "differ",
    "differs",
    "different",
    "difference",
    "differences",
    "influence",
    "influences",
    "influenced",
    "impact",
    "impacts",
    "risk",
    "risks",
    "cause",
    "causes",
    "caused",
    "contribute",
    "contributes",
}


def clean_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())


def terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", clean_text(text).lower())
    return {token for token in tokens if token not in STOPWORDS}


def has_effect_intent(item: dict) -> bool:
    question_terms = terms(item.get("question", ""))
    requirement_terms = set()
    for value in (item.get("requirements") or {}).get("expanded_terms", []):
        requirement_terms.update(terms(value))
    return bool((question_terms | requirement_terms) & EFFECT_INTENT_TERMS)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choose_answer(base: dict, candidate: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    base_answer = clean_text(base.get("prediction", ""))
    candidate_answer = clean_text(candidate.get("prediction", ""))
    base_len = len(base_answer)
    candidate_len = len(candidate_answer)
    base_terms = terms(base_answer)
    candidate_terms = terms(candidate_answer)

    reasons = []
    if not has_effect_intent(base):
        reasons.append("not_effect_or_difference_question")
    if base_len > args.max_base_chars:
        reasons.append("base_not_short")
    if candidate_len < base_len + args.min_candidate_gain_chars:
        reasons.append("candidate_not_richer")
    if candidate_len > args.max_candidate_chars:
        reasons.append("candidate_too_long")
    if len(candidate_terms) < len(base_terms) + args.min_candidate_gain_terms:
        reasons.append("candidate_not_more_informative")
    if candidate.get("refiner_stats", {}).get("failed"):
        reasons.append("candidate_refinement_failed")

    use_candidate = not reasons
    merged = dict(candidate if use_candidate else base)
    merged["method"] = f"{base.get('method', 'base')}_selective_recovery"
    merged["selective_recovery"] = {
        "enabled": True,
        "action": "use_candidate" if use_candidate else "keep_base",
        "reasons": reasons,
        "base_file_method": base.get("method"),
        "candidate_file_method": candidate.get("method"),
        "base_answer_chars": base_len,
        "candidate_answer_chars": candidate_len,
        "base_term_count": len(base_terms),
        "candidate_term_count": len(candidate_terms),
        "effect_intent": has_effect_intent(base),
    }
    return merged, merged["selective_recovery"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-base-chars", type=int, default=400)
    parser.add_argument("--max-candidate-chars", type=int, default=650)
    parser.add_argument("--min-candidate-gain-chars", type=int, default=80)
    parser.add_argument("--min-candidate-gain-terms", type=int, default=3)
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT_DIR / args.base)
    candidate_by_line = {item.get("line_no"): item for item in load_jsonl(ROOT_DIR / args.candidate)}

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_rows = []
    reports = []
    with output_path.open("w", encoding="utf-8") as out:
        for base in base_rows:
            candidate = candidate_by_line.get(base.get("line_no"))
            if candidate is None:
                merged = dict(base)
                report = {
                    "enabled": True,
                    "action": "keep_base",
                    "reasons": ["missing_candidate"],
                }
                merged["selective_recovery"] = report
            else:
                merged, report = choose_answer(base, candidate, args)
            merged_rows.append(merged)
            reports.append((base.get("line_no"), report))
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    used = sum(1 for _, report in reports if report.get("action") == "use_candidate")
    avg_chars = sum(len(item.get("prediction", "")) for item in merged_rows) / max(1, len(merged_rows))
    lines = [
        "# SAGE Selective Recovery Summary\n\n",
        f"- Base: `{args.base}`\n",
        f"- Candidate: `{args.candidate}`\n",
        f"- Output: `{args.output}`\n",
        f"- Samples: {len(merged_rows)}\n",
        f"- Candidate-used samples: {used}\n",
        f"- Avg answer chars: {avg_chars:.1f}\n\n",
        "| Line | Action | Reasons | Base chars | Candidate chars |\n",
        "|---:|---|---|---:|---:|\n",
    ]
    for line_no, report in reports:
        lines.append(
            f"| {line_no} | {report.get('action')} | "
            f"{', '.join(report.get('reasons', [])) or 'accepted'} | "
            f"{report.get('base_answer_chars', '')} | {report.get('candidate_answer_chars', '')} |\n"
        )
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved selective recovery results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
