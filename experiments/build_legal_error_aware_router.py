import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


UNCERTAINTY_PATTERNS = [
    r"\bcannot be determined\b",
    r"\bnot specified\b",
    r"\bdoes not specify\b",
    r"\bnot disclosed\b",
    r"\bno specific information\b",
]


def clean_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_uncertainty(text: str) -> bool:
    lowered = clean_text(text).lower()
    return any(re.search(pattern, lowered) for pattern in UNCERTAINTY_PATTERNS)


def load_source(spec: str) -> tuple[str, dict[int, dict]]:
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError("--source must use name:results_jsonl")
    name, path = parts
    return name, {row.get("line_no"): row for row in load_jsonl(ROOT_DIR / path)}


def purpose_specificity_score(answer: str) -> float:
    lowered = clean_text(answer).lower()
    score = 0.0
    if re.search(r"\$[\d,.]+", answer):
        score += 2.0
    if re.search(r"\b\d+(?:\.\d+)?%", answer):
        score += 1.5
    if re.search(r"\bdue\s+\d{4}\b", lowered):
        score += 1.5
    for cue in ["senior notes", "aggregate principal", "initial purchasers", "issuers", "guarantors"]:
        if cue in lowered:
            score += 0.6
    length = len(answer)
    if length > 1200:
        score -= 1.5
    elif length < 450:
        score += 0.8
    if has_uncertainty(answer):
        score -= 3.0
    return score


def indemnification_direction_score(answer: str) -> float:
    text = clean_text(answer)
    lowered = text.lower()
    score = 0.0

    positive_patterns = [
        r"partnership parties.{0,120}indemnif.{0,120}initial purchaser",
        r"indemnif.{0,80}hold harmless.{0,120}initial purchaser",
        r"hold harmless.{0,120}initial purchaser",
        r"core obligation rests with the partnership parties",
    ]
    negative_patterns = [
        r"initial purchaser.{0,120}indemnif.{0,120}partnership parties",
        r"each initial purchaser.{0,120}indemnif.{0,120}partnership parties",
        r"initial purchasers agree.{0,120}indemnify",
    ]

    for pattern in positive_patterns:
        if re.search(pattern, lowered):
            score += 3.0
    for pattern in negative_patterns:
        if re.search(pattern, lowered):
            score -= 4.0

    for cue in ["loss", "claim", "damage", "liability", "expense", "untrue statement", "omission"]:
        if cue in lowered:
            score += 0.3
    for cue in [
        "jointly and severally",
        "affiliates",
        "directors",
        "officers",
        "employees",
        "selling agents",
        "controlling persons",
        "reimburse",
        "legal or other expenses",
        "contribution",
    ]:
        if cue in lowered:
            score += 0.5
    if "only to the extent" in lowered and "furnished by that initial purchaser" in lowered:
        score -= 3.0
    if len(text) > 1200:
        score -= 1.0
    if has_uncertainty(answer):
        score -= 3.0
    return score


def choose_purpose_candidate(candidates: dict[str, dict]) -> tuple[str | None, list[str]]:
    scored = []
    for name, row in candidates.items():
        answer = row.get("prediction", "")
        scored.append((purpose_specificity_score(answer), name, answer))
    scored.sort(reverse=True)
    best_score, best_name, _ = scored[0]
    base_score = next(score for score, name, _ in scored if name == "base")
    if best_name != "base" and best_score >= base_score + 1.2 and best_score >= 3.2:
        return best_name, [f"purpose_specificity:{best_score:.2f}>{base_score:.2f}"]
    return None, [f"keep_base_purpose_specificity:{base_score:.2f}"]


def choose_indemnification_candidate(candidates: dict[str, dict]) -> tuple[str | None, list[str]]:
    scored = []
    for name, row in candidates.items():
        answer = row.get("prediction", "")
        scored.append((indemnification_direction_score(answer), name, answer))
    scored.sort(reverse=True)
    best_score, best_name, _ = scored[0]
    base_score = next(score for score, name, _ in scored if name == "base")
    if best_name != "base" and best_score >= base_score + 3.0 and best_score >= 2.5:
        return best_name, [f"indemnification_direction:{best_score:.2f}>{base_score:.2f}"]
    return None, [f"keep_base_indemnification_direction:{base_score:.2f}"]


def route_item(base: dict, candidates: dict[str, dict]) -> tuple[str | None, str, list[str]]:
    question = clean_text(base.get("question")).lower()
    if "purpose" in question and "purchase agreement" in question:
        chosen, reasons = choose_purpose_candidate(candidates)
        return chosen, "purchase_agreement_purpose", reasons
    if "indemnification" in question and "initial purchaser" in question:
        chosen, reasons = choose_indemnification_candidate(candidates)
        return chosen, "initial_purchaser_indemnification_direction", reasons
    return None, "none", ["no_error_pattern"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT_DIR / args.base)
    sources = dict(load_source(spec) for spec in args.source)

    output_rows = []
    reports = []
    for base in base_rows:
        candidates = {"base": base}
        for source_name, rows_by_line in sources.items():
            row = rows_by_line.get(base.get("line_no"))
            if row and clean_text(row.get("prediction", "")):
                candidates[source_name] = row

        chosen_name, pattern, reasons = route_item(base, candidates)
        if chosen_name and chosen_name in candidates:
            chosen = candidates[chosen_name]
            new_row = dict(base)
            new_row["prediction_before_candidate_router"] = base.get("prediction", "")
            new_row["prediction"] = chosen.get("prediction", "")
            new_row["method"] = f"{base.get('method', 'base')}_legal_error_router"
            action = f"use_{chosen_name}"
        else:
            new_row = dict(base)
            action = "keep_base"

        new_row["legal_error_router"] = {
            "enabled": True,
            "action": action,
            "pattern": pattern,
            "reasons": reasons,
            "candidate_sources": sorted(candidates.keys()),
            "base_answer_chars": len(base.get("prediction", "")),
            "final_answer_chars": len(new_row.get("prediction", "")),
        }
        output_rows.append(new_row)
        reports.append((new_row.get("line_no"), new_row["legal_error_router"]))

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in output_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    actions = Counter(report["action"] for _, report in reports)
    lines = [
        "# Legal Error-Aware Router Summary\n\n",
        f"- Base: `{args.base}`\n",
        f"- Output: `{args.output}`\n",
        f"- Samples: {len(output_rows)}\n\n",
        "## Action Counts\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")

    lines.extend([
        "\n## Per-Sample Routing\n\n",
        "| Line | Action | Pattern | Reasons | Base chars | Final chars |\n",
        "|---:|---|---|---|---:|---:|\n",
    ])
    for line_no, report in reports:
        lines.append(
            f"| {line_no} | {report.get('action')} | {report.get('pattern')} | "
            f"{', '.join(report.get('reasons', []))} | "
            f"{report.get('base_answer_chars')} | {report.get('final_answer_chars')} |\n"
        )

    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved routed results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
