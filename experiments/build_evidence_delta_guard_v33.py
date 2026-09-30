from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_claim_evidence_agent import content_tokens, split_claims
from build_evidence_delta_guard import build_guard as build_v32_guard
from build_evidence_delta_guard import load_jsonl, supported_ratio, write_jsonl
from build_evidence_preserving_compressor import semantic_claim_coverage


ROOT_DIR = Path(__file__).resolve().parents[1]

REFUSAL_MARKERS = (
    "cannot be determined",
    "cannot determine",
    "not described",
    "not provided",
    "not specified",
    "does not contain",
    "no information",
    "insufficient information",
    "not explicitly",
)


def has_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def token_preservation(original: str, candidate: str) -> float:
    original_tokens = content_tokens(original)
    if not original_tokens:
        return 1.0
    candidate_tokens = content_tokens(candidate)
    return len(original_tokens & candidate_tokens) / len(original_tokens)


def build_guard(row: dict) -> dict:
    guard = build_v32_guard(row)
    guard["version"] = "v33"
    guard.setdefault("checks", {})
    guard.setdefault("metrics", {})

    original = str(row.get("prediction_before_contract_completion") or "")
    candidate = str(row.get("prediction") or "")
    completion = row.get("contract_completion_agent") or {}
    contract = completion.get("contract") or {}
    pre = completion.get("pre_claim_metrics") or {}

    original_claims = split_claims(original)
    candidate_claims = split_claims(candidate)
    semantic_preservation, claim_similarities = semantic_claim_coverage(original_claims, candidate_claims)
    token_overlap = token_preservation(original, candidate)
    minimum_points = max(1, int(contract.get("minimum_distinct_points") or 1))
    original_refusal = has_refusal(original)

    high_content_original = bool(
        not original_refusal
        and len(original) >= 900
        and len(original_claims) >= max(minimum_points + 1, 4)
        and float(pre.get("mean_support") or 0.0) >= 0.58
        and supported_ratio(pre) >= 0.60
        and int(pre.get("unsupported_claims") or 0) <= 1
    )
    low_preservation_rewrite = bool(semantic_preservation < 0.25 and token_overlap < 0.25)
    high_content_original_drift = bool(high_content_original and low_preservation_rewrite)

    guard["checks"]["high_content_original_preserved"] = not high_content_original_drift
    guard["metrics"].update(
        {
            "v33_original_refusal": original_refusal,
            "v33_original_chars": len(original),
            "v33_candidate_chars": len(candidate),
            "v33_original_claims": len(original_claims),
            "v33_candidate_claims": len(candidate_claims),
            "v33_semantic_preservation": round(float(semantic_preservation), 6),
            "v33_min_claim_similarity": round(min(claim_similarities), 6) if claim_similarities else 1.0,
            "v33_token_preservation": round(float(token_overlap), 6),
            "v33_high_content_original": high_content_original,
            "v33_low_preservation_rewrite": low_preservation_rewrite,
        }
    )

    if guard.get("enabled") and guard.get("action") == "retain_repair" and high_content_original_drift:
        guard["action"] = "retain_original"
        guard.setdefault("rejection_reasons", []).append("high_content_original_low_preservation")

    return guard


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    guards = [row.get("evidence_delta_guard") or {} for row in rows]
    active = [guard for guard in guards if guard.get("enabled")]
    retained = [guard for guard in active if guard.get("action") == "retain_repair"]
    reverted = [guard for guard in active if guard.get("action") == "retain_original"]
    reason_counts: dict[str, int] = {}
    for guard in reverted:
        for reason in guard.get("rejection_reasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    lines = [
        "# SAGE v33 Preservation Evidence-Delta Guard Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Starts from v32 deterministic evidence-delta checks.\n",
        "- Adds a narrow high-content original preservation check for non-refusal answers.\n",
        "- Reference answers, judge scores, and LLM consensus votes are not runtime inputs.\n\n",
        "| Decision | Count |\n",
        "|---|---:|\n",
        f"| Accepted repairs checked | {len(active)} |\n",
        f"| Repairs retained | {len(retained)} |\n",
        f"| Repairs reverted | {len(reverted)} |\n",
        f"| Extra LLM calls | 0 |\n",
    ]
    if reason_counts:
        lines.extend(["\n## Rejection Reasons\n\n", "| Reason | Count |\n", "|---|---:|\n"])
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"| `{reason}` | {count} |\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    source_path = ROOT_DIR / args.source
    output_rows = []
    for source_row in load_jsonl(source_path):
        row = dict(source_row)
        guard = build_guard(row)
        if guard["enabled"] and guard["action"] == "retain_original":
            row["rejected_evidence_delta_v33_prediction"] = row.get("prediction", "")
            row["prediction"] = row.get("prediction_before_contract_completion", row.get("prediction", ""))
        row["method"] = f"sage_v33_preservation_delta_{guard['action']}"
        row["evidence_delta_guard"] = guard
        output_rows.append(row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, source_path)
    print(f"Saved v33 preservation-delta results to {output_path}")


if __name__ == "__main__":
    main()
