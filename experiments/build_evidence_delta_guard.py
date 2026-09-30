import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def supported_ratio(metrics: dict) -> float:
    claims = as_int(metrics.get("claims"))
    return as_int(metrics.get("supported_claims")) / max(1, claims)


def build_guard(row: dict) -> dict:
    completion = row.get("contract_completion_agent") or {}
    router = row.get("learned_contract_gap_router") or {}
    repair_guard = completion.get("repair_guard") or {}
    pre = completion.get("pre_claim_metrics") or {}
    post = repair_guard.get("post_claim_metrics") or {}
    contract = completion.get("contract") or {}
    repair_is_active = bool(
        completion.get("repair_accepted")
        and row.get("prediction_before_contract_completion")
        and row.get("prediction") != row.get("prediction_before_contract_completion")
        and (not router.get("enabled") or router.get("repair_used"))
    )

    pre_claims = as_int(pre.get("claims"))
    post_claims = as_int(post.get("claims"))
    pre_supported = as_int(pre.get("supported_claims"))
    post_supported = as_int(post.get("supported_claims"))
    pre_mean = as_float(pre.get("mean_support"))
    post_mean = as_float(post.get("mean_support"))
    pre_unsupported = as_int(pre.get("unsupported_claims"))
    post_unsupported = as_int(post.get("unsupported_claims"))
    minimum_points = max(1, as_int(contract.get("minimum_distinct_points")))
    original = str(row.get("prediction_before_contract_completion") or "")
    candidate = str(row.get("prediction") or "")

    supported_capacity_retention = min(1.0, post_supported / max(1, pre_supported))
    evidence_gain = post_mean - pre_mean
    length_ratio = len(candidate) / max(1, len(original))
    high_confidence_original = bool(
        pre_mean >= 0.72
        and supported_ratio(pre) >= 0.75
        and pre_unsupported <= 1
    )
    supported_content_collapse = bool(
        high_confidence_original
        and supported_capacity_retention <= 0.50
        and evidence_gain < 0.15
    )
    satisfied_enumeration_expansion = bool(
        contract.get("type") == "enumeration"
        and pre_claims >= minimum_points
        and pre_unsupported == 0
        and pre_mean >= 0.65
        and post_claims >= minimum_points
        and length_ratio >= 0.70
    )
    candidate_evidence_supported = bool(
        post_claims > 0
        and post_mean >= 0.50
        and post_unsupported <= pre_unsupported
    )

    reasons = []
    if supported_content_collapse:
        reasons.append("high_support_content_collapse")
    if satisfied_enumeration_expansion:
        reasons.append("already_satisfied_enumeration_expanded")
    if not candidate_evidence_supported:
        reasons.append("candidate_evidence_regression")

    retain_repair = bool(repair_is_active and not reasons)
    action = "retain_repair" if retain_repair else "retain_original"
    if not repair_is_active:
        action = "not_applicable"

    return {
        "enabled": repair_is_active,
        "version": "v32",
        "action": action,
        "rejection_reasons": reasons,
        "checks": {
            "candidate_evidence_supported": candidate_evidence_supported,
            "supported_content_preserved": not supported_content_collapse,
            "no_redundant_enumeration_expansion": not satisfied_enumeration_expansion,
        },
        "metrics": {
            "pre_claims": pre_claims,
            "post_claims": post_claims,
            "pre_supported_claims": pre_supported,
            "post_supported_claims": post_supported,
            "pre_mean_support": round(pre_mean, 6),
            "post_mean_support": round(post_mean, 6),
            "evidence_gain": round(evidence_gain, 6),
            "supported_capacity_retention": round(supported_capacity_retention, 6),
            "answer_length_ratio": round(length_ratio, 6),
            "minimum_distinct_points": minimum_points,
        },
    }


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
        "# SAGE v32 Independent Evidence-Delta Guard Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- The guard uses deterministic claim-support statistics only.\n",
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
            row["rejected_evidence_delta_prediction"] = row.get("prediction", "")
            row["prediction"] = row.get("prediction_before_contract_completion", row.get("prediction", ""))
        row["method"] = f"sage_v32_evidence_delta_{guard['action']}"
        row["evidence_delta_guard"] = guard
        output_rows.append(row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, source_path)
    print(f"Saved v32 evidence-delta results to {output_path}")


if __name__ == "__main__":
    main()
