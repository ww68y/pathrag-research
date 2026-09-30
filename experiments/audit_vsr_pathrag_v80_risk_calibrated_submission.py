"""Audit VSR-PathRAG v80 risk-calibrated answer submission."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from api.features.chats.counterfactual_path_necessity import (  # noqa: E402
    certify_counterfactual_path_necessity,
)
from api.features.chats.minimal_proof_subgraph import (  # noqa: E402
    build_minimal_proof_subgraph,
)
from api.features.chats.risk_calibrated_submission import (  # noqa: E402
    calibrate_answer_submission,
)


SOURCE_JSONL = ROOT / "results" / "vsr_pathrag_v77_selective_30.jsonl"
OUT_JSON = ROOT / "results" / "vsr_pathrag_v80_risk_calibrated_submission_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v80_risk_calibrated_submission_report.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.fmean(values), 4)


def controlled_cases() -> list[dict[str, Any]]:
    cases = [
        {
            "tag": "supported_submit",
            "expected_action": "submit",
            "audit": {
                "risk_level": "low",
                "claim_count": 2,
                "unsupported_claim_count": 0,
                "claim_support_ratio": 1.0,
            },
            "proof": {
                "proof_sufficient": True,
                "claim_support_coverage_ratio": 1.0,
                "obligation_coverage_ratio": 0.75,
            },
            "necessity": {
                "certification_passed": True,
                "selected_path_count": 2,
                "replaceable_path_count": 0,
            },
            "graph": {"claims": [{"status": "supported"}]},
        },
        {
            "tag": "unsupported_block",
            "expected_action": "block_unsupported",
            "audit": {
                "risk_level": "medium",
                "claim_count": 3,
                "unsupported_claim_count": 1,
                "claim_support_ratio": 0.667,
            },
            "proof": {
                "proof_sufficient": False,
                "claim_support_coverage_ratio": 0.667,
                "obligation_coverage_ratio": 0.5,
            },
            "necessity": {
                "certification_passed": False,
                "selected_path_count": 1,
                "replaceable_path_count": 0,
            },
            "graph": {"claims": [{"status": "unknown"}]},
        },
        {
            "tag": "supported_monitor",
            "expected_action": "submit_with_monitoring",
            "audit": {
                "risk_level": "medium",
                "claim_count": 2,
                "unsupported_claim_count": 0,
                "claim_support_ratio": 0.9,
            },
            "proof": {
                "proof_sufficient": True,
                "claim_support_coverage_ratio": 0.9,
                "obligation_coverage_ratio": 0.7,
            },
            "necessity": {
                "certification_passed": True,
                "selected_path_count": 2,
                "replaceable_path_count": 1,
            },
            "graph": {"claims": [{"status": "supported"}]},
        },
    ]
    results = []
    for case in cases:
        policy = calibrate_answer_submission(
            case["audit"],
            case["proof"],
            case["necessity"],
            case["graph"],
            {"refusal_like": False},
        )
        results.append(
            {
                "tag": case["tag"],
                "expected_action": case["expected_action"],
                "action": policy["action"],
                "passed": policy["action"] == case["expected_action"],
                "calibrated_risk": policy["calibrated_risk"],
                "submission_allowed": policy["submission_allowed"],
            }
        )
    return results


def selective_30_audit() -> dict[str, Any]:
    rows = load_jsonl(SOURCE_JSONL)
    per_row = []
    for row in rows:
        trace = row.get("trace") or {}
        proof = build_minimal_proof_subgraph(
            row.get("question", ""),
            trace.get("query_graph") or {},
            row.get("answer", ""),
            trace.get("evidence_summary") or {},
            trace.get("claim_evidence_graph") or {},
            trace.get("evidence_audit") or {},
        )
        necessity = certify_counterfactual_path_necessity(
            row.get("question", ""),
            trace.get("query_graph") or {},
            row.get("answer", ""),
            trace.get("evidence_summary") or {},
            trace.get("claim_evidence_graph") or {},
            trace.get("evidence_audit") or {},
            proof,
        )
        policy = calibrate_answer_submission(
            trace.get("evidence_audit") or {},
            proof,
            necessity,
            trace.get("claim_evidence_graph") or {},
            {},
        )
        line_no = row.get("line_no") or row.get("_line_no") or row.get("index")
        per_row.append(
            {
                "line_no": line_no,
                "question": row.get("question", ""),
                "action": policy["action"],
                "calibrated_risk": policy["calibrated_risk"],
                "submission_allowed": policy["submission_allowed"],
                "monitoring_required": policy["monitoring_required"],
                "unsupported_claim_count": policy["unsupported_claim_count"],
                "proof_claim_coverage": policy["proof_claim_coverage"],
                "proof_obligation_coverage": policy[
                    "proof_obligation_coverage"
                ],
                "necessity_certified": policy["necessity_certified"],
            }
        )
    counts = Counter(row["action"] for row in per_row)
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "action_counts": dict(counts),
        "submission_allowed_count": sum(
            1 for row in per_row if row["submission_allowed"]
        ),
        "monitoring_required_count": sum(
            1 for row in per_row if row["monitoring_required"]
        ),
        "blocked_unsupported_count": counts.get("block_unsupported", 0),
        "mean_calibrated_risk": mean(
            [row["calibrated_risk"] for row in per_row]
        ),
        "mean_proof_claim_coverage": mean(
            [row["proof_claim_coverage"] for row in per_row]
        ),
        "mean_proof_obligation_coverage": mean(
            [row["proof_obligation_coverage"] for row in per_row]
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_cases"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v80 Risk-Calibrated Submission Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V80 converts evidence audit, minimal proof coverage, and "
        "counterfactual necessity into a calibrated answer-submission "
        "decision. Unsupported claims have a zero budget and are blocked.",
        "",
        "## Controlled Audit",
        "",
        "| Case | Expected | Action | Pass | Risk | Allowed |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in controlled:
        lines.append(
            "| {tag} | {expected} | {action} | {passed} | {risk:.4f} | {allowed} |".format(
                tag=row["tag"],
                expected=row["expected_action"],
                action=row["action"],
                passed=row["passed"],
                risk=row["calibrated_risk"],
                allowed=row["submission_allowed"],
            )
        )
    lines.extend(
        [
            "",
            "## Offline Selective-30 Calibration",
            "",
            f"- Source file: `{selective['source_file']}`",
            f"- Samples: `{selective['sample_count']}`",
            f"- Action counts: `{selective['action_counts']}`",
            f"- Submission allowed: `{selective['submission_allowed_count']}/{selective['sample_count']}`",
            f"- Monitoring required: `{selective['monitoring_required_count']}/{selective['sample_count']}`",
            f"- Blocked unsupported: `{selective['blocked_unsupported_count']}`",
            f"- Mean calibrated risk: `{selective['mean_calibrated_risk']}`",
            f"- Mean proof claim coverage: `{selective['mean_proof_claim_coverage']}`",
            f"- Mean proof obligation coverage: `{selective['mean_proof_obligation_coverage']}`",
            "",
            "## Interpretation",
            "",
            "This is not a judged quality score. It is the final submission "
            "control layer for the pre-ablation innovation stack: supported "
            "answers can be submitted, replaceable-path answers are monitored, "
            "and unsupported claims are blocked before final response commit.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = {
        "agent_version": "vsr-pathrag-v80",
        "protocol": "risk_calibrated_answer_submission_audit",
        "ablation": False,
        "reference_answers_used": False,
        "judge_used": False,
        "controlled_cases": controlled_cases(),
        "selective_30": selective_30_audit(),
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(result)
    controlled_pass = sum(1 for row in result["controlled_cases"] if row["passed"])
    selective = result["selective_30"]
    print("VSR-PathRAG v80 risk-calibrated submission audit")
    print(f"controlled expected outcomes: {controlled_pass}/{len(result['controlled_cases'])}")
    print(f"action counts: {selective['action_counts']}")
    print(
        "submission allowed: "
        f"{selective['submission_allowed_count']}/{selective['sample_count']}; "
        "blocked unsupported: "
        f"{selective['blocked_unsupported_count']}"
    )
    print(f"mean calibrated risk: {selective['mean_calibrated_risk']}")
    print(f"written: {OUT_JSON}")
    print(f"report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
