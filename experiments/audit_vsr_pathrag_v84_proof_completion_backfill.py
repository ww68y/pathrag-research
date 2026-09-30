"""Audit VSR-PathRAG v84 proof-completion backfill."""

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

from api.features.chats.agent_controller import assess_terminal_answer  # noqa: E402
from api.features.chats.proof_completion_backfill import (  # noqa: E402
    build_proof_completion_backfill,
)
from api.features.chats.proof_source_quorum import (  # noqa: E402
    certify_proof_source_quorum,
)


SOURCE_JSONL = ROOT / "results" / "vsr_pathrag_v77_selective_30.jsonl"
OUT_JSON = ROOT / "results" / "vsr_pathrag_v84_proof_completion_backfill_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v84_proof_completion_backfill_report.md"


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


def controlled_case() -> dict[str, Any]:
    question = "How did Aurora improve battery safety?"
    query_graph = {
        "anchors": ["Aurora", "battery safety"],
        "relation_intents": ["impact"],
        "required_facets": ["thermal sensors", "operator training"],
    }
    evidence_summary = {
        "source_excerpts": [
            {
                "source_id": "doc-a#1",
                "text": "Aurora improved battery safety with thermal sensors.",
            },
            {
                "source_id": "doc-b#1",
                "text": (
                    "Aurora trained operators to respond to battery "
                    "temperature alarms."
                ),
            },
        ]
    }
    claim_graph = {
        "claims": [
            {"claim_id": "claim-0", "status": "supported"},
            {"claim_id": "claim-1", "status": "supported"},
        ],
        "edges": [
            {
                "claim_id": "claim-0",
                "evidence_id": "evidence-0",
                "relation": "supports",
                "score": 0.8,
                "source_ids": ["doc-a#1"],
            },
            {
                "claim_id": "claim-1",
                "evidence_id": "evidence-1",
                "relation": "supports",
                "score": 0.7,
                "source_ids": ["doc-b#1"],
                "evidence_window": (
                    "Aurora trained operators to respond to battery "
                    "temperature alarms."
                ),
            },
        ],
    }
    proof = {
        "enabled": True,
        "controller": "minimal_proof_subgraph_v1",
        "proof_sufficient": False,
        "selected_count": 1,
        "estimated_tokens": 20,
        "covered_claims": ["claim-0"],
        "required_claim_count": 2,
        "covered_obligations": ["anchor:AURORA", "anchor:BATTERY SAFETY"],
        "required_obligation_count": 4,
        "claim_support_coverage_ratio": 0.5,
        "obligation_coverage_ratio": 0.5,
        "contradicted_claims_in_proof": [],
        "selected_evidence": [
            {
                "evidence_id": "evidence-0",
                "kind": "source_excerpt",
                "label": "doc-a#1",
                "source_ids": ["doc-a#1"],
                "supported_claim_ids": ["claim-0"],
                "obligation_hits": [
                    {"obligation_id": "anchor:AURORA"},
                    {"obligation_id": "anchor:BATTERY SAFETY"},
                ],
                "max_claim_support_score": 0.8,
                "estimated_tokens": 20,
                "text_preview": "Aurora improved battery safety.",
            }
        ],
    }
    completion = build_proof_completion_backfill(
        question,
        query_graph,
        evidence_summary,
        claim_graph,
        {"unsupported_claim_count": 0},
        proof,
    )
    return {
        "tag": "missing_supported_claim_completed",
        "passed": bool(
            completion.get("applied")
            and not completion.get("before_proof_sufficient")
            and completion.get("after_proof_sufficient")
        ),
        "before_proof_sufficient": completion.get("before_proof_sufficient"),
        "after_proof_sufficient": completion.get("after_proof_sufficient"),
        "selected_addition_count": completion.get("selected_addition_count"),
        "missing_claims_after": completion.get("missing_claims_after"),
    }


def selective_30_audit() -> dict[str, Any]:
    rows = load_jsonl(SOURCE_JSONL)
    per_row = []
    for row in rows:
        trace = row.get("trace") or {}
        question = row.get("question", "")
        evidence_summary = trace.get("evidence_summary") or {}
        query_graph = trace.get("query_graph") or {}
        assessment = assess_terminal_answer(
            question,
            row.get("answer", ""),
            evidence_summary,
            trace.get("structured_confidence"),
            query_graph,
        )
        completion = assessment["proof_completion_backfill"]
        before_quorum = certify_proof_source_quorum(
            completion["base_minimal_proof_subgraph"],
            evidence_summary,
            assessment["claim_evidence_graph"],
        )
        after_quorum = assessment["proof_source_quorum"]
        line_no = row.get("line_no") or row.get("_line_no") or row.get("index")
        per_row.append(
            {
                "line_no": line_no,
                "question": question,
                "triggered": completion["triggered"],
                "applied": completion["applied"],
                "improved_proof": completion["improved_proof"],
                "before_proof_sufficient": completion[
                    "before_proof_sufficient"
                ],
                "after_proof_sufficient": completion[
                    "after_proof_sufficient"
                ],
                "selected_addition_count": completion[
                    "selected_addition_count"
                ],
                "candidate_count": completion["candidate_count"],
                "missing_claim_count_before": len(
                    completion["missing_claims_before"]
                ),
                "missing_claim_count_after": len(
                    completion["missing_claims_after"]
                ),
                "missing_obligation_count_before": len(
                    completion["missing_obligations_before"]
                ),
                "missing_obligation_count_after": len(
                    completion["missing_obligations_after"]
                ),
                "before_quorum_status": before_quorum["quorum_status"],
                "after_quorum_status": after_quorum["quorum_status"],
            }
        )
    before_quorum_counts = Counter(
        row["before_quorum_status"] for row in per_row
    )
    after_quorum_counts = Counter(row["after_quorum_status"] for row in per_row)
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "triggered_count": sum(1 for row in per_row if row["triggered"]),
        "applied_count": sum(1 for row in per_row if row["applied"]),
        "improved_count": sum(1 for row in per_row if row["improved_proof"]),
        "before_proof_sufficient_count": sum(
            1 for row in per_row if row["before_proof_sufficient"]
        ),
        "after_proof_sufficient_count": sum(
            1 for row in per_row if row["after_proof_sufficient"]
        ),
        "proof_sufficient_gain": sum(
            1 for row in per_row if row["after_proof_sufficient"]
        )
        - sum(1 for row in per_row if row["before_proof_sufficient"]),
        "before_quorum_status_counts": dict(before_quorum_counts),
        "after_quorum_status_counts": dict(after_quorum_counts),
        "independent_quorum_gain": after_quorum_counts.get(
            "independent_quorum",
            0,
        )
        - before_quorum_counts.get("independent_quorum", 0),
        "quorum_failed_reduction": before_quorum_counts.get("quorum_failed", 0)
        - after_quorum_counts.get("quorum_failed", 0),
        "total_selected_additions": sum(
            row["selected_addition_count"] for row in per_row
        ),
        "mean_selected_additions": mean(
            [row["selected_addition_count"] for row in per_row]
        ),
        "total_missing_claims_before": sum(
            row["missing_claim_count_before"] for row in per_row
        ),
        "total_missing_claims_after": sum(
            row["missing_claim_count_after"] for row in per_row
        ),
        "total_missing_obligations_before": sum(
            row["missing_obligation_count_before"] for row in per_row
        ),
        "total_missing_obligations_after": sum(
            row["missing_obligation_count_after"] for row in per_row
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_case"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v84 Proof-Completion Backfill Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V84 completes incomplete proof subgraphs by adding bounded, "
        "claim-local evidence windows for supported answer claims or question "
        "obligations that V78 did not select.",
        "",
        "## Controlled Audit",
        "",
        f"- Passed: `{controlled['passed']}`",
        f"- Proof sufficient: `{controlled['before_proof_sufficient']}` -> `{controlled['after_proof_sufficient']}`",
        f"- Selected additions: `{controlled['selected_addition_count']}`",
        f"- Remaining missing claims: `{controlled['missing_claims_after']}`",
        "",
        "## Offline Selective-30 Reconstruction",
        "",
        f"- Source file: `{selective['source_file']}`",
        f"- Samples: `{selective['sample_count']}`",
        f"- Triggered/applied/improved: `{selective['triggered_count']}` / `{selective['applied_count']}` / `{selective['improved_count']}`",
        f"- Proof sufficient: `{selective['before_proof_sufficient_count']}` -> `{selective['after_proof_sufficient_count']}`",
        f"- Proof sufficient gain: `{selective['proof_sufficient_gain']}`",
        f"- Before quorum counts: `{selective['before_quorum_status_counts']}`",
        f"- After quorum counts: `{selective['after_quorum_status_counts']}`",
        f"- Independent quorum gain: `{selective['independent_quorum_gain']}`",
        f"- Quorum-failed reduction: `{selective['quorum_failed_reduction']}`",
        f"- Total selected proof additions: `{selective['total_selected_additions']}`",
        f"- Missing claims before/after: `{selective['total_missing_claims_before']}` / `{selective['total_missing_claims_after']}`",
        f"- Missing obligations before/after: `{selective['total_missing_obligations_before']}` / `{selective['total_missing_obligations_after']}`",
        "",
        "## Interpretation",
        "",
        "This is a mechanism audit, not an ablation and not a judged quality "
        "score. It targets rows where the proof subgraph is incomplete despite "
        "available signed support, then lets V79-V83 operate on the completed "
        "proof.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = {
        "agent_version": "vsr-pathrag-v84",
        "protocol": "proof_completion_backfill_audit",
        "ablation": False,
        "reference_answers_used": False,
        "judge_used": False,
        "controlled_case": controlled_case(),
        "selective_30": selective_30_audit(),
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(result)
    controlled = result["controlled_case"]
    selective = result["selective_30"]
    print("VSR-PathRAG v84 proof-completion backfill audit")
    print(f"controlled passed: {controlled['passed']}")
    print(
        "triggered/applied/improved: "
        f"{selective['triggered_count']}/"
        f"{selective['applied_count']}/"
        f"{selective['improved_count']}"
    )
    print(
        "proof sufficient: "
        f"{selective['before_proof_sufficient_count']} -> "
        f"{selective['after_proof_sufficient_count']}"
    )
    print(f"before quorum: {selective['before_quorum_status_counts']}")
    print(f"after quorum: {selective['after_quorum_status_counts']}")
    print(f"written: {OUT_JSON}")
    print(f"report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
