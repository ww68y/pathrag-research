"""Audit VSR-PathRAG v83 source-quorum backfill."""

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
from api.features.chats.source_quorum_backfill import (  # noqa: E402
    build_source_quorum_backfill,
)
from api.features.chats.vsr_pathrag import (  # noqa: E402
    build_signed_claim_evidence_graph,
)


SOURCE_JSONL = ROOT / "results" / "vsr_pathrag_v77_selective_30.jsonl"
OUT_JSON = ROOT / "results" / "vsr_pathrag_v83_source_quorum_backfill_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v83_source_quorum_backfill_report.md"


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
                "document_id": "doc-a",
                "text": "Aurora improved battery safety with thermal sensors.",
            },
            {
                "source_id": "doc-b#1",
                "document_id": "doc-b",
                "text": "Aurora trained operators to respond to battery alarms.",
            },
            {
                "source_id": "doc-c#1",
                "document_id": "doc-c",
                "text": (
                    "A separate Aurora safety report says thermal sensors "
                    "and operator training reduced battery risk."
                ),
            },
        ]
    }
    evidence = [
        {
            "label": row["source_id"],
            "kind": "source_excerpt",
            "text": row["text"],
            "source_ids": [row["source_id"]],
        }
        for row in evidence_summary["source_excerpts"]
    ]
    claim_graph = build_signed_claim_evidence_graph(
        ["Aurora improved battery safety with thermal sensors."],
        evidence,
    )
    proof = {
        "proof_sufficient": True,
        "covered_claims": ["claim-0"],
        "required_claim_count": 1,
        "covered_obligations": ["facet:question_specific_bundle"],
        "required_obligation_count": 1,
        "contradicted_claims_in_proof": [],
        "estimated_tokens": 40,
        "selected_evidence": [
            {
                "evidence_id": "evidence-0",
                "kind": "source_excerpt",
                "label": "doc-a#1",
                "source_ids": ["doc-a#1"],
                "supported_claim_ids": ["claim-0"],
                "obligation_hits": [],
                "max_claim_support_score": 0.8,
                "estimated_tokens": 20,
                "text_preview": "Aurora improved battery safety.",
            },
            {
                "evidence_id": "evidence-1",
                "kind": "source_excerpt",
                "label": "doc-b#1",
                "source_ids": ["doc-b#1"],
                "supported_claim_ids": [],
                "obligation_hits": [
                    {"obligation_id": "facet:question_specific_bundle"},
                ],
                "max_claim_support_score": 0.0,
                "estimated_tokens": 20,
                "text_preview": "Aurora trained operators.",
            },
        ],
    }
    backfill = build_source_quorum_backfill(
        question,
        query_graph,
        evidence_summary,
        claim_graph,
        {"unsupported_claim_count": 0},
        proof,
    )
    return {
        "tag": "correlated_proof_upgraded_by_third_source",
        "passed": bool(
            backfill.get("applied")
            and backfill.get("before_quorum_status") == "correlated_monitor"
            and backfill.get("after_quorum_status") == "independent_quorum"
        ),
        "before_quorum_status": backfill.get("before_quorum_status"),
        "after_quorum_status": backfill.get("after_quorum_status"),
        "selected_addition_count": backfill.get("selected_addition_count"),
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
        backfill = assessment["source_quorum_backfill"]
        before_status = backfill["before_quorum_status"]
        after_status = backfill["after_quorum_status"]
        base_quorum = backfill.get("base_proof_source_quorum", {})
        before_single_claims = len(
            base_quorum.get(
                "single_family_claims",
                [],
            )
        )
        after_single_claims = len(
            assessment["proof_source_quorum"].get("single_family_claims", [])
        )
        line_no = row.get("line_no") or row.get("_line_no") or row.get("index")
        per_row.append(
            {
                "line_no": line_no,
                "question": question,
                "before_quorum_status": before_status,
                "after_quorum_status": after_status,
                "triggered": backfill["triggered"],
                "applied": backfill.get("applied", False),
                "improved_quorum": backfill.get("improved_quorum", False),
                "selected_addition_count": backfill[
                    "selected_addition_count"
                ],
                "candidate_count": backfill["candidate_count"],
                "after_independent_family_count": assessment[
                    "proof_source_quorum"
                ]["independent_family_count"],
                "before_single_family_claim_count": before_single_claims,
                "after_single_family_claim_count": after_single_claims,
            }
        )
    before_counts = Counter(row["before_quorum_status"] for row in per_row)
    after_counts = Counter(row["after_quorum_status"] for row in per_row)
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "triggered_count": sum(1 for row in per_row if row["triggered"]),
        "applied_count": sum(1 for row in per_row if row["applied"]),
        "improved_count": sum(1 for row in per_row if row["improved_quorum"]),
        "before_quorum_status_counts": dict(before_counts),
        "after_quorum_status_counts": dict(after_counts),
        "independent_quorum_gain": after_counts.get("independent_quorum", 0)
        - before_counts.get("independent_quorum", 0),
        "total_selected_additions": sum(
            row["selected_addition_count"] for row in per_row
        ),
        "mean_selected_additions": mean(
            [row["selected_addition_count"] for row in per_row]
        ),
        "mean_candidate_count": mean([row["candidate_count"] for row in per_row]),
        "mean_after_independent_family_count": mean(
            [row["after_independent_family_count"] for row in per_row]
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_case"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v83 Source-Quorum Backfill Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V83 uses the V82 source-quorum certificate to add a compact set of "
        "unselected evidence rows that supply independent source families for "
        "single-family claims or obligations.",
        "",
        "## Controlled Audit",
        "",
        f"- Passed: `{controlled['passed']}`",
        f"- Before/after quorum: `{controlled['before_quorum_status']}` -> `{controlled['after_quorum_status']}`",
        f"- Selected additions: `{controlled['selected_addition_count']}`",
        "",
        "## Offline Selective-30 Reconstruction",
        "",
        f"- Source file: `{selective['source_file']}`",
        f"- Samples: `{selective['sample_count']}`",
        f"- Triggered/applied/improved: `{selective['triggered_count']}` / `{selective['applied_count']}` / `{selective['improved_count']}`",
        f"- Before status counts: `{selective['before_quorum_status_counts']}`",
        f"- After status counts: `{selective['after_quorum_status_counts']}`",
        f"- Independent quorum gain: `{selective['independent_quorum_gain']}`",
        f"- Total selected additions: `{selective['total_selected_additions']}`",
        f"- Mean selected additions: `{selective['mean_selected_additions']}`",
        f"- Mean candidate count: `{selective['mean_candidate_count']}`",
        f"- Mean after independent family count: `{selective['mean_after_independent_family_count']}`",
        "",
        "## Interpretation",
        "",
        "This is still an innovation-stage mechanism, not an ablation. It "
        "turns provenance certification into a selective proof augmentation "
        "step, strengthening claims whose support previously depended on a "
        "single source family.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = {
        "agent_version": "vsr-pathrag-v83",
        "protocol": "source_quorum_backfill_audit",
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
    print("VSR-PathRAG v83 source-quorum backfill audit")
    print(f"controlled passed: {controlled['passed']}")
    print(
        "triggered/applied/improved: "
        f"{selective['triggered_count']}/"
        f"{selective['applied_count']}/"
        f"{selective['improved_count']}"
    )
    print(f"before statuses: {selective['before_quorum_status_counts']}")
    print(f"after statuses: {selective['after_quorum_status_counts']}")
    print(f"written: {OUT_JSON}")
    print(f"report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
