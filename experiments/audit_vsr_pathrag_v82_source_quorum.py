"""Audit VSR-PathRAG v82 source-independent proof quorum."""

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
from api.features.chats.proof_source_quorum import (  # noqa: E402
    certify_proof_source_quorum,
)


SOURCE_JSONL = ROOT / "results" / "vsr_pathrag_v77_selective_30.jsonl"
OUT_JSON = ROOT / "results" / "vsr_pathrag_v82_source_quorum_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v82_source_quorum_report.md"


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
    independent_proof = {
        "proof_sufficient": True,
        "covered_claims": ["claim-0"],
        "covered_obligations": ["facet:safety"],
        "contradicted_claims_in_proof": [],
        "selected_evidence": [
            {
                "evidence_id": "evidence-0",
                "label": "doc-a#1",
                "source_ids": ["doc-a#1"],
                "supported_claim_ids": ["claim-0"],
                "obligation_hits": [{"obligation_id": "facet:safety"}],
            },
            {
                "evidence_id": "evidence-1",
                "label": "doc-b#1",
                "source_ids": ["doc-b#1"],
                "supported_claim_ids": ["claim-0"],
                "obligation_hits": [{"obligation_id": "facet:safety"}],
            },
        ],
    }
    independent_summary = {
        "source_excerpts": [
            {
                "source_id": "doc-a#1",
                "document_id": "doc-a",
                "text": "Aurora added thermal sensors.",
            },
            {
                "source_id": "doc-b#1",
                "document_id": "doc-b",
                "text": "A separate report says sensors reduced risk.",
            },
        ]
    }
    correlated_proof = {
        "proof_sufficient": True,
        "covered_claims": ["claim-0"],
        "covered_obligations": ["facet:training"],
        "contradicted_claims_in_proof": [],
        "selected_evidence": [
            {
                "evidence_id": "evidence-0",
                "label": "doc-a#1",
                "source_ids": ["doc-a#1"],
                "supported_claim_ids": ["claim-0"],
                "obligation_hits": [],
            },
            {
                "evidence_id": "evidence-1",
                "label": "doc-b#1",
                "source_ids": ["doc-b#1"],
                "supported_claim_ids": [],
                "obligation_hits": [{"obligation_id": "facet:training"}],
            },
        ],
    }
    correlated_summary = {
        "source_excerpts": [
            {
                "source_id": "doc-a#1",
                "document_id": "doc-a",
                "text": "Aurora added thermal sensors.",
            },
            {
                "source_id": "doc-b#1",
                "document_id": "doc-b",
                "text": "Aurora trained operators on alarms.",
            },
        ]
    }
    cases = [
        {
            "tag": "independent_two_family_claim_and_obligation",
            "expected": "independent_quorum",
            "quorum": certify_proof_source_quorum(
                independent_proof,
                independent_summary,
                {"claims": []},
            ),
        },
        {
            "tag": "two_sources_but_single_family_targets",
            "expected": "correlated_monitor",
            "quorum": certify_proof_source_quorum(
                correlated_proof,
                correlated_summary,
                {"claims": []},
            ),
        },
    ]
    for case in cases:
        case["passed"] = (
            case["quorum"].get("quorum_status") == case["expected"]
        )
    return cases


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
        quorum = certify_proof_source_quorum(
            assessment["minimal_proof_subgraph"],
            evidence_summary,
            assessment["claim_evidence_graph"],
        )
        line_no = row.get("line_no") or row.get("_line_no") or row.get("index")
        per_row.append(
            {
                "line_no": line_no,
                "question": question,
                "quorum_status": quorum["quorum_status"],
                "quorum_passed": quorum["quorum_passed"],
                "proof_sufficient": quorum["proof_sufficient"],
                "independent_family_count": quorum[
                    "independent_family_count"
                ],
                "selected_evidence_count": quorum[
                    "selected_evidence_count"
                ],
                "single_family_claim_count": len(
                    quorum["single_family_claims"]
                ),
                "single_family_obligation_count": len(
                    quorum["single_family_obligations"]
                ),
                "monitoring_required": quorum["monitoring_required"],
            }
        )
    status_counts = Counter(row["quorum_status"] for row in per_row)
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "proof_sufficient_count": sum(
            1 for row in per_row if row["proof_sufficient"]
        ),
        "quorum_passed_count": sum(
            1 for row in per_row if row["quorum_passed"]
        ),
        "monitoring_required_count": sum(
            1 for row in per_row if row["monitoring_required"]
        ),
        "quorum_status_counts": dict(status_counts),
        "mean_independent_family_count": mean(
            [row["independent_family_count"] for row in per_row]
        ),
        "mean_selected_evidence_count": mean(
            [row["selected_evidence_count"] for row in per_row]
        ),
        "total_single_family_claims": sum(
            row["single_family_claim_count"] for row in per_row
        ),
        "total_single_family_obligations": sum(
            row["single_family_obligation_count"] for row in per_row
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_cases"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v82 Source-Quorum Proof Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V82 certifies whether the selected V78 minimal proof subgraph is "
        "supported by independent source families, instead of merely by "
        "multiple chunks that may share the same provenance.",
        "",
        "## Controlled Audit",
        "",
    ]
    for case in controlled:
        quorum = case["quorum"]
        lines.extend(
            [
                f"- `{case['tag']}` passed: `{case['passed']}`",
                f"  - Expected/status: `{case['expected']}` / `{quorum['quorum_status']}`",
                f"  - Families: `{quorum['independent_family_count']}`",
                f"  - Single-family claims/obligations: `{len(quorum['single_family_claims'])}` / `{len(quorum['single_family_obligations'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Offline Selective-30 Reconstruction",
            "",
            f"- Source file: `{selective['source_file']}`",
            f"- Samples: `{selective['sample_count']}`",
            f"- Proof sufficient: `{selective['proof_sufficient_count']}`",
            f"- Quorum passed: `{selective['quorum_passed_count']}`",
            f"- Monitoring required: `{selective['monitoring_required_count']}`",
            f"- Quorum status counts: `{selective['quorum_status_counts']}`",
            f"- Mean independent family count: `{selective['mean_independent_family_count']}`",
            f"- Mean selected evidence count: `{selective['mean_selected_evidence_count']}`",
            f"- Total single-family claims: `{selective['total_single_family_claims']}`",
            f"- Total single-family obligations: `{selective['total_single_family_obligations']}`",
            "",
            "## Interpretation",
            "",
            "This is a proof-certification mechanism, not an ablation. It "
            "adds a provenance-independence signal that can distinguish an "
            "actually multi-source proof from a proof whose apparent support "
            "comes from correlated chunks of the same source family.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = {
        "agent_version": "vsr-pathrag-v82",
        "protocol": "source_independent_proof_quorum_audit",
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
    selective = result["selective_30"]
    print("VSR-PathRAG v82 source-quorum proof audit")
    print(
        "controlled passed: "
        f"{sum(1 for case in result['controlled_cases'] if case['passed'])}/"
        f"{len(result['controlled_cases'])}"
    )
    print(f"selective-30 statuses: {selective['quorum_status_counts']}")
    print(
        "quorum passed/monitoring: "
        f"{selective['quorum_passed_count']}/"
        f"{selective['monitoring_required_count']}"
    )
    print(f"written: {OUT_JSON}")
    print(f"report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
