"""Audit VSR-PathRAG v78 minimal proof subgraph innovation.

This audit is not an ablation and does not call a judge or read reference
answers. It validates the new v78 mechanism in two ways:
1. controlled cross-domain proof construction, and
2. offline proof-subgraph reconstruction on the existing 30 selective
   agent outputs.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from api.features.chats.minimal_proof_subgraph import (  # noqa: E402
    build_minimal_proof_subgraph,
)
from api.features.chats.vsr_pathrag import (  # noqa: E402
    build_signed_claim_evidence_graph,
)


SOURCE_JSONL = ROOT / "results" / "vsr_pathrag_v77_selective_30.jsonl"
OUT_JSON = ROOT / "results" / "vsr_pathrag_v78_minimal_proof_subgraph_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v78_minimal_proof_subgraph_report.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(rows: list[float]) -> float:
    if not rows:
        return 0.0
    return round(statistics.fmean(rows), 4)


def controlled_cases() -> list[dict[str, Any]]:
    cases = [
        {
            "tag": "legal_priority",
            "question": "What is the purpose of the Intercreditor Agreement?",
            "answer": (
                "The Intercreditor Agreement establishes priorities among "
                "secured parties and covers obligations under the senior "
                "credit facility."
            ),
            "query_graph": {
                "anchors": ["Intercreditor Agreement", "secured parties"],
                "relation_intents": ["fact"],
                "required_facets": ["purpose", "priorities", "obligations"],
            },
            "evidence_summary": {
                "source_excerpts": [
                    {
                        "source_id": "legal-priority#1",
                        "text": (
                            "The purpose of the Intercreditor Agreement is "
                            "to establish priorities among secured parties."
                        ),
                    },
                    {
                        "source_id": "legal-scope#2",
                        "text": (
                            "Its scope includes all obligations under the "
                            "senior credit facility."
                        ),
                    },
                ]
            },
            "claims": [
                "The Intercreditor Agreement establishes priorities among secured parties.",
                "The Intercreditor Agreement covers obligations under the senior credit facility.",
            ],
        },
        {
            "tag": "biology_mechanism",
            "question": "How did Aurora improve battery safety?",
            "answer": (
                "Aurora improved battery safety by adding thermal sensors "
                "that reduced overheating risk."
            ),
            "query_graph": {
                "anchors": ["Aurora", "battery safety"],
                "relation_intents": ["impact"],
                "required_facets": ["thermal sensors", "overheating risk"],
            },
            "evidence_summary": {
                "source_excerpts": [
                    {
                        "source_id": "battery-safety#1",
                        "text": (
                            "Aurora improved battery safety by adding "
                            "thermal sensors that reduced overheating risk."
                        ),
                    },
                    {
                        "source_id": "battery-duplicate#2",
                        "text": (
                            "Aurora improved battery safety by adding "
                            "thermal sensors that reduced overheating risk."
                        ),
                    },
                ]
            },
            "claims": [
                "Aurora improved battery safety by adding thermal sensors that reduced overheating risk.",
            ],
        },
        {
            "tag": "history_role",
            "question": "What role did the harbor reforms play in Odessa?",
            "answer": (
                "The harbor reforms helped Odessa expand trade and become "
                "a more important Black Sea port."
            ),
            "query_graph": {
                "anchors": ["harbor reforms", "Odessa"],
                "relation_intents": ["role"],
                "required_facets": ["trade", "Black Sea port"],
            },
            "evidence_summary": {
                "source_excerpts": [
                    {
                        "source_id": "odessa-trade#1",
                        "text": (
                            "The harbor reforms helped Odessa expand trade "
                            "and become a more important Black Sea port."
                        ),
                    }
                ]
            },
            "claims": [
                "The harbor reforms helped Odessa expand trade and become a more important Black Sea port.",
            ],
        },
    ]
    results = []
    for case in cases:
        evidence = [
            {
                "label": row["source_id"],
                "kind": "source_excerpt",
                "text": row["text"],
                "source_ids": [row["source_id"]],
            }
            for row in case["evidence_summary"]["source_excerpts"]
        ]
        claim_graph = build_signed_claim_evidence_graph(
            case["claims"],
            evidence,
        )
        proof = build_minimal_proof_subgraph(
            case["question"],
            case["query_graph"],
            case["answer"],
            case["evidence_summary"],
            claim_graph,
            {"unsupported_claim_count": 0},
        )
        results.append(
            {
                "tag": case["tag"],
                "proof_sufficient": proof["proof_sufficient"],
                "selected_count": proof["selected_count"],
                "claim_support_coverage_ratio": proof[
                    "claim_support_coverage_ratio"
                ],
                "obligation_coverage_ratio": proof[
                    "obligation_coverage_ratio"
                ],
                "redundant_evidence_pruned_count": proof[
                    "redundant_evidence_pruned_count"
                ],
                "minimality_all_selected_necessary": proof[
                    "minimality_certificate"
                ]["all_selected_necessary"],
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
        line_no = row.get("line_no") or row.get("_line_no") or row.get("index")
        per_row.append(
            {
                "line_no": line_no,
                "question": row.get("question", ""),
                "proof_sufficient": proof["proof_sufficient"],
                "selected_count": proof["selected_count"],
                "claim_support_coverage_ratio": proof[
                    "claim_support_coverage_ratio"
                ],
                "obligation_coverage_ratio": proof[
                    "obligation_coverage_ratio"
                ],
                "joint_objective": proof["joint_objective"],
                "redundant_evidence_pruned_count": proof[
                    "redundant_evidence_pruned_count"
                ],
                "selected_source_ids": [
                    source_id
                    for evidence in proof["selected_evidence"]
                    for source_id in evidence["source_ids"]
                ],
            }
        )
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "proof_sufficient_count": sum(
            1 for row in per_row if row["proof_sufficient"]
        ),
        "mean_selected_count": mean(
            [float(row["selected_count"]) for row in per_row]
        ),
        "mean_claim_support_coverage_ratio": mean(
            [row["claim_support_coverage_ratio"] for row in per_row]
        ),
        "mean_obligation_coverage_ratio": mean(
            [row["obligation_coverage_ratio"] for row in per_row]
        ),
        "mean_joint_objective": mean(
            [row["joint_objective"] for row in per_row]
        ),
        "total_redundant_evidence_pruned": sum(
            row["redundant_evidence_pruned_count"] for row in per_row
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_cases"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v78 Minimal Proof Subgraph Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V78 adds a deterministic minimal proof subgraph optimizer that "
        "jointly covers question obligations and signed answer claims. It "
        "does not call a judge and does not read reference answers.",
        "",
        "## Controlled Cross-Domain Audit",
        "",
        "| Case | Proof sufficient | Selected | Claim coverage | Obligation coverage | Redundant pruned | Minimality |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in controlled:
        lines.append(
            "| {tag} | {proof} | {selected} | {claim:.3f} | {obl:.3f} | {pruned} | {minimal} |".format(
                tag=row["tag"],
                proof=row["proof_sufficient"],
                selected=row["selected_count"],
                claim=row["claim_support_coverage_ratio"],
                obl=row["obligation_coverage_ratio"],
                pruned=row["redundant_evidence_pruned_count"],
                minimal=row["minimality_all_selected_necessary"],
            )
        )
    lines.extend(
        [
            "",
            "## Offline Selective-30 Proof Reconstruction",
            "",
            f"- Source file: `{selective['source_file']}`",
            f"- Samples: `{selective['sample_count']}`",
            f"- Proof sufficient: `{selective['proof_sufficient_count']}/{selective['sample_count']}`",
            f"- Mean selected evidence count: `{selective['mean_selected_count']}`",
            f"- Mean claim-support coverage: `{selective['mean_claim_support_coverage_ratio']}`",
            f"- Mean obligation coverage: `{selective['mean_obligation_coverage_ratio']}`",
            f"- Mean joint objective: `{selective['mean_joint_objective']}`",
            f"- Redundant evidence pruned: `{selective['total_redundant_evidence_pruned']}`",
            "",
            "## Interpretation",
            "",
            "This is a mechanism validation, not a performance claim. The "
            "proof subgraph exposes which evidence nodes are jointly needed "
            "for answer claims and question obligations. Rows marked "
            "`proof_incomplete_monitor` are not failures of retrieval by "
            "themselves; they identify where V79 counterfactual necessity "
            "and V80 risk calibration should intervene before final claims "
            "are submitted.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = {
        "agent_version": "vsr-pathrag-v78",
        "protocol": "minimal_proof_subgraph_joint_obligation_claim_audit",
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
    controlled_pass = sum(
        1 for row in result["controlled_cases"] if row["proof_sufficient"]
    )
    selective = result["selective_30"]
    print("VSR-PathRAG v78 minimal proof subgraph audit")
    print(f"controlled proof sufficient: {controlled_pass}/{len(result['controlled_cases'])}")
    print(
        "selective-30 proof sufficient: "
        f"{selective['proof_sufficient_count']}/{selective['sample_count']}"
    )
    print(
        "mean claim coverage: "
        f"{selective['mean_claim_support_coverage_ratio']}; "
        "mean obligation coverage: "
        f"{selective['mean_obligation_coverage_ratio']}"
    )
    print(f"written: {OUT_JSON}")
    print(f"report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
