"""Audit VSR-PathRAG v79 counterfactual path necessity certification."""

from __future__ import annotations

import json
import statistics
import sys
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
from api.features.chats.vsr_pathrag import (  # noqa: E402
    build_signed_claim_evidence_graph,
)


SOURCE_JSONL = ROOT / "results" / "vsr_pathrag_v77_selective_30.jsonl"
OUT_JSON = ROOT / "results" / "vsr_pathrag_v79_counterfactual_necessity_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v79_counterfactual_necessity_report.md"


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


def _claim_graph(claims: list[str], evidence_summary: dict[str, Any]) -> dict[str, Any]:
    evidence = [
        {
            "label": row["source_id"],
            "kind": "source_excerpt",
            "text": row["text"],
            "source_ids": [row["source_id"]],
        }
        for row in evidence_summary.get("source_excerpts", [])
    ]
    return build_signed_claim_evidence_graph(claims, evidence)


def controlled_cases() -> list[dict[str, Any]]:
    cases = [
        {
            "tag": "essential_two_path",
            "question": "How did Aurora improve battery safety?",
            "answer": (
                "Aurora improved battery safety by adding thermal sensors "
                "that reduced overheating risk. Aurora trained operators "
                "to respond to battery temperature alarms."
            ),
            "query_graph": {
                "anchors": ["Aurora", "battery safety"],
                "relation_intents": ["impact"],
                "required_facets": ["thermal sensors", "operator training"],
            },
            "evidence_summary": {
                "source_excerpts": [
                    {
                        "source_id": "sensor#1",
                        "text": (
                            "Aurora improved battery safety by adding "
                            "thermal sensors that reduced overheating risk."
                        ),
                    },
                    {
                        "source_id": "training#2",
                        "text": (
                            "Aurora trained operators to respond to battery "
                            "temperature alarms."
                        ),
                    },
                ]
            },
            "claims": [
                "Aurora improved battery safety by adding thermal sensors that reduced overheating risk.",
                "Aurora trained operators to respond to battery temperature alarms.",
            ],
            "expected": "necessity_certified",
        },
        {
            "tag": "replaceable_duplicate",
            "question": "How did Aurora improve battery safety?",
            "answer": (
                "Aurora improved battery safety by adding thermal sensors "
                "that reduced overheating risk."
            ),
            "query_graph": {
                "anchors": ["Aurora", "battery safety"],
                "relation_intents": ["impact"],
                "required_facets": ["thermal sensors"],
            },
            "evidence_summary": {
                "source_excerpts": [
                    {
                        "source_id": "sensor#1",
                        "text": (
                            "Aurora improved battery safety by adding "
                            "thermal sensors that reduced overheating risk."
                        ),
                    },
                    {
                        "source_id": "sensor#2",
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
            "expected": "necessity_certified_with_replacements",
        },
    ]
    results = []
    for case in cases:
        claim_graph = _claim_graph(case["claims"], case["evidence_summary"])
        proof = build_minimal_proof_subgraph(
            case["question"],
            case["query_graph"],
            case["answer"],
            case["evidence_summary"],
            claim_graph,
            {"unsupported_claim_count": 0},
        )
        cert = certify_counterfactual_path_necessity(
            case["question"],
            case["query_graph"],
            case["answer"],
            case["evidence_summary"],
            claim_graph,
            {"unsupported_claim_count": 0},
            proof,
        )
        results.append(
            {
                "tag": case["tag"],
                "expected": case["expected"],
                "guidance": cert["submission_guidance"],
                "passed": cert["submission_guidance"] == case["expected"],
                "certification_passed": cert["certification_passed"],
                "selected_path_count": cert["selected_path_count"],
                "essential_path_count": cert["essential_path_count"],
                "replaceable_path_count": cert["replaceable_path_count"],
                "mean_necessity_score": cert["mean_necessity_score"],
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
        cert = certify_counterfactual_path_necessity(
            row.get("question", ""),
            trace.get("query_graph") or {},
            row.get("answer", ""),
            trace.get("evidence_summary") or {},
            trace.get("claim_evidence_graph") or {},
            trace.get("evidence_audit") or {},
            proof,
        )
        line_no = row.get("line_no") or row.get("_line_no") or row.get("index")
        per_row.append(
            {
                "line_no": line_no,
                "question": row.get("question", ""),
                "base_proof_sufficient": cert["base_proof_sufficient"],
                "certification_passed": cert["certification_passed"],
                "selected_path_count": cert["selected_path_count"],
                "essential_path_count": cert["essential_path_count"],
                "replaceable_path_count": cert["replaceable_path_count"],
                "mean_necessity_score": cert["mean_necessity_score"],
            }
        )
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "base_proof_sufficient_count": sum(
            1 for row in per_row if row["base_proof_sufficient"]
        ),
        "certification_passed_count": sum(
            1 for row in per_row if row["certification_passed"]
        ),
        "mean_selected_path_count": mean(
            [float(row["selected_path_count"]) for row in per_row]
        ),
        "total_essential_paths": sum(
            row["essential_path_count"] for row in per_row
        ),
        "total_replaceable_paths": sum(
            row["replaceable_path_count"] for row in per_row
        ),
        "mean_necessity_score": mean(
            [row["mean_necessity_score"] for row in per_row]
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_cases"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v79 Counterfactual Necessity Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V79 removes each selected proof path by intervention and reruns the "
        "v78 minimal proof optimizer. The resulting loss in claim coverage, "
        "obligation coverage, and joint objective certifies whether a path "
        "is essential or replaceable.",
        "",
        "## Controlled Audit",
        "",
        "| Case | Expected | Guidance | Pass | Essential | Replaceable | Mean necessity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in controlled:
        lines.append(
            "| {tag} | {expected} | {guidance} | {passed} | {essential} | {replaceable} | {score:.4f} |".format(
                tag=row["tag"],
                expected=row["expected"],
                guidance=row["guidance"],
                passed=row["passed"],
                essential=row["essential_path_count"],
                replaceable=row["replaceable_path_count"],
                score=row["mean_necessity_score"],
            )
        )
    lines.extend(
        [
            "",
            "## Offline Selective-30 Necessity Reconstruction",
            "",
            f"- Source file: `{selective['source_file']}`",
            f"- Samples: `{selective['sample_count']}`",
            f"- Base proof sufficient: `{selective['base_proof_sufficient_count']}/{selective['sample_count']}`",
            f"- Necessity certified: `{selective['certification_passed_count']}/{selective['sample_count']}`",
            f"- Mean selected path count: `{selective['mean_selected_path_count']}`",
            f"- Total essential paths: `{selective['total_essential_paths']}`",
            f"- Total replaceable paths: `{selective['total_replaceable_paths']}`",
            f"- Mean necessity score: `{selective['mean_necessity_score']}`",
            "",
            "## Interpretation",
            "",
            "Rows that are proof-sufficient but not necessity-certified have "
            "replaceable or low-impact support paths. That is useful for "
            "V80: risk calibration can submit strongly necessary proofs "
            "confidently and monitor replaceable or incomplete proofs.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = {
        "agent_version": "vsr-pathrag-v79",
        "protocol": "counterfactual_evidence_path_necessity_audit",
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
    print("VSR-PathRAG v79 counterfactual necessity audit")
    print(f"controlled expected outcomes: {controlled_pass}/{len(result['controlled_cases'])}")
    print(
        "selective-30 necessity certified: "
        f"{selective['certification_passed_count']}/{selective['sample_count']}"
    )
    print(
        "essential paths: "
        f"{selective['total_essential_paths']}; "
        "replaceable paths: "
        f"{selective['total_replaceable_paths']}; "
        "mean necessity: "
        f"{selective['mean_necessity_score']}"
    )
    print(f"written: {OUT_JSON}")
    print(f"report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
