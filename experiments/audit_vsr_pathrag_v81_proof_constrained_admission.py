"""Audit VSR-PathRAG v81 proof-constrained claim admission."""

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

from api.features.chats.agent_controller import (  # noqa: E402
    assess_terminal_answer,
    terminal_assessment_score,
)
from api.features.chats.proof_constrained_admission import (  # noqa: E402
    build_proof_constrained_admission,
)


SOURCE_JSONL = ROOT / "results" / "vsr_pathrag_v77_selective_30.jsonl"
OUT_JSON = ROOT / "results" / "vsr_pathrag_v81_proof_constrained_admission_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v81_proof_constrained_admission_report.md"


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
    evidence_summary = {
        "source_excerpts": [
            {
                "source_id": "aurora#sensors",
                "text": (
                    "Aurora improved battery safety by adding thermal "
                    "sensors that reduced overheating risk."
                ),
            }
        ]
    }
    query_graph = {
        "anchors": ["Aurora", "battery safety"],
        "relation_intents": ["impact"],
        "required_facets": ["thermal sensors", "overheating risk"],
    }
    original_answer = (
        "Aurora improved battery safety by adding thermal sensors that "
        "reduced overheating risk. The archive states that cobalt mining "
        "subsidies in Chile doubled municipal tax revenue during 2014."
    )
    original = assess_terminal_answer(
        question,
        original_answer,
        evidence_summary,
        None,
        query_graph,
    )
    admission = build_proof_constrained_admission(
        original_answer,
        original["claim_evidence_graph"],
        original["minimal_proof_subgraph"],
        original["counterfactual_path_necessity"],
    )
    candidate = assess_terminal_answer(
        question,
        admission["candidate_answer"],
        evidence_summary,
        None,
        query_graph,
    )
    original_action = original["risk_calibrated_submission"]["action"]
    candidate_action = candidate["risk_calibrated_submission"]["action"]
    return {
        "tag": "unsupported_claim_pruned_to_proof_answer",
        "passed": bool(
            admission["changed"]
            and admission["rejected_claim_count"] == 1
            and "cobalt mining" not in admission["candidate_answer"]
            and original_action == "block_unsupported"
            and candidate_action in {"submit", "submit_with_monitoring"}
        ),
        "original_action": original_action,
        "candidate_action": candidate_action,
        "original_score": terminal_assessment_score(original),
        "candidate_score": terminal_assessment_score(candidate),
        "admitted_claim_count": admission["admitted_claim_count"],
        "rejected_claim_count": admission["rejected_claim_count"],
    }


def _would_select(
    current: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    current_action = current["risk_calibrated_submission"]["action"]
    candidate_action = candidate["risk_calibrated_submission"]["action"]
    current_unsupported = current["evidence_audit"]["unsupported_claim_count"]
    candidate_unsupported = candidate["evidence_audit"]["unsupported_claim_count"]
    if candidate_action not in {"submit", "submit_with_monitoring"}:
        return False
    if candidate_unsupported > current_unsupported:
        return False
    return bool(
        current_action in {"block_unsupported", "defer_or_fallback"}
        or (candidate_action == "submit" and current_action != "submit")
        or candidate_unsupported < current_unsupported
        or terminal_assessment_score(candidate)
        >= terminal_assessment_score(current) - 0.10
    )


def selective_30_audit() -> dict[str, Any]:
    rows = load_jsonl(SOURCE_JSONL)
    per_row = []
    for row in rows:
        trace = row.get("trace") or {}
        question = row.get("question", "")
        evidence_summary = trace.get("evidence_summary") or {}
        query_graph = trace.get("query_graph") or {}
        current = assess_terminal_answer(
            question,
            row.get("answer", ""),
            evidence_summary,
            trace.get("structured_confidence"),
            query_graph,
        )
        admission = build_proof_constrained_admission(
            row.get("answer", ""),
            current["claim_evidence_graph"],
            current["minimal_proof_subgraph"],
            current["counterfactual_path_necessity"],
        )
        candidate = None
        selected = False
        if admission.get("changed"):
            candidate = assess_terminal_answer(
                question,
                admission["candidate_answer"],
                evidence_summary,
                trace.get("structured_confidence"),
                query_graph,
            )
            selected = _would_select(current, candidate)
        line_no = row.get("line_no") or row.get("_line_no") or row.get("index")
        per_row.append(
            {
                "line_no": line_no,
                "question": question,
                "changed": bool(admission.get("changed")),
                "would_select": selected,
                "original_action": current["risk_calibrated_submission"][
                    "action"
                ],
                "candidate_action": (
                    candidate["risk_calibrated_submission"]["action"]
                    if candidate
                    else None
                ),
                "original_score": terminal_assessment_score(current),
                "candidate_score": (
                    terminal_assessment_score(candidate)
                    if candidate
                    else None
                ),
                "admitted_claim_count": admission["admitted_claim_count"],
                "rejected_claim_count": admission["rejected_claim_count"],
                "necessary_claim_count": admission["necessary_claim_count"],
            }
        )
    action_pairs = Counter(
        (
            row["original_action"],
            row["candidate_action"] or "none",
        )
        for row in per_row
    )
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "candidate_changed_count": sum(1 for row in per_row if row["changed"]),
        "candidate_would_select_count": sum(
            1 for row in per_row if row["would_select"]
        ),
        "total_rejected_claims": sum(
            row["rejected_claim_count"] for row in per_row
        ),
        "total_admitted_claims": sum(
            row["admitted_claim_count"] for row in per_row
        ),
        "mean_original_score": mean([row["original_score"] for row in per_row]),
        "mean_candidate_score": mean(
            [
                row["candidate_score"]
                for row in per_row
                if row["candidate_score"] is not None
            ]
        ),
        "action_pair_counts": {
            f"{left}->{right}": count
            for (left, right), count in action_pairs.items()
        },
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_case"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v81 Proof-Constrained Admission Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V81 admits final answer claims only when they are supported by the "
        "signed claim graph and covered by the V78 minimal proof subgraph. "
        "Unsupported or proof-uncovered claims are rejected before terminal "
        "submission.",
        "",
        "## Controlled Audit",
        "",
        f"- Passed: `{controlled['passed']}`",
        f"- Original action: `{controlled['original_action']}`",
        f"- Candidate action: `{controlled['candidate_action']}`",
        f"- Original score: `{controlled['original_score']}`",
        f"- Candidate score: `{controlled['candidate_score']}`",
        f"- Admitted/rejected claims: `{controlled['admitted_claim_count']}` / `{controlled['rejected_claim_count']}`",
        "",
        "## Offline Selective-30 Admission Reconstruction",
        "",
        f"- Source file: `{selective['source_file']}`",
        f"- Samples: `{selective['sample_count']}`",
        f"- Candidate changed: `{selective['candidate_changed_count']}`",
        f"- Candidate would select: `{selective['candidate_would_select_count']}`",
        f"- Total admitted claims: `{selective['total_admitted_claims']}`",
        f"- Total rejected claims: `{selective['total_rejected_claims']}`",
        f"- Mean original terminal score: `{selective['mean_original_score']}`",
        f"- Mean candidate terminal score: `{selective['mean_candidate_score']}`",
        f"- Action transitions: `{selective['action_pair_counts']}`",
        "",
        "## Interpretation",
        "",
        "This is a generation-control mechanism, not an ablation. It turns "
        "the proof graph into a claim admission policy so that final answers "
        "can be shortened or corrected by removing claims that lack proof "
        "coverage.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = {
        "agent_version": "vsr-pathrag-v81",
        "protocol": "proof_constrained_claim_admission_audit",
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
    print("VSR-PathRAG v81 proof-constrained admission audit")
    print(f"controlled passed: {controlled['passed']}")
    print(
        "selective-30 changed/selectable: "
        f"{selective['candidate_changed_count']}/"
        f"{selective['candidate_would_select_count']}"
    )
    print(
        "admitted/rejected claims: "
        f"{selective['total_admitted_claims']}/"
        f"{selective['total_rejected_claims']}"
    )
    print(f"written: {OUT_JSON}")
    print(f"report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
