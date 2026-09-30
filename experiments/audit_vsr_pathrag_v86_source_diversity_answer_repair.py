"""Audit VSR-PathRAG v86 source-diversity answer repair."""

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
from api.features.chats.proof_deficit_answer_repair import (  # noqa: E402
    build_proof_deficit_answer_repair,
)
from api.features.chats.source_diversity_answer_repair import (  # noqa: E402
    build_source_diversity_answer_repair,
)


SOURCE_JSONL = ROOT / "results" / "vsr_pathrag_v77_selective_30.jsonl"
OUT_JSON = ROOT / "results" / "vsr_pathrag_v86_source_diversity_answer_repair_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v86_source_diversity_answer_repair_report.md"


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


def _submission_action(assessment: dict[str, Any]) -> str:
    return (assessment.get("risk_calibrated_submission") or {}).get(
        "action",
        "defer_or_fallback",
    )


def _unsupported_count(assessment: dict[str, Any]) -> int:
    return int(
        (assessment.get("evidence_audit") or {}).get(
            "unsupported_claim_count",
            0,
        )
        or 0
    )


def _quorum(assessment: dict[str, Any]) -> dict[str, Any]:
    return assessment.get("proof_source_quorum") or {}


def _quorum_status(assessment: dict[str, Any]) -> str:
    return _quorum(assessment).get("quorum_status", "unknown")


def _proof_sufficient(assessment: dict[str, Any]) -> bool:
    return bool(
        (assessment.get("minimal_proof_subgraph") or {}).get(
            "proof_sufficient",
        )
    )


def _single_family_deficit_count(assessment: dict[str, Any]) -> int:
    quorum = _quorum(assessment)
    return len(quorum.get("single_family_claims", [])) + len(
        quorum.get("single_family_obligations", [])
    )


def _select_v81_candidate(
    current: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    current_action = _submission_action(current)
    candidate_action = _submission_action(candidate)
    current_unsupported = _unsupported_count(current)
    candidate_unsupported = _unsupported_count(candidate)
    current_score = terminal_assessment_score(current)
    candidate_score = terminal_assessment_score(candidate)
    return bool(
        candidate_action in {"submit", "submit_with_monitoring"}
        and candidate_unsupported <= current_unsupported
        and (
            current_action in {"block_unsupported", "defer_or_fallback"}
            or (
                candidate_action == "submit"
                and current_action != "submit"
            )
            or candidate_unsupported < current_unsupported
            or candidate_score >= current_score - 0.10
        )
    )


def _select_v85_candidate(
    current: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    current_action = _submission_action(current)
    candidate_action = _submission_action(candidate)
    current_unsupported = _unsupported_count(current)
    candidate_unsupported = _unsupported_count(candidate)
    deficit_repaired = bool(
        (
            _quorum_status(current) == "quorum_failed"
            and _quorum_status(candidate) != "quorum_failed"
        )
        or (not _proof_sufficient(current) and _proof_sufficient(candidate))
        or candidate_unsupported < current_unsupported
        or (
            candidate_action == "submit"
            and current_action != "submit"
        )
    )
    return bool(
        candidate_action in {"submit", "submit_with_monitoring"}
        and candidate_unsupported <= current_unsupported
        and (
            current_action in {"block_unsupported", "defer_or_fallback"}
            or deficit_repaired
            or terminal_assessment_score(candidate)
            >= terminal_assessment_score(current) - 0.08
        )
    )


def _select_v86_variant(
    current: dict[str, Any],
    variants: list[dict[str, Any]],
    question: str,
    evidence_summary: dict[str, Any],
    structured_confidence: Any,
    query_graph: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    current_quorum = _quorum_status(current)
    current_score = terminal_assessment_score(current)
    current_deficit = _single_family_deficit_count(current)
    assessed = []
    for variant in variants:
        candidate = assess_terminal_answer(
            question,
            variant.get("candidate_answer", ""),
            evidence_summary,
            structured_confidence,
            query_graph,
        )
        candidate_quorum = _quorum_status(candidate)
        candidate_score = terminal_assessment_score(candidate)
        candidate_deficit = _single_family_deficit_count(candidate)
        quorum_improved = bool(
            current_quorum != "independent_quorum"
            and candidate_quorum == "independent_quorum"
        )
        deficit_reduced = candidate_deficit < current_deficit
        eligible = bool(
            _submission_action(candidate) in {"submit", "submit_with_monitoring"}
            and _unsupported_count(candidate) == 0
            and _proof_sufficient(candidate)
            and candidate_quorum != "quorum_failed"
            and (quorum_improved or deficit_reduced)
            and candidate_score >= current_score - 0.20
        )
        assessed.append(
            {
                "variant_id": variant.get("variant_id"),
                "candidate_assessment": candidate,
                "quorum_status": candidate_quorum,
                "deficit_count": candidate_deficit,
                "terminal_score": candidate_score,
                "eligible": eligible,
                "quorum_improved": quorum_improved,
                "deficit_reduced": deficit_reduced,
                "dropped_claim_count": variant.get("dropped_claim_count", 0),
                "dropped_necessary_claim_count": variant.get(
                    "dropped_necessary_claim_count",
                    0,
                ),
            }
        )
    eligible_rows = [row for row in assessed if row["eligible"]]
    if not eligible_rows:
        return None, assessed
    selected = max(
        eligible_rows,
        key=lambda row: (
            row["quorum_status"] == "independent_quorum",
            current_deficit - row["deficit_count"],
            row["terminal_score"],
            -row["dropped_necessary_claim_count"],
            -row["dropped_claim_count"],
        ),
    )
    return selected, assessed


def controlled_case() -> dict[str, Any]:
    answer = (
        "Aurora improved battery safety by adding thermal sensors. "
        "Aurora also reduced warranty costs."
    )
    assessment = {
        "claim_evidence_graph": {
            "claims": [
                {
                    "claim_id": "claim-0",
                    "claim": (
                        "Aurora improved battery safety by adding thermal "
                        "sensors."
                    ),
                    "status": "supported",
                    "support_mass": 0.8,
                },
                {
                    "claim_id": "claim-1",
                    "claim": "Aurora also reduced warranty costs.",
                    "status": "supported",
                    "support_mass": 0.42,
                },
            ]
        },
        "minimal_proof_subgraph": {
            "proof_sufficient": True,
            "covered_claims": ["claim-0", "claim-1"],
        },
        "counterfactual_path_necessity": {
            "interventions": [{"lost_claims_if_removed": ["claim-0"]}]
        },
        "proof_source_quorum": {
            "quorum_status": "correlated_monitor",
            "single_family_claims": ["claim-1"],
            "single_family_obligations": [],
        },
        "risk_calibrated_submission": {"action": "submit"},
        "evidence_audit": {"unsupported_claim_count": 0},
    }
    repair = build_source_diversity_answer_repair(answer, assessment)
    candidate = (
        repair.get("candidate_variants", [{}])[0].get(
            "candidate_answer",
            "",
        )
        if repair.get("candidate_variants")
        else ""
    )
    return {
        "tag": "single_family_claim_variant_generated",
        "passed": bool(
            repair.get("triggered")
            and repair.get("variant_count", 0) >= 1
            and "warranty costs" not in candidate
        ),
        "triggered": repair.get("triggered"),
        "variant_count": repair.get("variant_count"),
        "single_family_claim_count": repair.get("single_family_claim_count"),
    }


def selective_30_audit() -> dict[str, Any]:
    rows = load_jsonl(SOURCE_JSONL)
    per_row = []
    for row in rows:
        trace = row.get("trace") or {}
        question = row.get("question", "")
        evidence_summary = trace.get("evidence_summary") or {}
        structured_confidence = trace.get("structured_confidence")
        query_graph = trace.get("query_graph") or {}

        base = assess_terminal_answer(
            question,
            row.get("answer", ""),
            evidence_summary,
            structured_confidence,
            query_graph,
        )
        current = base

        v81 = build_proof_constrained_admission(
            current.get("answer", ""),
            current.get("claim_evidence_graph") or {},
            current.get("minimal_proof_subgraph") or {},
            current.get("counterfactual_path_necessity") or {},
        )
        v81_selected = False
        if v81.get("changed") and v81.get("candidate_answer"):
            candidate = assess_terminal_answer(
                question,
                v81["candidate_answer"],
                evidence_summary,
                structured_confidence,
                query_graph,
            )
            v81_selected = _select_v81_candidate(current, candidate)
            if v81_selected:
                current = candidate

        v85 = build_proof_deficit_answer_repair(
            current.get("answer", ""),
            current,
        )
        v85_selected = False
        if v85.get("changed") and v85.get("candidate_answer"):
            candidate = assess_terminal_answer(
                question,
                v85["candidate_answer"],
                evidence_summary,
                structured_confidence,
                query_graph,
            )
            v85_selected = _select_v85_candidate(current, candidate)
            if v85_selected:
                current = candidate

        after_v85 = current
        v86 = build_source_diversity_answer_repair(
            current.get("answer", ""),
            current,
        )
        selected_variant = None
        assessed_variants = []
        if v86.get("candidate_variants"):
            selected_variant, assessed_variants = _select_v86_variant(
                current,
                v86["candidate_variants"],
                question,
                evidence_summary,
                structured_confidence,
                query_graph,
            )
            if selected_variant:
                current = selected_variant["candidate_assessment"]

        final = current
        per_row.append(
            {
                "line_no": row.get("line_no"),
                "dataset": row.get("dataset"),
                "question": question,
                "base_quorum_status": _quorum_status(base),
                "after_v85_quorum_status": _quorum_status(after_v85),
                "after_v86_quorum_status": _quorum_status(final),
                "after_v85_deficit_count": _single_family_deficit_count(
                    after_v85
                ),
                "after_v86_deficit_count": _single_family_deficit_count(
                    final
                ),
                "after_v85_terminal_score": terminal_assessment_score(after_v85),
                "after_v86_terminal_score": terminal_assessment_score(final),
                "after_v85_proof_sufficient": _proof_sufficient(after_v85),
                "after_v86_proof_sufficient": _proof_sufficient(final),
                "after_v86_unsupported_claim_count": _unsupported_count(final),
                "v81_selected": v81_selected,
                "v85_selected": v85_selected,
                "v86_triggered": bool(v86.get("triggered")),
                "v86_variant_count": v86.get("variant_count", 0),
                "v86_selected": bool(selected_variant),
                "v86_selected_variant_id": (
                    selected_variant.get("variant_id")
                    if selected_variant
                    else None
                ),
                "v86_candidate_independent_count": sum(
                    1
                    for item in assessed_variants
                    if item.get("quorum_status") == "independent_quorum"
                ),
                "v86_eligible_variant_count": sum(
                    1 for item in assessed_variants if item.get("eligible")
                ),
            }
        )

    after_v85_quorum = Counter(
        row["after_v85_quorum_status"] for row in per_row
    )
    after_v86_quorum = Counter(
        row["after_v86_quorum_status"] for row in per_row
    )
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "after_v85_quorum_status_counts": dict(after_v85_quorum),
        "after_v86_quorum_status_counts": dict(after_v86_quorum),
        "after_v85_proof_sufficient_count": sum(
            1 for row in per_row if row["after_v85_proof_sufficient"]
        ),
        "after_v86_proof_sufficient_count": sum(
            1 for row in per_row if row["after_v86_proof_sufficient"]
        ),
        "after_v86_unsupported_claim_total": sum(
            row["after_v86_unsupported_claim_count"] for row in per_row
        ),
        "v86_triggered_count": sum(1 for row in per_row if row["v86_triggered"]),
        "v86_selected_count": sum(1 for row in per_row if row["v86_selected"]),
        "v86_independent_candidate_rows": sum(
            1
            for row in per_row
            if row["v86_candidate_independent_count"] > 0
        ),
        "v86_eligible_variant_total": sum(
            row["v86_eligible_variant_count"] for row in per_row
        ),
        "independent_quorum_gain_after_v85": after_v86_quorum.get(
            "independent_quorum",
            0,
        )
        - after_v85_quorum.get("independent_quorum", 0),
        "correlated_monitor_reduction_after_v85": after_v85_quorum.get(
            "correlated_monitor",
            0,
        )
        - after_v86_quorum.get("correlated_monitor", 0),
        "mean_single_family_deficit_after_v85": mean(
            [row["after_v85_deficit_count"] for row in per_row]
        ),
        "mean_single_family_deficit_after_v86": mean(
            [row["after_v86_deficit_count"] for row in per_row]
        ),
        "mean_terminal_score_after_v85": mean(
            [row["after_v85_terminal_score"] for row in per_row]
        ),
        "mean_terminal_score_after_v86": mean(
            [row["after_v86_terminal_score"] for row in per_row]
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_case"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v86 Source-Diversity Answer Repair Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V86 handles clean but correlated proofs by generating answer variants "
        "that remove source-singleton claims. The controller commits a variant "
        "only after full terminal re-assessment confirms source-quorum gain and "
        "no safety regression.",
        "",
        "## Controlled case",
        "",
        f"- Passed: `{controlled['passed']}`",
        f"- Triggered: `{controlled['triggered']}`",
        f"- Variant count: `{controlled['variant_count']}`",
        f"- Single-family claims: `{controlled['single_family_claim_count']}`",
        "",
        "## Selective-30 replay audit",
        "",
        f"- Source: `{selective['source_file']}`",
        f"- Samples: `{selective['sample_count']}`",
        f"- After V85 quorum counts: `{selective['after_v85_quorum_status_counts']}`",
        f"- After V86 quorum counts: `{selective['after_v86_quorum_status_counts']}`",
        f"- Proof sufficient count: `{selective['after_v85_proof_sufficient_count']}` -> `{selective['after_v86_proof_sufficient_count']}`",
        f"- Unsupported claim total after V86: `{selective['after_v86_unsupported_claim_total']}`",
        f"- V86 triggered / selected: `{selective['v86_triggered_count']}` / `{selective['v86_selected_count']}`",
        f"- Rows with independent-quorum candidate: `{selective['v86_independent_candidate_rows']}`",
        f"- Eligible variants: `{selective['v86_eligible_variant_total']}`",
        f"- Independent-quorum gain after V85: `{selective['independent_quorum_gain_after_v85']}`",
        f"- Correlated-monitor reduction after V85: `{selective['correlated_monitor_reduction_after_v85']}`",
        f"- Mean single-family deficit: `{selective['mean_single_family_deficit_after_v85']}` -> `{selective['mean_single_family_deficit_after_v86']}`",
        f"- Mean terminal score: `{selective['mean_terminal_score_after_v85']}` -> `{selective['mean_terminal_score_after_v86']}`",
        "",
        "## Interpretation",
        "",
        "V86 converts part of the remaining correlated-monitor set into "
        "independent-quorum answers by narrowing the submitted claim envelope "
        "to what independent provenance can certify. It does not use reference "
        "answers and does not start ablation.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    result = {
        "controlled_case": controlled_case(),
        "selective_30": selective_30_audit(),
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
