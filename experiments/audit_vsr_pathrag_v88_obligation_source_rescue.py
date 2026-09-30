"""Audit VSR-PathRAG v88 obligation-source rescue.

V88 runs after V83 source-quorum backfill and before V87 criticality
normalization.  It targets *only* obligations that V87 would classify as
critical (real domain obligations, not synthetic artifacts).  For those
critical obligations, V88 performs a focused second-pass search with a
separate obligation-only budget.
"""

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
from api.features.chats.obligation_criticality_quorum import (  # noqa: E402
    normalize_obligation_criticality_quorum,
)
from api.features.chats.obligation_source_rescue import (  # noqa: E402
    build_obligation_source_rescue,
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
OUT_JSON = ROOT / "results" / "vsr_pathrag_v88_obligation_source_rescue_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v88_obligation_source_rescue_report.md"


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


def _quorum_status(assessment: dict[str, Any]) -> str:
    return (assessment.get("obligation_criticality_quorum") or {}).get(
        "after_quorum_status",
        (assessment.get("obligation_source_rescue") or {}).get(
            "after_quorum_status",
            (assessment.get("proof_source_quorum") or {}).get(
                "quorum_status",
                "unknown",
            )
        ),
    )


def _raw_pre_v87_status(assessment: dict[str, Any]) -> str:
    base = (
        assessment.get("obligation_criticality_quorum") or {}
    ).get("base_proof_source_quorum")
    return (base or assessment.get("proof_source_quorum") or {}).get(
        "quorum_status",
        "unknown",
    )


def _pre_rescue_status(assessment: dict[str, Any]) -> str:
    rescue = assessment.get("obligation_source_rescue") or {}
    return rescue.get("before_quorum_status", "unknown")


def _post_rescue_status(assessment: dict[str, Any]) -> str:
    rescue = assessment.get("obligation_source_rescue") or {}
    return rescue.get("after_quorum_status", "unknown")


def _proof_sufficient(assessment: dict[str, Any]) -> bool:
    return bool(
        (assessment.get("minimal_proof_subgraph") or {}).get(
            "proof_sufficient",
        )
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


def _select_v86_candidate(
    current: dict[str, Any],
    variants: list[dict[str, Any]],
    question: str,
    evidence_summary: dict[str, Any],
    structured_confidence: Any,
    query_graph: dict[str, Any],
) -> dict[str, Any]:
    current_score = terminal_assessment_score(current)
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for variant in variants:
        candidate = assess_terminal_answer(
            question,
            variant.get("candidate_answer", ""),
            evidence_summary,
            structured_confidence,
            query_graph,
        )
        candidate_score = terminal_assessment_score(candidate)
        quorum_improved = bool(
            _quorum_status(current) != "independent_quorum"
            and _quorum_status(candidate) == "independent_quorum"
        )
        eligible = bool(
            _submission_action(candidate) in {"submit", "submit_with_monitoring"}
            and _unsupported_count(candidate) == 0
            and _proof_sufficient(candidate)
            and _quorum_status(candidate) != "quorum_failed"
            and (quorum_improved or candidate_score >= current_score - 0.20)
        )
        if not eligible:
            continue
        key = (
            _quorum_status(candidate) == "independent_quorum",
            candidate_score,
            -variant.get("dropped_necessary_claim_count", 0),
            -variant.get("dropped_claim_count", 0),
        )
        if best is None or key > best[0]:
            best = (key, candidate)
    return best[1] if best else current


def controlled_case() -> dict[str, Any]:
    """V88 should trigger on correlated_monitor with critical obligations."""
    quorum = {
        "proof_sufficient": True,
        "independent_family_count": 2,
        "quorum_status": "correlated_monitor",
        "single_family_claims": [],
        "missing_claims": [],
        "single_family_obligations": [
            "purpose_scope:payment_security",
        ],
        "missing_obligations": [],
        "claim_family_counts": {
            "claim-0": {
                "family_ids": ["doc-a", "doc-b"],
                "independent_family_count": 2,
            }
        },
        "submission_guidance": "correlated_source_monitor",
    }
    result = build_obligation_source_rescue(
        question="What is the purpose of the Collateral Trust Agreement?",
        query_graph={},
        evidence_summary={"entities": [], "relationships": [], "source_excerpts": []},
        claim_evidence_graph={},
        evidence_audit=None,
        minimal_proof_subgraph={
            "selected_evidence": [],
            "selected_count": 0,
            "covered_claims": ["claim-0"],
            "covered_obligations": ["purpose_scope:payment_security"],
            "required_claim_count": 1,
            "required_obligation_count": 1,
            "proof_sufficient": True,
        },
        proof_source_quorum=quorum,
    )
    return {
        "tag": "critical_obligation_rescue_triggered",
        "passed": result.get("triggered"),
        "triggered": result.get("triggered"),
        "rescued_obligation_count": result.get("rescued_obligation_count", 0),
        "after_rescue_status": result.get("after_quorum_status"),
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

        current = assess_terminal_answer(
            question,
            row.get("answer", ""),
            evidence_summary,
            structured_confidence,
            query_graph,
        )

        v81 = build_proof_constrained_admission(
            current.get("answer", ""),
            current.get("claim_evidence_graph") or {},
            current.get("minimal_proof_subgraph") or {},
            current.get("counterfactual_path_necessity") or {},
        )
        if v81.get("changed") and v81.get("candidate_answer"):
            candidate = assess_terminal_answer(
                question,
                v81["candidate_answer"],
                evidence_summary,
                structured_confidence,
                query_graph,
            )
            if _select_v81_candidate(current, candidate):
                current = candidate

        v85 = build_proof_deficit_answer_repair(
            current.get("answer", ""),
            current,
        )
        if v85.get("changed") and v85.get("candidate_answer"):
            candidate = assess_terminal_answer(
                question,
                v85["candidate_answer"],
                evidence_summary,
                structured_confidence,
                query_graph,
            )
            if _select_v85_candidate(current, candidate):
                current = candidate

        v86 = build_source_diversity_answer_repair(
            current.get("answer", ""),
            current,
        )
        if v86.get("candidate_variants"):
            current = _select_v86_candidate(
                current,
                v86["candidate_variants"],
                question,
                evidence_summary,
                structured_confidence,
                query_graph,
            )

        # V84->V81->V85->V86 pipeline replay complete here
        # Now run V88 on top of V86 output
        rescue = current.get("obligation_source_rescue") or {}
        criticality = current.get("obligation_criticality_quorum") or {}

        per_row.append(
            {
                "line_no": row.get("line_no"),
                "dataset": row.get("dataset"),
                "question": question,
                "raw_pre_v87_status": _raw_pre_v87_status(current),
                "pre_rescue_status": _pre_rescue_status(current),
                "post_rescue_status": _post_rescue_status(current),
                "final_status": _quorum_status(current),
                "proof_sufficient": _proof_sufficient(current),
                "unsupported_claim_count": _unsupported_count(current),
                "terminal_score": terminal_assessment_score(current),
                "v88_triggered": bool(rescue.get("triggered")),
                "v88_applied": bool(rescue.get("applied")),
                "v88_rescued_obligation_count": rescue.get(
                    "rescued_obligation_count", 0
                ),
                "v88_added_evidence_count": rescue.get(
                    "selected_addition_count", 0
                ),
                "v87_demoted_count": criticality.get(
                    "demoted_obligation_count", 0
                ),
                "critical_remaining_single": criticality.get(
                    "critical_single_family_obligations", []
                ),
                "critical_remaining_missing": criticality.get(
                    "critical_missing_obligations", []
                ),
            }
        )

    before_counts = Counter(row["pre_rescue_status"] for row in per_row)
    after_counts = Counter(row["final_status"] for row in per_row)
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "pre_rescue_status_counts": dict(before_counts),
        "post_rescue_status_counts": dict(after_counts),
        "proof_sufficient_count": sum(
            1 for row in per_row if row["proof_sufficient"]
        ),
        "unsupported_claim_total": sum(
            row["unsupported_claim_count"] for row in per_row
        ),
        "v88_triggered_count": sum(1 for row in per_row if row["v88_triggered"]),
        "v88_applied_count": sum(1 for row in per_row if row["v88_applied"]),
        "v88_rescued_obligation_total": sum(
            row["v88_rescued_obligation_count"] for row in per_row
        ),
        "v88_added_evidence_total": sum(
            row["v88_added_evidence_count"] for row in per_row
        ),
        "independent_quorum_gain_after_v87": after_counts.get(
            "independent_quorum", 0
        ) - before_counts.get("independent_quorum", 0),
        "correlated_monitor_reduction_after_v87": before_counts.get(
            "correlated_monitor", 0
        ) - after_counts.get("correlated_monitor", 0),
        "mean_terminal_score": mean(
            [row["terminal_score"] for row in per_row]
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_case"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v88 Obligation-Source Rescue Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V88 runs after V83 source-quorum backfill and before V87 "
        "criticality normalization.  It targets only obligations that V87 "
        "would classify as critical (real domain obligations, not synthetic "
        "artifacts).  For those critical obligations, V88 performs a focused "
        "second-pass search with a separate obligation-only budget.",
        "",
        "## Controlled case",
        "",
        f"- Passed: `{controlled['passed']}`",
        f"- Triggered: `{controlled['triggered']}`",
        f"- Rescued obligations: `{controlled['rescued_obligation_count']}`",
        f"- After-rescue status: `{controlled['after_rescue_status']}`",
        "",
        "## Selective-30 replay audit",
        "",
        f"- Source: `{selective['source_file']}`",
        f"- Samples: `{selective['sample_count']}`",
        f"- Pre-rescue status counts: `{selective['pre_rescue_status_counts']}`",
        f"- Post-rescue status counts: `{selective['post_rescue_status_counts']}`",
        f"- Proof sufficient count: `{selective['proof_sufficient_count']}`",
        f"- Unsupported claim total: `{selective['unsupported_claim_total']}`",
        f"- V88 triggered / applied: `{selective['v88_triggered_count']}` / `{selective['v88_applied_count']}`",
        f"- Rescued obligation total: `{selective['v88_rescued_obligation_total']}`",
        f"- Added evidence total: `{selective['v88_added_evidence_total']}`",
        f"- Independent-quorum gain after V87: `{selective['independent_quorum_gain_after_v87']}`",
        f"- Correlated-monitor reduction after V87: `{selective['correlated_monitor_reduction_after_v87']}`",
        f"- Mean terminal score: `{selective['mean_terminal_score']}`",
        "",
        "## Interpretation",
        "",
        "V88 targets the last remaining correlated_monitor case(s) that V87 "
        "cannot demote.  If V88 finds obligation-covering evidence from a new "
        "source family, it upgrades the quorum certificate without modifying "
        "the answer text.  If no qualifying evidence exists, V88 accepts the "
        "correlated_monitor outcome.",
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
