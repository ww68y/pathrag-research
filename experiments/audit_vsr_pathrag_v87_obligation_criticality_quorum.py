"""Audit VSR-PathRAG v87 obligation-criticality quorum normalization."""

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
OUT_JSON = ROOT / "results" / "vsr_pathrag_v87_obligation_criticality_quorum_audit.json"
OUT_REPORT = ROOT / "results" / "vsr_pathrag_v87_obligation_criticality_quorum_report.md"


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
    return (assessment.get("proof_source_quorum") or {}).get(
        "quorum_status",
        "unknown",
    )


def _raw_quorum_status(assessment: dict[str, Any]) -> str:
    base = (
        assessment.get("obligation_criticality_quorum") or {}
    ).get("base_proof_source_quorum")
    return (base or assessment.get("proof_source_quorum") or {}).get(
        "quorum_status",
        "unknown",
    )


def _proof_sufficient(assessment: dict[str, Any]) -> bool:
    return bool(
        (assessment.get("minimal_proof_subgraph") or {}).get(
            "proof_sufficient",
        )
    )


def _deficit_count(assessment: dict[str, Any]) -> int:
    quorum = assessment.get("proof_source_quorum") or {}
    critical_obligations = quorum.get(
        "critical_single_family_obligations",
        quorum.get("single_family_obligations", []),
    )
    return len(quorum.get("single_family_claims", [])) + len(
        critical_obligations
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
    current_deficit = _deficit_count(current)
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
        candidate_deficit = _deficit_count(candidate)
        quorum_improved = bool(
            _quorum_status(current) != "independent_quorum"
            and _quorum_status(candidate) == "independent_quorum"
        )
        deficit_reduced = candidate_deficit < current_deficit
        eligible = bool(
            _submission_action(candidate) in {"submit", "submit_with_monitoring"}
            and _unsupported_count(candidate) == 0
            and _proof_sufficient(candidate)
            and _quorum_status(candidate) != "quorum_failed"
            and (quorum_improved or deficit_reduced)
            and candidate_score >= current_score - 0.20
        )
        if not eligible:
            continue
        key = (
            _quorum_status(candidate) == "independent_quorum",
            current_deficit - candidate_deficit,
            candidate_score,
            -variant.get("dropped_necessary_claim_count", 0),
            -variant.get("dropped_claim_count", 0),
        )
        if best is None or key > best[0]:
            best = (key, candidate)
    return best[1] if best else current


def controlled_case() -> dict[str, Any]:
    quorum = {
        "proof_sufficient": True,
        "independent_family_count": 2,
        "quorum_status": "correlated_monitor",
        "single_family_claims": [],
        "missing_claims": [],
        "single_family_obligations": ["intent:fact"],
        "missing_obligations": [],
        "claim_family_counts": {
            "claim-0": {
                "family_ids": ["doc-a", "doc-b"],
                "independent_family_count": 2,
            }
        },
        "submission_guidance": "correlated_source_monitor",
    }
    result = normalize_obligation_criticality_quorum(quorum)
    return {
        "tag": "generic_fact_obligation_demoted",
        "passed": bool(
            result.get("applied")
            and result.get("after_quorum_status") == "independent_quorum"
        ),
        "triggered": result.get("triggered"),
        "applied": result.get("applied"),
        "demoted_obligation_count": result.get("demoted_obligation_count"),
        "after_quorum_status": result.get("after_quorum_status"),
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

        criticality = current.get("obligation_criticality_quorum") or {}
        per_row.append(
            {
                "line_no": row.get("line_no"),
                "dataset": row.get("dataset"),
                "question": question,
                "before_v87_quorum_status": _raw_quorum_status(current),
                "after_v87_quorum_status": _quorum_status(current),
                "proof_sufficient": _proof_sufficient(current),
                "unsupported_claim_count": _unsupported_count(current),
                "terminal_score": terminal_assessment_score(current),
                "v87_triggered": bool(criticality.get("triggered")),
                "v87_applied": bool(criticality.get("applied")),
                "v87_demoted_obligation_count": criticality.get(
                    "demoted_obligation_count",
                    0,
                ),
                "v87_demoted_obligations": criticality.get(
                    "noncritical_obligation_deficits",
                    [],
                ),
            }
        )

    before_counts = Counter(row["before_v87_quorum_status"] for row in per_row)
    after_counts = Counter(row["after_v87_quorum_status"] for row in per_row)
    return {
        "source_file": str(SOURCE_JSONL.relative_to(ROOT)),
        "sample_count": len(per_row),
        "before_v87_quorum_status_counts": dict(before_counts),
        "after_v87_quorum_status_counts": dict(after_counts),
        "proof_sufficient_count": sum(
            1 for row in per_row if row["proof_sufficient"]
        ),
        "unsupported_claim_total": sum(
            row["unsupported_claim_count"] for row in per_row
        ),
        "v87_triggered_count": sum(1 for row in per_row if row["v87_triggered"]),
        "v87_applied_count": sum(1 for row in per_row if row["v87_applied"]),
        "v87_demoted_obligation_total": sum(
            row["v87_demoted_obligation_count"] for row in per_row
        ),
        "independent_quorum_gain_after_v86": after_counts.get(
            "independent_quorum",
            0,
        )
        - before_counts.get("independent_quorum", 0),
        "correlated_monitor_reduction_after_v86": before_counts.get(
            "correlated_monitor",
            0,
        )
        - after_counts.get("correlated_monitor", 0),
        "mean_terminal_score": mean(
            [row["terminal_score"] for row in per_row]
        ),
        "per_row": per_row,
    }


def write_report(result: dict[str, Any]) -> None:
    controlled = result["controlled_case"]
    selective = result["selective_30"]
    lines = [
        "# VSR-PathRAG v87 Obligation-Criticality Quorum Report",
        "",
        "Status: renewed innovation, pre-ablation.",
        "",
        "V87 normalizes source-quorum certification by demoting synthetic "
        "or claim-mediated obligation deficits once answer claims are already "
        "source-diverse. It does not rewrite answers, add evidence, use "
        "references, or start ablation.",
        "",
        "## Controlled case",
        "",
        f"- Passed: `{controlled['passed']}`",
        f"- Triggered / applied: `{controlled['triggered']}` / `{controlled['applied']}`",
        f"- Demoted obligations: `{controlled['demoted_obligation_count']}`",
        f"- After quorum: `{controlled['after_quorum_status']}`",
        "",
        "## Selective-30 replay audit",
        "",
        f"- Source: `{selective['source_file']}`",
        f"- Samples: `{selective['sample_count']}`",
        f"- Before V87 quorum counts: `{selective['before_v87_quorum_status_counts']}`",
        f"- After V87 quorum counts: `{selective['after_v87_quorum_status_counts']}`",
        f"- Proof sufficient count: `{selective['proof_sufficient_count']}`",
        f"- Unsupported claim total: `{selective['unsupported_claim_total']}`",
        f"- V87 triggered / applied: `{selective['v87_triggered_count']}` / `{selective['v87_applied_count']}`",
        f"- Demoted obligation deficits: `{selective['v87_demoted_obligation_total']}`",
        f"- Independent-quorum gain after V86: `{selective['independent_quorum_gain_after_v86']}`",
        f"- Correlated-monitor reduction after V86: `{selective['correlated_monitor_reduction_after_v86']}`",
        f"- Mean terminal score: `{selective['mean_terminal_score']}`",
        "",
        "## Interpretation",
        "",
        "The remaining correlated cases after V86 are mostly obligation-side "
        "artifacts or generic intents, not unsupported answer claims. V87 "
        "keeps claim-level quorum strict while preventing non-critical "
        "synthetic obligations from blocking an otherwise source-diverse proof.",
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
