"""VSR-PathRAG v80 risk-calibrated answer submission policy."""

from __future__ import annotations

from typing import Any


AUDIT_RISK_PRIOR = {
    "low": 0.14,
    "medium": 0.34,
    "high": 0.72,
}


def _hard_contradicted_count(claim_graph: dict[str, Any]) -> int:
    count = 0
    for row in claim_graph.get("claims", []):
        if row.get("status") != "contradicted":
            continue
        support = float(row.get("support_mass", 0.0) or 0.0)
        contradiction = float(row.get("contradiction_mass", 0.0) or 0.0)
        if contradiction - support >= 0.45:
            count += 1
    return count


def calibrate_answer_submission(
    evidence_audit: dict[str, Any],
    minimal_proof_subgraph: dict[str, Any],
    counterfactual_path_necessity: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    answer_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a calibrated final-submission decision.

    The policy is conservative about unsupported claims but tolerant of
    replaceable evidence paths: replaceability is monitoring information,
    not a reason to block a supported answer.
    """
    answer_state = answer_state or {}
    unsupported = int(evidence_audit.get("unsupported_claim_count", 0) or 0)
    claim_count = int(evidence_audit.get("claim_count", 0) or 0)
    claim_support = float(
        evidence_audit.get("claim_support_ratio", 0.0) or 0.0
    )
    audit_risk = str(evidence_audit.get("risk_level", "high"))
    audit_prior = AUDIT_RISK_PRIOR.get(audit_risk, 0.72)
    proof_claim = float(
        minimal_proof_subgraph.get("claim_support_coverage_ratio", 0.0)
        or 0.0
    )
    proof_obligation = float(
        minimal_proof_subgraph.get("obligation_coverage_ratio", 0.0) or 0.0
    )
    proof_sufficient = bool(minimal_proof_subgraph.get("proof_sufficient"))
    necessity_certified = bool(
        counterfactual_path_necessity.get("certification_passed")
    )
    selected_paths = int(
        counterfactual_path_necessity.get("selected_path_count", 0) or 0
    )
    replaceable_paths = int(
        counterfactual_path_necessity.get("replaceable_path_count", 0) or 0
    )
    replaceable_ratio = replaceable_paths / max(1, selected_paths)
    hard_contradicted = _hard_contradicted_count(claim_evidence_graph)
    refusal_like = bool(answer_state.get("refusal_like"))

    unsupported_risk = min(1.0, unsupported / max(1, claim_count))
    proof_gap_risk = 1.0 - (
        0.62 * proof_claim + 0.38 * proof_obligation
    )
    necessity_gap_risk = 0.0 if necessity_certified else 0.18
    replaceability_monitor_risk = 0.10 * replaceable_ratio
    contradiction_risk = min(1.0, hard_contradicted * 0.55)
    refusal_penalty = 0.08 if refusal_like else 0.0
    calibrated_risk = min(
        1.0,
        max(
            0.0,
            0.34 * audit_prior
            + 0.26 * unsupported_risk
            + 0.18 * proof_gap_risk
            + 0.10 * necessity_gap_risk
            + 0.07 * replaceability_monitor_risk
            + 0.05 * contradiction_risk
            + refusal_penalty,
        ),
    )

    if unsupported > 0 or hard_contradicted > 0:
        action = "block_unsupported"
        reason = "unsupported or contradicted claims exceed submission budget"
    elif (
        calibrated_risk <= 0.22
        and claim_support >= 0.95
        and proof_claim >= 0.90
        and proof_obligation >= 0.66
        and proof_sufficient
    ):
        action = "submit"
        reason = "low calibrated risk with sufficient claim and obligation proof"
    elif calibrated_risk <= 0.42 and claim_support >= 0.80:
        action = "submit_with_monitoring"
        reason = "claims are supported but proof or necessity signals need monitoring"
    else:
        action = "defer_or_fallback"
        reason = "calibrated support is below the direct submission threshold"

    return {
        "enabled": True,
        "controller": "risk_calibrated_submission_v1",
        "action": action,
        "reason": reason,
        "calibrated_risk": round(calibrated_risk, 4),
        "claim_support_ratio": round(claim_support, 3),
        "unsupported_claim_count": unsupported,
        "hard_contradicted_claim_count": hard_contradicted,
        "proof_claim_coverage": round(proof_claim, 3),
        "proof_obligation_coverage": round(proof_obligation, 3),
        "proof_sufficient": proof_sufficient,
        "necessity_certified": necessity_certified,
        "replaceable_path_ratio": round(replaceable_ratio, 3),
        "unsupported_claim_budget": 0,
        "submission_allowed": action in {"submit", "submit_with_monitoring"},
        "monitoring_required": action == "submit_with_monitoring"
        or replaceable_paths > 0,
        "risk_components": {
            "audit_prior": round(audit_prior, 3),
            "unsupported_risk": round(unsupported_risk, 3),
            "proof_gap_risk": round(proof_gap_risk, 3),
            "necessity_gap_risk": round(necessity_gap_risk, 3),
            "replaceability_monitor_risk": round(
                replaceability_monitor_risk,
                3,
            ),
            "contradiction_risk": round(contradiction_risk, 3),
            "refusal_penalty": round(refusal_penalty, 3),
        },
    }
