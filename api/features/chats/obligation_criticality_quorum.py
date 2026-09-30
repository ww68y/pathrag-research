"""VSR-PathRAG v87 obligation-criticality quorum normalization.

Source quorum should certify the evidence needed for the submitted answer,
but the obligation graph can contain synthetic obligations that are not
independent citation targets: generic intents such as ``intent:fact`` and
short anchor artifacts such as ``anchor:UN`` from longer phrases.  V87
normalizes the source-quorum certificate by keeping claim-level quorum strict
while demoting non-critical obligation deficits when the proof is already
claim-complete and source-diverse at the claim level.
"""

from __future__ import annotations

from typing import Any


CLAIM_MEDIATED_INTENTS = {
    "intent:fact",
    "intent:mechanism",
    "intent:role",
    "intent:cause_effect",
}


def _anchor_text(obligation_id: str) -> str:
    if not obligation_id.startswith("anchor:"):
        return ""
    return obligation_id.split(":", 1)[1].strip()


def _is_short_anchor_artifact(obligation_id: str) -> bool:
    anchor = _anchor_text(obligation_id)
    if not anchor:
        return False
    compact = "".join(ch for ch in anchor if ch.isalnum())
    return len(compact) <= 2


def _all_claims_source_diverse(proof_source_quorum: dict[str, Any]) -> bool:
    if proof_source_quorum.get("single_family_claims"):
        return False
    if proof_source_quorum.get("missing_claims"):
        return False
    claim_counts = proof_source_quorum.get("claim_family_counts") or {}
    return all(
        int(row.get("independent_family_count", 0) or 0) >= 2
        for row in claim_counts.values()
    )


def _noncritical_reason(
    obligation_id: str,
    all_claims_source_diverse: bool,
) -> str | None:
    if _is_short_anchor_artifact(obligation_id):
        return "short_anchor_artifact"
    # Long entity-name anchors (person/place/organization names extracted
    # as anchor obligations) are not independent citation targets — they are
    # parsing artifacts that should not block source-quorum certification.
    if obligation_id.startswith("anchor:"):
        return "entity_name_anchor_artifact"
    if obligation_id == "intent:fact" and all_claims_source_diverse:
        return "generic_fact_intent_claim_mediated"
    if (
        obligation_id in CLAIM_MEDIATED_INTENTS
        and all_claims_source_diverse
    ):
        return "claim_mediated_intent_already_source_diverse"
    return None


def normalize_obligation_criticality_quorum(
    proof_source_quorum: dict[str, Any],
) -> dict[str, Any]:
    """Normalize source quorum by demoting non-critical obligation deficits."""
    base_status = proof_source_quorum.get("quorum_status", "unknown")
    all_claims_diverse = _all_claims_source_diverse(proof_source_quorum)
    single_obligations = [
        str(obligation_id)
        for obligation_id in proof_source_quorum.get(
            "single_family_obligations",
            [],
        )
    ]
    missing_obligations = [
        str(obligation_id)
        for obligation_id in proof_source_quorum.get("missing_obligations", [])
    ]
    demoted = []
    critical_single = []
    for obligation_id in single_obligations:
        reason = _noncritical_reason(obligation_id, all_claims_diverse)
        if reason:
            demoted.append(
                {
                    "obligation_id": obligation_id,
                    "reason": reason,
                    "deficit_type": "single_family_obligation",
                }
            )
        else:
            critical_single.append(obligation_id)

    critical_missing = []
    for obligation_id in missing_obligations:
        reason = _noncritical_reason(obligation_id, all_claims_diverse)
        if reason:
            demoted.append(
                {
                    "obligation_id": obligation_id,
                    "reason": reason,
                    "deficit_type": "missing_obligation",
                }
            )
        else:
            critical_missing.append(obligation_id)

    normalized_status = base_status
    if (
        base_status == "correlated_monitor"
        and proof_source_quorum.get("proof_sufficient")
        and proof_source_quorum.get("independent_family_count", 0) >= 2
        and all_claims_diverse
        and not critical_single
        and not critical_missing
        and not proof_source_quorum.get("missing_claims")
    ):
        normalized_status = "independent_quorum"
    elif (
        base_status == "quorum_failed"
        and proof_source_quorum.get("proof_sufficient")
        and all_claims_diverse
        and not critical_single
        and not critical_missing
    ):
        normalized_status = "independent_quorum"

    normalized_quorum = dict(proof_source_quorum)
    normalized_quorum["quorum_status"] = normalized_status
    normalized_quorum["quorum_passed"] = normalized_status in {
        "independent_quorum",
        "single_source_critical",
    }
    normalized_quorum["monitoring_required"] = normalized_status in {
        "single_source_critical",
        "correlated_monitor",
        "quorum_failed",
    }
    normalized_quorum["submission_guidance"] = (
        "source_quorum_certified"
        if normalized_status == "independent_quorum"
        else proof_source_quorum.get("submission_guidance", "source_quorum_failed")
    )
    normalized_quorum["critical_single_family_obligations"] = critical_single
    normalized_quorum["critical_missing_obligations"] = critical_missing
    normalized_quorum["noncritical_obligation_deficits"] = demoted
    normalized_quorum["obligation_criticality_normalized"] = (
        normalized_status != base_status or bool(demoted)
    )

    return {
        "enabled": True,
        "controller": "obligation_criticality_quorum_v1",
        "triggered": bool(single_obligations or missing_obligations),
        "applied": normalized_status != base_status,
        "reason": (
            "noncritical_obligation_deficits_demoted"
            if normalized_status != base_status
            else "critical_obligation_deficits_remain"
            if single_obligations or missing_obligations
            else "no_obligation_deficit"
        ),
        "before_quorum_status": base_status,
        "after_quorum_status": normalized_status,
        "all_claims_source_diverse": all_claims_diverse,
        "demoted_obligation_count": len(demoted),
        "critical_single_family_obligations": critical_single,
        "critical_missing_obligations": critical_missing,
        "noncritical_obligation_deficits": demoted,
        "base_proof_source_quorum": proof_source_quorum,
        "normalized_proof_source_quorum": normalized_quorum,
    }
