"""VSR-PathRAG v86 source-diversity answer repair.

V82/V83 certify whether a proof depends on correlated provenance families.
After V85 removes unsupported claims, some answers remain clean but still
contain claims supported by only one source family.  V86 builds deterministic
answer-pruning variants that remove source-singleton claims, then lets the
controller re-assess and select only variants that improve source quorum
without breaking terminal safety.
"""

from __future__ import annotations

from typing import Any


def _clean_claim(text: str) -> str:
    return " ".join(str(text or "").strip().split()).strip(" .")


def _necessary_claims(
    counterfactual_path_necessity: dict[str, Any],
) -> set[str]:
    necessary = set()
    for row in counterfactual_path_necessity.get("interventions", []):
        for claim_id in row.get("lost_claims_if_removed", []):
            necessary.add(str(claim_id))
    return necessary


def _claim_rows(
    claim_evidence_graph: dict[str, Any],
    covered_claims: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for row in claim_evidence_graph.get("claims", []):
        claim_id = str(row.get("claim_id", ""))
        claim = _clean_claim(row.get("claim", ""))
        if not claim:
            continue
        rows.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "status": str(row.get("status", "unknown")),
                "support_mass": round(
                    float(row.get("support_mass", 0.0) or 0.0),
                    3,
                ),
                "proof_covered": claim_id in covered_claims,
            }
        )
    return rows


def _build_variant(
    variant_id: str,
    description: str,
    claims: list[dict[str, Any]],
    drop_claim_ids: set[str],
    single_family_claims: set[str],
    necessary_claim_ids: set[str],
    answer: str,
) -> dict[str, Any] | None:
    retained = []
    dropped = []
    for row in claims:
        if row["claim_id"] in drop_claim_ids:
            dropped.append(
                {
                    **row,
                    "source_singleton": row["claim_id"] in single_family_claims,
                    "necessary": row["claim_id"] in necessary_claim_ids,
                }
            )
        else:
            retained.append(
                {
                    **row,
                    "source_singleton": row["claim_id"] in single_family_claims,
                    "necessary": row["claim_id"] in necessary_claim_ids,
                }
            )
    if not retained or not dropped:
        return None
    candidate = ". ".join(row["claim"] for row in retained) + "."
    if _clean_claim(candidate) == _clean_claim(answer):
        return None
    return {
        "variant_id": variant_id,
        "description": description,
        "candidate_answer": candidate,
        "retained_claim_count": len(retained),
        "dropped_claim_count": len(dropped),
        "dropped_single_family_claim_count": sum(
            1 for row in dropped if row["source_singleton"]
        ),
        "dropped_necessary_claim_count": sum(
            1 for row in dropped if row["necessary"]
        ),
        "retained_claims": retained,
        "dropped_claims": dropped,
    }


def build_source_diversity_answer_repair(
    answer: str,
    terminal_assessment: dict[str, Any],
) -> dict[str, Any]:
    """Build source-diversity repair variants for correlated clean proofs."""
    claim_graph = terminal_assessment.get("claim_evidence_graph") or {}
    minimal_proof_subgraph = terminal_assessment.get("minimal_proof_subgraph") or {}
    proof_source_quorum = terminal_assessment.get("proof_source_quorum") or {}
    counterfactual_path_necessity = (
        terminal_assessment.get("counterfactual_path_necessity") or {}
    )
    risk_calibrated_submission = (
        terminal_assessment.get("risk_calibrated_submission") or {}
    )
    evidence_audit = terminal_assessment.get("evidence_audit") or {}

    quorum_status = proof_source_quorum.get("quorum_status", "unknown")
    single_family_claims = {
        str(claim_id)
        for claim_id in proof_source_quorum.get("single_family_claims", [])
    }
    single_family_obligations = {
        str(obligation_id)
        for obligation_id in proof_source_quorum.get(
            "single_family_obligations",
            [],
        )
    }
    covered_claims = {
        str(claim_id)
        for claim_id in minimal_proof_subgraph.get("covered_claims", [])
    }
    claims = _claim_rows(claim_graph, covered_claims)
    necessary = _necessary_claims(counterfactual_path_necessity)
    unsupported = int(evidence_audit.get("unsupported_claim_count", 0) or 0)

    should_attempt = bool(
        quorum_status == "correlated_monitor"
        and single_family_claims
        and unsupported == 0
        and risk_calibrated_submission.get("action")
        in {"submit", "submit_with_monitoring"}
    )
    if not should_attempt:
        return {
            "enabled": True,
            "controller": "source_diversity_answer_repair_v1",
            "triggered": False,
            "reason": "no_clean_correlated_claim_deficit",
            "candidate_selected": False,
            "candidate_variants": [],
            "variant_count": 0,
            "selected_variant_id": None,
            "current_quorum_status": quorum_status,
            "single_family_claim_count": len(single_family_claims),
            "single_family_obligation_count": len(single_family_obligations),
            "selection_reason": None,
        }

    variants = []
    nonnecessary_singletons = single_family_claims - necessary
    nonnecessary_variant = _build_variant(
        "drop_nonnecessary_single_family_claims",
        "drop source-singleton claims that are not counterfactually necessary",
        claims,
        nonnecessary_singletons,
        single_family_claims,
        necessary,
        answer,
    )
    if nonnecessary_variant:
        variants.append(nonnecessary_variant)

    all_singleton_variant = _build_variant(
        "drop_all_single_family_claims",
        "drop all source-singleton claims and let terminal assessment verify sufficiency",
        claims,
        single_family_claims,
        single_family_claims,
        necessary,
        answer,
    )
    if all_singleton_variant and all_singleton_variant.get(
        "candidate_answer"
    ) not in {row.get("candidate_answer") for row in variants}:
        variants.append(all_singleton_variant)

    return {
        "enabled": True,
        "controller": "source_diversity_answer_repair_v1",
        "triggered": True,
        "reason": (
            "source_singleton_claim_variants_available"
            if variants
            else "no_prunable_source_singleton_claims"
        ),
        "candidate_selected": False,
        "candidate_variants": variants,
        "variant_count": len(variants),
        "selected_variant_id": None,
        "current_quorum_status": quorum_status,
        "single_family_claim_count": len(single_family_claims),
        "single_family_obligation_count": len(single_family_obligations),
        "necessary_single_family_claim_count": len(
            single_family_claims & necessary
        ),
        "selection_reason": None,
    }
