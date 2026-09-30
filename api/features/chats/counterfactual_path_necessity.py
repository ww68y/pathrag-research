"""VSR-PathRAG v79 counterfactual evidence-path necessity certification."""

from __future__ import annotations

from typing import Any

from api.features.chats.minimal_proof_subgraph import (
    build_minimal_proof_subgraph,
)


def _necessity_label(
    base: dict[str, Any],
    counterfactual: dict[str, Any],
    claim_loss: float,
    obligation_loss: float,
    objective_loss: float,
) -> str:
    if claim_loss > 0:
        return "claim_essential"
    if base.get("proof_sufficient") and not counterfactual.get(
        "proof_sufficient"
    ):
        return "sufficiency_critical"
    if obligation_loss > 0:
        return "obligation_critical"
    if objective_loss >= 0.05:
        return "efficiency_critical"
    return "replaceable"


def _selected_ids(proof: dict[str, Any]) -> set[str]:
    return {
        str(row.get("evidence_id"))
        for row in proof.get("selected_evidence", [])
        if row.get("evidence_id")
    }


def certify_counterfactual_path_necessity(
    question: str,
    query_graph: dict[str, Any] | None,
    answer: str,
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    evidence_audit: dict[str, Any] | None = None,
    minimal_proof_subgraph: dict[str, Any] | None = None,
    max_evidence: int = 6,
    token_budget: int = 720,
) -> dict[str, Any]:
    """Certify whether selected proof paths are necessary.

    Each selected evidence node is removed by intervention, then the v78
    optimizer is rerun on the remaining evidence. The delta tells us
    whether the original path was claim-essential, obligation-critical,
    merely efficiency-critical, or replaceable.
    """
    base = minimal_proof_subgraph or build_minimal_proof_subgraph(
        question,
        query_graph or {},
        answer,
        evidence_summary,
        claim_evidence_graph,
        evidence_audit,
        max_evidence=max_evidence,
        token_budget=token_budget,
    )
    interventions = []
    base_claim = float(base.get("claim_support_coverage_ratio", 0.0) or 0.0)
    base_obligation = float(
        base.get("obligation_coverage_ratio", 0.0) or 0.0
    )
    base_objective = float(base.get("joint_objective", 0.0) or 0.0)
    for evidence in base.get("selected_evidence", []):
        evidence_id = str(evidence.get("evidence_id", ""))
        if not evidence_id:
            continue
        counterfactual = build_minimal_proof_subgraph(
            question,
            query_graph or {},
            answer,
            evidence_summary,
            claim_evidence_graph,
            evidence_audit,
            max_evidence=max_evidence,
            token_budget=token_budget,
            excluded_evidence_ids={evidence_id},
        )
        cf_claim = float(
            counterfactual.get("claim_support_coverage_ratio", 0.0) or 0.0
        )
        cf_obligation = float(
            counterfactual.get("obligation_coverage_ratio", 0.0) or 0.0
        )
        cf_objective = float(counterfactual.get("joint_objective", 0.0) or 0.0)
        claim_loss = max(0.0, base_claim - cf_claim)
        obligation_loss = max(0.0, base_obligation - cf_obligation)
        objective_loss = max(0.0, base_objective - cf_objective)
        label = _necessity_label(
            base,
            counterfactual,
            claim_loss,
            obligation_loss,
            objective_loss,
        )
        interventions.append(
            {
                "removed_evidence_id": evidence_id,
                "label": evidence.get("label", ""),
                "source_ids": evidence.get("source_ids", []),
                "necessity": label,
                "claim_coverage_loss": round(claim_loss, 3),
                "obligation_coverage_loss": round(obligation_loss, 3),
                "joint_objective_loss": round(objective_loss, 4),
                "counterfactual_proof_sufficient": counterfactual.get(
                    "proof_sufficient",
                    False,
                ),
                "replacement_selected_ids": sorted(
                    _selected_ids(counterfactual) - {evidence_id}
                ),
                "lost_claims_if_removed": sorted(
                    set(base.get("covered_claims", []))
                    - set(counterfactual.get("covered_claims", []))
                ),
                "lost_obligations_if_removed": sorted(
                    set(base.get("covered_obligations", []))
                    - set(counterfactual.get("covered_obligations", []))
                ),
            }
        )

    essential_labels = {
        "claim_essential",
        "sufficiency_critical",
        "obligation_critical",
    }
    essential = [
        row for row in interventions if row["necessity"] in essential_labels
    ]
    replaceable = [
        row for row in interventions if row["necessity"] == "replaceable"
    ]
    mean_necessity = 0.0
    if interventions:
        mean_necessity = sum(
            0.55 * row["claim_coverage_loss"]
            + 0.35 * row["obligation_coverage_loss"]
            + 0.10 * row["joint_objective_loss"]
            for row in interventions
        ) / len(interventions)
    certification_passed = bool(
        base.get("proof_sufficient")
        and interventions
        and not base.get("contradicted_claims_in_proof")
    )
    strong_necessity_certified = bool(certification_passed and not replaceable)
    if not certification_passed:
        submission_guidance = "necessity_monitor"
    elif replaceable:
        submission_guidance = "necessity_certified_with_replacements"
    else:
        submission_guidance = "necessity_certified"
    return {
        "enabled": True,
        "controller": "counterfactual_path_necessity_v1",
        "base_proof_sufficient": bool(base.get("proof_sufficient")),
        "selected_path_count": len(base.get("selected_evidence", [])),
        "intervention_count": len(interventions),
        "essential_path_count": len(essential),
        "replaceable_path_count": len(replaceable),
        "mean_necessity_score": round(mean_necessity, 4),
        "certification_passed": certification_passed,
        "strong_necessity_certified": strong_necessity_certified,
        "submission_guidance": submission_guidance,
        "interventions": interventions,
    }
