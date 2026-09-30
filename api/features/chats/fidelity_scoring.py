"""VSR-PathRAG v91 fidelity-weighted answer quality scoring.

V91 introduces fine-grained, continuous answer quality metrics that go beyond
the binary pass/fail of the evidence audit.  These scores enable:
- Multi-candidate answer ranking with nuanced quality comparison.
- Identification of weak claim components for targeted refinement (V92).
- Composite Answer Quality (CAQ) score for submission calibration.

Core scores:
1. Claim-Fidelity Score (CFS): Per-claim and aggregate evidence support strength.
2. Numerical-Precision Score (NPS): Accuracy of numerical/exact answers.
3. Source-Attribution Score (SAS): Traceability and citation completeness.
4. Composite Answer Quality (CAQ): Weighted combination of above.

All scores are computed from the VSR pipeline's verified claim-evidence graph,
requiring no external LLM-as-judge calls.
"""

from __future__ import annotations

import re
from typing import Any


def _extract_numerical_claims(answer: str) -> list[dict[str, Any]]:
    """Extract numerical claims from the answer text."""
    patterns = [
        # Percentages: 42%, 3.5%
        (r"(\d+(?:\.\d+)?)\s*%", "percentage"),
        # Fractions: 1/2, 3/4
        (r"(\d+)\s*/\s*(\d+)", "fraction"),
        # Numbers with units: 100 km, $50 million, 2023
        (r"(\d+(?:\.\d+)?)\s*(?:%|km|m|s|kg|gb|mb|\$|€|£|billion|million|thousand|years?|days?|hours?|minutes?)", "measured_quantity"),
        # Dates: 1990, January 15, 20th century
        (r"\b(19|20)\d{2}\b|\b\d{1,2}(?:st|nd|rd|th)?\s*(?:january|february|march|april|may|june|july|august|september|october|november|december)?", "date"),
        # Standalone numbers in context
        (r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\b", "number"),
    ]

    numerical_claims: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern, claim_type in patterns:
        for match in re.finditer(pattern, answer, re.IGNORECASE):
            span = match.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)
            numerical_claims.append({
                "text": match.group(),
                "type": claim_type,
                "start": span[0],
                "end": span[1],
                "value": match.group(1) if match.lastindex and match.group(1) else match.group(),
            })

    return numerical_claims


def _compute_claim_fidelity_scores(
    claim_evidence_graph: dict[str, Any],
    minimal_proof_subgraph: dict[str, Any],
) -> dict[str, Any]:
    """Compute Claim-Fidelity Score (CFS) for each supported claim.

    CFS per claim = weighted combination of:
    - Support strength (max edge score among supporting evidence)
    - Evidence diversity (number of independent sources)
    - Proof necessity (is this claim in the minimal proof?)
    """
    claims = claim_evidence_graph.get("claims", [])
    edges = claim_evidence_graph.get("edges", [])
    covered_claims = set(minimal_proof_subgraph.get("covered_claims", []))
    selected_evidence_ids = {
        row["evidence_id"]
        for row in minimal_proof_subgraph.get("selected_evidence", []) or []
    }

    # Build claim -> edges mapping
    claim_edges: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        claim_id = str(edge.get("claim_id", ""))
        if claim_id:
            claim_edges.setdefault(claim_id, []).append(edge)

    # Source diversity per claim
    claim_sources: dict[str, set[str]] = {}
    for row in minimal_proof_subgraph.get("selected_evidence", []) or []:
        for claim_id in row.get("supported_claim_ids", []):
            claim_sources.setdefault(claim_id, set()).update(row.get("source_ids", []))

    per_claim_cfs: dict[str, float] = {}
    total_support_strength = 0.0
    total_diversity_bonus = 0.0
    claim_count = 0

    for claim in claims:
        claim_id = str(claim.get("claim_id", ""))
        if not claim_id:
            continue

        claim_edge_list = claim_edges.get(claim_id, [])
        support_edges = [e for e in claim_edge_list if e.get("relation") == "supports"]

        if not support_edges:
            per_claim_cfs[claim_id] = 0.0
            continue

        claim_count += 1

        # Support strength: max score among supporters
        max_support = max(
            (float(e.get("score", 0.0)) or 0.0) for e in support_edges
        )
        total_support_strength += max_support

        # Diversity bonus: log-scaled source count
        source_count = len(claim_sources.get(claim_id, set()))
        diversity_bonus = min(1.0, (source_count - 1) / 2.0) if source_count > 1 else 0.0
        total_diversity_bonus += diversity_bonus

        # Necessity bonus: is this claim in the minimal proof?
        necessity_bonus = 0.10 if claim_id in covered_claims else 0.0

        cfs = (
            0.60 * min(1.0, max_support)
            + 0.25 * diversity_bonus
            + 0.15 * necessity_bonus
        )
        per_claim_cfs[claim_id] = round(max(0.0, cfs), 4)

    # Aggregate CFS
    aggregate_cfs = round(
        total_support_strength / max(1, claim_count) * 0.70
        + total_diversity_bonus / max(1, claim_count) * 0.30,
        4,
    ) if claim_count > 0 else 0.0

    return {
        "per_claim_cfs": per_claim_cfs,
        "aggregate_cfs": aggregate_cfs,
        "claim_count": claim_count,
        "supported_claim_count": len([c for c in per_claim_cfs.values() if c > 0]),
        "mean_cfs": round(
            sum(per_claim_cfs.values()) / max(1, len(per_claim_cfs)), 4
        ) if per_claim_cfs else 0.0,
        "weak_claims": sorted(
            [cid for cid, score in per_claim_cfs.items() if 0 < score < 0.40]
        ),
        "strong_claims": sorted(
            [cid for cid, score in per_claim_cfs.items() if score >= 0.75]
        ),
    }


def _compute_numerical_precision_score(
    answer: str,
    evidence_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compute Numerical-Precision Score (NPS).

    NPS measures how well numerical values in the answer are grounded in evidence.
    This is a conservative heuristic that checks whether extracted numbers appear
    in the evidence text (not exact semantic matching).
    """
    numerical_claims = _extract_numerical_claims(answer)

    if not numerical_claims:
        return {
            "nps": 1.0,  # No numerical claims → perfect by default
            "numerical_claim_count": 0,
            "grounded_count": 0,
            "numerical_details": [],
        }

    # Collect all evidence text for grounding check
    evidence_texts: list[str] = []
    for entity in evidence_summary.get("entities", []):
        desc = str(entity.get("description", ""))
        name = str(entity.get("name", ""))
        if desc:
            evidence_texts.append(desc.lower())
        if name:
            evidence_texts.append(name.lower())
    for rel in evidence_summary.get("relationships", []):
        desc = str(rel.get("description", ""))
        keywords = str(rel.get("keywords", ""))
        if desc:
            evidence_texts.append(desc.lower())
        if keywords:
            evidence_texts.append(keywords.lower())
    for excerpt in evidence_summary.get("source_excerpts", []):
        text = str(excerpt.get("text", ""))
        if text:
            evidence_texts.append(text.lower())

    combined_evidence = " ".join(evidence_texts)

    grounded_count = 0
    details: list[dict[str, Any]] = []

    for nc in numerical_claims:
        value = nc["value"].lower()
        # Check if value appears in evidence (substring match)
        is_grounded = value in combined_evidence
        if is_grounded:
            grounded_count += 1

        details.append({
            "value": nc["text"],
            "type": nc["type"],
            "grounded": is_grounded,
        })

    nps = round(grounded_count / max(1, len(numerical_claims)), 4)

    return {
        "nps": nps,
        "numerical_claim_count": len(numerical_claims),
        "grounded_count": grounded_count,
        "numerical_details": details,
    }


def _compute_source_attribution_score(
    minimal_proof_subgraph: dict[str, Any],
    proof_source_quorum: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute Source-Attribution Score (SAS).

    SAS measures how well the answer can be traced back to independent sources.
    Higher SAS = more diverse, independently sourced evidence.
    """
    selected = minimal_proof_subgraph.get("selected_evidence", []) or []
    unique_sources: set[str] = set()
    total_source_references = 0
    evidence_with_sources = 0

    for row in selected:
        sources = row.get("source_ids", [])
        if sources:
            evidence_with_sources += 1
            unique_sources.update(sources)
            total_source_references += len(sources)

    # Quorum status contribution
    quorum_status = (proof_source_quorum or {}).get("quorum_status", "unknown")
    quormap = {
        "independent_quorum": 1.0,
        "single_source_critical": 0.5,
        "correlated_monitor": 0.35,
        "quorum_failed": 0.0,
    }
    quorum_component = quormap.get(quorum_status, 0.25)

    # Source diversity component
    diversity_component = min(1.0, len(unique_sources) / max(1, len(selected)))

    # Coverage component (what fraction of evidence has source IDs)
    coverage_component = (
        evidence_with_sources / max(1, len(selected))
    )

    sas = round(
        0.45 * quorum_component
        + 0.35 * diversity_component
        + 0.20 * coverage_component,
        4,
    )

    return {
        "sas": sas,
        "unique_source_count": len(unique_sources),
        "total_source_references": total_source_references,
        "evidence_with_source_ids": evidence_with_sources,
        "total_evidence_count": len(selected),
        "quorum_status": quorum_status,
        "quorum_component": round(quorum_component, 4),
        "diversity_component": round(diversity_component, 4),
        "coverage_component": round(coverage_component, 4),
    }


def build_fidelity_scoring(
    question: str,
    answer: str,
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    minimal_proof_subgraph: dict[str, Any],
    proof_source_quorum: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive fidelity-weighted answer quality scores.

    Returns CAQ (Composite Answer Quality) and sub-scores for use in
    candidate ranking and weak-claim identification.
    """
    # 1. Claim-Fidelity Score (CFS)
    cfs_result = _compute_claim_fidelity_scores(
        claim_evidence_graph, minimal_proof_subgraph
    )

    # 2. Numerical-Precision Score (NPS)
    nps_result = _compute_numerical_precision_score(answer, evidence_summary)

    # 3. Source-Attribution Score (SAS)
    sas_result = _compute_source_attribution_score(
        minimal_proof_subgraph, proof_source_quorum
    )

    # 4. Composite Answer Quality (CAQ)
    caq = round(
        0.45 * cfs_result["aggregate_cfs"]
        + 0.25 * nps_result["nps"]
        + 0.30 * sas_result["sas"],
        4,
    )

    return {
        "enabled": True,
        "controller": "fidelity_scoring_v1",
        "triggered": True,
        "composite_answer_quality": caq,
        "claim_fidelity_score": cfs_result,
        "numerical_precision_score": nps_result,
        "source_attribution_score": sas_result,
        "score_weights": {
            "cfs_weight": 0.45,
            "nps_weight": 0.25,
            "sas_weight": 0.30,
        },
        "quality_tier": (
            "excellent" if caq >= 0.80
            else "good" if caq >= 0.60
            else "acceptable" if caq >= 0.40
            else "needs_improvement"
        ),
        "refinement_recommendation": (
            "none"
            if caq >= 0.80
            else "target_weak_claims"
            if cfs_result["weak_claims"]
            else "increase_evidence_diversity"
            if sas_result["sas"] < 0.50
            else "verify_numerical_claims"
            if nps_result["nps"] < 0.60
            else "general_refinement"
        ),
    }
