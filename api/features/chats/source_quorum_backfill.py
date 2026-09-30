"""VSR-PathRAG v83 source-quorum backfill.

V82 certifies whether a proof is independently sourced.  V83 uses that
certificate to add a small number of unselected evidence rows that supply a
new source family for single-family claims or obligations.
"""

from __future__ import annotations

from typing import Any

from api.features.chats.minimal_proof_subgraph import (
    _candidate_rows,
    _question_obligations,
)
from api.features.chats.proof_source_quorum import (
    _families_for_evidence,
    _lineage_index,
    certify_proof_source_quorum,
)


def _target_family_sets(
    quorum: dict[str, Any],
    key: str,
) -> dict[str, set[str]]:
    rows = quorum.get(key, {}) or {}
    return {
        target_id: set(row.get("family_ids", []))
        for target_id, row in rows.items()
    }


def _obligation_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(hit.get("obligation_id", ""))
        for hit in row.get("obligation_hits", [])
        if hit.get("obligation_id")
    }


def _public_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": row["evidence_id"],
        "kind": row["kind"],
        "label": row["label"],
        "source_ids": row["source_ids"],
        "obligation_hits": row["obligation_hits"],
        "supported_claim_ids": row["supported_claim_ids"],
        "max_claim_support_score": row["max_claim_support_score"],
        "estimated_tokens": row["estimated_tokens"],
        "marginal_utility": row.get("marginal_utility", 0.0),
        "text_preview": row["text_preview"],
    }


def build_source_quorum_backfill(
    question: str,
    query_graph: dict[str, Any] | None,
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    evidence_audit: dict[str, Any] | None,
    minimal_proof_subgraph: dict[str, Any],
    proof_source_quorum: dict[str, Any] | None = None,
    max_additional_evidence: int = 3,
    token_budget: int = 360,
) -> dict[str, Any]:
    """Add independent-family evidence for V82 single-family targets."""
    base_quorum = proof_source_quorum or certify_proof_source_quorum(
        minimal_proof_subgraph,
        evidence_summary,
        claim_evidence_graph,
    )
    before_status = base_quorum.get("quorum_status", "unknown")
    selected_ids = {
        str(row.get("evidence_id"))
        for row in minimal_proof_subgraph.get("selected_evidence", [])
        if row.get("evidence_id")
    }
    deficit_claims = set(base_quorum.get("single_family_claims", [])) | set(
        base_quorum.get("missing_claims", [])
    )
    deficit_obligations = set(
        base_quorum.get("single_family_obligations", [])
    ) | set(base_quorum.get("missing_obligations", []))
    should_attempt = bool(
        before_status in {"correlated_monitor", "single_source_critical"}
        and (deficit_claims or deficit_obligations)
    )
    if not should_attempt:
        return {
            "enabled": True,
            "controller": "source_quorum_backfill_v1",
            "triggered": False,
            "reason": "proof_source_quorum_does_not_require_backfill",
            "before_quorum_status": before_status,
            "after_quorum_status": before_status,
            "candidate_count": 0,
            "selected_addition_count": 0,
            "selected_additions": [],
            "base_proof_source_quorum": base_quorum,
            "augmented_minimal_proof_subgraph": minimal_proof_subgraph,
            "augmented_proof_source_quorum": base_quorum,
        }

    obligations = _question_obligations(question, query_graph or {})
    candidates = _candidate_rows(
        evidence_summary,
        obligations,
        claim_evidence_graph,
        excluded_evidence_ids=selected_ids,
    )
    source_index = _lineage_index(evidence_summary or {})
    claim_family_sets = _target_family_sets(
        base_quorum,
        "claim_family_counts",
    )
    obligation_family_sets = _target_family_sets(
        base_quorum,
        "obligation_family_counts",
    )
    proof_families = {
        family_id
        for row in base_quorum.get("selected_evidence_lineage", [])
        for family_id in row.get("family_ids", [])
    }
    additions = []
    used_tokens = 0
    remaining = list(candidates)

    while remaining and len(additions) < max_additional_evidence:
        best = None
        best_score = 0.0
        best_detail: dict[str, Any] = {}
        for row in remaining:
            if row.get("contradicted_claim_ids"):
                continue
            token_cost = int(row.get("estimated_tokens", 0) or 0)
            if used_tokens + token_cost > token_budget:
                continue
            lineage = _families_for_evidence(row, source_index)
            families = set(lineage.get("family_ids", []))
            if not families:
                continue
            claim_gains = []
            for claim_id in set(row.get("supported_claim_ids", [])) & deficit_claims:
                existing = claim_family_sets.setdefault(claim_id, set())
                if families - existing:
                    claim_gains.append(claim_id)
            obligation_gains = []
            for obligation_id in _obligation_ids(row) & deficit_obligations:
                existing = obligation_family_sets.setdefault(
                    obligation_id,
                    set(),
                )
                if families - existing:
                    obligation_gains.append(obligation_id)
            if not claim_gains and not obligation_gains:
                continue
            new_family_count = len(families - proof_families)
            score = (
                0.56 * len(claim_gains)
                + 0.34 * len(obligation_gains)
                + 0.08 * min(1, new_family_count)
                + 0.06 * float(row.get("max_claim_support_score", 0.0) or 0.0)
                - 0.012 * token_cost / 100.0
            )
            if score > best_score:
                best = row
                best_score = score
                best_detail = {
                    "families": sorted(families),
                    "claim_gains": sorted(claim_gains),
                    "obligation_gains": sorted(obligation_gains),
                    "new_family_count": new_family_count,
                }
        if best is None:
            break
        remaining.remove(best)
        used_tokens += int(best.get("estimated_tokens", 0) or 0)
        addition = _public_evidence_row(best)
        allowed_claims = set(base_quorum.get("single_family_claims", [])) | set(
            minimal_proof_subgraph.get("covered_claims", [])
        )
        allowed_obligations = set(
            base_quorum.get("single_family_obligations", [])
        ) | set(minimal_proof_subgraph.get("covered_obligations", []))
        addition["supported_claim_ids"] = [
            claim_id
            for claim_id in addition.get("supported_claim_ids", [])
            if claim_id in allowed_claims
        ]
        addition["obligation_hits"] = [
            hit
            for hit in addition.get("obligation_hits", [])
            if hit.get("obligation_id") in allowed_obligations
        ]
        addition["backfill_score"] = round(best_score, 4)
        addition["backfill_family_ids"] = best_detail["families"]
        addition["backfilled_claim_ids"] = best_detail["claim_gains"]
        addition["backfilled_obligation_ids"] = best_detail[
            "obligation_gains"
        ]
        additions.append(addition)
        for claim_id in best_detail["claim_gains"]:
            claim_family_sets.setdefault(claim_id, set()).update(
                best_detail["families"]
            )
        for obligation_id in best_detail["obligation_gains"]:
            obligation_family_sets.setdefault(obligation_id, set()).update(
                best_detail["families"]
            )
        proof_families.update(best_detail["families"])
        deficit_claims = {
            claim_id
            for claim_id in deficit_claims
            if len(claim_family_sets.get(claim_id, set())) < 2
        }
        deficit_obligations = {
            obligation_id
            for obligation_id in deficit_obligations
            if len(obligation_family_sets.get(obligation_id, set())) < 2
        }
        if not deficit_claims and not deficit_obligations:
            break

    augmented = dict(minimal_proof_subgraph)
    augmented_selected = (
        list(minimal_proof_subgraph.get("selected_evidence", [])) + additions
    )
    augmented["selected_evidence"] = augmented_selected
    augmented["selected_count"] = len(augmented_selected)
    augmented["estimated_tokens"] = int(
        minimal_proof_subgraph.get("estimated_tokens", 0) or 0
    ) + used_tokens
    covered_claims = sorted(
        {
            claim_id
            for row in augmented_selected
            for claim_id in row.get("supported_claim_ids", [])
        }
    )
    covered_obligations = sorted(
        {
            hit.get("obligation_id")
            for row in augmented_selected
            for hit in row.get("obligation_hits", [])
            if hit.get("obligation_id")
        }
    )
    covered_sources = sorted(
        {
            source_id
            for row in augmented_selected
            for source_id in row.get("source_ids", [])
        }
    )
    augmented["covered_claims"] = covered_claims
    augmented["covered_obligations"] = covered_obligations
    augmented["unique_source_count"] = len(covered_sources)
    required_claim_count = int(
        minimal_proof_subgraph.get("required_claim_count", 0) or 0
    )
    required_obligation_count = int(
        minimal_proof_subgraph.get("required_obligation_count", 0) or 0
    )
    claim_coverage = len(covered_claims) / max(1, required_claim_count)
    obligation_coverage = len(covered_obligations) / max(
        1,
        required_obligation_count,
    )
    unsupported = int((evidence_audit or {}).get("unsupported_claim_count", 0) or 0)
    contradicted = augmented.get("contradicted_claims_in_proof", []) or []
    augmented["claim_support_coverage_ratio"] = round(claim_coverage, 3)
    augmented["obligation_coverage_ratio"] = round(obligation_coverage, 3)
    augmented["proof_sufficient"] = bool(
        claim_coverage >= 1.0
        and (obligation_coverage >= 0.66 or required_obligation_count == 0)
        and unsupported == 0
        and not contradicted
    )
    after_quorum = certify_proof_source_quorum(
        augmented,
        evidence_summary,
        claim_evidence_graph,
    )
    improved = (
        before_status != after_quorum.get("quorum_status")
        or len(after_quorum.get("single_family_claims", []))
        < len(base_quorum.get("single_family_claims", []))
        or len(after_quorum.get("single_family_obligations", []))
        < len(base_quorum.get("single_family_obligations", []))
    )
    return {
        "enabled": True,
        "controller": "source_quorum_backfill_v1",
        "triggered": True,
        "reason": (
            "independent_family_evidence_added"
            if additions
            else "no_independent_family_candidate_available"
        ),
        "before_quorum_status": before_status,
        "after_quorum_status": after_quorum.get("quorum_status", before_status),
        "candidate_count": len(candidates),
        "selected_addition_count": len(additions),
        "selected_additions": additions,
        "used_token_budget": used_tokens,
        "max_additional_evidence": max_additional_evidence,
        "remaining_single_family_claims": after_quorum.get(
            "single_family_claims",
            [],
        ),
        "remaining_single_family_obligations": after_quorum.get(
            "single_family_obligations",
            [],
        ),
        "improved_quorum": bool(improved),
        "applied": bool(additions and improved),
        "base_proof_source_quorum": base_quorum,
        "augmented_minimal_proof_subgraph": augmented,
        "augmented_proof_source_quorum": after_quorum,
    }
