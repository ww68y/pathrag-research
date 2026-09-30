"""VSR-PathRAG v88 obligation-source rescue.

V83 backfills single-family claims and obligations from unselected evidence, but it
operates under a tight token budget (360 tokens, 3 rows) and treats claims and
obligations with the same scoring formula.  When V87 cannot demote a remaining
obligation deficit (i.e. the obligation is *critical* — not a synthetic artifact,
not a claim-mediated intent), it means the answer has a real coverage gap at the
obligation level that is not independently sourced.

V88 runs after V83 backfill and before V87 criticality normalization.  It
targets *only* the obligations that V87 would later classify as critical:
obligations that are neither short anchor artifacts nor claim-mediated generic
intents.  For those critical obligations, V88 performs a focused second-pass
search with a separate obligation-only budget, preferring evidence that:

1. Covers a critical obligation via its ``obligation_hits``.
2. Comes from a source family not already in the augmented proof.
3. Does not contradict existing supported claims.

If V88 finds qualifying evidence, it augments the proof further and the
downstream V87 quorum check may upgrade ``correlated_monitor`` to
``independent_quorum``.  If no qualifying evidence exists, V88 accepts the
V87 outcome without modifying the answer text.
"""

from __future__ import annotations

from typing import Any

from api.features.chats.minimal_proof_subgraph import (
    _candidate_rows,
    _question_obligations,
)
from api.features.chats.obligation_criticality_quorum import (
    CLAIM_MEDIATED_INTENTS,
    _is_short_anchor_artifact,
)
from api.features.chats.proof_source_quorum import (
    _families_for_evidence,
    _lineage_index,
    certify_proof_source_quorum,
)


def _is_critical_obligation(obligation_id: str) -> bool:
    """Return True if V87 would classify this obligation as critical (non-demotable).

    Critical obligations are real domain-specific obligations that represent
    independent citation targets.  Non-critical obligations include:
    - Short anchor artifacts (≤2 alphanumeric chars) — synthetic parsing noise
    - Long entity-name anchors — person/place/organization names extracted as
      anchors, not independent factual claims
    - Claim-mediated generic intents — already covered by claim-level quorum
    """
    if _is_short_anchor_artifact(obligation_id):
        return False
    if obligation_id in CLAIM_MEDIATED_INTENTS:
        return False
    if obligation_id == "intent:fact":
        return False
    # Long entity-name anchors (e.g. "anchor:LAWRENCE SCOTT SHEETS",
    # "anchor:United Nations") are extraction artifacts, not independent
    # citation targets.  They should be demoted by V87, not rescued by V88.
    if obligation_id.startswith("anchor:"):
        return False
    # Anything else is a real domain-specific obligation
    return bool(obligation_id)


def _obligation_ids_from_row(row: dict[str, Any]) -> set[str]:
    return {
        str(hit.get("obligation_id", ""))
        for hit in row.get("obligation_hits", [])
        if hit.get("obligation_id")
    }


def _proof_family_ids(
    minimal_proof_subgraph: dict[str, Any],
    evidence_summary: dict[str, Any],
) -> set[str]:
    """Collect all source family IDs currently in the proof."""
    selected = minimal_proof_subgraph.get("selected_evidence", []) or []
    source_index = _lineage_index(evidence_summary or {})
    families: set[str] = set()
    for evidence in selected:
        lineage = _families_for_evidence(evidence, source_index)
        families.update(lineage.get("family_ids", []))
    return families


def _public_rescue_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": row["evidence_id"],
        "kind": row.get("kind", "unknown"),
        "label": row.get("label", ""),
        "source_ids": row.get("source_ids", []),
        "obligation_hits": row.get("obligation_hits", []),
        "supported_claim_ids": row.get("supported_claim_ids", []),
        "contradicted_claim_ids": row.get("contradicted_claim_ids", []),
        "max_claim_support_score": row.get("max_claim_support_score", 0.0),
        "estimated_tokens": row.get("estimated_tokens", 0),
        "text_preview": row.get("text_preview", ""),
    }


def build_obligation_source_rescue(
    question: str,
    query_graph: dict[str, Any] | None,
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    evidence_audit: dict[str, Any] | None,
    minimal_proof_subgraph: dict[str, Any],
    proof_source_quorum: dict[str, Any] | None = None,
    max_additional_evidence: int = 4,
    token_budget: int = 480,
) -> dict[str, Any]:
    """Rescue critical obligation deficits with obligation-focused evidence.

    This runs after V83 source-quorum backfill and targets only obligations
    that V87 would classify as critical (real domain obligations, not
    synthetic artifacts).  It uses a separate obligation-focused budget and
    scoring that prioritizes obligation coverage over claim coverage.
    """
    base_quorum = proof_source_quorum or certify_proof_source_quorum(
        minimal_proof_subgraph,
        evidence_summary,
        claim_evidence_graph,
    )
    base_status = base_quorum.get("quorum_status", "unknown")

    # Only activate when there are critical obligation deficits
    all_single_obligations = [
        str(o) for o in base_quorum.get("single_family_obligations", [])
    ]
    all_missing_obligations = [
        str(o) for o in base_quorum.get("missing_obligations", [])
    ]
    critical_single = [o for o in all_single_obligations if _is_critical_obligation(o)]
    critical_missing = [o for o in all_missing_obligations if _is_critical_obligation(o)]

    should_attempt = bool(
        base_status in {"correlated_monitor", "single_source_critical"}
        and (critical_single or critical_missing)
    )
    if not should_attempt:
        return {
            "enabled": True,
            "controller": "obligation_source_rescue_v1",
            "triggered": False,
            "reason": "no_critical_obligation_deficit",
            "before_quorum_status": base_status,
            "after_quorum_status": base_status,
            "critical_obligation_count": 0,
            "rescued_obligation_count": 0,
            "selected_addition_count": 0,
            "selected_additions": [],
            "base_proof_source_quorum": base_quorum,
            "rescued_minimal_proof_subgraph": minimal_proof_subgraph,
            "rescued_proof_source_quorum": base_quorum,
        }

    selected_ids = {
        str(row.get("evidence_id"))
        for row in minimal_proof_subgraph.get("selected_evidence", [])
        if row.get("evidence_id")
    }
    obligations = _question_obligations(question, query_graph or {})
    candidates = _candidate_rows(
        evidence_summary,
        obligations,
        claim_evidence_graph,
        excluded_evidence_ids=selected_ids,
    )
    source_index = _lineage_index(evidence_summary or {})
    proof_families = _proof_family_ids(minimal_proof_subgraph, evidence_summary)

    # Score candidates specifically for critical obligation rescue
    additions = []
    used_tokens = 0
    rescued_obligations: set[str] = set()
    remaining = list(candidates)

    while remaining and len(additions) < max_additional_evidence:
        best = None
        best_score = 0.0
        best_detail: dict[str, Any] = {}

        for row in remaining:
            # Skip contradicted evidence
            if row.get("contradicted_claim_ids"):
                continue
            token_cost = int(row.get("estimated_tokens", 0) or 0)
            if used_tokens + token_cost > token_budget:
                continue

            lineage = _families_for_evidence(row, source_index)
            families = set(lineage.get("family_ids", []))
            if not families:
                continue

            # Count new families this evidence would add
            new_families = families - proof_families - {
                f for a in additions for f in a.get("rescue_family_ids", [])
            }
            new_family_count = len(new_families)

            # Check which critical obligations this covers
            row_obligation_ids = _obligation_ids_from_row(row)
            covered_critical_single = [
                o for o in critical_single
                if o in row_obligation_ids and o not in rescued_obligations
            ]
            covered_critical_missing = [
                o for o in critical_missing
                if o in row_obligation_ids and o not in rescued_obligations
            ]

            # Obligation-rescue scoring: heavily weight obligation coverage
            # and new source families
            score = (
                0.45 * len(covered_critical_single)
                + 0.40 * len(covered_critical_missing)
                + 0.08 * min(1, new_family_count)
                + 0.04 * float(row.get("max_claim_support_score", 0.0) or 0.0)
                - 0.008 * token_cost / 100.0
            )

            if score > best_score:
                best = row
                best_score = score
                best_detail = {
                    "families": sorted(families),
                    "new_families": sorted(new_families),
                    "covered_critical_single": sorted(covered_critical_single),
                    "covered_critical_missing": sorted(covered_critical_missing),
                    "new_family_count": new_family_count,
                }

        if best is None or best_score <= 0:
            break

        remaining.remove(best)
        used_tokens += int(best.get("estimated_tokens", 0) or 0)

        addition = _public_rescue_row(best)
        addition["rescue_score"] = round(best_score, 4)
        addition["rescue_family_ids"] = best_detail["families"]
        addition["rescued_obligation_ids"] = (
            best_detail["covered_critical_single"]
            + best_detail["covered_critical_missing"
        ]
        )
        additions.append(addition)
        rescued_obligations.update(best_detail["covered_critical_single"])
        rescued_obligations.update(best_detail["covered_critical_missing"])
        proof_families.update(best_detail["families"])

        # Update critical lists
        critical_single = [o for o in critical_single if o not in rescued_obligations]
        critical_missing = [o for o in critical_missing if o not in rescued_obligations]

        if not critical_single and not critical_missing:
            break

    # Build augmented proof
    augmented = dict(minimal_proof_subgraph)
    augmented_selected = (
        list(minimal_proof_subgraph.get("selected_evidence", [])) + additions
    )
    augmented["selected_evidence"] = augmented_selected
    augmented["selected_count"] = len(augmented_selected)
    augmented["estimated_tokens"] = int(
        minimal_proof_subgraph.get("estimated_tokens", 0) or 0
    ) + used_tokens

    covered_claims = sorted({
        cid for row in augmented_selected for cid in row.get("supported_claim_ids", [])
    })
    covered_obligations = sorted({
        hit.get("obligation_id")
        for row in augmented_selected
        for hit in row.get("obligation_hits", [])
        if hit.get("obligation_id")
    })
    covered_sources = sorted({
        sid for row in augmented_selected for sid in row.get("source_ids", [])
    })
    augmented["covered_claims"] = covered_claims
    augmented["covered_obligations"] = covered_obligations
    augmented["unique_source_count"] = len(covered_sources)

    required_claim_count = int(minimal_proof_subgraph.get("required_claim_count", 0) or 0)
    required_obligation_count = int(
        minimal_proof_subgraph.get("required_obligation_count", 0) or 0
    )
    claim_coverage = len(covered_claims) / max(1, required_claim_count)
    obligation_coverage = len(covered_obligations) / max(1, required_obligation_count)
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

    # Re-certify quorum on the rescued proof
    after_quorum = certify_proof_source_quorum(
        augmented, evidence_summary, claim_evidence_graph
    )

    improved = (
        base_status != after_quorum.get("quorum_status")
        or len(after_quorum.get("single_family_obligations", []))
        < len(base_quorum.get("single_family_obligations", []))
    )

    return {
        "enabled": True,
        "controller": "obligation_source_rescue_v1",
        "triggered": True,
        "reason": (
            "critical_obligation_evidence_added"
            if additions
            else "no_critical_obligation_candidate_available"
        ),
        "before_quorum_status": base_status,
        "after_quorum_status": after_quorum.get("quorum_status", base_status),
        "critical_obligation_count": len(all_single_obligations) + len(all_missing_obligations),
        "rescued_obligation_count": len(rescued_obligations),
        "remaining_critical_single": critical_single,
        "remaining_critical_missing": critical_missing,
        "candidate_count": len(candidates),
        "selected_addition_count": len(additions),
        "selected_additions": additions,
        "used_token_budget": used_tokens,
        "improved_quorum": bool(improved),
        "applied": bool(additions and improved),
        "base_proof_source_quorum": base_quorum,
        "rescued_minimal_proof_subgraph": augmented,
        "rescued_proof_source_quorum": after_quorum,
    }
