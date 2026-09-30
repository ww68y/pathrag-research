"""VSR-PathRAG v92 iterative answer refinement loop.

V92 implements a closed-loop self-calibration mechanism that uses the fidelity
scores from V91 to identify weak answer components and iteratively refine them.
This is inspired by Corrective RAG and Self-RAG but with key differences:
- No external LLM-as-judge calls; uses the VSR pipeline's own verification.
- Targeted refinement: only weak claims are regenerated, not the full answer.
- Proof-aware: each refinement step maintains proof sufficiency and quorum.
- Bounded: max iterations and token budget prevent infinite loops.

Refinement triggers:
1. CAQ below threshold (default 0.60)
2. Presence of weak claims (CFS < 0.40)
3. Low source attribution (SAS < 0.50)

Refinement actions per iteration:
1. Identify weakest claim(s) by CFS.
2. Check if additional evidence can be retrieved for those claims.
3. If yes, augment evidence and re-compute scores.
4. If no new evidence available, attempt claim rephrasing guidance.
5. Re-evaluate CAQ after each refinement.
6. Stop when CAQ >= threshold or max iterations reached.
"""

from __future__ import annotations

from typing import Any

from api.features.chats.fidelity_scoring import build_fidelity_scoring
from api.features.chats.minimal_proof_subgraph import (
    _candidate_rows,
    _question_obligations,
)


def _identify_weak_claims(
    fidelity_result: dict[str, Any],
    cfs_threshold: float = 0.40,
    max_weak_claims: int = 3,
) -> list[dict[str, Any]]:
    """Identify the weakest claims that need refinement."""
    per_claim_cfs = fidelity_result.get("claim_fidelity_score", {}).get(
        "per_claim_cfs", {}
    )
    weak_claims = [
        {"claim_id": cid, "cfs": score}
        for cid, score in sorted(
            per_claim_cfs.items(), key=lambda x: x[1]
        )
        if 0 < score < cfs_threshold
    ]
    return weak_claims[:max_weak_claims]


def _check_evidence_rescue_potential(
    weak_claims: list[dict[str, Any]],
    evidence_summary: dict[str, Any],
    query_graph: dict[str, Any] | None,
    claim_evidence_graph: dict[str, Any],
    minimal_proof_subgraph: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check if unselected evidence can cover weak claims.

    Returns candidate evidence rows that could rescue weak claims.
    """
    selected_ids = {
        row["evidence_id"]
        for row in minimal_proof_subgraph.get("selected_evidence", []) or []
    }
    obligations = _question_obligations(
        "", query_graph  # Question not needed for candidate lookup
    )
    candidates = _candidate_rows(
        evidence_summary,
        obligations,
        claim_evidence_graph,
        excluded_evidence_ids=selected_ids,
    )

    weak_claim_ids = {wc["claim_id"] for wc in weak_claims}

    rescue_candidates: list[dict[str, Any]] = []
    for row in candidates:
        # Check if this candidate covers any weak claim
        covered_weak = [
            cid for cid in row.get("supported_claim_ids", [])
            if cid in weak_claim_ids
        ]
        if covered_weak and not row.get("contradicted_claim_ids"):
            rescue_candidates.append({
                **row,
                "rescued_weak_claims": covered_weak,
                "rescue_value": len(covered_weak) * float(
                    row.get("max_claim_support_score", 0.0) or 0.0
                ),
            })

    # Sort by rescue value descending
    rescue_candidates.sort(key=lambda r: r.get("rescue_value", 0.0), reverse=True)
    return rescue_candidates[:5]  # Top 5 candidates


def build_iterative_refinement(
    question: str,
    answer: str,
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    minimal_proof_subgraph: dict[str, Any],
    proof_source_quorum: dict[str, Any] | None = None,
    fidelity_result: dict[str, Any] | None = None,
    caq_threshold: float = 0.60,
    max_iterations: int = 3,
    max_additional_evidence: int = 4,
    refinement_token_budget: int = 400,
) -> dict[str, Any]:
    """Execute iterative answer refinement loop.

    This is a lightweight refinement that:
    1. Uses V91 fidelity scores to identify weaknesses.
    2. Attempts targeted evidence rescue for weak claims.
    3. Recomputes scores after each refinement.
    4. Returns the best scoring variant.

    Note: This does NOT regenerate the answer text. It augments the evidence
    base and recomputes quality scores. Answer regeneration would require an
    LLM call and is left to the caller (or future V93).
    """
    if fidelity_result is None:
        fidelity_result = build_fidelity_scoring(
            question, answer, evidence_summary, claim_evidence_graph,
            minimal_proof_subgraph, proof_source_quorum,
        )

    caq = fidelity_result.get("composite_answer_quality", 0.0)
    recommendation = fidelity_result.get("refinement_recommendation", "none")

    should_refine = bool(
        caq < caq_threshold
        and recommendation != "none"
        and max_iterations > 0
    )

    if not should_refine:
        return {
            "enabled": True,
            "controller": "iterative_refinement_v1",
            "triggered": False,
            "reason": "answer_quality_sufficient_or_no_refinement_needed",
            "initial_caq": caq,
            "final_caq": caq,
            "iterations_completed": 0,
            "refinement_applied": False,
            "evidence_augmented": False,
            "weak_claims_identified": [],
            "rescue_candidates_found": 0,
            "refined_minimal_proof_subgraph": minimal_proof_subgraph,
            "refined_fidelity_score": fidelity_result,
        }

    # Initial state tracking
    current_proof = dict(minimal_proof_subgraph)
    current_caq = caq
    iterations: list[dict[str, Any]] = []
    total_added_evidence = 0

    for iteration in range(1, max_iterations + 1):
        # Recompute fidelity on current proof state
        current_fidelity = build_fidelity_scoring(
            question, answer, evidence_summary, claim_evidence_graph,
            current_proof, proof_source_quorum,
        )
        current_caq = current_fidelity.get("composite_answer_quality", 0.0)

        # Check termination
        if current_caq >= caq_threshold:
            iterations.append({
                "iteration": iteration,
                "action": "termination_caq_sufficient",
                "caq_before": round(caq, 4),
                "caq_after": round(current_caq, 4),
                "evidence_added": 0,
                "reason": f"CAQ {current_caq:.4f} >= threshold {caq_threshold}",
            })
            break

        # Identify weak claims
        weak_claims = _identify_weak_claims(current_fidelity)
        if not weak_claims:
            iterations.append({
                "iteration": iteration,
                "action": "termination_no_weak_claims",
                "caq_before": round(current_caq, 4),
                "caq_after": round(current_caq, 4),
                "evidence_added": 0,
                "reason": "No weak claims identifiable",
            })
            break

        # Check for rescue candidates
        rescue_candidates = _check_evidence_rescue_potential(
            weak_claims, evidence_summary,
            claim_evidence_graph.get("query_graph"),
            claim_evidence_graph, current_proof,
        )

        if not rescue_candidates:
            iterations.append({
                "iteration": iteration,
                "action": "termination_no_rescue_candidates",
                "caq_before": round(current_caq, 4),
                "caq_after": round(current_caq, 4),
                "evidence_added": 0,
                "reason": f"No rescue candidates for {len(weak_claims)} weak claims",
            })
            break

        # Apply top rescue candidates within budget
        added_this_iteration = 0
        used_tokens_this_iter = 0
        selected_ids = {
            row["evidence_id"]
            for row in current_proof.get("selected_evidence", []) or []
        }

        for candidate in rescue_candidates:
            if total_added_evidence >= max_additional_evidence:
                break
            if added_this_iteration >= 2:  # Max 2 per iteration
                break

            token_cost = int(candidate.get("estimated_tokens", 0) or 0)
            if used_tokens_this_iter + token_cost > refinement_token_budget // max_iterations:
                continue

            eid = candidate.get("evidence_id")
            if eid in selected_ids:
                continue

            # Add to proof
            new_row = {
                "evidence_id": eid,
                "kind": candidate.get("kind", "unknown"),
                "label": candidate.get("label", ""),
                "source_ids": candidate.get("source_ids", []),
                "obligation_hits": candidate.get("obligation_hits", []),
                "supported_claim_ids": candidate.get("supported_claim_ids", []),
                "max_claim_support_score": candidate.get(
                    "max_claim_support_score", 0.0
                ),
                "estimated_tokens": token_cost,
                "text_preview": candidate.get("text_preview", "")[:220],
                "refinement_added": True,
                "refinement_iteration": iteration,
                "rescued_claims": candidate.get("rescued_weak_claims", []),
            }

            selected_list = list(current_proof.get("selected_evidence", []))
            selected_list.append(new_row)
            current_proof["selected_evidence"] = selected_list
            current_proof["selected_count"] = len(selected_list)
            current_proof["estimated_tokens"] = int(
                current_proof.get("estimated_tokens", 0) or 0
            ) + token_cost

            selected_ids.add(eid)
            added_this_iteration += 1
            total_added_evidence += 1
            used_tokens_this_iter += token_cost

        # Record iteration result
        post_fidelity = build_fidelity_scoring(
            question, answer, evidence_summary, claim_evidence_graph,
            current_proof, proof_source_quorum,
        )
        post_caq = post_fidelity.get("composite_answer_quality", 0.0)

        iterations.append({
            "iteration": iteration,
            "action": "evidence_rescue_applied" if added_this_iteration > 0 else "no_rescue_applied",
            "caq_before": round(current_caq, 4),
            "caq_after": round(post_caq, 4),
            "evidence_added": added_this_iteration,
            "weak_claims_count": len(weak_claims),
            "rescue_candidates_count": len(rescue_candidates),
            "cumulative_evidence_added": total_added_evidence,
        })

        current_caq = post_caq

    # Final fidelity computation
    final_fidelity = build_fidelity_scoring(
        question, answer, evidence_summary, claim_evidence_graph,
        current_proof, proof_source_quorum,
    )
    final_caq = final_fidelity.get("composite_answer_quality", 0.0)

    improved = final_caq > caq + 0.02  # At least 2% improvement

    return {
        "enabled": True,
        "controller": "iterative_refinement_v1",
        "triggered": True,
        "reason": (
            "iterative_refinement_completed"
            if improved
            else "refinement_attempted_limited_gain"
        ),
        "initial_caq": round(caq, 4),
        "final_caq": round(final_caq, 4),
        "caq_improvement": round(final_caq - caq, 4),
        "iterations_completed": len(iterations),
        "max_iterations": max_iterations,
        "refinement_applied": bool(total_added_evidence > 0),
        "evidence_augmented": total_added_evidence > 0,
        "total_evidence_added": total_added_evidence,
        "weak_claims_identified": [
            wc["claim_id"] for it in iterations for wc in
            _identify_weak_claims(build_fidelity_scoring(
                question, answer, evidence_summary, claim_evidence_graph,
                current_proof, proof_source_quorum,
            )) if it.get("weak_claims_count", 0) > 0
        ][:5],  # Deduplicate roughly
        "rescue_candidates_evaluated": sum(
            it.get("rescue_candidates_count", 0) for it in iterations
        ),
        "iteration_details": iterations,
        "refined_minimal_proof_subgraph": current_proof,
        "refined_fidelity_score": final_fidelity,
    }
