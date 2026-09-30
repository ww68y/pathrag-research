"""VSR-PathRAG v85 proof-deficit answer repair.

V81 admits claims from the signed claim graph when they are supported and
covered by the proof subgraph.  V85 adds a stricter terminal repair pass for
cases where the independent terminal audit still marks answer claims as
unsupported after proof completion.  The repair is deterministic and
reference-free: it keeps only claims that pass the signed graph, completed
proof coverage, and audit-unsupported filters.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def _clean_claim(text: str) -> str:
    return " ".join(str(text or "").strip().split()).strip(" .")


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _claim_similarity(left: str, right: str) -> float:
    left_clean = _clean_claim(left).lower()
    right_clean = _clean_claim(right).lower()
    if not left_clean or not right_clean:
        return 0.0
    left_tokens = _token_set(left_clean)
    right_tokens = _token_set(right_clean)
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = overlap / max(1, union)
    containment = min(
        overlap / max(1, len(left_tokens)),
        overlap / max(1, len(right_tokens)),
    )
    sequence = SequenceMatcher(None, left_clean, right_clean).ratio()
    return max(jaccard, containment, sequence)


def _claim_texts(rows: Any) -> list[str]:
    texts = []
    if not isinstance(rows, list):
        return texts
    for row in rows:
        if isinstance(row, dict):
            value = row.get("claim") or row.get("text") or row.get("answer")
        else:
            value = row
        cleaned = _clean_claim(str(value or ""))
        if cleaned and cleaned not in texts:
            texts.append(cleaned)
    return texts


def _audit_unsupported_claims(
    evidence_audit: dict[str, Any],
    claim_verification: dict[str, Any],
) -> list[str]:
    texts = []
    for source in (
        evidence_audit.get("unsupported_claim_examples", []),
        claim_verification.get("unsupported_claim_examples", []),
        claim_verification.get("unsupported_claims", []),
    ):
        for claim in _claim_texts(source):
            if claim not in texts:
                texts.append(claim)
    return texts


def _best_unsupported_match(claim: str, unsupported_claims: list[str]) -> dict[str, Any]:
    best_text = ""
    best_score = 0.0
    for unsupported in unsupported_claims:
        score = _claim_similarity(claim, unsupported)
        if score > best_score:
            best_score = score
            best_text = unsupported
    return {
        "score": round(best_score, 3),
        "claim": best_text,
    }


def _deficit_types(
    evidence_audit: dict[str, Any],
    minimal_proof_subgraph: dict[str, Any],
    proof_completion_backfill: dict[str, Any],
    proof_source_quorum: dict[str, Any],
    risk_calibrated_submission: dict[str, Any],
) -> list[str]:
    deficits = []
    if proof_source_quorum.get("quorum_status") == "quorum_failed":
        deficits.append("quorum_failed")
    if not minimal_proof_subgraph.get("proof_sufficient"):
        deficits.append("proof_incomplete")
    if int(evidence_audit.get("unsupported_claim_count", 0) or 0) > 0:
        deficits.append("audit_unsupported_claims")
    if proof_completion_backfill.get("missing_claims_after"):
        deficits.append("claim_gap_after_completion")
    if proof_completion_backfill.get("missing_obligations_after"):
        deficits.append("obligation_gap_after_completion")
    if risk_calibrated_submission.get("action") in {
        "block_unsupported",
        "defer_or_fallback",
    }:
        deficits.append("submission_blocked")
    return list(dict.fromkeys(deficits))


def build_proof_deficit_answer_repair(
    answer: str,
    terminal_assessment: dict[str, Any],
    unsupported_similarity_threshold: float = 0.78,
) -> dict[str, Any]:
    """Build a proof-deficit repaired answer candidate.

    The candidate is stricter than V81: a claim can be signed-graph supported
    and proof-covered, but still removed if the independent evidence audit
    marks the same claim as unsupported.
    """
    claim_graph = terminal_assessment.get("claim_evidence_graph") or {}
    evidence_audit = terminal_assessment.get("evidence_audit") or {}
    claim_verification = terminal_assessment.get("claim_verification") or {}
    minimal_proof_subgraph = terminal_assessment.get("minimal_proof_subgraph") or {}
    proof_completion_backfill = (
        terminal_assessment.get("proof_completion_backfill") or {}
    )
    proof_source_quorum = terminal_assessment.get("proof_source_quorum") or {}
    risk_calibrated_submission = (
        terminal_assessment.get("risk_calibrated_submission") or {}
    )

    deficits = _deficit_types(
        evidence_audit,
        minimal_proof_subgraph,
        proof_completion_backfill,
        proof_source_quorum,
        risk_calibrated_submission,
    )
    covered_claims = {
        str(claim_id)
        for claim_id in minimal_proof_subgraph.get("covered_claims", [])
    }
    missing_after_completion = {
        str(claim_id)
        for claim_id in proof_completion_backfill.get(
            "missing_claims_after",
            [],
        )
    }
    unsupported_claims = _audit_unsupported_claims(
        evidence_audit,
        claim_verification,
    )

    should_attempt = bool(
        deficits
        and (
            proof_source_quorum.get("quorum_status") == "quorum_failed"
            or risk_calibrated_submission.get("action")
            in {"block_unsupported", "defer_or_fallback"}
            or unsupported_claims
            or missing_after_completion
        )
    )
    if not should_attempt:
        return {
            "enabled": True,
            "controller": "proof_deficit_answer_repair_v1",
            "triggered": False,
            "reason": "no_terminal_proof_deficit",
            "candidate_available": False,
            "candidate_selected": False,
            "changed": False,
            "candidate_answer": "",
            "deficit_types": deficits,
            "current_quorum_status": proof_source_quorum.get(
                "quorum_status",
                "unknown",
            ),
            "current_submission_action": risk_calibrated_submission.get(
                "action",
                "unknown",
            ),
            "current_unsupported_claim_count": int(
                evidence_audit.get("unsupported_claim_count", 0) or 0
            ),
            "admitted_claim_count": 0,
            "rejected_claim_count": 0,
            "audit_unsupported_match_count": 0,
            "admitted_claims": [],
            "rejected_claims": [],
            "selection_reason": None,
        }

    admitted = []
    rejected = []
    audit_match_count = 0
    for row in claim_graph.get("claims", []):
        claim_id = str(row.get("claim_id", ""))
        claim = _clean_claim(row.get("claim", ""))
        if not claim:
            continue
        status = str(row.get("status", "unknown"))
        support = float(row.get("support_mass", 0.0) or 0.0)
        contradiction = float(row.get("contradiction_mass", 0.0) or 0.0)
        unsupported_match = _best_unsupported_match(
            claim,
            unsupported_claims,
        )
        reasons = []
        if status != "supported":
            reasons.append(f"claim_status:{status}")
        if claim_id not in covered_claims:
            reasons.append("not_covered_by_completed_proof")
        if claim_id in missing_after_completion:
            reasons.append("missing_after_proof_completion")
        if contradiction > support and contradiction >= 0.24:
            reasons.append("contradiction_dominates_support")
        if unsupported_match["score"] >= unsupported_similarity_threshold:
            reasons.append("audit_unsupported_claim")
            audit_match_count += 1

        if reasons:
            rejected.append(
                {
                    "claim_id": claim_id,
                    "claim": claim,
                    "reasons": reasons,
                    "support_mass": round(support, 3),
                    "contradiction_mass": round(contradiction, 3),
                    "audit_unsupported_similarity": unsupported_match[
                        "score"
                    ],
                    "audit_unsupported_claim": unsupported_match["claim"],
                }
            )
            continue
        admitted.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "support_mass": round(support, 3),
            }
        )

    candidate = ""
    if admitted:
        candidate = ". ".join(row["claim"] for row in admitted) + "."
    changed = bool(
        candidate
        and rejected
        and _clean_claim(candidate) != _clean_claim(answer)
    )
    return {
        "enabled": True,
        "controller": "proof_deficit_answer_repair_v1",
        "triggered": True,
        "reason": (
            "proof_deficit_claims_pruned"
            if changed
            else "no_prunable_claim_preserving_answer"
        ),
        "candidate_available": bool(candidate),
        "candidate_selected": False,
        "changed": changed,
        "candidate_answer": candidate,
        "deficit_types": deficits,
        "current_quorum_status": proof_source_quorum.get(
            "quorum_status",
            "unknown",
        ),
        "current_submission_action": risk_calibrated_submission.get(
            "action",
            "unknown",
        ),
        "current_unsupported_claim_count": int(
            evidence_audit.get("unsupported_claim_count", 0) or 0
        ),
        "current_proof_sufficient": bool(
            minimal_proof_subgraph.get("proof_sufficient")
        ),
        "missing_claims_after_completion": sorted(missing_after_completion),
        "missing_obligations_after_completion": (
            proof_completion_backfill.get("missing_obligations_after", [])
        ),
        "admitted_claim_count": len(admitted),
        "rejected_claim_count": len(rejected),
        "audit_unsupported_match_count": audit_match_count,
        "admitted_claims": admitted,
        "rejected_claims": rejected,
        "selection_reason": None,
    }
