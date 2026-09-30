"""VSR-PathRAG v81 proof-constrained claim admission."""

from __future__ import annotations

from typing import Any


def _clean_claim(text: str) -> str:
    return " ".join(str(text or "").strip().split()).strip(" .")


def _claim_status_map(claim_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("claim_id")): row
        for row in claim_graph.get("claims", [])
        if row.get("claim_id")
    }


def build_proof_constrained_admission(
    answer: str,
    claim_evidence_graph: dict[str, Any],
    minimal_proof_subgraph: dict[str, Any],
    counterfactual_path_necessity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a candidate answer using only proof-admitted claims.

    A claim is admitted only if it is supported by the signed claim graph,
    appears in the v78 covered-claim set, and is not contradicted. The
    result is intentionally deterministic and reference-free.
    """
    claim_map = _claim_status_map(claim_evidence_graph)
    covered_claims = {
        str(claim_id)
        for claim_id in minimal_proof_subgraph.get("covered_claims", [])
    }
    necessary_claims = set()
    for row in (counterfactual_path_necessity or {}).get(
        "interventions",
        [],
    ):
        for claim_id in row.get("lost_claims_if_removed", []):
            necessary_claims.add(str(claim_id))

    admitted = []
    rejected = []
    for claim_id, row in claim_map.items():
        claim = _clean_claim(row.get("claim", ""))
        if not claim:
            continue
        status = str(row.get("status", "unknown"))
        support = float(row.get("support_mass", 0.0) or 0.0)
        contradiction = float(row.get("contradiction_mass", 0.0) or 0.0)
        reasons = []
        if status != "supported":
            reasons.append(f"claim_status:{status}")
        if claim_id not in covered_claims:
            reasons.append("not_covered_by_minimal_proof")
        if contradiction > support and contradiction >= 0.24:
            reasons.append("contradiction_dominates_support")
        if reasons:
            rejected.append(
                {
                    "claim_id": claim_id,
                    "claim": claim,
                    "reasons": reasons,
                    "support_mass": round(support, 3),
                    "contradiction_mass": round(contradiction, 3),
                }
            )
            continue
        admitted.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "necessary": claim_id in necessary_claims,
                "support_mass": round(support, 3),
            }
        )

    if not admitted:
        candidate = ""
    else:
        # Preserve original claim order from the graph.
        candidate = ". ".join(row["claim"] for row in admitted) + "."

    original_claim_count = len(
        [_clean_claim(row.get("claim", "")) for row in claim_map.values()]
    )
    removed_count = len(rejected)
    changed = bool(
        candidate
        and _clean_claim(candidate) != _clean_claim(answer)
        and removed_count > 0
    )
    return {
        "enabled": True,
        "controller": "proof_constrained_claim_admission_v1",
        "candidate_available": bool(candidate),
        "candidate_selected": False,
        "changed": changed,
        "original_claim_count": original_claim_count,
        "admitted_claim_count": len(admitted),
        "rejected_claim_count": removed_count,
        "necessary_claim_count": sum(1 for row in admitted if row["necessary"]),
        "proof_covered_claim_count": len(covered_claims),
        "candidate_answer": candidate,
        "admitted_claims": admitted,
        "rejected_claims": rejected,
        "selection_reason": None,
    }
