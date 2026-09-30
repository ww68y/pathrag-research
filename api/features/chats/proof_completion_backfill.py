"""VSR-PathRAG v84 proof-completion backfill.

V78 builds a compact proof subgraph.  V84 adds a bounded completion pass for
cases where the compact proof misses supported answer claims or question
obligations even though suitable unselected evidence is already available.
"""

from __future__ import annotations

from typing import Any

from api.features.chats.minimal_proof_subgraph import (
    _candidate_rows,
    _minimality_certificate,
    _question_obligations,
)


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


def _covered_claims(selected: list[dict[str, Any]]) -> set[str]:
    return {
        claim_id
        for row in selected
        for claim_id in row.get("supported_claim_ids", [])
    }


def _covered_obligations(selected: list[dict[str, Any]]) -> set[str]:
    return {
        hit.get("obligation_id")
        for row in selected
        for hit in row.get("obligation_hits", [])
        if hit.get("obligation_id")
    }


def _covered_sources(selected: list[dict[str, Any]]) -> set[str]:
    return {
        source_id
        for row in selected
        for source_id in row.get("source_ids", [])
    }


def _support_window_index(
    claim_evidence_graph: dict[str, Any],
) -> dict[tuple[str, str], str]:
    windows: dict[tuple[str, str], str] = {}
    for edge in claim_evidence_graph.get("edges", []):
        if edge.get("relation") != "supports":
            continue
        evidence_id = str(edge.get("evidence_id", ""))
        claim_id = str(edge.get("claim_id", ""))
        window = str(edge.get("evidence_window", "")).strip()
        if evidence_id and claim_id and window:
            windows[(evidence_id, claim_id)] = window
    return windows


def _focused_completion_row(
    row: dict[str, Any],
    claim_ids: list[str],
    support_windows: dict[tuple[str, str], str],
) -> dict[str, Any]:
    windows = [
        support_windows[(row["evidence_id"], claim_id)]
        for claim_id in claim_ids
        if (row["evidence_id"], claim_id) in support_windows
    ]
    preview = " ".join(windows) or str(row.get("text_preview", ""))
    preview = " ".join(preview.split())[:520]
    focused = dict(row)
    focused["text_preview"] = preview
    focused["estimated_tokens"] = max(8, (len(preview) + 3) // 4)
    focused["completion_focused_window"] = bool(windows)
    return focused


def _proof_metrics(
    selected: list[dict[str, Any]],
    required_claims: set[str],
    required_obligations: set[str],
    evidence_audit: dict[str, Any] | None,
    contradicted_claims: list[str],
) -> dict[str, Any]:
    covered_claims = _covered_claims(selected)
    covered_obligations = _covered_obligations(selected)
    covered_sources = _covered_sources(selected)
    claim_coverage = len(covered_claims) / max(1, len(required_claims))
    obligation_coverage = len(covered_obligations) / max(
        1,
        len(required_obligations),
    )
    unsupported = int((evidence_audit or {}).get("unsupported_claim_count", 0) or 0)
    proof_sufficient = bool(
        claim_coverage >= 1.0
        and (obligation_coverage >= 0.66 or not required_obligations)
        and unsupported == 0
        and not contradicted_claims
    )
    joint_objective = (
        0.52 * claim_coverage
        + 0.34 * obligation_coverage
        + 0.08 * min(1.0, len(covered_sources) / max(1, len(selected)))
        + 0.06 * (1.0 if proof_sufficient else 0.0)
    )
    return {
        "covered_claims": sorted(covered_claims),
        "covered_obligations": sorted(covered_obligations),
        "unique_source_count": len(covered_sources),
        "claim_support_coverage_ratio": round(claim_coverage, 3),
        "obligation_coverage_ratio": round(obligation_coverage, 3),
        "joint_objective": round(joint_objective, 4),
        "proof_sufficient": proof_sufficient,
    }


def build_proof_completion_backfill(
    question: str,
    query_graph: dict[str, Any] | None,
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    evidence_audit: dict[str, Any] | None,
    minimal_proof_subgraph: dict[str, Any],
    max_additional_evidence: int = 4,
    token_budget: int = 520,
) -> dict[str, Any]:
    """Complete missing claim/obligation proof coverage from available evidence."""
    obligations = _question_obligations(question, query_graph or {})
    required_obligations = {row["id"] for row in obligations}
    required_claims = {
        str(row.get("claim_id"))
        for row in claim_evidence_graph.get("claims", [])
        if row.get("claim_id")
    }
    selected = list(minimal_proof_subgraph.get("selected_evidence", []) or [])
    selected_ids = {
        str(row.get("evidence_id"))
        for row in selected
        if row.get("evidence_id")
    }
    before_claims = set(minimal_proof_subgraph.get("covered_claims", []))
    before_obligations = set(
        minimal_proof_subgraph.get("covered_obligations", [])
    )
    missing_claims = set(required_claims - before_claims)
    missing_obligations = set(required_obligations - before_obligations)
    before_proof_sufficient = bool(
        minimal_proof_subgraph.get("proof_sufficient")
    )
    should_attempt = bool(
        not before_proof_sufficient and (missing_claims or missing_obligations)
    )
    if not should_attempt:
        return {
            "enabled": True,
            "controller": "proof_completion_backfill_v1",
            "triggered": False,
            "reason": "proof_already_sufficient_or_no_coverable_gap",
            "before_proof_sufficient": before_proof_sufficient,
            "after_proof_sufficient": before_proof_sufficient,
            "candidate_count": 0,
            "selected_addition_count": 0,
            "selected_additions": [],
            "missing_claims_before": sorted(missing_claims),
            "missing_obligations_before": sorted(missing_obligations),
            "missing_claims_after": sorted(missing_claims),
            "missing_obligations_after": sorted(missing_obligations),
            "improved_proof": False,
            "applied": False,
            "base_minimal_proof_subgraph": minimal_proof_subgraph,
            "augmented_minimal_proof_subgraph": minimal_proof_subgraph,
        }

    candidates = _candidate_rows(
        evidence_summary,
        obligations,
        claim_evidence_graph,
        excluded_evidence_ids=selected_ids,
    )
    support_windows = _support_window_index(claim_evidence_graph)
    additions = []
    used_tokens = 0
    covered_claims = set(before_claims)
    covered_obligations = set(before_obligations)
    remaining = list(candidates)
    while remaining and len(additions) < max_additional_evidence:
        best = None
        best_original = None
        best_score = 0.0
        best_detail: dict[str, Any] = {}
        for row in remaining:
            if row.get("contradicted_claim_ids"):
                continue
            row_claims = set(row.get("supported_claim_ids", []))
            row_obligations = {
                hit.get("obligation_id")
                for hit in row.get("obligation_hits", [])
                if hit.get("obligation_id")
            }
            new_claims = sorted(row_claims & missing_claims)
            new_obligations = sorted(row_obligations & missing_obligations)
            if not new_claims and not new_obligations:
                continue
            candidate_row = row
            token_cost = int(row.get("estimated_tokens", 0) or 0)
            if used_tokens + token_cost > token_budget:
                candidate_row = _focused_completion_row(
                    row,
                    new_claims,
                    support_windows,
                )
                token_cost = int(
                    candidate_row.get("estimated_tokens", 0) or 0
                )
                if used_tokens + token_cost > token_budget:
                    continue
            source_novelty = len(
                set(row.get("source_ids", [])) - _covered_sources(selected + additions)
            )
            score = (
                0.58 * len(new_claims) / max(1, len(required_claims))
                + 0.34
                * len(new_obligations)
                / max(1, len(required_obligations))
                + 0.08 * min(1, source_novelty)
                + 0.08 * float(row.get("max_claim_support_score", 0.0) or 0.0)
                - 0.012 * token_cost / 100.0
            )
            if score > best_score:
                best = candidate_row
                best_original = row
                best_score = score
                best_detail = {
                    "new_claims": new_claims,
                    "new_obligations": new_obligations,
                }
        if best is None:
            break
        remaining.remove(best_original)
        addition = _public_evidence_row(best)
        addition["completion_score"] = round(best_score, 4)
        addition["completed_claim_ids"] = best_detail["new_claims"]
        addition["completed_obligation_ids"] = best_detail["new_obligations"]
        additions.append(addition)
        used_tokens += int(best.get("estimated_tokens", 0) or 0)
        covered_claims.update(best_detail["new_claims"])
        covered_obligations.update(best_detail["new_obligations"])
        missing_claims = required_claims - covered_claims
        missing_obligations = required_obligations - covered_obligations
        if not missing_claims and (
            len(covered_obligations) / max(1, len(required_obligations)) >= 0.66
            or not required_obligations
        ):
            break

    augmented = dict(minimal_proof_subgraph)
    augmented_selected = selected + additions
    contradicted_claims = sorted(
        {
            claim_id
            for row in augmented_selected
            for claim_id in row.get("contradicted_claim_ids", [])
        }
        | set(minimal_proof_subgraph.get("contradicted_claims_in_proof", []))
    )
    metrics = _proof_metrics(
        augmented_selected,
        required_claims,
        required_obligations,
        evidence_audit,
        contradicted_claims,
    )
    augmented.update(metrics)
    augmented["selected_evidence"] = augmented_selected
    augmented["selected_count"] = len(augmented_selected)
    augmented["estimated_tokens"] = int(
        minimal_proof_subgraph.get("estimated_tokens", 0) or 0
    ) + used_tokens
    augmented["contradicted_claims_in_proof"] = contradicted_claims
    augmented["completion_backfilled"] = bool(additions)
    augmented["completion_addition_count"] = len(additions)
    augmented["submission_guidance"] = (
        "proof_sufficient"
        if augmented["proof_sufficient"]
        else "proof_incomplete_monitor"
    )
    augmented["minimality_certificate"] = _minimality_certificate(
        augmented_selected
    )
    after_missing_claims = required_claims - set(augmented["covered_claims"])
    after_missing_obligations = required_obligations - set(
        augmented["covered_obligations"]
    )
    improved = bool(
        len(after_missing_claims) < len(required_claims - before_claims)
        or len(after_missing_obligations)
        < len(required_obligations - before_obligations)
        or augmented["proof_sufficient"] != before_proof_sufficient
    )
    return {
        "enabled": True,
        "controller": "proof_completion_backfill_v1",
        "triggered": True,
        "reason": (
            "missing_proof_targets_completed"
            if additions
            else "no_completion_candidate_available"
        ),
        "before_proof_sufficient": before_proof_sufficient,
        "after_proof_sufficient": augmented["proof_sufficient"],
        "candidate_count": len(candidates),
        "selected_addition_count": len(additions),
        "selected_additions": additions,
        "used_token_budget": used_tokens,
        "max_additional_evidence": max_additional_evidence,
        "missing_claims_before": sorted(required_claims - before_claims),
        "missing_obligations_before": sorted(
            required_obligations - before_obligations
        ),
        "missing_claims_after": sorted(after_missing_claims),
        "missing_obligations_after": sorted(after_missing_obligations),
        "claim_coverage_before": minimal_proof_subgraph.get(
            "claim_support_coverage_ratio",
            0.0,
        ),
        "claim_coverage_after": augmented["claim_support_coverage_ratio"],
        "obligation_coverage_before": minimal_proof_subgraph.get(
            "obligation_coverage_ratio",
            0.0,
        ),
        "obligation_coverage_after": augmented["obligation_coverage_ratio"],
        "improved_proof": improved,
        "applied": bool(additions and improved),
        "base_minimal_proof_subgraph": minimal_proof_subgraph,
        "augmented_minimal_proof_subgraph": augmented,
    }
