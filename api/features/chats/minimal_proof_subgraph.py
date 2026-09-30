"""VSR-PathRAG v78 minimal proof subgraph optimizer.

The v78 mechanism joins two previously separate signals:
1. question-obligation coverage from the query graph; and
2. signed claim support from the final answer's evidence graph.

It greedily selects a compact set of evidence nodes that covers both
surfaces, then prunes redundant nodes with a leave-one-out certificate.
The module is deterministic and reference-free, so it remains in the
pre-ablation innovation stage.
"""

from __future__ import annotations

import hashlib
from typing import Any

from api.features.chats.vsr_pathrag import (
    lexical_terms,
    normalize_label,
    normalize_text,
    split_source_ids,
)


INTENT_HINT_TERMS = {
    "cause_effect": {"cause", "effect", "impact", "result", "because"},
    "causal_explanation": {"cause", "effect", "impact", "result"},
    "mechanism": {"mechanism", "process", "through", "using", "method"},
    "comparison": {"compare", "difference", "contrast", "between"},
    "role": {"role", "significance", "responsibility", "contribution"},
    "time": {"time", "date", "year", "timeline", "when"},
    "quantity": {"amount", "number", "rate", "percentage"},
    "enumeration": {"factor", "type", "step", "item", "list"},
    "fact": {"fact", "property", "definition", "description"},
    "impact": {"impact", "effect", "influence", "significance"},
}


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:12]}"


def _evidence_items(evidence_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Mirror the controller evidence order used by the signed graph."""
    items: list[dict[str, Any]] = []
    for entity in evidence_summary.get("entities", []):
        name = normalize_label(entity.get("name", ""))
        text = normalize_text(
            " ".join(
                [
                    name,
                    str(entity.get("type", "")),
                    str(entity.get("description", "")),
                ]
            )
        )
        items.append(
            {
                "item_id": f"entity:{name}",
                "kind": "entity",
                "label": name,
                "text": text,
                "source_ids": split_source_ids(
                    entity.get("source_ids") or entity.get("source_id")
                ),
                "repair_id": entity.get("repair_id"),
            }
        )
    for relation in evidence_summary.get("relationships", []):
        src = normalize_label(relation.get("src", ""))
        tgt = normalize_label(relation.get("tgt", ""))
        label = f"{src} -> {tgt}"
        text = normalize_text(
            " ".join(
                [
                    src,
                    tgt,
                    str(relation.get("keywords", "")),
                    str(relation.get("description", "")),
                ]
            )
        )
        items.append(
            {
                "item_id": relation.get("repair_id")
                or f"relationship:{src}->{tgt}",
                "kind": "relationship",
                "label": label,
                "text": text,
                "source_ids": split_source_ids(
                    relation.get("source_ids") or relation.get("source_id")
                ),
                "repair_id": relation.get("repair_id"),
            }
        )
    for excerpt in evidence_summary.get("source_excerpts", []):
        source_id = normalize_text(excerpt.get("source_id", ""))
        text = normalize_text(excerpt.get("text", ""))
        if not source_id or not text:
            continue
        items.append(
            {
                "item_id": f"source:{source_id}",
                "kind": "source_excerpt",
                "label": source_id,
                "text": text,
                "source_ids": [source_id],
                "repair_id": None,
            }
        )
    for index, item in enumerate(items):
        item["evidence_id"] = f"evidence-{index}"
        item["terms"] = lexical_terms(item.get("text", ""))
        item["estimated_tokens"] = max(
            8,
            (len(item.get("text", "")) + 3) // 4,
        )
    return items


def _question_obligations(
    question: str,
    query_graph: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    graph = query_graph or {}
    obligations: list[dict[str, Any]] = []
    seen_anchor_terms: set[tuple[str, ...]] = set()
    for anchor in graph.get("anchors", [])[:8]:
        terms = lexical_terms(anchor)
        key = tuple(sorted(terms))
        if not terms or key in seen_anchor_terms:
            continue
        seen_anchor_terms.add(key)
        obligations.append(
            {
                "id": f"anchor:{normalize_label(anchor)}",
                "type": "query_anchor",
                "target": normalize_text(anchor),
                "terms": terms,
                "priority": 0.92,
            }
        )
    for intent in graph.get("relation_intents", [])[:8]:
        terms = lexical_terms(intent) | INTENT_HINT_TERMS.get(
            str(intent),
            set(),
        )
        if terms:
            obligations.append(
                {
                    "id": f"intent:{intent}",
                    "type": "relation_intent",
                    "target": str(intent),
                    "terms": terms,
                    "priority": 0.76,
                }
            )
    facet_terms: set[str] = set()
    facet_labels = []
    for facet in graph.get("required_facets", [])[:12]:
        terms = lexical_terms(facet)
        if terms:
            facet_terms.update(terms)
            facet_labels.append(normalize_text(facet))
    if facet_terms:
        obligations.append(
            {
                "id": "facet:question_specific_bundle",
                "type": "required_facet_bundle",
                "target": " ".join(facet_labels[:12]),
                "terms": facet_terms,
                "priority": 0.84,
            }
        )
    if obligations:
        unique = {row["id"]: row for row in obligations}
        return list(unique.values())

    question_terms = sorted(lexical_terms(question))[:10]
    if not question_terms:
        return []
    return [
        {
            "id": "question:lexical_core",
            "type": "question_core",
            "target": " ".join(question_terms),
            "terms": set(question_terms),
            "priority": 0.65,
        }
    ]


def _support_indexes(
    claim_graph: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    supports: dict[str, list[dict[str, Any]]] = {}
    contradicts: dict[str, list[dict[str, Any]]] = {}
    for edge in claim_graph.get("edges", []):
        evidence_id = edge.get("evidence_id")
        if not evidence_id:
            continue
        target = supports if edge.get("relation") == "supports" else contradicts
        target.setdefault(str(evidence_id), []).append(edge)
    return supports, contradicts


def _candidate_rows(
    evidence_summary: dict[str, Any],
    obligations: list[dict[str, Any]],
    claim_graph: dict[str, Any],
    excluded_evidence_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    support_edges, contradiction_edges = _support_indexes(claim_graph)
    excluded_evidence_ids = excluded_evidence_ids or set()
    rows = []
    for item in _evidence_items(evidence_summary):
        if item["evidence_id"] in excluded_evidence_ids:
            continue
        terms = set(item.get("terms", set()))
        obligation_hits = []
        for obligation in obligations:
            obligation_terms = set(obligation["terms"])
            overlap_terms = terms & obligation_terms
            if not overlap_terms:
                continue
            obligation_hits.append(
                {
                    "obligation_id": obligation["id"],
                    "overlap": round(
                        len(overlap_terms) / max(1, len(obligation_terms)),
                        3,
                    ),
                    "terms": sorted(overlap_terms)[:8],
                }
            )
        supports = support_edges.get(item["evidence_id"], [])
        contradicts = contradiction_edges.get(item["evidence_id"], [])
        max_support = max(
            (float(edge.get("score", 0.0) or 0.0) for edge in supports),
            default=0.0,
        )
        rows.append(
            {
                "evidence_id": item["evidence_id"],
                "item_id": item["item_id"],
                "kind": item["kind"],
                "label": item["label"],
                "source_ids": item["source_ids"],
                "repair_id": item.get("repair_id"),
                "estimated_tokens": item["estimated_tokens"],
                "obligation_hits": obligation_hits,
                "supported_claim_ids": sorted(
                    {
                        str(edge.get("claim_id"))
                        for edge in supports
                        if edge.get("claim_id")
                    }
                ),
                "contradicted_claim_ids": sorted(
                    {
                        str(edge.get("claim_id"))
                        for edge in contradicts
                        if edge.get("claim_id")
                    }
                ),
                "max_claim_support_score": round(max_support, 3),
                "text_preview": normalize_text(item.get("text", ""))[:220],
            }
        )
    return rows


def _coverage(
    selected: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    obligations = {
        hit["obligation_id"]
        for row in selected
        for hit in row.get("obligation_hits", [])
    }
    claims = {
        claim_id
        for row in selected
        for claim_id in row.get("supported_claim_ids", [])
    }
    sources = {
        source_id
        for row in selected
        for source_id in row.get("source_ids", [])
    }
    return obligations, claims, sources


def _greedy_select(
    candidates: list[dict[str, Any]],
    required_obligations: set[str],
    required_claims: set[str],
    max_evidence: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    used_tokens = 0
    covered_obligations: set[str] = set()
    covered_claims: set[str] = set()
    covered_sources: set[str] = set()

    while remaining and len(selected) < max_evidence:
        best = None
        best_utility = 0.0
        best_detail: dict[str, Any] = {}
        for row in remaining:
            token_cost = int(row.get("estimated_tokens", 0) or 0)
            if used_tokens + token_cost > token_budget:
                continue
            row_obligations = {
                hit["obligation_id"]
                for hit in row.get("obligation_hits", [])
            }
            row_claims = set(row.get("supported_claim_ids", []))
            new_obligations = row_obligations - covered_obligations
            new_claims = row_claims - covered_claims
            new_sources = set(row.get("source_ids", [])) - covered_sources
            contradiction_penalty = 0.18 * len(
                set(row.get("contradicted_claim_ids", []))
            )
            utility = (
                0.52 * len(new_claims) / max(1, len(required_claims))
                + 0.34
                * len(new_obligations)
                / max(1, len(required_obligations))
                + 0.08 * min(1.0, len(new_sources))
                + 0.10 * float(row.get("max_claim_support_score", 0.0) or 0.0)
                - 0.015 * token_cost / 100.0
                - contradiction_penalty
            )
            if utility > best_utility:
                best = row
                best_utility = utility
                best_detail = {
                    "new_obligations": sorted(new_obligations),
                    "new_claims": sorted(new_claims),
                    "new_sources": sorted(new_sources),
                }
        if best is None or best_utility <= 0.0:
            break
        remaining.remove(best)
        chosen = {**best}
        chosen["marginal_utility"] = round(best_utility, 4)
        chosen["new_obligations"] = best_detail["new_obligations"]
        chosen["new_claims"] = best_detail["new_claims"]
        chosen["new_sources"] = best_detail["new_sources"]
        selected.append(chosen)
        used_tokens += int(best.get("estimated_tokens", 0) or 0)
        covered_obligations.update(best_detail["new_obligations"])
        covered_claims.update(best_detail["new_claims"])
        covered_sources.update(best_detail["new_sources"])
        if (
            covered_claims >= required_claims
            and covered_obligations >= required_obligations
        ):
            break
    return selected


def _prune_redundant(
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    pruned = list(selected)
    removed = 0
    changed = True
    while changed:
        changed = False
        full_obligations, full_claims, _sources = _coverage(pruned)
        for row in list(pruned):
            trial = [candidate for candidate in pruned if candidate is not row]
            trial_obligations, trial_claims, _trial_sources = _coverage(trial)
            if (
                trial_obligations >= full_obligations
                and trial_claims >= full_claims
            ):
                pruned.remove(row)
                removed += 1
                changed = True
                break
    return pruned, removed


def _minimality_certificate(
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    full_obligations, full_claims, _sources = _coverage(selected)
    necessary = []
    for row in selected:
        trial = [candidate for candidate in selected if candidate is not row]
        trial_obligations, trial_claims, _trial_sources = _coverage(trial)
        lost_obligations = sorted(full_obligations - trial_obligations)
        lost_claims = sorted(full_claims - trial_claims)
        necessary.append(
            {
                "evidence_id": row["evidence_id"],
                "necessary": bool(lost_obligations or lost_claims),
                "lost_obligations_if_removed": lost_obligations,
                "lost_claims_if_removed": lost_claims,
            }
        )
    return {
        "removal_tested_count": len(selected),
        "all_selected_necessary": all(row["necessary"] for row in necessary),
        "necessary_evidence_ids": [
            row["evidence_id"] for row in necessary if row["necessary"]
        ],
        "leave_one_out": necessary,
    }


def build_minimal_proof_subgraph(
    question: str,
    query_graph: dict[str, Any] | None,
    answer: str,
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    evidence_audit: dict[str, Any] | None = None,
    max_evidence: int = 6,
    token_budget: int = 720,
    excluded_evidence_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build the v78 joint obligation/claim proof subgraph."""
    obligations = _question_obligations(question, query_graph)
    candidates = _candidate_rows(
        evidence_summary,
        obligations,
        claim_evidence_graph,
        excluded_evidence_ids=excluded_evidence_ids,
    )
    required_obligations = {row["id"] for row in obligations}
    required_claims = {
        str(row.get("claim_id"))
        for row in claim_evidence_graph.get("claims", [])
        if row.get("claim_id")
    }
    selected = _greedy_select(
        candidates,
        required_obligations,
        required_claims,
        max_evidence=max_evidence,
        token_budget=token_budget,
    )
    selected, redundant_count = _prune_redundant(selected)
    covered_obligations, covered_claims, covered_sources = _coverage(selected)
    estimated_tokens = sum(
        int(row.get("estimated_tokens", 0) or 0) for row in selected
    )
    obligation_coverage = len(covered_obligations) / max(
        1,
        len(required_obligations),
    )
    claim_coverage = len(covered_claims) / max(1, len(required_claims))
    contradiction_claims = sorted(
        {
            claim_id
            for row in selected
            for claim_id in row.get("contradicted_claim_ids", [])
        }
    )
    audit = evidence_audit or {}
    unsupported = int(audit.get("unsupported_claim_count", 0) or 0)
    proof_sufficient = bool(
        claim_coverage >= 1.0
        and (obligation_coverage >= 0.66 or not required_obligations)
        and unsupported == 0
        and not contradiction_claims
    )
    joint_objective = (
        0.52 * claim_coverage
        + 0.34 * obligation_coverage
        + 0.08 * min(1.0, len(covered_sources) / max(1, len(selected)))
        + 0.06 * (1.0 if proof_sufficient else 0.0)
    )
    return {
        "enabled": True,
        "controller": "minimal_proof_subgraph_v1",
        "question": normalize_text(question),
        "answer_digest": _stable_id("answer", answer),
        "candidate_count": len(candidates),
        "excluded_evidence_ids": sorted(excluded_evidence_ids or []),
        "selected_count": len(selected),
        "max_evidence": max_evidence,
        "token_budget": token_budget,
        "estimated_tokens": estimated_tokens,
        "covered_obligations": sorted(covered_obligations),
        "required_obligation_count": len(required_obligations),
        "obligation_coverage_ratio": round(obligation_coverage, 3),
        "covered_claims": sorted(covered_claims),
        "required_claim_count": len(required_claims),
        "claim_support_coverage_ratio": round(claim_coverage, 3),
        "unique_source_count": len(covered_sources),
        "contradicted_claims_in_proof": contradiction_claims,
        "joint_objective": round(joint_objective, 4),
        "proof_sufficient": proof_sufficient,
        "submission_guidance": (
            "proof_sufficient"
            if proof_sufficient
            else "proof_incomplete_monitor"
        ),
        "redundant_evidence_pruned_count": redundant_count,
        "selected_evidence": [
            {
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
            for row in selected
        ],
        "minimality_certificate": _minimality_certificate(selected),
    }
