"""VSR-PathRAG v82 source-independent proof quorum certification.

The v78 proof optimizer can find a compact proof, but compact evidence may
still be over-correlated: several chunks can come from the same document,
same derived source, or the same provenance family.  V82 certifies that the
selected proof has independent source support, or marks the answer for
monitoring when the proof depends on a single source family.
"""

from __future__ import annotations

import re
from typing import Any

from api.features.chats.vsr_pathrag import normalize_text, split_source_ids


LINEAGE_KEYS = (
    "root_document_id",
    "document_id",
    "full_doc_id",
    "parent_document_id",
    "ancestor_ids",
    "provenance_chain",
    "lineage_ids",
    "correlation_ids",
    "source_family_ids",
    "family_ids",
    "derived_from_ids",
    "parent_source_ids",
)


def _identity_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = re.split(r"[|,;]", str(value))
    values = []
    for candidate in candidates:
        cleaned = normalize_text(candidate)
        if cleaned and cleaned != "UNKNOWN" and cleaned not in values:
            values.append(cleaned)
    return values


def _source_family_from_id(source_id: str) -> str:
    source_id = normalize_text(source_id)
    if not source_id:
        return ""
    if "#" in source_id:
        return source_id.split("#", 1)[0]
    match = re.match(r"^(.*?)(?:[:/_-]?(?:chunk|part|section)[-_:]?\d+)$", source_id)
    if match and match.group(1):
        return match.group(1).rstrip(":/_-")
    return source_id


def _lineage_for_row(row: dict[str, Any], fallback_ids: list[str]) -> dict[str, list[str]]:
    origin_ids = _identity_values(
        row.get("origin_ids")
        or row.get("source_ids")
        or row.get("source_id")
        or fallback_ids
    )
    ancestor_ids = []
    correlation_ids = []
    for key in (
        "root_document_id",
        "document_id",
        "full_doc_id",
        "parent_document_id",
        "ancestor_ids",
        "provenance_chain",
        "lineage_ids",
    ):
        for identity in _identity_values(row.get(key)):
            if identity not in ancestor_ids:
                ancestor_ids.append(identity)
    for key in (
        "correlation_ids",
        "source_family_ids",
        "family_ids",
        "derived_from_ids",
        "parent_source_ids",
    ):
        for identity in _identity_values(row.get(key)):
            if identity not in correlation_ids:
                correlation_ids.append(identity)
    family_ids = list(dict.fromkeys(ancestor_ids + correlation_ids))
    if not family_ids:
        family_ids = [_source_family_from_id(source_id) for source_id in origin_ids]
        family_ids = [source_id for source_id in dict.fromkeys(family_ids) if source_id]
    if not family_ids:
        family_ids = [source_id for source_id in origin_ids if source_id]
    return {
        "origin_ids": origin_ids,
        "ancestor_ids": ancestor_ids,
        "correlation_ids": correlation_ids,
        "family_ids": family_ids,
    }


def _lineage_index(evidence_summary: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    index: dict[str, dict[str, list[str]]] = {}

    def add(row: dict[str, Any], fallback_ids: list[str]) -> None:
        lineage = _lineage_for_row(row, fallback_ids)
        keys = set(fallback_ids)
        keys.update(lineage["origin_ids"])
        label = normalize_text(
            row.get("source_id")
            or row.get("id")
            or row.get("label")
            or row.get("name")
            or ""
        )
        if label:
            keys.add(label)
        for key in keys:
            if key:
                index[key] = lineage

    for entity in evidence_summary.get("entities", []):
        add(
            entity,
            split_source_ids(entity.get("source_ids") or entity.get("source_id")),
        )
    for relation in evidence_summary.get("relationships", []):
        add(
            relation,
            split_source_ids(
                relation.get("source_ids") or relation.get("source_id")
            ),
        )
    for excerpt in evidence_summary.get("source_excerpts", []):
        source_id = normalize_text(excerpt.get("source_id", ""))
        if source_id:
            add(excerpt, [source_id])
    return index


def _families_for_evidence(
    evidence: dict[str, Any],
    source_index: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    source_ids = split_source_ids(evidence.get("source_ids"))
    lookup_keys = list(source_ids)
    for key in (
        normalize_text(evidence.get("label", "")),
        normalize_text(evidence.get("item_id", "")),
        normalize_text(evidence.get("evidence_id", "")),
    ):
        if key:
            lookup_keys.append(key)
    lineages = []
    for key in lookup_keys:
        if key in source_index:
            lineages.append(source_index[key])
    if not lineages:
        lineages.append(_lineage_for_row(evidence, source_ids or lookup_keys))

    family_ids = []
    origin_ids = []
    ancestor_ids = []
    correlation_ids = []
    for lineage in lineages:
        for target, values in (
            (family_ids, lineage.get("family_ids", [])),
            (origin_ids, lineage.get("origin_ids", [])),
            (ancestor_ids, lineage.get("ancestor_ids", [])),
            (correlation_ids, lineage.get("correlation_ids", [])),
        ):
            for value in values:
                if value and value not in target:
                    target.append(value)
    if not family_ids:
        family_ids = [_source_family_from_id(source_id) for source_id in source_ids]
        family_ids = [source_id for source_id in dict.fromkeys(family_ids) if source_id]
    if not family_ids and evidence.get("label"):
        family_ids = [normalize_text(evidence.get("label"))]
    return {
        "evidence_id": normalize_text(evidence.get("evidence_id", "")),
        "label": normalize_text(evidence.get("label", "")),
        "source_ids": source_ids,
        "origin_ids": origin_ids or source_ids,
        "ancestor_ids": ancestor_ids,
        "correlation_ids": correlation_ids,
        "family_ids": family_ids,
    }


def _family_counts_by_target(
    selected: list[dict[str, Any]],
    target_key: str,
) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for row in selected:
        families = row.get("family_ids", [])
        for target in row.get(target_key, []):
            if isinstance(target, dict):
                target_id = normalize_text(target.get("obligation_id", ""))
            else:
                target_id = normalize_text(target)
            if not target_id:
                continue
            record = counts.setdefault(
                target_id,
                {"family_ids": [], "evidence_ids": []},
            )
            for family_id in families:
                if family_id not in record["family_ids"]:
                    record["family_ids"].append(family_id)
            evidence_id = row.get("evidence_id")
            if evidence_id and evidence_id not in record["evidence_ids"]:
                record["evidence_ids"].append(evidence_id)
    for record in counts.values():
        record["independent_family_count"] = len(record["family_ids"])
    return counts


def _correlated_groups(
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in selected:
        for family_id in row.get("family_ids", []):
            group = groups.setdefault(
                family_id,
                {
                    "family_id": family_id,
                    "evidence_ids": [],
                    "labels": [],
                    "source_ids": [],
                },
            )
            for key, values in (
                ("evidence_ids", [row.get("evidence_id", "")]),
                ("labels", [row.get("label", "")]),
                ("source_ids", row.get("source_ids", [])),
            ):
                for value in values:
                    if value and value not in group[key]:
                        group[key].append(value)
    return sorted(
        groups.values(),
        key=lambda row: (-len(row["evidence_ids"]), row["family_id"]),
    )


def certify_proof_source_quorum(
    minimal_proof_subgraph: dict[str, Any],
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Certify independent provenance for the selected proof evidence."""
    selected_input = minimal_proof_subgraph.get("selected_evidence", []) or []
    source_index = _lineage_index(evidence_summary or {})
    selected = []
    for evidence in selected_input:
        lineage = _families_for_evidence(evidence, source_index)
        selected.append({**evidence, **lineage})

    claim_counts = _family_counts_by_target(selected, "supported_claim_ids")
    obligation_counts = _family_counts_by_target(selected, "obligation_hits")
    required_claims = {
        normalize_text(claim_id)
        for claim_id in minimal_proof_subgraph.get("covered_claims", [])
    }
    if not required_claims and claim_evidence_graph:
        required_claims = {
            normalize_text(row.get("claim_id", ""))
            for row in claim_evidence_graph.get("claims", [])
            if row.get("status") == "supported"
        }
    required_claims = {claim_id for claim_id in required_claims if claim_id}
    required_obligations = {
        normalize_text(obligation_id)
        for obligation_id in minimal_proof_subgraph.get("covered_obligations", [])
        if obligation_id
    }

    single_family_claims = sorted(
        claim_id
        for claim_id in required_claims
        if claim_counts.get(claim_id, {}).get("independent_family_count", 0) < 2
    )
    single_family_obligations = sorted(
        obligation_id
        for obligation_id in required_obligations
        if obligation_counts.get(obligation_id, {}).get(
            "independent_family_count",
            0,
        )
        < 2
    )
    all_family_ids = sorted(
        {
            family_id
            for row in selected
            for family_id in row.get("family_ids", [])
            if family_id
        }
    )
    selected_count = len(selected)
    proof_sufficient = bool(minimal_proof_subgraph.get("proof_sufficient"))
    contradiction_count = len(
        minimal_proof_subgraph.get("contradicted_claims_in_proof", []) or []
    )
    missing_claims = sorted(
        claim_id for claim_id in required_claims if claim_id not in claim_counts
    )
    missing_obligations = sorted(
        obligation_id
        for obligation_id in required_obligations
        if obligation_id not in obligation_counts
    )

    if not proof_sufficient or not selected:
        quorum_status = "quorum_failed"
    elif (
        len(all_family_ids) >= 2
        and not single_family_claims
        and not single_family_obligations
        and not missing_claims
        and not missing_obligations
        and contradiction_count == 0
    ):
        quorum_status = "independent_quorum"
    elif len(all_family_ids) <= 1 or selected_count <= 1:
        quorum_status = "single_source_critical"
    else:
        quorum_status = "correlated_monitor"

    monitoring_required = quorum_status in {
        "single_source_critical",
        "correlated_monitor",
        "quorum_failed",
    }
    if quorum_status == "independent_quorum":
        submission_guidance = "source_quorum_certified"
    elif quorum_status == "single_source_critical":
        submission_guidance = "single_source_monitor"
    elif quorum_status == "correlated_monitor":
        submission_guidance = "correlated_source_monitor"
    else:
        submission_guidance = "source_quorum_failed"

    return {
        "enabled": True,
        "controller": "source_independent_proof_quorum_v1",
        "proof_sufficient": proof_sufficient,
        "selected_evidence_count": selected_count,
        "independent_family_count": len(all_family_ids),
        "quorum_status": quorum_status,
        "quorum_passed": quorum_status in {
            "independent_quorum",
            "single_source_critical",
        },
        "monitoring_required": monitoring_required,
        "submission_guidance": submission_guidance,
        "single_family_claims": single_family_claims,
        "single_family_obligations": single_family_obligations,
        "missing_claims": missing_claims,
        "missing_obligations": missing_obligations,
        "claim_family_counts": claim_counts,
        "obligation_family_counts": obligation_counts,
        "correlated_evidence_groups": _correlated_groups(selected),
        "selected_evidence_lineage": [
            {
                "evidence_id": row.get("evidence_id", ""),
                "label": row.get("label", ""),
                "source_ids": row.get("source_ids", []),
                "family_ids": row.get("family_ids", []),
                "ancestor_ids": row.get("ancestor_ids", []),
                "correlation_ids": row.get("correlation_ids", []),
            }
            for row in selected
        ],
    }
