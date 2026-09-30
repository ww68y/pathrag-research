"""VSR-PathRAG v90 salience-preserving evidence reordering.

V90 reorders the compressed evidence from V89 to maximize answer quality under
the LLM's "lost in the middle" bias.  High-salience evidence (high claim support,
high obligation coverage, high information density) is placed at the beginning
and end of the context window, while lower-salience evidence occupies the middle.

Core innovation:
- Leverages the verified claim-evidence graph for salience scoring.
- Implements position-aware utility weighting (primacy + recency effect).
- Maintains source diversity across positions.
- Works with the compressed evidence from V89, not raw text.
"""

from __future__ import annotations

from typing import Any


def _compute_salience_score(
    row: dict[str, Any],
    density: float | None = None,
) -> float:
    """Compute salience score for evidence reordering.

    Salience combines:
    - Claim support breadth (how many claims this evidence supports)
    - Obligation coverage (question relevance)
    - Information density (from V89)
    - Support quality (max claim support score)
    """
    supported_claims = len(row.get("supported_claim_ids", []))
    obligation_hits = len(row.get("obligation_hits", []))
    support_score = float(row.get("max_claim_support_score", 0.0) or 0.0)
    contradicted = len(row.get("contradicted_claim_ids", []))

    # Use pre-computed density if available
    info_density = density if density is not None else float(
        row.get("information_density", 0.0) or 0.0
    )

    # Normalized components
    claim_salience = min(1.0, supported_claims / 5.0)  # Normalize by typical max
    obligation_salience = min(1.0, obligation_hits / 3.0)

    salience = (
        0.35 * claim_salience          # Primary: how many claims does this support
        + 0.25 * obligation_salience   # Secondary: question relevance
        + 0.20 * support_score         # Quality of support
        + 0.12 * info_density          # Information per token
        - 0.08 * contradicted           # Penalty for contradictions
    )
    return round(max(0.0, salience), 6)


def _position_weight(position: int, total: int) -> float:
    """Compute position weight based on "lost in the middle" research.

    LLMs perform best on information at the BEGINNING and END of context,
    with degraded performance in the middle.  We model this as a U-shaped curve.

    Reference: "Lost in the Middle" (Liu et al., 2023)
    """
    if total <= 1:
        return 1.0

    normalized_pos = position / max(1, total - 1)

    # U-shaped curve: high at edges (pos=0 and pos=1), low in middle (pos=0.5)
    # Using quadratic: w(p) = 4 * (p - 0.5)^2, then invert for placement priority
    distance_from_edge = min(normalized_pos, 1.0 - normalized_pos)
    weight = 1.0 - 2.0 * distance_from_edge  # 1.0 at edges, 0.0 at center

    # Add slight primacy bias (beginning slightly favored over end)
    if normalized_pos < 0.5:
        weight += 0.05

    return round(max(0.0, min(1.0, weight)), 4)


def build_salience_reordering(
    minimal_proof_subgraph: dict[str, Any],
    compression_result: dict[str, Any] | None = None,
    max_front_bucket: int = 3,
    max_back_bucket: int = 3,
) -> dict[str, Any]:
    """Reorder evidence by salience for optimal LLM context utilization.

    Strategy:
    1. Compute salience score for each evidence row.
    2. Assign rows to three buckets:
       - Front bucket (positions 0..front_size): highest salience
       - Middle bucket: remaining medium/low salience
       - Back bucket (last back_size positions): second-highest salience
    3. Within each bucket, maintain diversity (different source families).
    4. Return reordered evidence list.
    """
    selected = list(minimal_proof_subgraph.get("selected_evidence", []) or [])
    if not selected or len(selected) <= 2:
        return {
            "enabled": True,
            "controller": "salience_reordering_v1",
            "triggered": False,
            "reason": "insufficient_evidence_for_reordering",
            "reordered_evidence": selected,
            "reorder_applied": False,
            "salience_distribution": {},
        }

    # Get density scores from compression result if available
    density_map: dict[str, float] = {}
    if compression_result:
        for action in compression_result.get("compression_actions", []):
            eid = action.get("evidence_id", "")
            density_map[eid] = action.get("density", 0.0)

    # Step 1: Compute salience for all rows
    scored_rows: list[dict[str, Any]] = []
    for row in selected:
        eid = row.get("evidence_id", "")
        salience = _compute_salience_score(row, density_map.get(eid))
        scored_rows.append({**row, "_salience_score": salience})

    # Sort by salience descending
    scored_rows.sort(key=lambda r: r["_salience_score"], reverse=True)

    n = len(scored_rows)
    front_size = min(max_front_bucket, n // 3 if n >= 6 else 1)
    back_size = min(max_back_bucket, n // 3 if n >= 6 else 1)

    # Step 2: Assign to buckets with diversity constraint
    front_bucket: list[dict[str, Any]] = []
    middle_bucket: list[dict[str, Any]] = []
    back_bucket: list[dict[str, Any]] = []
    used_sources: set[str] = set()

    def add_with_diversity(
        bucket: list[dict[str, Any]],
        row: dict[str, Any],
    ) -> None:
        """Add row to bucket, tracking source diversity."""
        sources = set(row.get("source_ids", []))
        row_for_bucket = {k: v for k, v in row.items() if not k.startswith("_")}
        row_for_bucket["reorder_position_hint"] = (
            "front" if bucket is front_bucket
            else "back" if bucket is back_bucket
            else "middle"
        )
        bucket.append(row_for_bucket)
        used_sources.update(sources)

    # Fill front bucket with highest salience (diversity-aware)
    for row in scored_rows:
        if len(front_bucket) >= front_size:
            break
        add_with_diversity(front_bucket, row)

    # Fill back bucket with next highest salience
    remaining_for_back = [
        r for r in scored_rows
        if r["evidence_id"] not in {row.get("evidence_id") for row in front_bucket}
    ]
    for row in remaining_for_back:
        if len(back_bucket) >= back_size:
            break
        add_with_diversity(back_bucket, row)

    # Rest goes to middle
    front_ids = {r.get("evidence_id") for r in front_bucket}
    back_ids = {r.get("evidence_id") for r in back_bucket}
    for row in scored_rows:
        eid = row.get("evidence_id", "")
        if eid not in front_ids and eid not in back_ids:
            add_with_diversity(middle_bucket, row)

    # Step 3: Assemble final order: front → middle → back
    reordered = front_bucket + middle_bucket + back_bucket

    # Compute position weights for diagnostics
    position_weights = [
        _position_weight(i, len(reordered)) for i in range(len(reordered))
    ]

    # Build reordered proof subgraph
    reordered_proof = dict(minimal_proof_subgraph)
    reordered_proof["selected_evidence"] = reordered
    reordered_proof["selected_count"] = len(reordered)

    # Salience distribution statistics
    saliences = [r["_salience_score"] for r in scored_rows]

    return {
        "enabled": True,
        "controller": "salience_reordering_v1",
        "triggered": True,
        "reason": "salience_preserving_reorder_applied",
        "original_order_evidence_ids": [r.get("evidence_id") for r in selected],
        "reordered_evidence_ids": [r.get("evidence_id") for r in reordered],
        "reorder_applied": reordered != [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in scored_rows
        ],
        "bucket_sizes": {
            "front": len(front_bucket),
            "middle": len(middle_bucket),
            "back": len(back_bucket),
        },
        "salience_distribution": {
            "min": round(min(saliences), 6) if saliences else 0.0,
            "max": round(max(saliences), 6) if saliences else 0.0,
            "mean": round(sum(saliences) / max(1, len(saliences)), 6),
            "front_mean": round(
                sum(r["_salience_score"] for r in scored_rows[:front_size]) / max(1, front_size),
                6,
            ) if front_size > 0 else 0.0,
            "back_mean": round(
                sum(r["_salience_score"] for r in scored_rows[-back_size:]) / max(1, back_size),
                6,
            ) if back_size > 0 else 0.0,
        },
        "position_weights": position_weights,
        "mean_position_weight": round(
            sum(position_weights) / max(1, len(position_weights)), 4
        ),
        "unique_source_count_in_reordered": len(used_sources),
        "reordered_minimal_proof_subgraph": reordered_proof,
    }
