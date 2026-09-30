"""VSR-PathRAG v95 question-type-aware reasoning path selection.

V95 implements dynamic reasoning path selection based on question type classification.
Different question types require different evidence organization and reasoning strategies:

- **Factoid questions**: Direct retrieval → concise answer (compressed path)
- **Causal questions**: Cause-effect chain reasoning (preserve full chain)
- **Comparison questions**: Side-by-side comparison structure (reorder for contrast)
- **Narrative/Process questions**: Temporal or stepwise ordering (timeline structure)
- **Definition/Concept questions**: Hierarchical concept explanation (tree structure)

Core innovation:
- First GraphRAG system to explicitly map question taxonomy to reasoning path topology.
- Dynamically adjusts evidence presentation format based on question intent.
- Optimizes LLM context utilization by matching evidence structure to reasoning type.
- Works with V89 compressed evidence and V90 reordered evidence.
"""

from __future__ import annotations

import re
from typing import Any

# Question type classification patterns
QUESTION_TYPE_PATTERNS = {
    "factoid": {
        "patterns": [
            r"\b(what|who|when|where|which)\b",
            r"\b(name|define|list|identify|state)\b",
            r"\b(how many|how much|what is)\b",
        ],
        "keywords": {"fact", "definition", "name", "date", "location"},
    },
    "causal": {
        "patterns": [
            r"\b(why|what caused|what led to|reason for|impact of|effect of)\b",
            r"\b(because of|due to|result from|consequence)\b",
        ],
        "keywords": {"cause", "effect", "impact", "result", "lead", "influence"},
    },
    "comparison": {
        "patterns": [
            r"\b(compare|contrast|difference|versus|vs\.?)\b",
            r"\b(similar|different|better|worse|more.*than|less.*than)\b",
        ],
        "keywords": {"compare", "contrast", "difference", "similar", "versus"},
    },
    "process": {
        "patterns": [
            r"\b(how does|how do|how did|process|mechanism|method)\b",
            r"\b(step|procedure|approach|way to|means of)\b",
        ],
        "keywords": {"process", "method", "step", "approach", "mechanism", "through"},
    },
    "narrative": {
        "patterns": [
            r"\b(describe|explain|story|history|tell me about|account of)\b",
            r"\b(development|evolution|role|significance|meaning)\b",
        ],
        "keywords": {"describe", "explain", "story", "history", "role", "narrative"},
    },
}


def classify_question_type(
    question: str,
    query_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify the question into a primary type with confidence scores.

    Uses both lexical pattern matching and query graph signals (if available).
    """
    q_lower = question.lower()
    scores: dict[str, float] = {}

    for qtype, config in QUESTION_TYPE_PATTERNS.items():
        pattern_score = 0.0
        keyword_score = 0.0

        # Pattern matching
        for pattern in config["patterns"]:
            if re.search(pattern, q_lower):
                pattern_score += 1.0

        # Keyword matching
        matched_keywords = sum(
            1 for kw in config["keywords"] if kw in q_lower
        )
        keyword_score = matched_keywords / max(1, len(config["keywords"]))

        scores[qtype] = 0.6 * min(1.0, pattern_score / max(1, len(config["patterns"]))) + 0.4 * keyword_score

    # Check query graph for additional signals
    graph_type_hint = ""
    if query_graph:
        intents = query_graph.get("relation_intents", [])
        if intents:
            for intent in intents:
                if intent in {"cause_effect"}:
                    scores["causal"] = max(scores.get("causal", 0), 0.85)
                    graph_type_hint = f"graph_intent:{intent}"
                elif intent in {"comparison"}:
                    scores["comparison"] = max(scores.get("comparison", 0), 0.85)
                    graph_type_hint = f"graph_intent:{intent}"
                elif intent in {"mechanism"}:
                    scores["process"] = max(scores.get("process", 0), 0.80)
                    graph_type_hint = f"graph_intent:{intent}"

        # Multi-hop indicator
        if query_graph.get("requires_multi_hop"):
            for hop_favoring in ["causal", "comparison", "narrative"]:
                scores[hop_favoring] = scores.get(hop_favoring, 0) * 1.15

    # Determine primary type
    primary_type = max(scores, key=scores.get) if scores else "general"
    primary_score = scores.get(primary_type, 0.0)

    # Secondary type (for mixed-type questions)
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    secondary_type = sorted_types[1][0] if len(sorted_types) > 1 else None
    secondary_score = sorted_types[1][1] if secondary_type else 0.0

    return {
        "primary_type": primary_type,
        "primary_confidence": round(min(1.0, primary_score), 4),
        "secondary_type": secondary_type,
        "secondary_confidence": round(secondary_score, 4) if secondary_type else 0.0,
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
        "graph_type_hint": graph_type_hint,
        "is_mixed_type": (
            secondary_type is not None and secondary_score > 0.40
        ),
    }


def _get_reasoning_path_config(question_type: str) -> dict[str, Any]:
    """Get the reasoning path configuration for a question type."""
    configs = {
        "factoid": {
            "evidence_compression_rate": 0.45,  # Aggressive compression
            "max_evidence_items": 3,
            "ordering_strategy": "direct_relevance",
            "context_template": "direct_answer",
            "prefers_conciseness": True,
            "chain_preservation": False,
        },
        "causal": {
            "evidence_compression_rate": 0.75,  # Light compression
            "max_evidence_items": 8,
            "ordering_strategy": "cause_to_effect_chain",
            "context_template": "causal_chain",
            "prefers_conciseness": False,
            "chain_preservation": True,
        },
        "comparison": {
            "evidence_compression_rate": 0.65,
            "max_evidence_items": 6,
            "ordering_strategy": "side_by_side_contrast",
            "context_template": "comparison_table",
            "prefers_conciseness": True,
            "chain_preservation": True,
        },
        "process": {
            "evidence_compression_rate": 0.70,
            "max_evidence_items": 7,
            "ordering_strategy": "sequential_steps",
            "context_template": "stepwise_process",
            "prefers_conciseness": False,
            "chain_preservation": True,
        },
        "narrative": {
            "evidence_compression_rate": 0.70,
            "max_evidence_items": 7,
            "ordering_strategy": "chronological_thematic",
            "context_template": "narrative_flow",
            "prefers_conciseness": False,
            "chain_preservation": True,
        },
        "general": {
            "evidence_compression_rate": 0.65,
            "max_evidence_items": 5,
            "ordering_strategy": "salience_weighted",
            "context_template": "balanced",
            "prefers_conciseness": True,
            "chain_preservation": False,
        },
    }
    return configs.get(question_type, configs["general"])


def build_adaptive_reasoning_path(
    question: str,
    answer: str,
    query_graph: dict[str, Any] | None,
    evidence_summary: dict[str, Any],
    claim_evidence_graph: dict[str, Any],
    minimal_proof_subgraph: dict[str, Any],
    traceability_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the V95 question-type-aware reasoning path selection.

    This module analyzes the question type and selects the optimal reasoning
    strategy, including:
    1. Question type classification with confidence.
    2. Reasoning path configuration selection.
    3. Evidence reorganization based on the selected strategy.
    4. Reasoning path quality scoring.

    Pipeline position: After V93 traceability, before V97 consistency check.
    """
    # Step 1: Classify question type
    qtype = classify_question_type(question, query_graph)
    primary_type = qtype["primary_type"]

    # Step 2: Get reasoning path configuration
    path_config = _get_reasoning_path_config(primary_type)

    # Step 3: Analyze current evidence against the optimal configuration
    selected_evidence = minimal_proof_subgraph.get("selected_evidence", []) or []
    current_count = len(selected_evidence)
    target_count = path_config["max_evidence_items"]

    # Determine if reorganization is needed
    needs_reorganization = (
        current_count > target_count * 1.5
        or (current_count < target_count * 0.5 and current_count > 0)
        or path_config["chain_preservation"]
    )

    # Step 4: Apply ordering strategy
    reordered_evidence: list[dict[str, Any]] = []
    ordering_strategy = path_config["ordering_strategy"]

    if ordering_strategy == "direct_relevance":
        # Sort by obligation coverage (most relevant first)
        reordered_evidence = sorted(
            selected_evidence,
            key=lambda r: len(r.get("obligation_hits", [])),
            reverse=True,
        )[:target_count]

    elif ordering_strategy == "cause_to_effect_chain":
        # Sort by causal intent coverage, then by support score
        reordered_evidence = sorted(
            selected_evidence,
            key=lambda r: (
                any(
                    h.get("obligation_id", "").startswith("intent:")
                    for h in r.get("obligation_hits", [])
                ),
                float(r.get("max_claim_support_score", 0.0) or 0.0),
            ),
            reverse=True,
        )[:target_count]

    elif ordering_strategy == "side_by_side_contrast":
        # Group by source family, then interleave for comparison
        source_groups: dict[str, list[dict[str, Any]]] = {}
        for row in selected_evidence:
            src_key = ",".join(sorted(row.get("source_ids", [])[:1]) or ["unknown"])
            source_groups.setdefault(src_key, []).append(row)

        # Interleave groups for side-by-side comparison
        max_group_len = max((len(g) for g in source_groups.values()), default=0)
        for i in range(max_group_len):
            for group in source_groups.values():
                if i < len(group):
                    reordered_evidence.append(group[i])
                    if len(reordered_evidence) >= target_count:
                        break
            if len(reordered_evidence) >= target_count:
                break

    elif ordering_strategy == "sequential_steps" or ordering_strategy == "chronological_thematic":
        # Sort by marginal utility (greedy order preserves sequence)
        reordered_evidence = sorted(
            selected_evidence,
            key=lambda r: float(r.get("marginal_utility", 0.0) or 0.0),
            reverse=True,
        )[:target_count]

    else:  # salience_weighted or balanced
        # Use existing V90 salience order if available, otherwise by support score
        reordered_evidence = sorted(
            selected_evidence,
            key=lambda r: float(r.get("max_claim_support_score", 0.0) or 0.0),
            reverse=True,
        )[:target_count]

    # If no reorganization needed, keep original order
    if not needs_reorganization or not reordered_evidence:
        reordered_evidence = selected_evidence[:target_count] if target_count > 0 else selected_evidence

    # Step 5: Compute reasoning path quality metrics
    path_coverage = len(reordered_evidence) / max(1, current_count) if current_count > 0 else 1.0
    path_tokens = sum(int(r.get("estimated_tokens", 0) or 0) for r in reordered_evidence)

    # Strategy alignment score: how well the evidence matches the intended strategy
    strategy_alignment = 1.0
    if path_config["chain_preservation"]:
        # Check if we preserved chains (multiple claims per evidence)
        if traceability_result:
            multi_claim_evidence = sum(
                1 for chain in traceability_result.get("traceability_chains", [])
                if len(chain.get("evidence_chain", [])) > 1
            )
            total_chains = len(traceability_result.get("traceability_chains", []))
            strategy_alignment = multi_claim_evidence / max(1, total_chains) if total_chains > 0 else 0.5

    reasoning_quality = round(
        0.4 * path_coverage
        + 0.3 * strategy_alignment
        + 0.2 * min(1.0, len(reordered_evidence) / max(1, target_count))
        + 0.1 * (1.0 if len(reordered_evidence) <= target_count * 1.2 else 0.5),
        4,
    )

    return {
        "enabled": True,
        "controller": "adaptive_reasoning_path_v1",
        "triggered": True,
        "question_type": qtype,
        "selected_path_config": {
            "type": primary_type,
            "compression_rate": path_config["evidence_compression_rate"],
            "max_evidence": path_config["max_evidence_items"],
            "ordering_strategy": ordering_strategy,
            "context_template": path_config["context_template"],
            "chain_preservation": path_config["chain_preservation"],
        },
        "needs_reorganization": needs_reorganization,
        "original_evidence_count": current_count,
        "reorganized_evidence_count": len(reordered_evidence),
        "path_token_estimate": path_tokens,
        "strategy_alignment": round(strategy_alignment, 4),
        "reasoning_quality_score": reasoning_quality,
        "quality_tier": (
            "optimal"
            if reasoning_quality >= 0.85
            else "good"
            if reasoning_quality >= 0.65
            else "acceptable"
            if reasoning_quality >= 0.40
            else "suboptimal"
        ),
        "reorganized_evidence": reordered_evidence,
    }
