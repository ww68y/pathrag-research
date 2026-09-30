#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score a frozen impact-aware escalation policy on new dry-completion rows."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from analyze_impact_aware_escalation import (
    CATEGORICAL_FEATURES as IMPACT_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES as IMPACT_NUMERIC_FEATURES,
)
from build_answer_contract_agent import clean_text, load_jsonl
from build_claim_evidence_agent import content_tokens


ROOT = Path(__file__).resolve().parents[1]
IMPACT_MODEL = ROOT / "results" / "evidence_contract_depth" / "impact_aware_escalation_policy_v1.joblib"
IMPACT_META = ROOT / "results" / "evidence_contract_depth" / "impact_aware_escalation_policy_v1.json"
RISK_MODEL = ROOT / "results" / "models" / "sage_v31_contract_gap_router.joblib"

RISK_NUMERIC_FEATURES = [
    "question_chars", "answer_chars", "answer_claims", "mean_support",
    "unsupported_ratio", "supported_ratio", "partial_ratio", "minimum_points",
    "required_facets", "top_text_score", "top_graph_score", "question_answer_overlap",
    "answer_question_length_ratio", "uncertainty_flag", "low_support_flag",
    "overlong_flag", "shape_flag",
]
RISK_CATEGORICAL_FEATURES = ["source_dataset", "task_dataset", "contract_type"]


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def numeric_mentions(text: str) -> int:
    return len(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))


def marker_count(text: str, markers: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(lowered.count(marker) for marker in markers)


def top_score(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return as_float(items[0].get("score"))


def build_feature_rows(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    risk_rows: list[dict[str, Any]] = []
    impact_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        answer = clean_text(row.get("prediction", ""))
        question = clean_text(row.get("question") or row.get("input") or "")
        completion = row.get("contract_completion_agent") or {}
        contract_agent = row.get("answer_contract_agent") or row.get("text_first_contract_agent") or {}
        contract = completion.get("contract") or contract_agent.get("contract") or {}
        signals = contract_agent.get("signals") or {}
        metrics = completion.get("pre_claim_metrics") or {}
        reasons = set(completion.get("risk_reasons") or [])
        text_hits = completion.get("question_text_evidence") or []
        graph_hits = completion.get("question_graph_evidence") or []
        claims = as_float(metrics.get("claims"), as_float(signals.get("estimated_distinct_points")))
        supported = as_float(metrics.get("supported_claims"))
        partial = as_float(metrics.get("partial_claims"))
        unsupported = as_float(metrics.get("unsupported_claims"))
        mean_support = as_float(metrics.get("mean_support"), as_float(signals.get("sufficiency_score"), 0.0))
        minimum_points = max(1.0, as_float(contract.get("minimum_distinct_points"), 1.0))
        required_facets = len(contract.get("required_facets") or [])
        q_tokens = content_tokens(question)
        a_tokens = content_tokens(answer)
        overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))
        risk_rows.append({
            "question_chars": len(question),
            "answer_chars": len(answer),
            "answer_claims": claims,
            "mean_support": mean_support,
            "unsupported_ratio": unsupported / max(1.0, claims),
            "supported_ratio": supported / max(1.0, claims),
            "partial_ratio": partial / max(1.0, claims),
            "minimum_points": minimum_points,
            "required_facets": required_facets,
            "top_text_score": top_score(text_hits),
            "top_graph_score": top_score(graph_hits),
            "question_answer_overlap": overlap,
            "answer_question_length_ratio": len(answer) / max(1, len(question)),
            "uncertainty_flag": int("uncertainty_or_hedging" in reasons),
            "low_support_flag": int("low_claim_support" in reasons),
            "overlong_flag": int("overlong_answer_may_drift" in reasons),
            "shape_flag": int("answer_shape_below_contract" in reasons),
            "source_dataset": "impact_aware_new48",
            "task_dataset": clean_text(row.get("dataset") or row.get("label") or "unknown"),
            "contract_type": clean_text(contract.get("type") or "general"),
        })

        estimated_points = as_float(signals.get("estimated_distinct_points"), claims)
        term_coverage = as_float(signals.get("term_coverage"), overlap)
        structural_gap = max(0.0, minimum_points - estimated_points) / minimum_points
        term_gap = max(0.0, 1.0 - term_coverage)
        graph_nodes = as_float(row.get("graph_nodes"))
        graph_edges = as_float(row.get("graph_edges"))
        base_answer_chars = as_float(
            signals.get("answer_chars"),
        ) or len(answer)
        impact_rows.append({
            "question_chars": max(1.0, float(len(question))),
            "base_answer_chars": base_answer_chars,
            "answer_question_length_ratio": base_answer_chars / max(1.0, float(len(question))),
            "sentence_count": as_float(signals.get("sentence_count"), sentence_count(answer)),
            "estimated_distinct_points": estimated_points,
            "minimum_points": minimum_points,
            "required_facets": required_facets,
            "structural_gap": structural_gap,
            "term_gap": term_gap,
            "term_coverage": term_coverage,
            "numeric_mentions": as_float(signals.get("numeric_mentions"), numeric_mentions(question)),
            "causal_markers": as_float(signals.get("causal_markers"), marker_count(question, ("why", "cause", "reason", "because"))),
            "contrast_markers": as_float(signals.get("contrast_markers"), marker_count(question, ("compare", "contrast", "difference", "versus"))),
            "sufficiency_score": mean_support,
            "top_text_score": top_score(text_hits),
            "top_graph_score": top_score(graph_hits),
            "context_chars": as_float(row.get("context_chars"), len(str(row.get("context") or ""))),
            "num_chunks": as_float(row.get("num_chunks"), len(text_hits)),
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "router_probability": 0.0,
            "router_eligible": 1.0,
            "uncertainty_flag": int("uncertainty_or_hedging" in reasons),
            "compound_question": int(bool(contract.get("compound_question"))),
            "comparison_required": int(bool(contract.get("comparison_required"))),
            "numeric_answer_expected": int(bool(contract.get("numeric_answer_expected"))),
            "requires_graph": int(bool(completion.get("requires_graph"))),
            "contract_type": clean_text(contract.get("type") or "general"),
            "dataset": clean_text(row.get("dataset") or row.get("label") or "unknown"),
        })
        metadata.append({
            "line_no": int(row.get("line_no", 0)),
            "question": question,
            "risk_candidate": bool(completion.get("risk_candidate")),
            "risk_reasons": sorted(reasons),
        })
    return (
        pd.DataFrame(risk_rows, columns=RISK_NUMERIC_FEATURES + RISK_CATEGORICAL_FEATURES),
        pd.DataFrame(impact_rows, columns=IMPACT_NUMERIC_FEATURES + IMPACT_CATEGORICAL_FEATURES),
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--risk-lines", required=True)
    parser.add_argument("--impact-lines", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--impact-model", default=str(IMPACT_MODEL))
    parser.add_argument("--impact-meta", default=str(IMPACT_META))
    args = parser.parse_args()

    rows = load_jsonl(ROOT / args.input)
    if not rows:
        raise RuntimeError("No rows in input")
    risk_frame, impact_frame, metadata = build_feature_rows(rows)

    risk_artifact = joblib.load(RISK_MODEL)
    risk_model = risk_artifact["model"]
    risk_probabilities = risk_model.predict_proba(risk_frame)[:, 1]

    impact_model_path = Path(args.impact_model)
    impact_meta_path = Path(args.impact_meta)
    if not impact_model_path.is_absolute():
        impact_model_path = ROOT / impact_model_path
    if not impact_meta_path.is_absolute():
        impact_meta_path = ROOT / impact_meta_path
    impact_model = joblib.load(impact_model_path)
    impact_meta = json.loads(impact_meta_path.read_text(encoding="utf-8"))
    # The frozen utility model expects the runtime risk signal as an input.
    # Feed the new-row risk-router probabilities instead of the training-time
    # default so scoring matches the frozen policy contract.
    impact_frame = impact_frame.copy()
    impact_frame["router_probability"] = risk_probabilities
    impact_frame["router_eligible"] = 1.0
    impact_scores = impact_model.predict(impact_frame)
    budget_rate = float(impact_meta["budget_policy"]["reference_budget_rate"])
    budget = max(1, min(len(rows), int(round(budget_rate * len(rows)))))

    risk_order = sorted(range(len(rows)), key=lambda i: (-risk_probabilities[i], int(rows[i].get("line_no", i))))
    impact_order = sorted(range(len(rows)), key=lambda i: (-impact_scores[i], int(rows[i].get("line_no", i))))
    risk_indices = risk_order[:budget]
    impact_indices = impact_order[:budget]
    risk_lines = sorted(int(rows[i].get("line_no", i)) for i in risk_indices)
    impact_lines = sorted(int(rows[i].get("line_no", i)) for i in impact_indices)

    scored_rows = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        new_row["impact_aware_policy"] = {
            "risk_probability": round(float(risk_probabilities[i]), 6),
            "predicted_utility": round(float(impact_scores[i]), 6),
            "risk_selected": i in set(risk_indices),
            "impact_selected": i in set(impact_indices),
            "budget": budget,
            "budget_rate": round(budget / len(rows), 6),
            "risk_model": str(RISK_MODEL.relative_to(ROOT)),
            "impact_model": str(impact_model_path.relative_to(ROOT)),
        }
        scored_rows.append(new_row)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    (ROOT / args.risk_lines).write_text(",".join(map(str, risk_lines)) + "\n", encoding="utf-8")
    (ROOT / args.impact_lines).write_text(",".join(map(str, impact_lines)) + "\n", encoding="utf-8")

    selected_meta = {item["line_no"]: item for item in metadata}
    summary = {
        "input": args.input,
        "rows": len(rows),
        "budget": budget,
        "budget_rate": budget / len(rows),
        "risk_lines": risk_lines,
        "impact_lines": impact_lines,
        "risk_selected_candidates": sum(bool(selected_meta[line]["risk_candidate"]) for line in risk_lines),
        "impact_selected_candidates": sum(bool(selected_meta[line]["risk_candidate"]) for line in impact_lines),
        "risk_probability_range": [float(np.min(risk_probabilities)), float(np.max(risk_probabilities))],
        "predicted_utility_range": [float(np.min(impact_scores)), float(np.max(impact_scores))],
        "causal_note": "Scores are frozen policy predictions; no reference answers or evaluation labels enter feature construction.",
    }
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
