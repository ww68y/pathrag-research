#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrospective prototype for impact-aware escalation.

This is an offline utility-proxy audit, not a causal treatment/control test.
It uses runtime-available features from frozen SAGE v33 Mix130 rows and the
observed SAGE-vs-VSR lexical F1 delta as a historical proxy label. The report
compares a risk-only router, a cross-fitted impact score, always/never
escalation, and an oracle top-k upper bound at the risk-router budget.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAGE_PATH = ROOT / "results" / "mse_pathrag_sage_v33_preservation_delta_mix130.jsonl"
VSR_PATH = ROOT / "results" / "enhanced_ablation_aligned" / "full_v2_rescue_aligned_n130.json"
OUT_DIR = ROOT / "results" / "evidence_contract_depth"
OUT_JSON = OUT_DIR / "impact_aware_escalation_mix130.json"
OUT_MD = OUT_DIR / "IMPACT_AWARE_ESCALATION_MIX130.md"

NUMERIC_FEATURES = [
    "question_chars",
    "base_answer_chars",
    "answer_question_length_ratio",
    "sentence_count",
    "estimated_distinct_points",
    "minimum_points",
    "required_facets",
    "structural_gap",
    "term_gap",
    "term_coverage",
    "numeric_mentions",
    "causal_markers",
    "contrast_markers",
    "sufficiency_score",
    "top_text_score",
    "top_graph_score",
    "context_chars",
    "num_chunks",
    "graph_nodes",
    "graph_edges",
    "router_probability",
    "router_eligible",
    "uncertainty_flag",
    "compound_question",
    "comparison_required",
    "numeric_answer_expected",
    "requires_graph",
]
CATEGORICAL_FEATURES = ["contract_type", "dataset"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_corr(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def make_model() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    features = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", features),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def f1_for_prediction(prediction: str, references: list[str]) -> float:
    from experiments.run_enhanced_experiments import compute_f1

    valid = [str(reference) for reference in references if str(reference).strip()]
    return max((compute_f1(prediction, reference) for reference in valid), default=0.0)


def build_records() -> list[dict[str, Any]]:
    sage_rows = {int(row["line_no"]): row for row in read_jsonl(SAGE_PATH)}
    vsr_payload = json.loads(VSR_PATH.read_text(encoding="utf-8"))
    vsr_rows: dict[int, dict[str, Any]] = {}
    for row in vsr_payload.get("results", []):
        question_id = str(row.get("question_id", ""))
        try:
            line_no = int(question_id.rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            continue
        vsr_rows[line_no] = row

    records: list[dict[str, Any]] = []
    for line_no, sage in sorted(sage_rows.items()):
        vsr = vsr_rows.get(line_no)
        if not vsr:
            continue
        agent = sage.get("text_first_contract_agent") or {}
        contract = agent.get("contract") or {}
        signals = agent.get("signals") or {}
        router = sage.get("learned_contract_gap_router") or {}
        retrieved = sage.get("retrieved_chunks") or []
        question = str(sage.get("question") or "")
        references = sage.get("reference_answers") or [vsr.get("gold_answer", "")]
        sage_f1 = f1_for_prediction(str(sage.get("prediction") or ""), references)
        vsr_f1 = f1_for_prediction(str(vsr.get("prediction") or ""), references)

        minimum_points = max(1.0, safe_float(contract.get("minimum_distinct_points"), 1.0))
        estimated_points = safe_float(signals.get("estimated_distinct_points"))
        term_coverage = safe_float(signals.get("term_coverage"), 1.0)
        structural_gap = max(0.0, minimum_points - estimated_points) / minimum_points
        term_gap = max(0.0, 1.0 - term_coverage)
        probability_raw = router.get("oof_probability")
        router_eligible = int(probability_raw is not None)
        probability = safe_float(probability_raw, 0.0)
        top_text_score = safe_float(retrieved[0].get("score")) if retrieved else 0.0
        top_graph_score = safe_float(sage.get("graph_nodes"))
        base_answer_chars = safe_float(agent.get("base_answer_chars"), safe_float(signals.get("answer_chars")))
        question_chars = max(1.0, float(len(question)))
        features = {
            "question_chars": question_chars,
            "base_answer_chars": base_answer_chars,
            "answer_question_length_ratio": base_answer_chars / question_chars,
            "sentence_count": safe_float(signals.get("sentence_count")),
            "estimated_distinct_points": estimated_points,
            "minimum_points": minimum_points,
            "required_facets": len(contract.get("required_facets") or []),
            "structural_gap": structural_gap,
            "term_gap": term_gap,
            "term_coverage": term_coverage,
            "numeric_mentions": safe_float(signals.get("numeric_mentions")),
            "causal_markers": safe_float(signals.get("causal_markers")),
            "contrast_markers": safe_float(signals.get("contrast_markers")),
            "sufficiency_score": safe_float(signals.get("sufficiency_score"), 1.0),
            "top_text_score": top_text_score,
            "top_graph_score": top_graph_score,
            "context_chars": safe_float(sage.get("context_chars")),
            "num_chunks": safe_float(sage.get("num_chunks")),
            "graph_nodes": safe_float(sage.get("graph_nodes")),
            "graph_edges": safe_float(sage.get("graph_edges")),
            "router_probability": probability,
            "router_eligible": router_eligible,
            "uncertainty_flag": int(bool(signals.get("uncertainty_patterns"))),
            "compound_question": int(bool(contract.get("compound_question"))),
            "comparison_required": int(bool(contract.get("comparison_required"))),
            "numeric_answer_expected": int(bool(contract.get("numeric_answer_expected"))),
            "requires_graph": int(bool(agent.get("requires_graph"))),
            "contract_type": str(contract.get("type") or "general"),
            "dataset": str(sage.get("dataset") or "unknown"),
        }
        records.append({
            "line_no": line_no,
            "features": features,
            "router_probability": probability,
            "router_eligible": router_eligible,
            "observed_route_selected": bool(router.get("verification_selected"))
            or bool(agent.get("graph_rescue_attempted")),
            "sage_f1": sage_f1,
            "vsr_f1": vsr_f1,
            "f1_delta": sage_f1 - vsr_f1,
            "positive_delta": int(sage_f1 - vsr_f1 > 1e-12),
        })
    return records


def policy_summary(
    name: str,
    selected: np.ndarray,
    deltas: np.ndarray,
    base_f1: np.ndarray,
    budget: int | None = None,
) -> dict[str, Any]:
    selected = selected.astype(bool)
    selected_deltas = deltas[selected]
    return {
        "policy": name,
        "n": int(len(deltas)),
        "selected": int(selected.sum()),
        "selection_rate": round(float(selected.mean()), 6),
        "budget_target": budget,
        "observed_gain_if_selected": round(float(selected_deltas.mean()), 6)
        if selected_deltas.size
        else 0.0,
        "mean_gain_proxy": round(float(deltas[selected].sum() / len(deltas)), 6),
        "proxy_final_f1": round(float(base_f1.mean() + deltas[selected].sum() / len(deltas)), 6),
        "positive_gain_selected": int((selected_deltas > 0).sum()),
        "negative_gain_selected": int((selected_deltas < 0).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    if len(records) < 20:
        raise RuntimeError(f"Need at least 20 aligned rows, found {len(records)}")

    frame = pd.DataFrame([record["features"] for record in records])
    deltas = np.array([record["f1_delta"] for record in records], dtype=float)
    base_f1 = np.array([record["vsr_f1"] for record in records], dtype=float)
    groups = np.array([f"row:{record['line_no']}" for record in records])
    n_splits = min(5, len(set(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    oof_predictions = np.zeros(len(records), dtype=float)
    for train_idx, test_idx in splitter.split(frame, deltas, groups):
        model = make_model()
        model.fit(frame.iloc[train_idx], deltas[train_idx])
        oof_predictions[test_idx] = model.predict(frame.iloc[test_idx])

    risk_scores = np.array([record["router_probability"] for record in records], dtype=float)
    thresholds = [
        record["router_probability"]
        for record in records
        if record["router_eligible"] and record["router_probability"] > 0
    ]
    risk_threshold = 0.13
    risk_selected = risk_scores >= risk_threshold
    budget = int(risk_selected.sum())
    impact_selected = np.zeros(len(records), dtype=bool)
    if budget:
        top_indices = np.argsort(-oof_predictions)[:budget]
        impact_selected[top_indices] = True
    observed_selected = np.array([record["observed_route_selected"] for record in records], dtype=bool)
    oracle_selected = np.zeros(len(records), dtype=bool)
    if budget:
        oracle_selected[np.argsort(-deltas)[:budget]] = True

    policies = [
        policy_summary("never_escalate", np.zeros(len(records), dtype=bool), deltas, base_f1),
        policy_summary("risk_only_router", risk_selected, deltas, base_f1, budget),
        policy_summary("impact_aware_oof_topk", impact_selected, deltas, base_f1, budget),
        policy_summary("observed_route", observed_selected, deltas, base_f1, int(observed_selected.sum())),
        policy_summary("always_escalate", np.ones(len(records), dtype=bool), deltas, base_f1, len(records)),
        policy_summary("oracle_topk_upper_bound", oracle_selected, deltas, base_f1, budget),
    ]

    payload = {
        "scope": "Retrospective impact-aware escalation proxy on frozen SAGE v33 Mix130",
        "offline": True,
        "causal_caveat": "Observed SAGE-vs-VSR F1 deltas are a historical utility proxy, not randomized escalation treatment effects.",
        "runtime_feature_caveat": "Features exclude reference answers, F1, judge scores, and post-decision guard fields.",
        "n_aligned": len(records),
        "positive_delta_rows": int((deltas > 0).sum()),
        "negative_delta_rows": int((deltas < 0).sum()),
        "risk_threshold": risk_threshold,
        "risk_budget": budget,
        "observed_route_budget": int(observed_selected.sum()),
        "cross_fitted_model": "RandomForestRegressor + GroupKFold by row id",
        "oof_mae": round(float(mean_absolute_error(deltas, oof_predictions)), 6),
        "oof_r2": round(float(r2_score(deltas, oof_predictions)), 6),
        "oof_pred_vs_delta_correlation": compute_corr(oof_predictions.tolist(), deltas.tolist()),
        "policies": policies,
        "rows": [
            {
                "line_no": record["line_no"],
                "f1_delta": round(record["f1_delta"], 6),
                "oof_predicted_gain": round(float(oof_predictions[index]), 6),
                "router_probability": round(record["router_probability"], 6),
                "risk_selected": bool(risk_selected[index]),
                "impact_selected": bool(impact_selected[index]),
                "observed_route_selected": bool(observed_selected[index]),
            }
            for index, record in enumerate(records)
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Impact-Aware Escalation Proxy on Mix130",
        "",
        "> Offline retrospective audit; no API calls and no locked-holdout writes.",
        "",
        "## Protocol",
        "",
        "This prototype uses only runtime-available features from frozen SAGE v33 rows. The observed SAGE-vs-VSR lexical F1 delta is used as a historical utility proxy. It is not a randomized estimate of escalation treatment effect, so the policy table is hypothesis-generating rather than a final paper result.",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Aligned rows | {payload['n_aligned']} |",
        f"| Positive historical deltas | {payload['positive_delta_rows']} |",
        f"| Negative historical deltas | {payload['negative_delta_rows']} |",
        f"| Risk-only threshold | {risk_threshold:.2f} |",
        f"| Risk-only budget | {budget} / {len(records)} |",
        f"| OOF MAE | {payload['oof_mae']:.4f} |",
        f"| OOF R2 | {payload['oof_r2']:.4f} |",
        f"| OOF prediction/delta correlation | {payload['oof_pred_vs_delta_correlation']:.4f} |",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Selected | Rate | Mean observed gain on selected | Mean gain proxy | Proxy final F1 | Positive / negative selected |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in policies:
        lines.append(
            f"| `{policy['policy']}` | {policy['selected']} | {policy['selection_rate']:.1%} | {policy['observed_gain_if_selected']:.4f} | {policy['mean_gain_proxy']:.4f} | {policy['proxy_final_f1']:.4f} | {policy['positive_gain_selected']} / {policy['negative_gain_selected']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The impact-aware score is evaluated with out-of-fold predictions, so its ranking does not use the row's own observed delta during selection.",
        "- The oracle top-k row is only an upper bound; it is not available at runtime.",
        "- A positive result here would justify a new frozen-policy experiment, not replace it. The next valid test must use new source data or a newly locked holdout with fresh online outputs.",
        "- If impact-aware routing does not beat risk-only routing at matched budget, the evidence-contract router should remain a risk detector rather than being claimed as an upgrade-utility estimator.",
        "",
        "## Reproducibility",
        "",
        f"- SAGE input: `{SAGE_PATH.relative_to(ROOT)}`",
        f"- VSR input: `{VSR_PATH.relative_to(ROOT)}`",
        f"- JSON output: `{OUT_JSON.relative_to(ROOT)}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps({"risk_budget": budget, "policies": policies}, ensure_ascii=False))


if __name__ == "__main__":
    main()
