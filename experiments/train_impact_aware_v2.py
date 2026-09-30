#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train and freeze impact-aware escalation v2 on matched development pairs.

v2 uses the same feature construction as new-row scoring, a matched
base/repair utility target, and source-context-grouped cross-fitting for the
development audit. The frozen model is still evaluated only on a new holdout.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.analyze_impact_aware_escalation import f1_for_prediction, make_model  # noqa: E402
from experiments.score_impact_aware_policy import (  # noqa: E402
    IMPACT_CATEGORICAL_FEATURES,
    IMPACT_NUMERIC_FEATURES,
    RISK_MODEL,
    build_feature_rows,
)

SAGE_PATH = ROOT / "results" / "mse_pathrag_sage_v33_preservation_delta_mix130.jsonl"
VSR_PATH = ROOT / "results" / "enhanced_ablation_aligned" / "full_v2_rescue_aligned_n130.json"
MIX_PATH = ROOT / "datasets" / "UltraDomain" / "mix.jsonl"
OUT_DIR = ROOT / "results" / "evidence_contract_depth"
MODEL_PATH = OUT_DIR / "impact_aware_escalation_policy_v2.joblib"
META_PATH = OUT_DIR / "impact_aware_escalation_policy_v2.json"
AUDIT_PATH = OUT_DIR / "IMPACT_AWARE_ESCALATION_V2_MIX130.json"
AUDIT_MD_PATH = OUT_DIR / "IMPACT_AWARE_ESCALATION_V2_MIX130.md"
REFERENCE_BUDGET_RATE = 32 / 130


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def context_group_index() -> dict[str, str]:
    groups: dict[str, str] = {}
    for row in read_jsonl(MIX_PATH):
        question = normalize(row.get("input") or row.get("question"))
        context = normalize(row.get("context"))
        if not question or not context:
            continue
        groups[question] = hashlib.sha1(context.encode("utf-8")).hexdigest()
    return groups


def load_matched_rows() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[str]]:
    sage_rows = read_jsonl(SAGE_PATH)
    vsr_payload = json.loads(VSR_PATH.read_text(encoding="utf-8"))
    vsr_rows = {
        int(str(row.get("question_id", "")).rsplit("_", 1)[-1]): row
        for row in vsr_payload.get("results", [])
        if str(row.get("question_id", "")).rsplit("_", 1)[-1].isdigit()
    }
    groups_by_question = context_group_index()
    feature_rows: list[dict[str, Any]] = []
    targets: list[float] = []
    groups: list[str] = []
    for sage in sorted(sage_rows, key=lambda row: int(row["line_no"])):
        line_no = int(sage["line_no"])
        base = vsr_rows.get(line_no)
        if not base:
            continue
        references = sage.get("reference_answers") or [base.get("gold_answer", "")]
        repair_f1 = f1_for_prediction(str(sage.get("prediction") or ""), references)
        base_f1 = f1_for_prediction(str(base.get("prediction") or ""), references)
        feature_rows.append(sage)
        targets.append(float(repair_f1 - base_f1))
        question = normalize(sage.get("question"))
        groups.append(groups_by_question.get(question, f"line:{line_no}"))
    if len(feature_rows) < 20:
        raise RuntimeError(f"Need at least 20 matched rows, found {len(feature_rows)}")
    return feature_rows, np.asarray(targets, dtype=float), np.asarray(groups), groups


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, target, group_array, group_list = load_matched_rows()
    risk_frame, impact_frame, _ = build_feature_rows(rows)
    risk_artifact = joblib.load(RISK_MODEL)
    risk_probabilities = risk_artifact["model"].predict_proba(risk_frame)[:, 1]
    impact_frame = impact_frame.copy()
    impact_frame["router_probability"] = risk_probabilities
    impact_frame["router_eligible"] = 1.0
    impact_frame = impact_frame[IMPACT_NUMERIC_FEATURES + IMPACT_CATEGORICAL_FEATURES]

    unique_groups = len(set(group_list))
    splitter = GroupKFold(n_splits=min(5, unique_groups))
    oof = np.zeros(len(rows), dtype=float)
    for train_idx, test_idx in splitter.split(impact_frame, target, group_array):
        model = make_model()
        model.fit(impact_frame.iloc[train_idx], target[train_idx])
        oof[test_idx] = model.predict(impact_frame.iloc[test_idx])

    frozen_model = make_model()
    frozen_model.fit(impact_frame, target)
    joblib.dump(frozen_model, MODEL_PATH)

    budget = int(round(REFERENCE_BUDGET_RATE * len(rows)))
    selected = np.zeros(len(rows), dtype=bool)
    selected[np.argsort(-oof)[:budget]] = True
    risk_selected = np.zeros(len(rows), dtype=bool)
    risk_selected[np.argsort(-risk_probabilities)[:budget]] = True
    oracle = np.zeros(len(rows), dtype=bool)
    oracle[np.argsort(-target)[:budget]] = True

    audit = {
        "scope": "Impact-aware escalation v2 matched-pair grouped cross-fit audit",
        "offline": True,
        "causal_caveat": "The matched repair-minus-base F1 target is observational utility, not a randomized treatment effect.",
        "n_rows": len(rows),
        "n_source_context_groups": unique_groups,
        "target": "matched repair F1 minus base F1",
        "grouping": "source-context SHA-1 recovered from mix.jsonl",
        "features": {
            "numeric": IMPACT_NUMERIC_FEATURES,
            "categorical": IMPACT_CATEGORICAL_FEATURES,
            "risk_probability_injected": True,
        },
        "budget": {"threshold_reference": 0.13, "rows": budget, "rate": budget / len(rows)},
        "oof": {
            "mae": float(mean_absolute_error(target, oof)),
            "r2": float(r2_score(target, oof)),
            "correlation": float(np.corrcoef(target, oof)[0, 1]) if np.std(target) and np.std(oof) else None,
        },
        "matched_budget": {
            "risk_only_mean_gain_proxy": float(target[risk_selected].sum() / len(target)),
            "impact_aware_v2_oof_mean_gain_proxy": float(target[selected].sum() / len(target)),
            "oracle_mean_gain_proxy": float(target[oracle].sum() / len(target)),
            "risk_selected_positive": int(np.sum(target[risk_selected] > 0)),
            "impact_selected_positive": int(np.sum(target[selected] > 0)),
            "oracle_positive": int(np.sum(target[oracle] > 0)),
        },
        "rows": [
            {
                "line_no": int(row["line_no"]),
                "source_context_group": group_list[index],
                "target_delta": round(float(target[index]), 6),
                "oof_predicted_utility": round(float(oof[index]), 6),
                "risk_probability": round(float(risk_probabilities[index]), 6),
                "risk_selected": bool(risk_selected[index]),
                "impact_selected": bool(selected[index]),
            }
            for index, row in enumerate(rows)
        ],
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "policy_name": "impact_aware_escalation_v2",
        "frozen_date": "2026-09-18",
        "training_scope": "SAGE v33 Mix130 matched base/repair development pairs",
        "training_rows": len(rows),
        "source_context_groups": unique_groups,
        "target": "matched repair F1 minus base F1 utility proxy",
        "grouped_cross_fitting": "GroupKFold by source-context hash recovered from mix.jsonl",
        "causal_caveat": audit["causal_caveat"],
        "runtime_feature_constraint": "No reference answers, judge scores, F1 values, or post-decision guard fields enter feature construction.",
        "model": {
            "family": "RandomForestRegressor",
            "n_estimators": 500,
            "max_depth": 6,
            "min_samples_leaf": 4,
            "random_state": 42,
        },
        "numeric_features": IMPACT_NUMERIC_FEATURES,
        "categorical_features": IMPACT_CATEGORICAL_FEATURES,
        "budget_policy": {
            "risk_router_threshold_reference": 0.13,
            "reference_budget_rows": 32,
            "reference_budget_rate": round(REFERENCE_BUDGET_RATE, 6),
            "selection_rule": "rank predicted escalation utility and select within the precommitted risk budget",
        },
        "inputs": {
            "sage_repair": str(SAGE_PATH.relative_to(ROOT)),
            "base": str(VSR_PATH.relative_to(ROOT)),
            "source_context_index": str(MIX_PATH.relative_to(ROOT)),
        },
        "oof_metrics": audit["oof"],
        "validation_requirement": "Use a new untouched source holdout; do not tune this artifact on New48.",
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Impact-Aware Escalation v2 Development Audit",
        "",
        "> Offline grouped cross-fit audit on matched base/repair pairs; no locked holdout is used.",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Matched rows | {len(rows)} |",
        f"| Source-context groups | {unique_groups} |",
        f"| Reference risk budget | {budget} / {len(rows)} |",
        f"| OOF MAE | {audit['oof']['mae']:.4f} |",
        f"| OOF R2 | {audit['oof']['r2']:.4f} |",
        f"| OOF correlation | {audit['oof']['correlation']:.4f} |",
        "",
        "## Matched Budget",
        "",
        "| Ranking | Mean gain proxy | Positive selected |",
        "|---|---:|---:|",
        f"| risk-only | {audit['matched_budget']['risk_only_mean_gain_proxy']:.4f} | {audit['matched_budget']['risk_selected_positive']} |",
        f"| impact-aware v2 OOF | {audit['matched_budget']['impact_aware_v2_oof_mean_gain_proxy']:.4f} | {audit['matched_budget']['impact_selected_positive']} |",
        f"| oracle upper bound | {audit['matched_budget']['oracle_mean_gain_proxy']:.4f} | {audit['matched_budget']['oracle_positive']} |",
        "",
        "## Interpretation",
        "",
        "- The target is constructed from matched repair and base outputs for the same question.",
        "- Grouped cross-fitting is used for development diagnostics; the frozen model is fit once on all development rows and must be tested only on a new holdout.",
        "- This audit is a gating step for v2 validation, not final evidence of causal escalation benefit.",
    ]
    AUDIT_MD_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "meta": str(META_PATH.relative_to(ROOT)),
        "audit": str(AUDIT_PATH.relative_to(ROOT)),
        "oof": audit["oof"],
        "matched_budget": audit["matched_budget"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
