#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Freeze the impact-aware escalation scorer trained on Mix130 development rows."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.analyze_impact_aware_escalation import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_records,
    make_model,
)

OUT_DIR = ROOT / "results" / "evidence_contract_depth"
MODEL_PATH = OUT_DIR / "impact_aware_escalation_policy_v1.joblib"
META_PATH = OUT_DIR / "impact_aware_escalation_policy_v1.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    if len(records) < 20:
        raise RuntimeError(f"Need at least 20 aligned development rows, found {len(records)}")

    import pandas as pd
    import numpy as np

    frame = pd.DataFrame([record["features"] for record in records])
    target = np.array([record["f1_delta"] for record in records], dtype=float)
    model = make_model()
    model.fit(frame, target)
    joblib.dump(model, MODEL_PATH)

    router_budget = sum(
        record["router_probability"] >= 0.13
        for record in records
    )
    metadata = {
        "policy_name": "impact_aware_escalation_v1",
        "frozen_date": date.today().isoformat(),
        "training_scope": "Frozen SAGE v33 Mix130 development artifacts only",
        "training_rows": len(records),
        "target": "historical SAGE-vs-VSR lexical F1 delta proxy",
        "causal_caveat": "The training target is not a randomized escalation treatment effect.",
        "runtime_feature_constraint": "No reference answers, judge scores, F1 values, or post-decision guard fields.",
        "model": {
            "family": "RandomForestRegressor",
            "n_estimators": 500,
            "max_depth": 6,
            "min_samples_leaf": 4,
            "random_state": 42,
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "budget_policy": {
            "risk_router_threshold_reference": 0.13,
            "reference_budget_rows": int(router_budget),
            "reference_budget_rate": round(router_budget / len(records), 6),
            "selection_rule": "rank predicted escalation utility and select only within a precommitted budget",
        },
        "inputs": {
            "sage": "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl",
            "vsr": "results/enhanced_ablation_aligned/full_v2_rescue_aligned_n130.json",
        },
        "validation_requirement": "Use new source data or a newly locked holdout; do not retune this artifact on completed holdouts.",
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {META_PATH}")
    print(json.dumps(metadata["budget_policy"], ensure_ascii=False))


if __name__ == "__main__":
    main()
