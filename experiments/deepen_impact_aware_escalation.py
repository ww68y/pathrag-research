#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deepen one mechanism: where should escalation budget be spent?

This remains an offline, retrospective utility-proxy analysis. It reuses the
runtime-only feature construction from analyze_impact_aware_escalation.py and
studies ranking quality across budgets, disagreement rows, contract types, and
simple cost penalties. No API calls or locked holdout rows are touched.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.analyze_impact_aware_escalation import build_records, make_model  # noqa: E402

OUT_DIR = ROOT / "results" / "evidence_contract_depth"
OUT_JSON = OUT_DIR / "IMPACT_AWARE_ESCALATION_DEPTH_MIX130.json"
OUT_MD = OUT_DIR / "IMPACT_AWARE_ESCALATION_DEPTH_MIX130.md"


def select_top(scores: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    if k > 0:
        mask[np.argsort(-scores)[:k]] = True
    return mask


def curve_point(name: str, mask: np.ndarray, deltas: np.ndarray) -> dict[str, Any]:
    selected = deltas[mask]
    return {
        "policy": name,
        "selected": int(mask.sum()),
        "selection_rate": round(float(mask.mean()), 6),
        "mean_gain_proxy": round(float(deltas[mask].sum() / len(deltas)), 6),
        "selected_mean_delta": round(float(selected.mean()), 6) if selected.size else 0.0,
        "positive_selected": int((selected > 0).sum()),
        "negative_selected": int((selected < 0).sum()),
    }


def auuc(points: list[dict[str, Any]]) -> float:
    ordered = sorted(points, key=lambda row: row["selection_rate"])
    xs = np.array([row["selection_rate"] for row in ordered], dtype=float)
    ys = np.array([row["mean_gain_proxy"] for row in ordered], dtype=float)
    return float(np.trapezoid(ys, xs))


def random_curve(deltas: np.ndarray, ks: list[int], repeats: int = 2000) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    rows = []
    for k in ks:
        if k == 0:
            rows.append(curve_point("random", np.zeros(len(deltas), dtype=bool), deltas))
            continue
        gains = []
        selected_means = []
        negatives = []
        for _ in range(repeats):
            indices = rng.choice(len(deltas), size=k, replace=False)
            picked = deltas[indices]
            gains.append(float(picked.sum() / len(deltas)))
            selected_means.append(float(picked.mean()))
            negatives.append(int((picked < 0).sum()))
        rows.append({
            "policy": "random",
            "selected": k,
            "selection_rate": round(k / len(deltas), 6),
            "mean_gain_proxy": round(float(np.mean(gains)), 6),
            "mean_gain_proxy_std": round(float(np.std(gains)), 6),
            "selected_mean_delta": round(float(np.mean(selected_means)), 6),
            "negative_selected": round(float(np.mean(negatives)), 3),
        })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    frame = pd.DataFrame([record["features"] for record in records])
    deltas = np.array([record["f1_delta"] for record in records], dtype=float)
    groups = np.array([f"row:{record['line_no']}" for record in records])
    router_scores = np.array([record["router_probability"] for record in records], dtype=float)
    contract_gap_scores = np.array([
        max(record["features"].get("structural_gap", 0.0), record["features"].get("term_gap", 0.0))
        for record in records
    ], dtype=float)

    oof_predictions = np.zeros(len(records), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(set(groups))))
    for train_idx, test_idx in splitter.split(frame, deltas, groups):
        model = make_model()
        model.fit(frame.iloc[train_idx], deltas[train_idx])
        oof_predictions[test_idx] = model.predict(frame.iloc[test_idx])

    ks = [0, 8, 16, 24, 32, 40, 52, 65, 78, 91, 104, 117, 130]
    curves: dict[str, list[dict[str, Any]]] = {
        "risk_probability": [curve_point("risk_probability", select_top(router_scores, k), deltas) for k in ks],
        "contract_gap": [curve_point("contract_gap", select_top(contract_gap_scores, k), deltas) for k in ks],
        "impact_aware_oof": [curve_point("impact_aware_oof", select_top(oof_predictions, k), deltas) for k in ks],
        "oracle": [curve_point("oracle", select_top(deltas, k), deltas) for k in ks],
        "random": random_curve(deltas, ks),
    }

    matched_k = 32
    masks = {
        "risk_probability": select_top(router_scores, matched_k),
        "contract_gap": select_top(contract_gap_scores, matched_k),
        "impact_aware_oof": select_top(oof_predictions, matched_k),
        "oracle": select_top(deltas, matched_k),
    }
    matched = [curve_point(name, mask, deltas) for name, mask in masks.items()]

    impact_mask = masks["impact_aware_oof"]
    risk_mask = masks["risk_probability"]
    disagreement_masks = {
        "both": impact_mask & risk_mask,
        "impact_only": impact_mask & ~risk_mask,
        "risk_only": risk_mask & ~impact_mask,
        "neither": ~impact_mask & ~risk_mask,
    }
    disagreement = []
    for name, mask in disagreement_masks.items():
        picked = deltas[mask]
        disagreement.append({
            "group": name,
            "n": int(mask.sum()),
            "mean_delta": round(float(picked.mean()), 6) if picked.size else 0.0,
            "positive": int((picked > 0).sum()),
            "negative": int((picked < 0).sum()),
            "line_nos": [records[index]["line_no"] for index in np.flatnonzero(mask)],
        })

    by_type: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, mask in masks.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for index, record in enumerate(records):
            if mask[index]:
                grouped[record["features"].get("contract_type", "unknown")].append(float(deltas[index]))
        for contract_type, values in sorted(grouped.items()):
            by_type[contract_type][name] = {
                "n": len(values),
                "mean_delta": round(float(np.mean(values)), 6),
                "negative": int(sum(value < 0 for value in values)),
            }

    lambdas = [0.05, 0.10, 0.20]
    net_utility = []
    for policy in matched:
        for penalty in lambdas:
            net_utility.append({
                "policy": policy["policy"],
                "lambda": penalty,
                "net_utility": round(policy["mean_gain_proxy"] - penalty * policy["selection_rate"], 6),
            })

    payload = {
        "scope": "Deep mechanism audit for impact-aware escalation on frozen SAGE v33 Mix130",
        "offline": True,
        "causal_caveat": "Observed SAGE-vs-VSR F1 deltas are historical utility proxies, not randomized escalation effects.",
        "n_aligned": len(records),
        "matched_budget": matched_k,
        "curves": curves,
        "auuc": {name: round(auuc(points), 8) for name, points in curves.items()},
        "matched_budget_comparison": matched,
        "disagreement": disagreement,
        "by_contract_type": dict(by_type),
        "net_utility": net_utility,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Deep Mechanism Audit: Where Should Escalation Budget Go?",
        "",
        "> Offline retrospective analysis; no API calls and no locked-holdout writes.",
        "",
        "## Narrow Question",
        "",
        "The analysis focuses on one claim only: whether an impact-aware ranking can spend a fixed escalation budget more effectively than the current risk ranking.",
        "",
        "Observed SAGE-vs-VSR F1 deltas are used as historical utility proxies. They are not randomized escalation treatment effects, so all conclusions are mechanism hypotheses rather than final causal evidence.",
        "",
        "## Budget Curves",
        "",
        "| Budget | Risk mean gain | Contract-gap mean gain | Impact-aware mean gain | Oracle mean gain | Random mean gain |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    curve_maps = {name: {row["selected"]: row for row in points} for name, points in curves.items()}
    for k in ks:
        lines.append(
            f"| {k} | {curve_maps['risk_probability'][k]['mean_gain_proxy']:.4f} | {curve_maps['contract_gap'][k]['mean_gain_proxy']:.4f} | {curve_maps['impact_aware_oof'][k]['mean_gain_proxy']:.4f} | {curve_maps['oracle'][k]['mean_gain_proxy']:.4f} | {curve_maps['random'][k]['mean_gain_proxy']:.4f} |"
        )
    lines.extend([
        "",
        "### Area under utility curve",
        "",
        "| Ranking | AUUC proxy |",
        "|---|---:|",
    ])
    for name, value in payload["auuc"].items():
        lines.append(f"| `{name}` | {value:.6f} |")
    lines.extend([
        "",
        "## Matched Budget: 32/130",
        "",
        "| Ranking | Mean gain proxy | Selected mean delta | Positive | Negative |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in matched:
        lines.append(
            f"| `{row['policy']}` | {row['mean_gain_proxy']:.4f} | {row['selected_mean_delta']:.4f} | {row['positive_selected']} | {row['negative_selected']} |"
        )
    lines.extend([
        "",
        "## Risk / Impact Disagreement",
        "",
        "| Group | N | Mean delta | Positive | Negative |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in disagreement:
        lines.append(f"| `{row['group']}` | {row['n']} | {row['mean_delta']:.4f} | {row['positive']} | {row['negative']} |")
    lines.extend([
        "",
        "The most important rows are the disagreement rows: `impact_only` is where a benefit ranker spends budget that the risk router would skip, while `risk_only` is where the current router spends budget that the impact ranker rejects.",
        "",
        "## Cost-Penalized Utility",
        "",
        "Net utility is defined as historical gain proxy minus `lambda * selection_rate`; lambda is a normalized cost penalty, not a dollar estimate.",
        "",
        "| Policy | lambda=0.05 | lambda=0.10 | lambda=0.20 |",
        "|---|---:|---:|---:|",
    ])
    utility_map = {(row["policy"], row["lambda"]): row["net_utility"] for row in net_utility}
    for policy in [row["policy"] for row in matched]:
        lines.append(
            f"| `{policy}` | {utility_map[(policy, 0.05)]:.4f} | {utility_map[(policy, 0.10)]:.4f} | {utility_map[(policy, 0.20)]:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If impact-aware stays above risk-only across the low-to-medium budget range, the deep contribution is selective escalation utility rather than generic risk detection.",
        "- If the curves converge or cross on a fresh holdout, the claim should be weakened to calibrated risk gating and the impact score should remain exploratory.",
        "- Contract-type tables and disagreement rows identify where the mechanism should be improved without adding another independent innovation module.",
        "",
        "## Reproducibility",
        "",
        "- Inputs: frozen SAGE v33 Mix130 and aligned `full_v2_rescue` artifacts.",
        f"- JSON output: `{OUT_JSON.relative_to(ROOT)}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps({"auuc": payload["auuc"], "matched": matched, "disagreement": disagreement}, ensure_ascii=False))


if __name__ == "__main__":
    main()
