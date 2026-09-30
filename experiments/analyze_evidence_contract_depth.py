#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit the evidence-contract mechanism behind frozen SAGE v33.

This is an offline mechanism audit, not a new API experiment. It aligns the
frozen SAGE v33 Mix130 rows with the aligned full_v2_rescue rows and asks:

1. Does the contract/routing signal identify the rows where adaptive SAGE is
   most useful relative to the VSR rescue layer?
2. How do contract type, structural deficit, routing probability, and rescue
   decisions relate to the observed row-level F1 delta?

The report is explicitly observational: route-selected and route-skipped rows
are not a randomized treatment/control split.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SAGE_PATH = ROOT / "results" / "mse_pathrag_sage_v33_preservation_delta_mix130.jsonl"
VSR_PATH = ROOT / "results" / "enhanced_ablation_aligned" / "full_v2_rescue_aligned_n130.json"
OUT_DIR = ROOT / "results" / "evidence_contract_depth"
OUT_JSON = OUT_DIR / "evidence_contract_depth_mix130.json"
OUT_MD = OUT_DIR / "EVIDENCE_CONTRACT_DEPTH_MIX130.md"

import sys

sys.path.insert(0, str(ROOT))
from experiments.run_enhanced_experiments import compute_f1  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: Iterable[float]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def f1_delta_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["f1_delta"]) for row in rows]
    return {
        "n": len(rows),
        "mean_f1_delta": mean(deltas),
        "wins": sum(delta > 1e-12 for delta in deltas),
        "losses": sum(delta < -1e-12 for delta in deltas),
        "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "mean_contract_gap": mean(row["contract_gap"] for row in rows),
        "mean_router_probability": mean(row["router_probability"] for row in rows),
        "route_selected": sum(bool(row["route_selected"]) for row in rows),
        "graph_rescue_accepted": sum(bool(row["graph_rescue_accepted"]) for row in rows),
        "guard_actions": dict(Counter(row["guard_action"] for row in rows)),
    }


def route_bucket(probability: float | None, selected: bool) -> str:
    if probability is None:
        return "not_eligible"
    if probability < 0.13:
        return "below_threshold"
    if probability < 0.30:
        return "threshold_to_0.30"
    if probability < 0.60:
        return "0.30_to_0.60"
    return "at_least_0.60"


def build_rows() -> list[dict[str, Any]]:
    sage_rows = {int(row["line_no"]): row for row in read_jsonl(SAGE_PATH)}
    vsr_payload = read_json(VSR_PATH)
    vsr_rows = {}
    for row in vsr_payload.get("results", []):
        question_id = str(row.get("question_id", ""))
        try:
            line_no = int(question_id.rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            continue
        vsr_rows[line_no] = row

    aligned: list[dict[str, Any]] = []
    for line_no, sage in sorted(sage_rows.items()):
        vsr = vsr_rows.get(line_no)
        if not vsr:
            continue
        references = sage.get("reference_answers") or [vsr.get("gold_answer", "")]
        prediction = str(sage.get("prediction") or "")
        vsr_prediction = str(vsr.get("prediction") or "")
        sage_f1 = max(compute_f1(prediction, reference) for reference in references if str(reference).strip())
        vsr_f1 = max(compute_f1(vsr_prediction, reference) for reference in references if str(reference).strip())

        contract_agent = sage.get("text_first_contract_agent") or {}
        contract = contract_agent.get("contract") or {}
        signals = contract_agent.get("signals") or {}
        router = sage.get("learned_contract_gap_router") or {}
        guard = sage.get("evidence_delta_guard") or {}
        rescue_guard = contract_agent.get("rescue_utility_guard") or {}

        min_points = max(1.0, float(contract.get("minimum_distinct_points") or 1.0))
        estimated_points = float(signals.get("estimated_distinct_points") or 0.0)
        term_coverage = float(signals.get("term_coverage") or 0.0)
        structural_gap = max(0.0, min_points - estimated_points) / min_points
        term_gap = max(0.0, 1.0 - term_coverage)
        probability = router.get("oof_probability")
        probability = float(probability) if probability is not None else None
        route_selected = bool(router.get("verification_selected")) or bool(contract_agent.get("graph_rescue_attempted"))

        aligned.append(
            {
                "line_no": line_no,
                "dataset": sage.get("dataset"),
                "contract_type": contract.get("type", "unknown"),
                "contract_gap": max(structural_gap, term_gap),
                "structural_gap": structural_gap,
                "term_gap": term_gap,
                "router_probability": probability,
                "route_selected": route_selected,
                "route_bucket": route_bucket(probability, route_selected),
                "router_action": router.get("action", "unknown"),
                "graph_rescue_attempted": bool(contract_agent.get("graph_rescue_attempted")),
                "graph_rescue_accepted": bool(contract_agent.get("graph_rescue_accepted")),
                "guard_action": guard.get("action", "unknown"),
                "rescue_trigger_reason": (contract_agent.get("rescue_utility_guard") or {}).get("trigger_reasons", []),
                "sage_f1": sage_f1,
                "vsr_rescue_f1": vsr_f1,
                "f1_delta": sage_f1 - vsr_f1,
                "final_answer_chars": int(contract_agent.get("final_answer_chars") or len(prediction)),
            }
        )
    return aligned


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) != len(x):
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denom = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denom if denom else None


def build_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"route_selected={row['route_selected']}"] .append(row)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["contract_type"]].append(row)

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[row["route_bucket"]].append(row)

    probabilities = [row["router_probability"] for row in rows if row["router_probability"] is not None]
    probability_deltas = [row["f1_delta"] for row in rows if row["router_probability"] is not None]
    return {
        "scope": "SAGE v33-preservation Mix130 aligned against VSR full_v2_rescue",
        "offline": True,
        "observational_caveat": "Route-selected and route-skipped rows are not randomized treatment/control groups.",
        "n_aligned": len(rows),
        "overall": f1_delta_stats(rows),
        "by_route": {key: f1_delta_stats(value) for key, value in grouped.items()},
        "by_contract_type": {key: f1_delta_stats(value) for key, value in sorted(by_type.items())},
        "by_route_bucket": {key: f1_delta_stats(value) for key, value in sorted(by_bucket.items())},
        "router_probability_vs_f1_delta_correlation": correlation(probabilities, probability_deltas),
        "contract_gap_vs_f1_delta_correlation": correlation([row["contract_gap"] for row in rows], [row["f1_delta"] for row in rows]),
        "route_action_counts": dict(Counter(row["router_action"] for row in rows)),
        "contract_type_counts": dict(Counter(row["contract_type"] for row in rows)),
        "guard_action_counts": dict(Counter(row["guard_action"] for row in rows)),
        "rescue_trigger_counts": dict(Counter(reason for row in rows for reason in row["rescue_trigger_reason"])),
        "rows": rows,
    }


def md_group_table(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| Group | N | Mean F1 Delta | Wins | Losses | Ties | Mean Contract Gap | Mean Router P |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in groups.items():
        lines.append(
            f"| {key} | {value['n']} | {fmt(value['mean_f1_delta'])} | {value['wins']} | {value['losses']} | {value['ties']} | {fmt(value['mean_contract_gap'])} | {fmt(value['mean_router_probability'])} |"
        )
    return lines + [""]


def write_report(payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    lines = [
        "# Evidence-Contract Depth Audit on Mix130",
        "",
        "> Offline mechanism audit; no API calls. Scope: frozen SAGE v33-preservation aligned against full_v2_rescue.",
        "",
        "## Core Question",
        "",
        "This audit tests whether one evidence-contract signal can explain the system's routing and rescue behavior: rows with larger contract deficit should be more likely to trigger verification/rescue and should show larger gains over the VSR rescue layer.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Aligned rows | {payload['n_aligned']} |",
        f"| Mean SAGE-vs-VSR F1 delta | {fmt(overall['mean_f1_delta'])} |",
        f"| Wins / losses / ties | {overall['wins']} / {overall['losses']} / {overall['ties']} |",
        f"| Mean contract gap | {fmt(overall['mean_contract_gap'])} |",
        f"| Mean router probability | {fmt(overall['mean_router_probability'])} |",
        f"| Graph rescue accepted | {overall['graph_rescue_accepted']} |",
        f"| Probability vs F1-delta correlation | {fmt(payload['router_probability_vs_f1_delta_correlation'])} |",
        f"| Contract-gap vs F1-delta correlation | {fmt(payload['contract_gap_vs_f1_delta_correlation'])} |",
        "",
    ]
    lines += md_group_table("Route Selection", payload["by_route"])
    lines += md_group_table("Contract Type", payload["by_contract_type"])
    lines += md_group_table("Router Probability Bucket", payload["by_route_bucket"])
    lines += [
        "## Interpretation",
        "",
        "- These groups are observational, so the tables describe mechanism behavior rather than randomized causal effects.",
        "- The primary depth test is whether route-selected rows have larger SAGE-vs-VSR deltas than route-skipped rows and whether the learned probability is aligned with that delta.",
        "- Contract-type breakdowns show whether the same mechanism behaves consistently across general, mechanism, enumeration, numeric, and compound questions.",
        "- Guard actions are included to separate routing value from repair-preservation value; a route can be useful even when the final guard keeps the original answer.",
        "",
        "## Counts",
        "",
        f"- Router actions: `{payload['route_action_counts']}`",
        f"- Contract types: `{payload['contract_type_counts']}`",
        f"- Guard actions: `{payload['guard_action_counts']}`",
        f"- Rescue triggers: `{payload['rescue_trigger_counts']}`",
        "",
        "## Reproducibility",
        "",
        f"- SAGE input: `{SAGE_PATH.relative_to(ROOT)}`",
        f"- VSR input: `{VSR_PATH.relative_to(ROOT)}`",
        f"- JSON payload: `{OUT_JSON.relative_to(ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    payload = build_payload(rows)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps(payload["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
