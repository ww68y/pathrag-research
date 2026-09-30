#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize cross-dataset rescue quality flags and simple gate candidates."""
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_enhanced_experiments import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    paired_t_test,
    wilcoxon_signed_rank_test,
)
from experiments.run_repair_v2_postverify import (  # noqa: E402
    detect_abstention_language,
    detect_meta_audit_leak,
)

BASE_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
OUTPUT_JSON = BASE_DIR / "cross_dataset_quality_flags_and_gates.json"
OUTPUT_MD = BASE_DIR / "CROSS_DATASET_QUALITY_FLAGS_AND_GATES.md"

DATASETS = [
    {"name": "UltraDomain mix", "slug": "ultradomain_mix", "dir": BASE_DIR, "n": 130},
    {"name": "Biology full context", "slug": "biology_fullcontext_50", "dir": BASE_DIR / "biology_fullcontext_50", "n": 50},
    {"name": "Legal full context", "slug": "legal_fullcontext_20", "dir": BASE_DIR / "legal_fullcontext_20", "n": 20},
    {"name": "Domain expansion windowed", "slug": "domain_expansion_windowed_50", "dir": BASE_DIR / "domain_expansion_windowed_50", "n": 50},
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(config_name: str) -> str:
    return config_name.replace("/", "_")


def result_file(dataset: Dict[str, Any], config_name: str) -> Path:
    return dataset["dir"] / f"{safe_name(config_name)}_aligned_n{dataset['n']}.json"


def load_rows(dataset: Dict[str, Any], config_name: str) -> List[Dict[str, Any]]:
    path = result_file(dataset, config_name)
    if not path.exists():
        return []
    return [row for row in load_json(path).get("results", []) if "error" not in row]


def by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["question_id"]: row for row in rows if "error" not in row}


def metric(row: Dict[str, Any], name: str, default: float = 0.0) -> float:
    return float(row.get("metrics", {}).get(name, default) or default)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_size": len(rows),
        "mean_f1": mean([metric(row, "f1") for row in rows]),
        "mean_rouge_l": mean([metric(row, "rouge_l") for row in rows]),
        "mean_faithfulness": mean([metric(row, "faithfulness") for row in rows]),
        "repair_rate": mean([1.0 if row.get("metrics", {}).get("repair_triggered") else 0.0 for row in rows]),
        "rescue_rate": mean([1.0 if row.get("metrics", {}).get("abstention_rescue_applied") else 0.0 for row in rows]),
        "abstention_rate": mean([1.0 if detect_abstention_language(row.get("prediction", "")) else 0.0 for row in rows]),
        "meta_leak_rate": mean([1.0 if detect_meta_audit_leak(row.get("prediction", "")) else 0.0 for row in rows]),
        "low_faithfulness_count": sum(1 for row in rows if metric(row, "faithfulness", 1.0) < 0.9),
    }


def paired_stats(left_rows: List[Dict[str, Any]], right_rows: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    left_by_id = by_id(left_rows)
    right_by_id = by_id(right_rows)
    ids = sorted(set(left_by_id) & set(right_by_id))
    if not ids:
        return {}
    left = [metric(left_by_id[qid], metric_name) for qid in ids]
    right = [metric(right_by_id[qid], metric_name) for qid in ids]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "n": len(ids),
        "mean_delta": mean(diffs),
        "wins": sum(1 for diff in diffs if diff > 1e-12),
        "losses": sum(1 for diff in diffs if diff < -1e-12),
        "ties": sum(1 for diff in diffs if abs(diff) <= 1e-12),
        "t_p_value": t_p,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def replace_with_full_v2(rescue_row: Dict[str, Any], full_v2_row: Dict[str, Any], policy_name: str) -> Dict[str, Any]:
    row = deepcopy(full_v2_row)
    row["method"] = policy_name
    row.setdefault("metrics", {})["abstention_rescue_applied"] = False
    row["metrics"]["secondary_gate_rejected_rescue"] = True
    return row


def keep_rescue(rescue_row: Dict[str, Any], policy_name: str) -> Dict[str, Any]:
    row = deepcopy(rescue_row)
    row["method"] = policy_name
    row.setdefault("metrics", {})["secondary_gate_rejected_rescue"] = False
    return row


def build_policy_rows(
    rescue_rows: List[Dict[str, Any]],
    full_v2_rows: List[Dict[str, Any]],
    policy_name: str,
    reject_rescue: Callable[[Dict[str, Any], Dict[str, Any]], bool],
) -> List[Dict[str, Any]]:
    full_v2_by_id = by_id(full_v2_rows)
    rows = []
    for rescue_row in rescue_rows:
        full_v2_row = full_v2_by_id.get(rescue_row["question_id"])
        if full_v2_row and reject_rescue(rescue_row, full_v2_row):
            rows.append(replace_with_full_v2(rescue_row, full_v2_row, policy_name))
        else:
            rows.append(keep_rescue(rescue_row, policy_name))
    return rows


def candidate_policies(rescue_rows: List[Dict[str, Any]], full_v2_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "current_full_v2_rescue": [keep_rescue(row, "current_full_v2_rescue") for row in rescue_rows],
        "gate_reject_low_faith_rescue": build_policy_rows(
            rescue_rows,
            full_v2_rows,
            "gate_reject_low_faith_rescue",
            lambda rescue, _base: bool(rescue.get("metrics", {}).get("abstention_rescue_applied"))
            and metric(rescue, "faithfulness", 1.0) < 0.9,
        ),
        "gate_reject_still_abstains": build_policy_rows(
            rescue_rows,
            full_v2_rows,
            "gate_reject_still_abstains",
            lambda rescue, _base: bool(rescue.get("metrics", {}).get("abstention_rescue_applied"))
            and detect_abstention_language(rescue.get("prediction", "")),
        ),
        "gate_reject_low_faith_or_still_abstains": build_policy_rows(
            rescue_rows,
            full_v2_rows,
            "gate_reject_low_faith_or_still_abstains",
            lambda rescue, _base: bool(rescue.get("metrics", {}).get("abstention_rescue_applied"))
            and (metric(rescue, "faithfulness", 1.0) < 0.9 or detect_abstention_language(rescue.get("prediction", ""))),
        ),
        "oracle_keep_higher_f1": build_policy_rows(
            rescue_rows,
            full_v2_rows,
            "oracle_keep_higher_f1",
            lambda rescue, base: metric(rescue, "f1") < metric(base, "f1"),
        ),
    }


def flagged_rows(dataset: Dict[str, Any], rescue_rows: List[Dict[str, Any]], full_v2_rows: List[Dict[str, Any]], no_repair_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    full_v2_by_id = by_id(full_v2_rows)
    no_repair_by_id = by_id(no_repair_rows)
    flags = []
    for row in rescue_rows:
        qid = row["question_id"]
        base = full_v2_by_id.get(qid, {})
        no_repair = no_repair_by_id.get(qid, {})
        delta_full_v2 = metric(row, "f1") - metric(base, "f1") if base else 0.0
        delta_no_repair = metric(row, "f1") - metric(no_repair, "f1") if no_repair else 0.0
        row_flags = []
        if metric(row, "faithfulness", 1.0) < 0.9:
            row_flags.append("low_faithfulness")
        if detect_abstention_language(row.get("prediction", "")):
            row_flags.append("final_abstention")
        if detect_meta_audit_leak(row.get("prediction", "")):
            row_flags.append("meta_audit_leak")
        if row.get("metrics", {}).get("abstention_rescue_applied") and delta_full_v2 < -1e-12:
            row_flags.append("rescue_worse_than_full_v2")
        if delta_no_repair < -1e-12:
            row_flags.append("worse_than_no_repair")
        if not row_flags:
            continue
        flags.append(
            {
                "dataset": dataset["name"],
                "dataset_slug": dataset["slug"],
                "question_id": qid,
                "domain": row.get("domain", ""),
                "question": row.get("question", ""),
                "flags": row_flags,
                "rescue_applied": bool(row.get("metrics", {}).get("abstention_rescue_applied")),
                "full_v2_rescue_f1": metric(row, "f1"),
                "full_v2_f1": metric(base, "f1") if base else None,
                "no_repair_f1": metric(no_repair, "f1") if no_repair else None,
                "delta_vs_full_v2": delta_full_v2,
                "delta_vs_no_repair": delta_no_repair,
                "faithfulness": metric(row, "faithfulness", 0.0),
                "abstention": detect_abstention_language(row.get("prediction", "")),
                "prediction_preview": " ".join(str(row.get("prediction", "")).split())[:240],
            }
        )
    return sorted(flags, key=lambda item: (item["delta_vs_full_v2"], item["faithfulness"]))


def summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    rescue_rows = load_rows(dataset, "full_v2_rescue")
    full_v2_rows = load_rows(dataset, "full_v2")
    no_repair_rows = load_rows(dataset, "w/o_repair")
    if not rescue_rows or not full_v2_rows:
        return {"name": dataset["name"], "available": False}
    policies = candidate_policies(rescue_rows, full_v2_rows)
    policy_summary = {
        name: {
            "aggregated": summarize_rows(rows),
            "vs_current_f1": paired_stats(rows, policies["current_full_v2_rescue"], "f1") if name != "current_full_v2_rescue" else {},
            "rejected_rescue_count": sum(1 for row in rows if row.get("metrics", {}).get("secondary_gate_rejected_rescue")),
        }
        for name, rows in policies.items()
    }
    return {
        "name": dataset["name"],
        "available": True,
        "n": len(rescue_rows),
        "result_dir": str(dataset["dir"]),
        "flagged_rows": flagged_rows(dataset, rescue_rows, full_v2_rows, no_repair_rows),
        "policy_summary": policy_summary,
    }


def write_report(payload: Dict[str, Any]) -> None:
    lines = [
        "# Cross-Dataset Quality Flags and Gate Diagnostics",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Flag Counts",
        "",
        "| Dataset | n | Flagged | Low Faith | Final Abstain | Rescue Worse | Worse Than No Repair | Meta Leak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slug, data in payload["datasets"].items():
        if not data.get("available"):
            lines.append(f"| {data['name']} | 0 | missing | missing | missing | missing | missing | missing |")
            continue
        flags = data["flagged_rows"]
        count = lambda name: sum(1 for item in flags if name in item["flags"])
        lines.append(
            f"| {data['name']} | {data['n']} | {len(flags)} | {count('low_faithfulness')} | "
            f"{count('final_abstention')} | {count('rescue_worse_than_full_v2')} | "
            f"{count('worse_than_no_repair')} | {count('meta_audit_leak')} |"
        )

    lines.extend([
        "",
        "## Simple Gate Candidates vs Current full_v2_rescue",
        "",
        "| Dataset | Candidate | F1 | Delta F1 | t-p | Wilcoxon-p | d | Rejected Rescue | LowFaith | Abstain% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for data in payload["datasets"].values():
        if not data.get("available"):
            continue
        for name, summary in data["policy_summary"].items():
            agg = summary["aggregated"]
            stats = summary["vs_current_f1"]
            if name == "current_full_v2_rescue":
                lines.append(
                    f"| {data['name']} | {name} | {agg['mean_f1']:.4f} | baseline | baseline | baseline | baseline | "
                    f"0 | {agg['low_faithfulness_count']} | {agg['abstention_rate']:.1%} |"
                )
                continue
            lines.append(
                f"| {data['name']} | {name} | {agg['mean_f1']:.4f} | {stats.get('mean_delta', 0):+.4f} | "
                f"{stats.get('t_p_value', 1):.4g} | {stats.get('wilcoxon_p_value', 1):.4g} | "
                f"{stats.get('cohens_d', 0):+.2f} | {summary['rejected_rescue_count']} | "
                f"{agg['low_faithfulness_count']} | {agg['abstention_rate']:.1%} |"
            )

    lines.extend([
        "",
        "## Highest-Priority Flagged Rows",
        "",
        "| Dataset | ID | Domain | Flags | Rescue F1 | Full_v2 F1 | Delta | Faith | Preview |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ])
    all_flags = []
    for data in payload["datasets"].values():
        if data.get("available"):
            all_flags.extend(data["flagged_rows"])
    all_flags = sorted(all_flags, key=lambda item: ("rescue_worse_than_full_v2" not in item["flags"], item["delta_vs_full_v2"]))
    for item in all_flags[:40]:
        preview = item["prediction_preview"].replace("|", "/")
        full_v2_f1 = item["full_v2_f1"] if item["full_v2_f1"] is not None else 0.0
        lines.append(
            f"| {item['dataset']} | {item['question_id']} | {item['domain']} | {', '.join(item['flags'])} | "
            f"{item['full_v2_rescue_f1']:.4f} | {full_v2_f1:.4f} | {item['delta_vs_full_v2']:+.4f} | "
            f"{item['faithfulness']:.2f} | {preview} |"
        )

    lines.extend([
        "",
        "## Takeaways",
        "",
        "- Gate candidates are diagnostic: the oracle uses gold F1 and is not deployable.",
        "- A deployable gate should reduce low-faithfulness and residual-abstention rows without sacrificing the paired F1 gains.",
        "- Rows listed above are the best targets for LLM audit or a forced-context diagnostic rerun.",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {dataset["slug"]: summarize_dataset(dataset) for dataset in DATASETS},
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
