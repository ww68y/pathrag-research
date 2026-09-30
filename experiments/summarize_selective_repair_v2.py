#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize deployable selective repair-v2 on the aligned 130-question set.

This is an offline consolidation step. It keeps the aligned full/no-repair
answer for rows where verification did not trigger repair, and substitutes the
repair_prompt_v2 answer only for rows where the aligned full branch triggered
repair. That matches the selective repair protocol without spending more API
calls.
"""
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_enhanced_experiments import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    paired_t_test,
    wilcoxon_signed_rank_test,
)


ALIGNED_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
V2_DIR = ROOT_DIR / "results" / "repair_v2_postverify"
NO_REPAIR_FILE = ALIGNED_DIR / "w_o_repair_aligned_n130.json"
FULL_V1_FILE = ALIGNED_DIR / "full_aligned_n130.json"
TRIGGERED_V2_FILE = V2_DIR / "repair_v2_postverify_triggered_n45.json"
OUTPUT_JSON = V2_DIR / "selective_repair_v2_aligned_n130.json"
OUTPUT_MD = V2_DIR / "selective_repair_v2_aligned_n130.md"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def paired_stats(rows: List[Dict[str, Any]], left_key: str, right_key: str) -> Dict[str, Any]:
    left = [float(row["metrics"][left_key]) for row in rows]
    right = [float(row["metrics"][right_key]) for row in rows]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "left": left_key,
        "right": right_key,
        "n": len(rows),
        "mean_diff": mean(diffs),
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs) if len(diffs) >= 2 else (0.0, 0.0),
    }


def build_rows() -> List[Dict[str, Any]]:
    no_rows = load_json(NO_REPAIR_FILE)["results"]
    full_rows = load_json(FULL_V1_FILE)["results"]
    v2_rows = load_json(TRIGGERED_V2_FILE)["results"]

    full_by_id = {row["question_id"]: row for row in full_rows}
    v2_by_id = {row["question_id"]: row for row in v2_rows if "error" not in row}
    triggered_ids = [row["question_id"] for row in full_rows if row["metrics"].get("repair_triggered")]
    missing = [qid for qid in triggered_ids if qid not in v2_by_id]
    if missing:
        raise RuntimeError(f"Missing repair-v2 rows for triggered ids: {missing[:10]}")

    rows: List[Dict[str, Any]] = []
    for no_row in no_rows:
        qid = no_row["question_id"]
        full_row = full_by_id[qid]
        no_m = no_row["metrics"]
        full_m = full_row["metrics"]
        repair_triggered = bool(full_m.get("repair_triggered"))

        if repair_triggered:
            v2_row = v2_by_id[qid]
            v2_m = v2_row["metrics"]
            prediction = v2_row["v2_prediction"]
            f1 = float(v2_m["v2_f1"])
            rouge_l = float(v2_m["v2_rouge_l"])
            exact_match = float(v2_m.get("v2_exact_match", 0.0))
            contains = float(v2_m.get("v2_contains", 0.0))
            faithfulness = float(v2_m.get("post_repair_faithfulness", 0.0))
            coverage = float(v2_m.get("post_repair_coverage", 0.0))
            meta_audit_leak = bool(v2_m.get("meta_audit_leak", False))
            abstention_language = bool(v2_m.get("abstention_language", False))
            runtime_seconds = float(v2_m.get("runtime_seconds", 0.0))
            branch = "repair_prompt_v2"
        else:
            prediction = full_row.get("prediction", no_row.get("prediction", ""))
            f1 = float(full_m["f1"])
            rouge_l = float(full_m["rouge_l"])
            exact_match = float(full_m.get("exact_match", 0.0))
            contains = float(full_m.get("contains", 0.0))
            faithfulness = float(full_m.get("faithfulness", 0.0))
            coverage = 1.0 if faithfulness >= 0.999 else faithfulness
            meta_audit_leak = False
            abstention_language = False
            runtime_seconds = float(full_m.get("runtime_seconds", 0.0))
            branch = "aligned_full_no_repair_needed"

        rows.append(
            {
                "question_id": qid,
                "question": no_row["question"],
                "gold_answer": no_row["gold_answer"],
                "domain": no_row.get("domain", ""),
                "dataset": no_row.get("dataset", "ultradomain"),
                "prediction": prediction,
                "source_branch": branch,
                "metrics": {
                    "f1": f1,
                    "rouge_l": rouge_l,
                    "exact_match": exact_match,
                    "contains": contains,
                    "faithfulness": faithfulness,
                    "post_or_original_faithfulness": faithfulness,
                    "coverage": coverage,
                    "runtime_seconds": runtime_seconds,
                    "repair_triggered": repair_triggered,
                    "repair_performed": repair_triggered,
                    "repair_prompt_version": "v2" if repair_triggered else "none",
                    "meta_audit_leak": meta_audit_leak,
                    "abstention_language": abstention_language,
                    "old_no_repair_f1": float(no_m["f1"]),
                    "old_full_f1": float(full_m["f1"]),
                    "old_no_repair_rouge_l": float(no_m["rouge_l"]),
                    "old_full_rouge_l": float(full_m["rouge_l"]),
                    "delta_vs_no_repair_f1": f1 - float(no_m["f1"]),
                    "delta_vs_v1_full_f1": f1 - float(full_m["f1"]),
                    "delta_vs_no_repair_rouge_l": rouge_l - float(no_m["rouge_l"]),
                    "delta_vs_v1_full_rouge_l": rouge_l - float(full_m["rouge_l"]),
                },
            }
        )
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    triggered = [row for row in rows if row["metrics"]["repair_triggered"]]
    non_triggered = [row for row in rows if not row["metrics"]["repair_triggered"]]
    return {
        "sample_size": len(rows),
        "repair_v2_applied_count": len(triggered),
        "non_triggered_passthrough_count": len(non_triggered),
        "mean_f1": mean([m["f1"] for m in metrics]),
        "mean_rouge_l": mean([m["rouge_l"] for m in metrics]),
        "mean_faithfulness": mean([m["faithfulness"] for m in metrics]),
        "mean_coverage": mean([m["coverage"] for m in metrics]),
        "mean_old_no_repair_f1": mean([m["old_no_repair_f1"] for m in metrics]),
        "mean_old_full_f1": mean([m["old_full_f1"] for m in metrics]),
        "mean_delta_vs_no_repair_f1": mean([m["delta_vs_no_repair_f1"] for m in metrics]),
        "mean_delta_vs_v1_full_f1": mean([m["delta_vs_v1_full_f1"] for m in metrics]),
        "mean_delta_vs_no_repair_rouge_l": mean([m["delta_vs_no_repair_rouge_l"] for m in metrics]),
        "mean_delta_vs_v1_full_rouge_l": mean([m["delta_vs_v1_full_rouge_l"] for m in metrics]),
        "improved_vs_no_repair": sum(1 for m in metrics if m["delta_vs_no_repair_f1"] > 1e-9),
        "worse_vs_no_repair": sum(1 for m in metrics if m["delta_vs_no_repair_f1"] < -1e-9),
        "improved_vs_v1_full": sum(1 for m in metrics if m["delta_vs_v1_full_f1"] > 1e-9),
        "worse_vs_v1_full": sum(1 for m in metrics if m["delta_vs_v1_full_f1"] < -1e-9),
        "triggered_improved_vs_v1_full": sum(
            1 for row in triggered if row["metrics"]["delta_vs_v1_full_f1"] > 1e-9
        ),
        "triggered_worse_vs_v1_full": sum(
            1 for row in triggered if row["metrics"]["delta_vs_v1_full_f1"] < -1e-9
        ),
        "meta_audit_leak_rate": mean([1.0 if m["meta_audit_leak"] else 0.0 for m in metrics]),
        "abstention_language_rate": mean([1.0 if m["abstention_language"] else 0.0 for m in metrics]),
    }


def write_report(payload: Dict[str, Any]) -> None:
    rows = payload["results"]
    agg = payload["aggregated"]
    stats = payload["stat_tests"]
    gains = sorted(rows, key=lambda r: r["metrics"]["delta_vs_v1_full_f1"], reverse=True)[:10]
    losses = sorted(rows, key=lambda r: r["metrics"]["delta_vs_v1_full_f1"])[:10]

    lines = [
        "# Selective Repair Prompt V2 on Aligned 130",
        "",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "> Protocol: use repair_prompt_v2 only on aligned v1 repair-triggered rows; pass through non-triggered rows.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sample size | {agg['sample_size']} |",
        f"| V2 repair applied | {agg['repair_v2_applied_count']} |",
        f"| Non-triggered passthrough | {agg['non_triggered_passthrough_count']} |",
        f"| Selective-v2 F1 | {agg['mean_f1']:.4f} |",
        f"| Old no-repair F1 | {agg['mean_old_no_repair_f1']:.4f} |",
        f"| Old v1 full F1 | {agg['mean_old_full_f1']:.4f} |",
        f"| Delta vs no-repair F1 | {agg['mean_delta_vs_no_repair_f1']:+.4f} |",
        f"| Delta vs v1 full F1 | {agg['mean_delta_vs_v1_full_f1']:+.4f} |",
        f"| Selective-v2 ROUGE-L | {agg['mean_rouge_l']:.4f} |",
        f"| Delta vs no-repair ROUGE-L | {agg['mean_delta_vs_no_repair_rouge_l']:+.4f} |",
        f"| Delta vs v1 full ROUGE-L | {agg['mean_delta_vs_v1_full_rouge_l']:+.4f} |",
        f"| Faithfulness | {agg['mean_faithfulness']:.4f} |",
        f"| Coverage | {agg['mean_coverage']:.4f} |",
        f"| Meta-audit leak rate | {agg['meta_audit_leak_rate']:.1%} |",
        f"| Abstention-language rate | {agg['abstention_language_rate']:.1%} |",
        f"| Improved / worse vs v1 full | {agg['improved_vs_v1_full']} / {agg['worse_vs_v1_full']} |",
        f"| Triggered improved / worse vs v1 full | {agg['triggered_improved_vs_v1_full']} / {agg['triggered_worse_vs_v1_full']} |",
        "",
        "## Paired Tests",
        "",
        "| Comparison | n | Mean diff | t-p | Wilcoxon-p | Cohen's d | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in stats.items():
        ci = row.get("bootstrap_ci", [0.0, 0.0])
        lines.append(
            f"| {label} | {row['n']} | {row['mean_diff']:+.4f} | {row['t_p_value']:.4f} | "
            f"{row['wilcoxon_p_value']:.4f} | {row['cohens_d']:+.2f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )

    lines.extend([
        "",
        "## Top Gains vs V1 Full",
        "",
        "| Question | Domain | Delta | Old Full F1 | Selective-v2 F1 | Branch |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in gains:
        m = row["metrics"]
        q = row["question"].replace("|", "/")[:90]
        lines.append(
            f"| {row['question_id']}: {q} | {row['domain']} | {m['delta_vs_v1_full_f1']:+.4f} | "
            f"{m['old_full_f1']:.4f} | {m['f1']:.4f} | {row['source_branch']} |"
        )

    lines.extend([
        "",
        "## Top Losses vs V1 Full",
        "",
        "| Question | Domain | Delta | Old Full F1 | Selective-v2 F1 | Branch | Abstain |",
        "|---|---|---:|---:|---:|---|---:|",
    ])
    for row in losses:
        m = row["metrics"]
        q = row["question"].replace("|", "/")[:90]
        lines.append(
            f"| {row['question_id']}: {q} | {row['domain']} | {m['delta_vs_v1_full_f1']:+.4f} | "
            f"{m['old_full_f1']:.4f} | {m['f1']:.4f} | {row['source_branch']} | {m['abstention_language']} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This report is the deployable selective protocol; it does not rewrite rows where repair was not triggered.",
        "- Faithfulness mixes post-repair verification for v2 rows with the original verification score for passthrough rows.",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    payload = {
        "experiment": "selective_repair_prompt_v2_aligned_130",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "no_repair_source": str(NO_REPAIR_FILE.relative_to(ROOT_DIR)),
            "v1_full_source": str(FULL_V1_FILE.relative_to(ROOT_DIR)),
            "v2_triggered_source": str(TRIGGERED_V2_FILE.relative_to(ROOT_DIR)),
            "description": "repair_prompt_v2 replaces v1 full only for aligned repair-triggered rows",
        },
        "aggregated": aggregate(rows),
        "stat_tests": {
            "selective_v2_vs_no_repair_f1": paired_stats(rows, "f1", "old_no_repair_f1"),
            "selective_v2_vs_v1_full_f1": paired_stats(rows, "f1", "old_full_f1"),
            "selective_v2_vs_no_repair_rouge_l": paired_stats(rows, "rouge_l", "old_no_repair_rouge_l"),
            "selective_v2_vs_v1_full_rouge_l": paired_stats(rows, "rouge_l", "old_full_rouge_l"),
        },
        "results": rows,
    }
    write_json(OUTPUT_JSON, payload)
    write_report(payload)
    agg = payload["aggregated"]
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(
        "selective_v2 "
        f"F1={agg['mean_f1']:.4f} "
        f"delta_vs_no_repair={agg['mean_delta_vs_no_repair_f1']:+.4f} "
        f"delta_vs_v1_full={agg['mean_delta_vs_v1_full_f1']:+.4f} "
        f"repair_applied={agg['repair_v2_applied_count']}"
    )


if __name__ == "__main__":
    main()
