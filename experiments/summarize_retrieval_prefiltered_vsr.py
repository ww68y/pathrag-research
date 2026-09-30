#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize retrieval-prefiltered VSR diagnostic runs."""
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_enhanced_experiments import bootstrap_ci, cohens_d, paired_t_test, wilcoxon_signed_rank_test  # noqa: E402
from experiments.summarize_retrieval_baselines import enrich_baseline_rows, load_jsonl  # noqa: E402

RESULTS_DIR = ROOT_DIR / "results"
PREFILTER_DIR = RESULTS_DIR / "retrieval_prefiltered_vsr"
BASELINE_DIR = RESULTS_DIR / "retrieval_baselines" / "legal_fullcontext_20"
FULL_CONTEXT_DIR = RESULTS_DIR / "enhanced_ablation_aligned" / "legal_fullcontext_20"
OUTPUT_JSON = PREFILTER_DIR / "retrieval_prefiltered_vsr_summary.json"
OUTPUT_MD = PREFILTER_DIR / "RETRIEVAL_PREFILTERED_VSR_SUMMARY.md"

RUNS = [
    {
        "name": "legal_retrieved_top5_default",
        "dir": PREFILTER_DIR / "legal_retrieved_top5_20",
        "n": 20,
        "note": "Top-5 retrieved context with default aligned 3.5k/3k context windows.",
    },
    {
        "name": "legal_retrieved_top5_ctx12k",
        "dir": PREFILTER_DIR / "legal_retrieved_top5_20_ctx12k",
        "n": 20,
        "note": "Top-5 retrieved context with 12k evidence/answer windows; resumed after 402 quota interruption and completed at N=20.",
    },
    {
        "name": "legal_retrieved_top5_prepended_full",
        "dir": PREFILTER_DIR / "legal_retrieved_top5_prepended_full_20",
        "n": 20,
        "note": "Top-5 retrieved chunks prepended before the full original context; tests retrieval-guided ordering without discarding source text.",
    },
]
METHODS = ["w/o_repair", "full_v2", "full_v2_rescue"]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(name: str) -> str:
    return name.replace("/", "_")


def load_vsr_rows(run_dir: Path, method: str, n: int) -> List[Dict[str, Any]]:
    path = run_dir / f"{safe_name(method)}_aligned_n{n}.json"
    if not path.exists():
        return []
    return [row for row in load_json(path).get("results", []) if "error" not in row]


def metric(row: Dict[str, Any], name: str) -> float:
    return float(row.get("metrics", {}).get(name, 0.0) or 0.0)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_size": len(rows),
        "mean_f1": mean([metric(row, "f1") for row in rows]),
        "mean_rouge_l": mean([metric(row, "rouge_l") for row in rows]),
        "mean_faithfulness": mean([metric(row, "faithfulness") for row in rows]),
        "mean_runtime_seconds": mean([metric(row, "runtime_seconds") for row in rows]),
        "repair_rate": mean([1.0 if row.get("metrics", {}).get("repair_triggered") else 0.0 for row in rows]),
        "rescue_rate": mean([1.0 if row.get("metrics", {}).get("abstention_rescue_applied") else 0.0 for row in rows]),
    }


def by_question(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row.get("question", ""): row for row in rows}


def compare(left_rows: List[Dict[str, Any]], right_rows: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    left_by_q = by_question(left_rows)
    right_by_q = by_question(right_rows)
    questions = sorted(set(left_by_q) & set(right_by_q))
    if not questions:
        return {}
    left = [metric(left_by_q[q], metric_name) for q in questions]
    right = [metric(right_by_q[q], metric_name) for q in questions]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "n": len(questions),
        "mean_delta": mean(diffs),
        "wins": sum(1 for diff in diffs if diff > 1e-12),
        "losses": sum(1 for diff in diffs if diff < -1e-12),
        "ties": sum(1 for diff in diffs if abs(diff) <= 1e-12),
        "t_p_value": t_p,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def baseline_rows() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "naive": enrich_baseline_rows(load_jsonl(BASELINE_DIR / "naive_aligned_n20.jsonl"), "naive"),
        "hyde": enrich_baseline_rows(load_jsonl(BASELINE_DIR / "hyde_aligned_n20.jsonl"), "hyde"),
        "full_context_full_v2_rescue": load_vsr_rows(FULL_CONTEXT_DIR, "full_v2_rescue", 20),
    }


def summarize() -> Dict[str, Any]:
    baselines = baseline_rows()
    payload: Dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
    for run in RUNS:
        methods = {method: load_vsr_rows(run["dir"], method, run["n"]) for method in METHODS}
        comparisons = {}
        candidate = methods.get("full_v2_rescue", [])
        if candidate:
            for baseline_name, rows in baselines.items():
                if rows:
                    comparisons[f"full_v2_rescue_minus_{baseline_name}"] = {
                        "f1": compare(candidate, rows, "f1"),
                        "rouge_l": compare(candidate, rows, "rouge_l"),
                    }
            for method in ["w/o_repair", "full_v2"]:
                if methods.get(method):
                    comparisons[f"full_v2_rescue_minus_{method}"] = {
                        "f1": compare(candidate, methods[method], "f1"),
                        "rouge_l": compare(candidate, methods[method], "rouge_l"),
                    }
        payload["runs"][run["name"]] = {
            "note": run["note"],
            "dir": str(run["dir"]),
            "aggregated": {method: aggregate(rows) for method, rows in methods.items() if rows},
            "comparisons": comparisons,
        }
    payload["baselines"] = {name: aggregate(rows) for name, rows in baselines.items() if rows}
    return payload


def write_report(payload: Dict[str, Any]) -> None:
    lines = [
        "# Retrieval-Prefiltered VSR Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Baselines",
        "",
        "| Method | n | F1 | ROUGE-L | Runtime(s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, agg in payload.get("baselines", {}).items():
        lines.append(f"| {name} | {agg['sample_size']} | {agg['mean_f1']:.4f} | {agg['mean_rouge_l']:.4f} | {agg['mean_runtime_seconds']:.1f} |")
    lines.extend(["", "## Prefiltered VSR Runs", "", "| Run | Method | n | F1 | ROUGE-L | Faith | Repair% | Rescue% | Runtime(s) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for run_name, run in payload["runs"].items():
        for method in METHODS:
            agg = run.get("aggregated", {}).get(method)
            if not agg:
                continue
            lines.append(
                f"| {run_name} | {method} | {agg['sample_size']} | {agg['mean_f1']:.4f} | {agg['mean_rouge_l']:.4f} | "
                f"{agg['mean_faithfulness']:.4f} | {agg['repair_rate']:.1%} | {agg['rescue_rate']:.1%} | {agg['mean_runtime_seconds']:.1f} |"
            )
    lines.extend(["", "## Candidate Deltas", "", "| Run | Comparison | Metric | n | Delta | Wins | Losses | Ties | t-p | d |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for run_name, run in payload["runs"].items():
        for comparison, metrics in run.get("comparisons", {}).items():
            for metric_name, stats in metrics.items():
                if not stats:
                    continue
                left, right = comparison.split("_minus_", 1)
                lines.append(
                    f"| {run_name} | {left} - {right} | {metric_name} | {stats['n']} | {stats['mean_delta']:+.4f} | "
                    f"{stats['wins']} | {stats['losses']} | {stats['ties']} | {stats['t_p_value']:.4g} | {stats['cohens_d']:+.2f} |"
                )
    ctx12k = payload.get("runs", {}).get("legal_retrieved_top5_ctx12k", {})
    ctx12k_rescue = ctx12k.get("aggregated", {}).get("full_v2_rescue", {})
    ctx12k_n = int(ctx12k_rescue.get("sample_size") or 0)
    prepended = payload.get("runs", {}).get("legal_retrieved_top5_prepended_full", {})
    prepended_rescue = prepended.get("aggregated", {}).get("full_v2_rescue", {})
    ctx12k_status = (
        "- The ctx12k run is now complete at N=20; it remains below Naive/HyDE and does not improve over the default retrieved-top5 setting."
        if ctx12k_n >= 20
        else "- The ctx12k run is incomplete; resume with --retry-errors once quota is restored."
    )
    prepended_status = (
        "- The top5+full prepend run is also negative: it preserves source text but still trails Naive/HyDE, full-context VSR, and the default top5-only VSR setting."
        if prepended_rescue
        else "- The top5+full prepend run has not been generated yet."
    )
    lines.extend([
        "",
        "## Takeaway",
        "",
        "- Top-k prefiltering alone did not solve the gap against Naive/HyDE on Legal.",
        ctx12k_status,
        prepended_status,
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
