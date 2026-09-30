#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run longer-context rescue diagnostics for audited cross-dataset quality flags."""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv  # noqa: E402

from experiments.audit_cross_dataset_quality_flags import DATASETS  # noqa: E402
from experiments.run_aligned_ablation_rerun import load_ultradomain_jsonl  # noqa: E402
from experiments.run_enhanced_experiments import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    get_api_config,
    paired_t_test,
    wilcoxon_signed_rank_test,
)
from experiments.run_full_v2_context_rescue import RUN_PROFILE, run_one  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

BASE_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
AUDIT_FILE = BASE_DIR / "cross_dataset_quality_flag_llm_audit.json"
LOGFILE = ROOT_DIR / "logs" / "cross_dataset_context_rescue_diagnostic.log"


def log(message: str) -> None:
    print(message, flush=True)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with LOGFILE.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(message + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_name(config_name: str) -> str:
    return config_name.replace("/", "_")


def result_file(dataset_slug: str, config_name: str) -> Path:
    dataset = DATASETS[dataset_slug]
    return dataset["dir"] / f"{safe_name(config_name)}_aligned_n{dataset['n']}.json"


def result_rows(dataset_slug: str, config_name: str) -> Dict[str, Dict[str, Any]]:
    return {
        row["question_id"]: row
        for row in load_json(result_file(dataset_slug, config_name)).get("results", [])
        if "error" not in row
    }


def question_rows(dataset_slug: str) -> Dict[str, Dict[str, Any]]:
    dataset = DATASETS[dataset_slug]
    return {
        row["question_id"]: row
        for row in load_ultradomain_jsonl(dataset["file"], dataset_slug)
    }


def metric(row: Dict[str, Any], name: str, default: float = 0.0) -> float:
    return float(row.get("metrics", {}).get(name, default) or default)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def select_targets(args: argparse.Namespace) -> List[Dict[str, Any]]:
    audit = load_json(AUDIT_FILE)
    rows = [row for row in audit.get("results", []) if "error" not in row]
    actions = set(args.actions)
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row.get("question_id") in wanted]
    else:
        rows = [row for row in rows if row.get("audit", {}).get("recommended_action") in actions]
    if not args.include_ultradomain:
        rows = [row for row in rows if row.get("dataset_slug") != "ultradomain_mix"]
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("audit", {}).get("recommended_action") != "rerun_with_more_context",
            row.get("delta_vs_full_v2", 0.0),
        ),
    )
    return rows[: args.limit] if args.limit else rows


def normalize_row(row: Dict[str, Any], target: Dict[str, Any], baseline: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    row["dataset"] = target.get("dataset", "")
    row["dataset_slug"] = target.get("dataset_slug", "")
    row["audit_issue"] = target.get("audit", {}).get("issue_type", "")
    row["audit_action"] = target.get("audit", {}).get("recommended_action", "")
    row["baseline_current_rescue_prediction"] = baseline.get("prediction", "")
    row["method"] = "cross_dataset_context_rescue_diagnostic"
    row["run_profile"] = profile
    metrics = row.setdefault("metrics", {})
    metrics["baseline_current_rescue_f1"] = metrics.pop("baseline_full_v2_f1", metric(baseline, "f1"))
    metrics["baseline_current_rescue_rouge_l"] = metrics.pop("baseline_full_v2_rouge_l", metric(baseline, "rouge_l"))
    metrics["baseline_current_rescue_faithfulness"] = metrics.pop(
        "baseline_full_v2_faithfulness",
        metric(baseline, "faithfulness"),
    )
    metrics["delta_vs_current_rescue_f1"] = metric(row, "f1") - metrics["baseline_current_rescue_f1"]
    metrics["delta_vs_current_rescue_rouge_l"] = metric(row, "rouge_l") - metrics["baseline_current_rescue_rouge_l"]
    return row


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    metrics = [row["metrics"] for row in valid]
    return {
        "sample_size": len(rows),
        "valid_sample_size": len(valid),
        "error_count": len(rows) - len(valid),
        "mean_f1": mean([m.get("f1", 0.0) for m in metrics]),
        "mean_baseline_current_rescue_f1": mean([m.get("baseline_current_rescue_f1", 0.0) for m in metrics]),
        "mean_delta_vs_current_rescue_f1": mean([m.get("delta_vs_current_rescue_f1", 0.0) for m in metrics]),
        "mean_rouge_l": mean([m.get("rouge_l", 0.0) for m in metrics]),
        "mean_delta_vs_current_rescue_rouge_l": mean([m.get("delta_vs_current_rescue_rouge_l", 0.0) for m in metrics]),
        "mean_faithfulness": mean([m.get("faithfulness", 0.0) for m in metrics]),
        "mean_coverage": mean([m.get("coverage", 0.0) for m in metrics]),
        "improved_vs_current_rescue": sum(1 for m in metrics if m.get("delta_vs_current_rescue_f1", 0.0) > 1e-12),
        "worse_vs_current_rescue": sum(1 for m in metrics if m.get("delta_vs_current_rescue_f1", 0.0) < -1e-12),
        "same_vs_current_rescue": sum(1 for m in metrics if abs(m.get("delta_vs_current_rescue_f1", 0.0)) <= 1e-12),
    }


def paired_stats(rows: List[Dict[str, Any]], metric_name: str, baseline_metric_name: str) -> Dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    if len(valid) < 2:
        return {}
    left = [metric(row, metric_name) for row in valid]
    right = [metric(row, baseline_metric_name) for row in valid]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "metric": metric_name,
        "baseline_metric": baseline_metric_name,
        "n": len(valid),
        "mean_diff": mean(diffs),
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    agg = payload["aggregated"]
    lines = [
        "# Cross-Dataset Longer-Context Rescue Diagnostic",
        "",
        f"> Generated: {payload['created_at']}",
        f"> Rows: {agg.get('valid_sample_size', 0)}/{payload['target_sample_size']}",
        f"> Evidence context chars: {payload['run_profile']['evidence_context_chars']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Error count | {agg.get('error_count', 0)} |",
        f"| Mean diagnostic F1 | {agg.get('mean_f1', 0):.4f} |",
        f"| Mean current rescue F1 | {agg.get('mean_baseline_current_rescue_f1', 0):.4f} |",
        f"| Delta vs current rescue F1 | {agg.get('mean_delta_vs_current_rescue_f1', 0):+.4f} |",
        f"| Delta vs current rescue ROUGE-L | {agg.get('mean_delta_vs_current_rescue_rouge_l', 0):+.4f} |",
        f"| Faithfulness | {agg.get('mean_faithfulness', 0):.4f} |",
        f"| Coverage | {agg.get('mean_coverage', 0):.4f} |",
        f"| Improved / worse / same | {agg.get('improved_vs_current_rescue', 0)} / {agg.get('worse_vs_current_rescue', 0)} / {agg.get('same_vs_current_rescue', 0)} |",
        "",
        "## Paired Tests",
        "",
        "| Metric | Mean Diff | t-p | Wilcoxon-p | Cohen's d | 95% Bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stats in payload.get("stat_tests", {}).values():
        if not stats:
            continue
        ci = stats["bootstrap_ci"]
        lines.append(
            f"| {stats['metric']} | {stats['mean_diff']:+.4f} | {stats['t_p_value']:.4g} | "
            f"{stats['wilcoxon_p_value']:.4g} | {stats['cohens_d']:+.2f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )
    lines.extend([
        "",
        "## Rows",
        "",
        "| Dataset | ID | Action | Current F1 | Diagnostic F1 | Delta | Faith | Coverage | Abstain |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(payload["results"], key=lambda item: item.get("metrics", {}).get("delta_vs_current_rescue_f1", -999), reverse=True):
        if "error" in row:
            lines.append(f"| {row.get('dataset', '')} | {row.get('question_id', '')} | error | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | False |")
            continue
        m = row["metrics"]
        lines.append(
            f"| {row.get('dataset', '')} | {row['question_id']} | {row.get('audit_action', '')} | "
            f"{m.get('baseline_current_rescue_f1', 0):.4f} | {m.get('f1', 0):.4f} | "
            f"{m.get('delta_vs_current_rescue_f1', 0):+.4f} | {m.get('faithfulness', 0):.2f} | "
            f"{m.get('coverage', 0):.2f} | {m.get('abstention_language', False)} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    profile = dict(RUN_PROFILE)
    profile["name"] = "cross_dataset_context_rescue_diagnostic"
    profile["evidence_context_chars"] = args.evidence_context_chars
    profile["answer_context_chars"] = args.answer_context_chars
    targets = select_targets(args)
    suffix = f"n{len(targets)}_ctx{args.evidence_context_chars}"
    output_file = BASE_DIR / f"cross_dataset_context_rescue_diagnostic_{suffix}.json"
    report_file = BASE_DIR / f"CROSS_DATASET_CONTEXT_RESCUE_DIAGNOSTIC_{suffix}.md"

    existing_by_key: Dict[str, Dict[str, Any]] = {}
    if output_file.exists() and not args.force:
        existing = load_json(output_file)
        existing_by_key = {
            f"{row.get('dataset_slug')}::{row.get('question_id')}": row
            for row in existing.get("results", [])
        }
        log(f"Loaded existing diagnostic rows: {len(existing_by_key)}/{len(targets)}")

    config = get_api_config()
    if not config.get("api_key"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    log("=" * 70)
    log(f"CROSS-DATASET CONTEXT RESCUE DIAGNOSTIC {suffix}")
    log("=" * 70)
    log(f"Model: {config.get('model')}")

    questions_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    baseline_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    rows_by_key = dict(existing_by_key)
    for index, target in enumerate(targets, 1):
        dataset_slug = target["dataset_slug"]
        qid = target["question_id"]
        key = f"{dataset_slug}::{qid}"
        existing = rows_by_key.get(key)
        if existing and ("error" not in existing or not args.retry_errors):
            log(f"[{index}/{len(targets)}] SKIP {qid}: already complete")
            continue
        questions_cache.setdefault(dataset_slug, question_rows(dataset_slug))
        baseline_cache.setdefault(dataset_slug, result_rows(dataset_slug, "full_v2_rescue"))
        question = questions_cache[dataset_slug][qid]
        baseline = baseline_cache[dataset_slug][qid]
        last_error: Optional[Exception] = None
        for attempt in range(1, args.max_retries + 1):
            try:
                log(f"[{index}/{len(targets)}] {target['dataset']} {qid}: {question['question'][:72]}...")
                start = time.time()
                row = await asyncio.wait_for(run_one(question, baseline, config, profile), timeout=args.question_timeout)
                row = normalize_row(row, target, baseline, profile)
                rows_by_key[key] = row
                m = row["metrics"]
                log(
                    f"  OK current={m['baseline_current_rescue_f1']:.3f} diag={m['f1']:.3f} "
                    f"delta={m['delta_vs_current_rescue_f1']:+.3f} faith={m['faithfulness']:.2f} "
                    f"time={time.time() - start:.1f}s"
                )
                break
            except Exception as exc:
                last_error = exc
                log(f"  ERROR attempt {attempt}/{args.max_retries}: {exc}")
                if attempt < args.max_retries:
                    await asyncio.sleep(5 * attempt)
        else:
            rows_by_key[key] = {
                "dataset": target.get("dataset", ""),
                "dataset_slug": dataset_slug,
                "question_id": qid,
                "audit_action": target.get("audit", {}).get("recommended_action", ""),
                "error": str(last_error),
                "metrics": {
                    "baseline_current_rescue_f1": metric(baseline, "f1"),
                    "delta_vs_current_rescue_f1": -metric(baseline, "f1"),
                },
            }

        ordered_rows = [rows_by_key[f"{target['dataset_slug']}::{target['question_id']}"] for target in targets if f"{target['dataset_slug']}::{target['question_id']}" in rows_by_key]
        payload = {
            "experiment": "cross_dataset_context_rescue_diagnostic",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_sample_size": len(targets),
            "target_actions": args.actions,
            "run_profile": profile,
            "aggregated": aggregate(ordered_rows),
            "stat_tests": {
                "f1": paired_stats(ordered_rows, "f1", "baseline_current_rescue_f1"),
                "rouge_l": paired_stats(ordered_rows, "rouge_l", "baseline_current_rescue_rouge_l"),
            },
            "results": ordered_rows,
        }
        write_json(output_file, payload)

    payload = load_json(output_file)
    payload["aggregated"] = aggregate(payload.get("results", []))
    payload["stat_tests"] = {
        "f1": paired_stats(payload.get("results", []), "f1", "baseline_current_rescue_f1"),
        "rouge_l": paired_stats(payload.get("results", []), "rouge_l", "baseline_current_rescue_rouge_l"),
    }
    write_json(output_file, payload)
    write_report(report_file, payload)
    log(f"Saved {output_file}")
    log(f"Report {report_file}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", nargs="+", default=["rerun_with_more_context", "improve_evidence_extraction"])
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--include-ultradomain", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--question-timeout", type=int, default=420)
    parser.add_argument("--evidence-context-chars", type=int, default=80000)
    parser.add_argument("--answer-context-chars", type=int, default=12000)
    args = parser.parse_args()
    LOGFILE.write_text("", encoding="utf-8")
    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
