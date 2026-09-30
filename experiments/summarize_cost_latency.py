from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cost_latency"
INSTRUMENTED_SAMPLE_PREFIXES = [
    ("Legal full-context cost sample", "legal_fullcontext_20"),
    ("Biology full-context cost sample", "biology_fullcontext_50"),
    ("Domain expansion windowed cost sample", "domain_expansion_windowed_50"),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def weighted_mean(rows: list[dict[str, Any]], value_key: str, weight_key: str = "valid_n") -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        value = row.get(value_key)
        weight = row.get(weight_key)
        if isinstance(value, (int, float)) and isinstance(weight, (int, float)) and weight > 0:
            weighted_sum += float(value) * float(weight)
            total_weight += float(weight)
    return weighted_sum / total_weight if total_weight else None


def mean_bool(rows: list[dict[str, Any]], *keys: str) -> float | None:
    if not rows:
        return None
    hits = 0
    for row in rows:
        trace = row.get("trace") or {}
        if any(bool(trace.get(key)) for key in keys):
            hits += 1
    return hits / len(rows)


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = (row.get("metrics") or {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def metric_sample_size(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if isinstance((row.get("metrics") or {}).get(key), (int, float)))


def metric_or_trace_rate(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        metrics = row.get("metrics") or {}
        trace = row.get("trace") or {}
        if not any(key in metrics or key in trace for key in keys):
            continue
        values.append(1.0 if any(bool(metrics.get(key, trace.get(key))) for key in keys) else 0.0)
    if not values:
        return None
    return sum(values) / len(values)


def rate_metric_true(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        metrics = row.get("metrics") or {}
        if key in metrics:
            values.append(1.0 if metrics.get(key) else 0.0)
    if not values:
        return None
    return sum(values) / len(values)


def read_vsr_run(label: str, method: str, path: Path) -> dict[str, Any]:
    data = load_json(path)
    rows = data.get("results") or []
    if not isinstance(rows, list):
        rows = []
    agg = data.get("aggregated") or {}
    valid_n = data.get("valid_sample_size") or agg.get("sample_size") or len(rows)
    target_n = data.get("target_sample_size") or data.get("sample_size") or valid_n
    return {
        "label": label,
        "method": method,
        "path": str(path.relative_to(ROOT)),
        "valid_n": int(valid_n or 0),
        "target_n": int(target_n or 0),
        "complete": bool(data.get("complete", True)),
        "mean_f1": agg.get("mean_f1"),
        "mean_rouge_l": agg.get("mean_rouge_l"),
        "mean_runtime_seconds": agg.get("mean_runtime_seconds"),
        "repair_rate": agg.get("repair_rate") if agg.get("repair_rate") is not None else mean_bool(rows, "repair_performed"),
        "rescue_rate": mean_bool(rows, "abstention_rescue_applied", "context_rescue_applied"),
        "rescue_trigger_rate": mean_bool(rows, "abstention_rescue_triggered", "context_rescue_triggered"),
        "abstention_rate": mean_bool(rows, "abstention_language"),
        "mean_llm_call_count": mean_metric(rows, "llm_call_count"),
        "llm_call_count_sample_size": metric_sample_size(rows, "llm_call_count"),
        "mean_llm_elapsed_seconds": mean_metric(rows, "llm_elapsed_seconds"),
        "llm_usage_available_rate": rate_metric_true(rows, "llm_usage_available"),
        "mean_llm_total_tokens": mean_metric(rows, "llm_total_tokens"),
        "error_count": int(data.get("error_count") or 0),
    }


def get_vsr_runs() -> list[dict[str, Any]]:
    specs = [
        (
            "UltraDomain mix",
            ROOT / "results" / "enhanced_ablation_aligned",
            130,
        ),
        (
            "Biology full context",
            ROOT / "results" / "enhanced_ablation_aligned" / "biology_fullcontext_50",
            50,
        ),
        (
            "Legal full context",
            ROOT / "results" / "enhanced_ablation_aligned" / "legal_fullcontext_20",
            20,
        ),
        (
            "Domain expansion windowed",
            ROOT / "results" / "enhanced_ablation_aligned" / "domain_expansion_windowed_50",
            50,
        ),
    ]
    method_files = {
        "w/o_repair": "w_o_repair_aligned_n{n}.json",
        "full_v2": "full_v2_aligned_n{n}.json",
        "full_v2_rescue": "full_v2_rescue_aligned_n{n}.json",
    }
    runs: list[dict[str, Any]] = []
    for label, directory, n in specs:
        for method, template in method_files.items():
            path = directory / template.format(n=n)
            if path.exists():
                runs.append(read_vsr_run(label, method, path))
    return runs


def summarize_internal_overhead(vsr_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for run in vsr_runs:
        by_label.setdefault(run["label"], {})[run["method"]] = run

    rows: list[dict[str, Any]] = []
    for label, methods in by_label.items():
        wo = methods.get("w/o_repair", {})
        full = methods.get("full_v2", {})
        rescue = methods.get("full_v2_rescue", {})
        rows.append(
            {
                "label": label,
                "valid_n": rescue.get("valid_n") or full.get("valid_n") or wo.get("valid_n"),
                "w_o_repair_runtime": wo.get("mean_runtime_seconds"),
                "full_v2_runtime": full.get("mean_runtime_seconds"),
                "full_v2_rescue_runtime": rescue.get("mean_runtime_seconds"),
                "rescue_vs_w_o_repair_runtime_ratio": ratio(
                    rescue.get("mean_runtime_seconds"), wo.get("mean_runtime_seconds")
                ),
                "rescue_vs_full_v2_runtime_ratio": ratio(
                    rescue.get("mean_runtime_seconds"), full.get("mean_runtime_seconds")
                ),
                "full_v2_rescue_f1": rescue.get("mean_f1"),
                "rescue_rate": rescue.get("rescue_rate"),
                "repair_rate": rescue.get("repair_rate"),
                "mean_llm_call_count": rescue.get("mean_llm_call_count"),
                "llm_call_count_sample_size": rescue.get("llm_call_count_sample_size"),
                "mean_llm_total_tokens": rescue.get("mean_llm_total_tokens"),
                "complete": rescue.get("complete", True),
            }
        )
    return rows


def summarize_retrieval_runtime() -> list[dict[str, Any]]:
    path = ROOT / "results" / "retrieval_baselines" / "retrieval_baseline_aligned_summary.json"
    if not path.exists():
        return []
    data = load_json(path)
    rows: list[dict[str, Any]] = []
    for slug, payload in (data.get("datasets") or {}).items():
        agg = payload.get("aggregated") or {}
        naive = agg.get("naive") or {}
        hyde = agg.get("hyde") or {}
        rescue = agg.get("full_v2_rescue") or {}
        best_retrieval_f1 = max(
            value
            for value in [naive.get("mean_f1"), hyde.get("mean_f1")]
            if value is not None
        )
        rows.append(
            {
                "slug": slug,
                "label": payload.get("name") or slug,
                "n": payload.get("n"),
                "naive_runtime": naive.get("mean_runtime_seconds"),
                "hyde_runtime": hyde.get("mean_runtime_seconds"),
                "full_v2_rescue_runtime": rescue.get("mean_runtime_seconds"),
                "vsr_vs_naive_runtime_ratio": ratio(
                    rescue.get("mean_runtime_seconds"), naive.get("mean_runtime_seconds")
                ),
                "vsr_vs_hyde_runtime_ratio": ratio(
                    rescue.get("mean_runtime_seconds"), hyde.get("mean_runtime_seconds")
                ),
                "naive_f1": naive.get("mean_f1"),
                "hyde_f1": hyde.get("mean_f1"),
                "full_v2_rescue_f1": rescue.get("mean_f1"),
                "vsr_minus_best_retrieval_f1": rescue.get("mean_f1") - best_retrieval_f1
                if rescue.get("mean_f1") is not None
                else None,
                "mean_num_chunks": naive.get("mean_num_chunks") or hyde.get("mean_num_chunks"),
            }
        )
    return rows


def get_prefiltered_runs() -> list[dict[str, Any]]:
    specs = [
        (
            "Legal retrieved-top5 default",
            ROOT / "results" / "retrieval_prefiltered_vsr" / "legal_retrieved_top5_20",
            20,
        ),
        (
            "Legal retrieved-top5 ctx12k",
            ROOT / "results" / "retrieval_prefiltered_vsr" / "legal_retrieved_top5_20_ctx12k",
            20,
        ),
        (
            "Legal retrieved-top5 prepended full",
            ROOT / "results" / "retrieval_prefiltered_vsr" / "legal_retrieved_top5_prepended_full_20",
            20,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, directory, n in specs:
        path = directory / f"full_v2_rescue_aligned_n{n}.json"
        if path.exists():
            rows.append(read_vsr_run(label, "full_v2_rescue", path))
    return rows


def get_instrumented_runs() -> list[dict[str, Any]]:
    sample_groups = instrumented_cost_sample_groups()
    specs = ["w/o_repair", "full_v2", "full_v2_rescue"]
    rows: list[dict[str, Any]] = []
    for label, sample_dirs in sample_groups.items():
        for method in specs:
            paths = [path for directory in sample_dirs for path in method_result_paths(directory, method)]
            if paths:
                rows.append(read_combined_vsr_run(label, method, paths))
    return rows


def summarize_instrumented_deltas(instrumented_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for run in instrumented_runs:
        by_label.setdefault(run["label"], {})[run["method"]] = run

    rows: list[dict[str, Any]] = []
    for label, methods in by_label.items():
        wo = methods.get("w/o_repair", {})
        full = methods.get("full_v2", {})
        rescue = methods.get("full_v2_rescue", {})
        if not rescue:
            continue
        rows.append(
            {
                "label": label,
                "n": rescue.get("valid_n"),
                "rescue_f1_delta_vs_w_o_repair": rescue.get("mean_f1") - wo.get("mean_f1")
                if rescue.get("mean_f1") is not None and wo.get("mean_f1") is not None
                else None,
                "rescue_f1_delta_vs_full_v2": rescue.get("mean_f1") - full.get("mean_f1")
                if rescue.get("mean_f1") is not None and full.get("mean_f1") is not None
                else None,
                "rescue_faithfulness_delta_vs_full_v2": rescue.get("mean_faithfulness") - full.get("mean_faithfulness")
                if rescue.get("mean_faithfulness") is not None and full.get("mean_faithfulness") is not None
                else None,
                "rescue_calls_delta_vs_w_o_repair": rescue.get("mean_llm_call_count") - wo.get("mean_llm_call_count")
                if rescue.get("mean_llm_call_count") is not None and wo.get("mean_llm_call_count") is not None
                else None,
                "rescue_token_ratio_vs_w_o_repair": ratio(
                    rescue.get("mean_llm_total_tokens"), wo.get("mean_llm_total_tokens")
                ),
                "rescue_rate": rescue.get("rescue_rate"),
                "repair_rate": rescue.get("repair_rate"),
            }
        )
    return rows


def instrumented_cost_sample_groups() -> dict[str, list[Path]]:
    base = ROOT / "results" / "cost_instrumented"
    if not base.exists():
        return {}
    groups: dict[str, list[Path]] = {}
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        if not any(directory.glob("full_v2_rescue_aligned_n*.json")):
            continue
        for label, prefix in INSTRUMENTED_SAMPLE_PREFIXES:
            if directory.name.startswith(prefix):
                groups.setdefault(label, []).append(directory)
                break
        else:
            groups.setdefault(directory.name, []).append(directory)
    return groups


def method_result_paths(directory: Path, method: str) -> list[Path]:
    return sorted(directory.glob(f"{method.replace('/', '_')}_aligned_n*.json"))


def read_combined_vsr_run(label: str, method: str, paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    target_n = 0
    complete = True
    error_count = 0
    source_paths = []
    for path in paths:
        data = load_json(path)
        source_paths.append(path.relative_to(ROOT).as_posix())
        payload_rows = data.get("results") or []
        if not isinstance(payload_rows, list):
            payload_rows = []
        rows.extend(payload_rows)
        target_n += int(data.get("target_sample_size") or data.get("sample_size") or len(payload_rows) or 0)
        complete = complete and bool(data.get("complete", True))
        error_count += int(data.get("error_count") or 0)

    valid_rows = [row for row in rows if "error" not in row]
    return {
        "label": label,
        "method": method,
        "path": "; ".join(source_paths),
        "source_paths": source_paths,
        "valid_n": len(valid_rows),
        "target_n": target_n or len(rows),
        "complete": complete and len(valid_rows) >= (target_n or len(valid_rows)),
        "mean_f1": mean_metric(valid_rows, "f1"),
        "mean_rouge_l": mean_metric(valid_rows, "rouge_l"),
        "mean_faithfulness": mean_metric(valid_rows, "faithfulness"),
        "mean_runtime_seconds": mean_metric(valid_rows, "runtime_seconds"),
        "repair_rate": metric_or_trace_rate(valid_rows, "repair_performed", "repair_triggered"),
        "rescue_rate": metric_or_trace_rate(valid_rows, "abstention_rescue_applied", "context_rescue_applied"),
        "rescue_trigger_rate": metric_or_trace_rate(valid_rows, "abstention_rescue_triggered", "context_rescue_triggered"),
        "abstention_rate": metric_or_trace_rate(valid_rows, "abstention_language"),
        "mean_llm_call_count": mean_metric(valid_rows, "llm_call_count"),
        "llm_call_count_sample_size": metric_sample_size(valid_rows, "llm_call_count"),
        "mean_llm_elapsed_seconds": mean_metric(valid_rows, "llm_elapsed_seconds"),
        "llm_usage_available_rate": rate_metric_true(valid_rows, "llm_usage_available"),
        "mean_llm_total_tokens": mean_metric(valid_rows, "llm_total_tokens"),
        "error_count": error_count,
    }


def summarize_stage_accounting() -> list[dict[str, Any]]:
    paths = [
        path
        for sample_dirs in instrumented_cost_sample_groups().values()
        for directory in sample_dirs
        for path in method_result_paths(directory, "full_v2_rescue")
    ]
    buckets: dict[str, dict[str, float]] = {}
    sample_counts: dict[str, int] = {}
    for path in paths:
        data = load_json(path)
        for row in data.get("results") or []:
            accounting = ((row.get("trace") or {}).get("cost_accounting") or {})
            by_stage = accounting.get("by_stage") or {}
            if not by_stage:
                continue
            for stage, stats in by_stage.items():
                bucket = buckets.setdefault(
                    stage,
                    {
                        "call_count": 0.0,
                        "elapsed_seconds": 0.0,
                        "prompt_tokens": 0.0,
                        "completion_tokens": 0.0,
                        "total_tokens": 0.0,
                    },
                )
                sample_counts[stage] = sample_counts.get(stage, 0) + 1
                for key in bucket:
                    bucket[key] += float(stats.get(key) or 0.0)
    order = [
        "evidence_extraction",
        "answer_generation",
        "verification",
        "repair_v2",
        "post_repair_verification",
        "rescue_evidence_extraction",
        "rescue_answer_generation",
        "rescue_verification",
        "rescue_repair",
        "rescue_postverification",
    ]
    rows = []
    for stage in order + sorted(set(buckets) - set(order)):
        if stage not in buckets:
            continue
        n = sample_counts[stage]
        bucket = buckets[stage]
        rows.append(
            {
                "stage": stage,
                "sample_size": n,
                "mean_call_count": bucket["call_count"] / n,
                "mean_elapsed_seconds": bucket["elapsed_seconds"] / n,
                "mean_prompt_tokens": bucket["prompt_tokens"] / n,
                "mean_completion_tokens": bucket["completion_tokens"] / n,
                "mean_total_tokens": bucket["total_tokens"] / n,
            }
        )
    return rows


def markdown_report(summary: dict[str, Any]) -> str:
    instrumented_rescue_rows = [
        row for row in summary.get("instrumented_vsr", []) if row.get("method") == "full_v2_rescue"
    ]
    instrumented_n = sum(int(row.get("valid_n") or 0) for row in instrumented_rescue_rows)
    instrumented_tokens = weighted_mean(instrumented_rescue_rows, "mean_llm_total_tokens")
    instrumented_calls = weighted_mean(instrumented_rescue_rows, "mean_llm_call_count")
    instrumented_faithfulness = weighted_mean(instrumented_rescue_rows, "mean_faithfulness")
    lines = [
        "# Cost and Latency Summary",
        "",
        f"> Generated: {summary['generated_at']}",
        "",
        "## Scope",
        "",
        "- This report uses only completed local result files; it does not call any LLM/API endpoints.",
        "- Runtime is measured as mean wall-clock seconds per question when available in result aggregates.",
        "- Older result files do not contain exact token and LLM-call accounting; future runs from the updated aligned runner will expose those fields where provider usage metadata is available.",
        "",
        "## VSR Internal Runtime Overhead",
        "",
        "| Dataset Slice | n | w/o Repair s/q | Full v2 s/q | Rescue s/q | Rescue / w/o | Rescue / Full v2 | Repair% | Rescue% | Cost n | LLM Calls/q | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["vsr_internal_overhead"]:
        lines.append(
            "| {label} | {n} | {wo} | {full} | {rescue} | {r_wo}x | {r_full}x | {repair} | {resc_rate} | {cost_n} | {calls} | {f1} |".format(
                label=row["label"],
                n=row.get("valid_n") or "n/a",
                wo=num(row.get("w_o_repair_runtime")),
                full=num(row.get("full_v2_runtime")),
                rescue=num(row.get("full_v2_rescue_runtime")),
                r_wo=num(row.get("rescue_vs_w_o_repair_runtime_ratio"), 2),
                r_full=num(row.get("rescue_vs_full_v2_runtime_ratio"), 2),
                repair=pct(row.get("repair_rate")),
                resc_rate=pct(row.get("rescue_rate")),
                cost_n=row.get("llm_call_count_sample_size") or 0,
                calls=num(row.get("mean_llm_call_count"), 1),
                f1=num(row.get("full_v2_rescue_f1"), 4),
            )
        )

    lines.extend(
        [
            "",
            "## Retrieval Baseline Runtime Comparison",
            "",
            "| Dataset Slice | n | Naive s/q | HyDE s/q | VSR Rescue s/q | VSR / Naive | VSR / HyDE | VSR F1 - Best Retrieval F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["retrieval_runtime"]:
        lines.append(
            "| {label} | {n} | {naive} | {hyde} | {vsr} | {r_naive}x | {r_hyde}x | {delta} |".format(
                label=row["label"],
                n=row.get("n") or "n/a",
                naive=num(row.get("naive_runtime")),
                hyde=num(row.get("hyde_runtime")),
                vsr=num(row.get("full_v2_rescue_runtime")),
                r_naive=num(row.get("vsr_vs_naive_runtime_ratio"), 2),
                r_hyde=num(row.get("vsr_vs_hyde_runtime_ratio"), 2),
                delta=num(row.get("vsr_minus_best_retrieval_f1"), 4),
            )
        )

    lines.extend(
        [
            "",
            "## Retrieval-Prefiltered VSR Runtime",
            "",
            "| Run | valid / target n | Complete | Rescue s/q | Repair% | Rescue% | Cost n | LLM Calls/q | F1 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["prefiltered_vsr"]:
        lines.append(
            "| {label} | {valid}/{target} | {complete} | {runtime} | {repair} | {rescue} | {cost_n} | {calls} | {f1} |".format(
                label=row["label"],
                valid=row.get("valid_n") or 0,
                target=row.get("target_n") or 0,
                complete="yes" if row.get("complete") else "no",
                runtime=num(row.get("mean_runtime_seconds")),
                repair=pct(row.get("repair_rate")),
                rescue=pct(row.get("rescue_rate")),
                cost_n=row.get("llm_call_count_sample_size") or 0,
                calls=num(row.get("mean_llm_call_count"), 1),
                f1=num(row.get("mean_f1"), 4),
            )
        )

    lines.extend(
        [
            "",
            "## Instrumented Full-Context Cost Sample",
            "",
            "| Sample | Method | n | Runtime s/q | Repair% | Rescue% | Cost n | LLM Calls/q | Tokens/q | F1 | Faithfulness |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["instrumented_vsr"]:
        lines.append(
            "| {label} | {method} | {n} | {runtime} | {repair} | {rescue} | {cost_n} | {calls} | {tokens} | {f1} | {faith} |".format(
                label=row["label"],
                method=row["method"],
                n=row.get("valid_n") or 0,
                runtime=num(row.get("mean_runtime_seconds")),
                repair=pct(row.get("repair_rate")),
                rescue=pct(row.get("rescue_rate")),
                cost_n=row.get("llm_call_count_sample_size") or 0,
                calls=num(row.get("mean_llm_call_count"), 1),
                tokens=num(row.get("mean_llm_total_tokens"), 0),
                f1=num(row.get("mean_f1"), 4),
                faith=num(row.get("mean_faithfulness"), 3),
            )
        )

    lines.extend(
        [
            "",
            "### Instrumented Cost/Quality Deltas",
            "",
            "| Sample | n | F1 vs w/o Repair | F1 vs Full v2 | Faith vs Full v2 | Calls vs w/o | Token Ratio vs w/o | Repair% | Rescue% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["instrumented_deltas"]:
        lines.append(
            "| {label} | {n} | {f1_wo} | {f1_full} | {faith_full} | {calls} | {token_ratio}x | {repair} | {rescue} |".format(
                label=row["label"],
                n=row.get("n") or 0,
                f1_wo=num(row.get("rescue_f1_delta_vs_w_o_repair"), 4),
                f1_full=num(row.get("rescue_f1_delta_vs_full_v2"), 4),
                faith_full=num(row.get("rescue_faithfulness_delta_vs_full_v2"), 3),
                calls=num(row.get("rescue_calls_delta_vs_w_o_repair"), 1),
                token_ratio=num(row.get("rescue_token_ratio_vs_w_o_repair"), 2),
                repair=pct(row.get("repair_rate")),
                rescue=pct(row.get("rescue_rate")),
            )
        )

    lines.extend(
        [
            "",
            "## Full-Context Rescue Stage Cost",
            "",
            "| Stage | Cost n | Calls/q | Seconds/q | Prompt Tokens/q | Completion Tokens/q | Total Tokens/q |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["stage_accounting"]:
        lines.append(
            "| {stage} | {n} | {calls} | {seconds} | {prompt} | {completion} | {total} |".format(
                stage=row["stage"],
                n=row["sample_size"],
                calls=num(row.get("mean_call_count"), 1),
                seconds=num(row.get("mean_elapsed_seconds"), 1),
                prompt=num(row.get("mean_prompt_tokens"), 0),
                completion=num(row.get("mean_completion_tokens"), 0),
                total=num(row.get("mean_total_tokens"), 0),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `full_v2_rescue` adds modest runtime over `full_v2` in the UltraDomain aggregate, but full VSR remains materially slower than Naive/HyDE retrieval baselines on the aligned slices.",
            "- The Legal full-context slice is the clearest warning: VSR rescue is roughly 9.9x slower than Naive and 4.1x slower than HyDE while trailing the best retrieval F1 by about 0.0264.",
            "- Retrieved-top5 Legal VSR is faster than Legal full-context VSR but also lower quality, so prefiltering has not yet solved the quality/cost tradeoff.",
            f"- The instrumented full-context samples currently cover N={instrumented_n} `full_v2_rescue` rows, averaging {num(instrumented_calls, 1)} LLM calls and {num(instrumented_tokens, 0)} tokens per question; rescue evidence extraction is the dominant token stage.",
            f"- The same instrumented rows flag a safety tradeoff: `full_v2_rescue` mean faithfulness is {num(instrumented_faithfulness, 3)}, so cost reporting should be paired with rescue-quality diagnostics.",
            "- Paper claims should quantify the cost of verification/repair/rescue and avoid framing the current system as a free F1 improvement over strong retrieval baselines.",
            "",
            "## Accounting Gaps to Fix",
            "",
            "- Backfill or rerun old rows with exact LLM call count by stage: evidence extraction, answer generation, verification, repair, rescue, and post-rescue verification.",
            "- Capture prompt/completion tokens by stage when the provider exposes usage metadata.",
            "- Separate method-specific stage cost from paired-run wall-clock time when several ablations are produced in one shared pass.",
            "- Add a no-op cost test that validates all future result files contain the same accounting fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vsr_runs = get_vsr_runs()
    instrumented_vsr = get_instrumented_runs()
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vsr_runs": vsr_runs,
        "vsr_internal_overhead": summarize_internal_overhead(vsr_runs),
        "retrieval_runtime": summarize_retrieval_runtime(),
        "prefiltered_vsr": get_prefiltered_runs(),
        "instrumented_vsr": instrumented_vsr,
        "instrumented_deltas": summarize_instrumented_deltas(instrumented_vsr),
        "stage_accounting": summarize_stage_accounting(),
        "accounting_gaps": [
            "exact_llm_call_count_by_stage",
            "prompt_completion_tokens_by_stage",
            "method_specific_stage_cost_for_paired_runs",
            "standardized_cost_schema_validation",
        ],
    }
    json_path = OUT_DIR / "cost_latency_summary.json"
    md_path = OUT_DIR / "COST_LATENCY_SUMMARY.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(summary), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
