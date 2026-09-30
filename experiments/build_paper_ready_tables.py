#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build consolidated paper-ready experiment tables from completed artifacts.

This script is intentionally offline: it only reads existing result JSON files
and writes paper-planning tables. It does not call any LLM/API provider.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "paper_tables"
sys.path.insert(0, str(ROOT))

from experiments.run_enhanced_experiments import compute_f1, compute_rouge_l  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def judge_average(path: Path, metric: str = "overall") -> Optional[float]:
    rows = read_jsonl(path)
    values = []
    for row in rows:
        scores = row.get("scores") or {}
        if metric in scores:
            try:
                values.append(float(scores[metric]))
            except (TypeError, ValueError):
                pass
    return sum(values) / len(values) if values else None


def judge_average_for_lines(path: Path, line_nos: List[int], metric: str = "overall") -> Optional[float]:
    rows = rows_by_line(path)
    values = []
    for line_no in line_nos:
        scores = (rows.get(line_no) or {}).get("scores") or {}
        if metric in scores:
            try:
                values.append(float(scores[metric]))
            except (TypeError, ValueError):
                pass
    return sum(values) / len(values) if values else None


def pairwise_pathrag_win_rate(path: Path) -> Optional[float]:
    rows = read_jsonl(path)
    if not rows:
        return None
    dimensions = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]
    rates = []
    for dimension in dimensions:
        pathrag_wins = sum(1 for row in rows if (row.get("winners") or {}).get(dimension) == "pathrag")
        ties = sum(1 for row in rows if (row.get("winners") or {}).get(dimension) == "tie")
        rates.append((pathrag_wins + 0.5 * ties) / len(rows))
    return sum(rates) / len(rates) if rates else None


def rows_by_line(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    for index, row in enumerate(read_jsonl(path)):
        line_no = row.get("line_no", row.get("_line_no", index))
        try:
            rows[int(line_no)] = row
        except (TypeError, ValueError):
            continue
    return rows


def prediction_text(row: Dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def reference_answers(row: Dict[str, Any]) -> List[str]:
    values = row.get("reference_answers") or row.get("answers") or row.get("gold_answer") or []
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values if str(value).strip()]


def score_same_lines(path: Path, line_nos: List[int]) -> Dict[str, Any]:
    rows = rows_by_line(path)
    scored = []
    for line_no in line_nos:
        row = rows.get(line_no)
        if not row:
            continue
        prediction = prediction_text(row)
        references = reference_answers(row)
        if not prediction or not references:
            continue
        scored.append(
            {
                "f1": max(compute_f1(prediction, reference) for reference in references),
                "rouge_l": max(compute_rouge_l(prediction, reference) for reference in references),
                "prediction_chars": len(prediction),
            }
        )
    if not scored:
        return {"n": 0, "f1": None, "rouge_l": None, "prediction_chars": None}
    return {
        "n": len(scored),
        "f1": statistics.mean(row["f1"] for row in scored),
        "rouge_l": statistics.mean(row["rouge_l"] for row in scored),
        "prediction_chars": statistics.mean(row["prediction_chars"] for row in scored),
    }


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_delta(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def coverage_status(coverage: Dict[str, Any], complete_key: str, rows_key: str) -> str:
    if coverage.get(complete_key):
        return "complete"
    expected_n = int(coverage.get("expected_n") or 0)
    rows = coverage.get(rows_key) or {}
    values = sorted(set(int(value or 0) for value in rows.values()))
    if len(values) == 1:
        return f"pending ({values[0]}/{expected_n} each)"
    return ", ".join(f"{key} {value}/{expected_n}" for key, value in rows.items())


def complete_judge_overall(judge_summaries: Dict[str, Any], method: str, expected_n: int) -> Any:
    row = judge_summaries.get(method) or {}
    if int(row.get("n") or 0) != int(expected_n or 0):
        return None
    return (row.get("means") or {}).get("overall")


def fmt_p(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0:
        return "<1e-15"
    if number < 0.001:
        return f"{number:.2e}"
    return f"{number:.4f}"


def md_table(headers: List[str], rows: Iterable[Iterable[Any]], numeric_cols: Optional[set[int]] = None) -> List[str]:
    numeric_cols = numeric_cols or set()
    lines = ["| " + " | ".join(headers) + " |"]
    aligns = ["---:" if idx in numeric_cols else "---" for idx, _ in enumerate(headers)]
    lines.append("|" + "|".join(aligns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def get_stat(stats: Dict[str, List[Dict[str, Any]]], metric: str, ablation: str) -> Dict[str, Any]:
    for row in stats.get(metric, []):
        if row.get("ablation") == ablation:
            return row
    return {}


def build_strict_ablation_tables(summary: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    configs = [
        "w/o_all",
        "w/o_evidence_extraction",
        "w/o_verification",
        "w/o_repair",
        "full",
        "full_v2",
        "full_v2_rescue",
    ]
    aggregated = summary.get("aggregated", {})
    sample_size = int(summary.get("sample_size", 0) or 0)

    aggregate_rows = []
    for name in configs:
        agg = aggregated.get(name, {})
        rows = int(agg.get("sample_size", 0) or 0)
        aggregate_rows.append(
            {
                "config": name,
                "rows": f"{rows}/{sample_size}",
                "status": "complete" if rows == sample_size else "partial",
                "f1": agg.get("mean_f1"),
                "rouge_l": agg.get("mean_rouge_l"),
                "faithfulness": agg.get("mean_faithfulness"),
                "repair_rate": agg.get("repair_rate"),
                "abstention_rate": agg.get("abstention_language_rate"),
                "runtime_seconds": agg.get("mean_runtime_seconds"),
            }
        )

    comparison_rows = []
    for name in ["w/o_all", "w/o_evidence_extraction", "w/o_verification", "w/o_repair", "full_v2", "full_v2_rescue"]:
        f1 = get_stat(stats, "f1", name)
        rouge = get_stat(stats, "rouge_l", name)
        faith = get_stat(stats, "faithfulness", name)
        comparison_rows.append(
            {
                "comparison": f"full - {name}",
                "n": f1.get("n_common"),
                "f1_delta": f1.get("mean_diff"),
                "f1_p": f1.get("t_p_value"),
                "f1_d": f1.get("cohens_d"),
                "rouge_delta": rouge.get("mean_diff"),
                "faith_delta": faith.get("mean_diff"),
                "faith_p": faith.get("t_p_value"),
            }
        )

    return {"aggregate": aggregate_rows, "comparisons": comparison_rows}


def build_vsr_cross_dataset_rows(cross: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for slug, item in cross.get("datasets", {}).items():
        agg = item.get("aggregated", {})
        comp = item.get("comparisons", {}).get("full_v2_rescue_minus_w/o_repair", {}).get("f1", {})
        rows.append(
            {
                "dataset": item.get("name", slug),
                "n": item.get("target_n"),
                "w_o_repair_f1": agg.get("w/o_repair", {}).get("mean_f1"),
                "full_v2_f1": agg.get("full_v2", {}).get("mean_f1"),
                "full_v2_rescue_f1": agg.get("full_v2_rescue", {}).get("mean_f1"),
                "rescue_delta": comp.get("mean_delta"),
                "p_value": comp.get("t_p_value"),
                "cohens_d": comp.get("cohens_d"),
            }
        )
    return rows


def build_retrieval_rows(retrieval: Dict[str, Any], sage: Dict[str, Any]) -> List[Dict[str, Any]]:
    sage_by_dataset = {row.get("dataset"): row for row in sage.get("external_slices", [])}
    rows = []
    for slug, item in retrieval.get("datasets", {}).items():
        agg = item.get("aggregated", {})
        label = item.get("name", slug)
        sage_row = sage_by_dataset.get(label, {})
        sage_methods = sage_row.get("methods", {})
        best = sage_row.get("best_retrieval")
        rows.append(
            {
                "dataset": label,
                "n": item.get("n"),
                "naive_f1": agg.get("naive", {}).get("mean_f1"),
                "hyde_f1": agg.get("hyde", {}).get("mean_f1"),
                "vsr_rescue_f1": agg.get("full_v2_rescue", {}).get("mean_f1"),
                "sage_v33_f1": sage_methods.get("sage_v33", {}).get("f1"),
                "best_retrieval": best,
                "sage_minus_best": sage_row.get("sage_vs_best_retrieval", {}).get("delta"),
                "sage_best_p": sage_row.get("sage_vs_best_retrieval", {}).get("t_p"),
                "vsr_minus_best": item.get("comparisons", {})
                .get(f"full_v2_rescue_minus_{best}", {})
                .get("f1", {})
                .get("mean_delta"),
            }
        )
    return rows


def build_sage_rows(sage: Dict[str, Any]) -> Dict[str, Any]:
    mix = sage.get("mix130", {})
    external = []
    for row in sage.get("external_slices", []):
        external.append(
            {
                "dataset": row.get("dataset"),
                "n": row.get("n"),
                "best_retrieval": row.get("best_retrieval"),
                "best_retrieval_f1": row.get("methods", {}).get(row.get("best_retrieval"), {}).get("f1"),
                "sage_v33_f1": row.get("methods", {}).get("sage_v33", {}).get("f1"),
                "sage_minus_best": row.get("sage_vs_best_retrieval", {}).get("delta"),
                "sage_minus_vsr": row.get("sage_vs_vsr_rescue", {}).get("delta"),
                "p_vs_best": row.get("sage_vs_best_retrieval", {}).get("t_p"),
            }
        )
    promotion = sage.get("external_promotion_replay", {}).get("external_weighted", {})
    return {"mix130": mix, "external": external, "promotion_weighted": promotion}


def build_cost_rows(cost: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "retrieval_runtime": cost.get("retrieval_runtime", []),
        "instrumented_vsr": cost.get("instrumented_vsr", []),
        "accounting_gaps": cost.get("accounting_gaps", []),
    }


def build_graph_lite_rows(legacy: Dict[str, Any]) -> List[Dict[str, Any]]:
    judge_paths = {
        "ultradomain_mix": ROOT / "results" / "ultradomain_mix_graphrag_lite_eval.jsonl",
        "legal_fullcontext_20": ROOT / "results" / "legal_fullcontext_20_graphrag_lite_eval.jsonl",
        "biology_fullcontext_50": ROOT / "results" / "biology_fullcontext_50_graphrag_lite_eval.jsonl",
        "domain_expansion_windowed_50": ROOT / "results" / "domain_expansion_windowed_50_graphrag_lite_eval.jsonl",
    }
    pairwise_paths = {
        "ultradomain_mix": ROOT / "results" / "ultradomain_mix_pairwise_pathrag_vs_graphrag_lite.jsonl",
        "legal_fullcontext_20": ROOT / "results" / "legal_fullcontext_20_pairwise_pathrag_vs_graphrag_lite.jsonl",
        "biology_fullcontext_50": ROOT / "results" / "biology_fullcontext_50_pairwise_pathrag_vs_graphrag_lite.jsonl",
        "domain_expansion_windowed_50": ROOT / "results" / "domain_expansion_windowed_50_pairwise_pathrag_vs_graphrag_lite.jsonl",
    }
    rows = []
    for slug, item in legacy.get("datasets", {}).items():
        methods = item.get("methods", {})
        graph = methods.get("legacy_graphrag_lite")
        if not graph:
            continue
        best_method = (item.get("best_methods") or [None])[0]
        best = methods.get(best_method, {}) if best_method else {}
        rows.append(
            {
                "dataset": item.get("name", slug),
                "n": item.get("n"),
                "graphrag_lite_f1": graph.get("mean_f1"),
                "graphrag_lite_rouge_l": graph.get("mean_rouge_l"),
                "graphrag_lite_judge_overall": judge_average(judge_paths.get(slug, Path(""))),
                "pathrag_pairwise_win_rate": pairwise_pathrag_win_rate(pairwise_paths.get(slug, Path(""))),
                "graphrag_lite_runtime": graph.get("mean_runtime_seconds"),
                "best_method": best_method,
                "best_f1": best.get("mean_f1"),
                "vsr_rescue_f1": methods.get("aligned_full_v2_rescue", {}).get("mean_f1"),
                "legacy_pathrag_f1": methods.get("legacy_pathrag", {}).get("mean_f1"),
            }
        )
    return rows


def build_lightrag_cached35_rows() -> List[Dict[str, Any]]:
    light_path = ROOT / "results" / "lightrag_current.jsonl"
    judge_path = ROOT / "results" / "lightrag_current_eval.jsonl"
    pairwise_path = ROOT / "results" / "pairwise_pathrag_vs_lightrag_current.jsonl"
    light_rows = rows_by_line(light_path)
    valid_light_rows = {
        line_no: row
        for line_no, row in light_rows.items()
        if not row.get("error") and prediction_text(row).strip()
    }
    judge_lines = set(rows_by_line(judge_path))
    pairwise_lines = set(rows_by_line(pairwise_path))
    line_nos = sorted(set(valid_light_rows) & judge_lines & pairwise_lines)
    if not line_nos:
        line_nos = sorted(valid_light_rows)
    subset_n = len(line_nos)
    valid_n = len(valid_light_rows)
    error_n = len(light_rows) - valid_n
    judge_n = len(set(valid_light_rows) & judge_lines)
    pairwise_n = len(set(valid_light_rows) & pairwise_lines)
    pending_judge_n = max(0, valid_n - judge_n)
    pending_pairwise_n = max(0, valid_n - pairwise_n)
    progress_note = ""
    if pending_judge_n or pending_pairwise_n or error_n:
        progress_note = (
            f" {pending_judge_n} additional valid prediction row(s) await judge; "
            f"{pending_pairwise_n} await pairwise; "
            f"{error_n} error row(s) await retry."
        )
    specs = [
        {
            "method": f"LightRAG current-{subset_n}",
            "path": light_path,
            "judge_path": judge_path,
            "pairwise_path": pairwise_path,
            "note": "Representative heavy graph-RAG baseline; evaluated current subset." + progress_note,
        },
        {
            "method": "SAGE v33-preservation",
            "path": ROOT / "results" / "mse_pathrag_sage_v33_preservation_delta_mix130.jsonl",
            "judge_path": None,
            "pairwise_path": None,
            "note": f"Same evaluated {subset_n}-line LightRAG subset; current headline candidate.",
        },
        {
            "method": "GraphRAG-lite",
            "path": ROOT / "results" / "ultradomain_mix_graphrag_lite.jsonl",
            "judge_path": None,
            "pairwise_path": None,
            "note": f"Same evaluated {subset_n}-line LightRAG subset; lightweight community baseline.",
        },
        {
            "method": "PathRAG current",
            "path": ROOT / "results" / "ultradomain_mix_repro_current.jsonl",
            "judge_path": None,
            "pairwise_path": None,
            "note": f"Same evaluated {subset_n}-line LightRAG subset; pairwise reference side.",
        },
    ]
    rows = []
    for spec in specs:
        scores = score_same_lines(spec["path"], line_nos)
        rows.append(
            {
                "method": spec["method"],
                "n": scores.get("n"),
                "f1": scores.get("f1"),
                "rouge_l": scores.get("rouge_l"),
                "judge_overall": judge_average_for_lines(spec["judge_path"], line_nos)
                if spec.get("judge_path")
                else None,
                "pathrag_pairwise_win_rate": pairwise_pathrag_win_rate(spec["pairwise_path"])
                if spec.get("pairwise_path")
                else None,
                "prediction_chars": scores.get("prediction_chars"),
                "note": spec["note"],
            }
        )
    return rows


def build_locked_unseen75_rows(lexical: Dict[str, Any], judge: Dict[str, Any], pairwise: Dict[str, Any]) -> Dict[str, Any]:
    methods = ["naive", "hyde", "sage_v33_preservation_delta"]
    aggregate = []
    for method in methods:
        lex_method = lexical.get("methods", {}).get(method, {})
        judge_method = judge.get("methods", {}).get(method, {})
        lex_agg = lex_method.get("aggregate", {})
        judge_agg = judge_method.get("aggregate", {})
        averages = judge_agg.get("averages", {})
        verdicts = judge_agg.get("verdict_counts", {})
        aggregate.append(
            {
                "method": method,
                "n": lex_agg.get("sample_size") or judge_agg.get("sample_size"),
                "f1": lex_agg.get("mean_f1"),
                "rouge_l": lex_agg.get("mean_rouge_l"),
                "judge_correctness": averages.get("correctness"),
                "judge_coverage": averages.get("coverage"),
                "judge_overall": averages.get("overall"),
                "excellent": verdicts.get("excellent", 0),
                "poor": verdicts.get("poor", 0),
            }
        )

    comparisons = []
    comparison_specs = [
        ("sage_v33_preservation_delta_minus_hyde", "SAGE v33 - HyDE"),
        ("sage_v33_preservation_delta_minus_naive", "SAGE v33 - Naive"),
    ]
    for key, label in comparison_specs:
        judge_comp = judge.get("comparisons", {}).get(key, {})
        lexical_comp = lexical.get("comparisons", {}).get(key, {})
        pairwise_comp = pairwise.get("comparisons", {}).get(key, {})
        comparisons.append(
            {
                "comparison": label,
                "n": judge_comp.get("overall", {}).get("n") or lexical_comp.get("f1", {}).get("n"),
                "f1_delta": lexical_comp.get("f1", {}).get("mean_delta"),
                "rouge_l_delta": lexical_comp.get("rouge_l", {}).get("mean_delta"),
                "overall_delta": judge_comp.get("overall", {}).get("mean_delta"),
                "overall_p": judge_comp.get("overall", {}).get("t_p_value"),
                "overall_d": judge_comp.get("overall", {}).get("cohens_d"),
                "pairwise_win_rate": pairwise_comp.get("average_target_win_rate"),
                "wins": judge_comp.get("overall", {}).get("wins"),
                "losses": judge_comp.get("overall", {}).get("losses"),
                "ties": judge_comp.get("overall", {}).get("ties"),
            }
        )
    return {"aggregate": aggregate, "comparisons": comparisons}


def build_gap_audit() -> List[Dict[str, str]]:
    light_rows = rows_by_line(ROOT / "results" / "lightrag_current.jsonl")
    valid_light_lines = {
        line_no
        for line_no, row in light_rows.items()
        if not row.get("error") and prediction_text(row).strip()
    }
    error_light_lines = {
        line_no
        for line_no, row in light_rows.items()
        if row.get("error") or not prediction_text(row).strip()
    }
    judge_lines = set(rows_by_line(ROOT / "results" / "lightrag_current_eval.jsonl"))
    pairwise_lines = set(rows_by_line(ROOT / "results" / "pairwise_pathrag_vs_lightrag_current.jsonl"))
    evaluated_light_n = len(valid_light_lines & judge_lines & pairwise_lines)
    judge_light_n = len(valid_light_lines & judge_lines)
    pairwise_light_n = len(valid_light_lines & pairwise_lines)
    valid_light_n = len(valid_light_lines)
    error_light_n = len(error_light_lines)
    pending_judge_n = max(0, valid_light_n - judge_light_n)
    pending_pairwise_n = max(0, valid_light_n - pairwise_light_n)
    if pending_judge_n == 0 and pending_pairwise_n == 0 and error_light_n == 0:
        light_status = (
            f"LightRAG current-{evaluated_light_n} is complete as an evaluated representative heavy baseline; "
            f"{valid_light_n} valid predictions have been generated in total, "
            f"with all {judge_light_n} judge-evaluated and all {pairwise_light_n} pairwise-evaluated; "
            "no judge, pairwise, or retry rows remain."
        )
    else:
        light_status = (
            f"LightRAG current-{evaluated_light_n} is complete as an evaluated representative heavy baseline; "
            f"{valid_light_n} valid predictions have been generated in total, "
            f"with {judge_light_n} judge-evaluated and {pairwise_light_n} pairwise-evaluated; "
            f"{pending_judge_n} await judge, {pending_pairwise_n} await pairwise, "
            f"and {error_light_n} await retry."
        )
    return [
        {
            "priority": "P0",
            "item": "Freeze final method definition",
            "status": "Complete for paper planning: freeze SAGE v33-preservation; keep v56 exploratory.",
            "api_intensity": "Offline / done unless promoting a new guard",
            "next_action": "Use the v33 freeze/holdout audit as the method boundary and reserve v56 for discussion/future work.",
        },
        {
            "priority": "P0",
            "item": "Same-protocol baselines",
            "status": f"Naive/HyDE complete on external slices; GraphRAG-lite covers Mix130, Legal, Biology, and Domain; {light_status} Full official GraphRAG/LightRAG comparability remains approximate.",
            "api_intensity": "Low for reporting completed GraphRAG-lite/LightRAG subset evidence; high only for official GraphRAG or full LightRAG reruns.",
            "next_action": f"Use GraphRAG-lite as the lightweight community baseline and LightRAG current-{evaluated_light_n} as the evaluated high-cost representative comparator unless budget supports full official reruns.",
        },
        {
            "priority": "P0",
            "item": "Strict ablation table",
            "status": "Complete: all seven UltraDomain mix130 aligned configs have 130/130 valid rows and paired tests.",
            "api_intensity": "Done / offline only",
            "next_action": "Use this as the main internal mechanism ablation table.",
        },
        {
            "priority": "P0",
            "item": "Multi-dataset validation",
            "status": "Complete: prior external slices plus locked unseen75 and Repeat1-8 strict holdouts now include lexical, same-rubric judge, and pairwise A/B evaluation.",
            "api_intensity": "Done / offline unless adding a different frozen method or more judge replications.",
            "next_action": "Use locked unseen75 as the primary final post-freeze holdout and Repeat1-8 as supplementary robustness/stress evidence; do not tune on any locked holdout.",
        },
        {
            "priority": "P0",
            "item": "Final paper tables",
            "status": "Consolidated in this output and exported as include-ready LaTeX fragments in `paper/generated_experiment_tables.tex`.",
            "api_intensity": "Offline",
            "next_action": "Input or copy the generated fragments into the paper draft after final claim wording is frozen.",
        },
        {
            "priority": "P1",
            "item": "Cost and latency",
            "status": "Runtime complete; exact token accounting now sampled across Legal N=10, Biology N=5, and Domain-windowed N=5 with per-stage call/token breakdown.",
            "api_intensity": "Offline for current summary; medium only if expanding any slice beyond the current sample.",
            "next_action": "Use cross-domain N=20 accounting now; optionally expand Domain/Biology only if reviewers require tighter cost intervals.",
        },
        {
            "priority": "P1",
            "item": "Error analysis",
            "status": "15-row LLM audit, longer-context diagnostic, cost-quality case audit, and exploratory i18n abstention/language guard audit are available.",
            "api_intensity": "Offline unless adding human/LLM judge cases.",
            "next_action": "Use the i18n guard only as future-validation framing; select representative cases for the paper discussion section.",
        },
        {
            "priority": "P1",
            "item": "Reproducibility manifest",
            "status": "Complete: manifest refreshed after Repeat8 judge and pairwise artifacts were added.",
            "api_intensity": "Offline",
            "next_action": "Refresh again only after future API reruns, table regeneration, or artifact changes.",
        },
    ]


def build_payload() -> Dict[str, Any]:
    strict_summary = read_json(ROOT / "results" / "enhanced_ablation_aligned" / "aligned_ablation_study_n130.json")
    strict_stats = read_json(ROOT / "results" / "enhanced_ablation_aligned" / "aligned_ablation_stat_tests_n130.json")
    cross = read_json(ROOT / "results" / "enhanced_ablation_aligned" / "cross_dataset_aligned_progress_summary.json")
    retrieval = read_json(ROOT / "results" / "retrieval_baselines" / "retrieval_baseline_aligned_summary.json")
    sage = read_json(ROOT / "results" / "sage_final_candidate" / "sage_final_candidate_summary.json")
    cost = read_json(ROOT / "results" / "cost_latency" / "cost_latency_summary.json")
    legacy = read_json(ROOT / "results" / "legacy_baseline_overlap" / "legacy_baseline_overlap_summary.json")
    locked_unseen75_lexical = read_json(
        ROOT / "results" / "sage_locked_unseen75" / "sage_locked_unseen75_baseline_summary.json"
    )
    locked_unseen75_judge = read_json(
        ROOT / "results" / "sage_locked_unseen75" / "sage_locked_unseen75_judge_summary.json"
    )
    locked_unseen75_pairwise = read_json(
        ROOT / "results" / "sage_locked_unseen75" / "sage_locked_unseen75_pairwise_summary.json"
    )
    locked_unseen25_repeat1 = read_optional_json(
        ROOT / "results" / "sage_locked_unseen25_repeat1" / "sage_locked_unseen25_repeat1_summary.json"
    )
    locked_unseen40_repeat2 = read_optional_json(
        ROOT / "results" / "sage_locked_unseen40_repeat2" / "sage_locked_unseen40_repeat2_summary.json"
    )
    locked_unseen24_repeat3 = read_optional_json(
        ROOT / "results" / "sage_locked_unseen24_repeat3" / "sage_locked_unseen24_repeat3_summary.json"
    )
    locked_unseen24_repeat4 = read_optional_json(
        ROOT / "results" / "sage_locked_unseen24_repeat4" / "sage_locked_unseen24_repeat4_summary.json"
    )
    locked_legal40_repeat5 = read_optional_json(
        ROOT / "results" / "sage_locked_legal40_repeat5" / "sage_locked_legal40_repeat5_summary.json"
    )
    locked_history4_repeat6 = read_optional_json(
        ROOT / "results" / "sage_locked_history4_repeat6" / "sage_locked_history4_repeat6_summary.json"
    )
    locked_legal40_repeat7 = read_optional_json(
        ROOT / "results" / "sage_locked_legal40_repeat7" / "sage_locked_legal40_repeat7_summary.json"
    )
    locked_legal54_repeat8 = read_optional_json(
        ROOT / "results" / "sage_locked_legal54_repeat8" / "sage_locked_legal54_repeat8_summary.json"
    )
    locked_unseen_meta = read_optional_json(
        ROOT / "results" / "sage_locked_unseen_meta" / "sage_locked_unseen_meta_summary.json"
    )

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_files": {
            "strict_ablation": "results/enhanced_ablation_aligned/aligned_ablation_study_n130.json",
            "strict_ablation_stats": "results/enhanced_ablation_aligned/aligned_ablation_stat_tests_n130.json",
            "cross_dataset_vsr": "results/enhanced_ablation_aligned/cross_dataset_aligned_progress_summary.json",
            "retrieval_baselines": "results/historical_baselines/retrieval_baseline_aligned_summary.json",
            "sage_final_candidate": "results/historical_sage/sage_final_candidate_summary.json",
            "cost_latency": "results/cost_latency/cost_latency_summary.json",
            "cost_quality_case_audit": "results/cost_latency/cost_quality_case_audit.json",
            "i18n_abstention_language_guard_audit": "results/cost_latency/i18n_abstention_language_guard_audit.json",
            "legacy_overlap": "results/legacy_baseline_overlap/legacy_baseline_overlap_summary.json",
            "lightrag_current_subset": "results/historical_baselines/lightrag_current_extension_report.md",
            "locked_unseen75_lexical": "results/historical_sage/sage_locked_unseen75_baseline_summary.json",
            "locked_unseen75_judge": "results/historical_sage/sage_locked_unseen75_judge_summary.json",
            "locked_unseen75_pairwise": "results/historical_sage/sage_locked_unseen75_pairwise_summary.json",
            "locked_unseen25_repeat1": "results/historical_sage/sage_locked_unseen25_repeat1_summary.json",
            "locked_unseen40_repeat2": "results/historical_sage/sage_locked_unseen40_repeat2_summary.json",
            "locked_unseen24_repeat3": "results/historical_sage/sage_locked_unseen24_repeat3_summary.json",
            "locked_unseen24_repeat4": "results/historical_sage/sage_locked_unseen24_repeat4_summary.json",
            "locked_legal40_repeat5": "results/historical_sage/sage_locked_legal40_repeat5_summary.json",
            "locked_history4_repeat6": "results/historical_sage/sage_locked_history4_repeat6_summary.json",
            "locked_legal40_repeat7": "results/historical_sage/sage_locked_legal40_repeat7_summary.json",
            "locked_legal54_repeat8": "results/historical_sage/sage_locked_legal54_repeat8_summary.json",
            "locked_unseen_meta": "results/sage_locked_unseen_meta/sage_locked_unseen_meta_summary.json",
        },
        "strict_ablation": build_strict_ablation_tables(strict_summary, strict_stats),
        "vsr_cross_dataset": build_vsr_cross_dataset_rows(cross),
        "retrieval_and_sage": build_retrieval_rows(retrieval, sage),
        "sage_candidate": build_sage_rows(sage),
        "graph_lite_legacy": build_graph_lite_rows(legacy),
        "lightrag_cached35": build_lightrag_cached35_rows(),
        "locked_unseen75": build_locked_unseen75_rows(
            locked_unseen75_lexical,
            locked_unseen75_judge,
            locked_unseen75_pairwise,
        ),
        "locked_unseen25_repeat1": locked_unseen25_repeat1,
        "locked_unseen40_repeat2": locked_unseen40_repeat2,
        "locked_unseen24_repeat3": locked_unseen24_repeat3,
        "locked_unseen24_repeat4": locked_unseen24_repeat4,
        "locked_legal40_repeat5": locked_legal40_repeat5,
        "locked_history4_repeat6": locked_history4_repeat6,
        "locked_legal40_repeat7": locked_legal40_repeat7,
        "locked_legal54_repeat8": locked_legal54_repeat8,
        "locked_unseen_meta": locked_unseen_meta,
        "cost_latency": build_cost_rows(cost),
        "gap_audit": build_gap_audit(),
    }


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines: List[str] = [
        "# Paper-Ready Experiment Tables",
        "",
        f"> Generated: {payload['generated_at']}",
        "> Scope: offline consolidation from completed result artifacts; no API calls.",
        "",
        "## Recommended Claim Framing",
        "",
        "- Headline candidate: `SAGE v33-preservation`, because it is reproducible, strong on Mix130, and safer than v56 despite v56's exploratory aggregate gains.",
        "- Final holdouts: locked unseen75 is the primary post-freeze zero-overlap holdout; Repeat1-8 add strict zero-overlap robustness/stress evidence with complete judge and pairwise coverage across 326 rows.",
        "- VSR contribution: use `full_v2_rescue` as the verification/repair/rescue mechanism layer, not as a blanket F1-superiority method against strong retrieval baselines.",
        "- Metric framing: report lexical F1/ROUGE-L together with faithfulness, abstention, semantic judge, and cost/latency; do not mix old LLM-judge baselines with current lexical scores as equivalent metrics.",
        "",
        "## Table 1. Strict VSR Ablation on UltraDomain Mix130",
        "",
    ]

    strict_rows = payload["strict_ablation"]["aggregate"]
    lines.extend(
        md_table(
            ["Config", "Rows", "Status", "F1", "ROUGE-L", "Faithfulness", "Repair%", "Abstain%", "Runtime s/q"],
            [
                [
                    row["config"],
                    row["rows"],
                    row["status"],
                    fmt_num(row["f1"]),
                    fmt_num(row["rouge_l"]),
                    fmt_num(row["faithfulness"]),
                    fmt_pct(row["repair_rate"]),
                    fmt_pct(row["abstention_rate"]),
                    fmt_num(row["runtime_seconds"], 1),
                ]
                for row in strict_rows
            ],
            numeric_cols={3, 4, 5, 6, 7, 8},
        )
    )

    lines.extend([
        "",
        "## Table 2. Full vs Ablations Paired Tests",
        "",
    ])
    lines.extend(
        md_table(
            ["Comparison", "N", "F1 Delta", "F1 p", "F1 d", "ROUGE-L Delta", "Faithfulness Delta", "Faithfulness p"],
            [
                [
                    row["comparison"],
                    row["n"],
                    fmt_delta(row["f1_delta"]),
                    fmt_p(row["f1_p"]),
                    fmt_delta(row["f1_d"], 2),
                    fmt_delta(row["rouge_delta"]),
                    fmt_delta(row["faith_delta"]),
                    fmt_p(row["faith_p"]),
                ]
                for row in payload["strict_ablation"]["comparisons"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 6, 7},
        )
    )

    lines.extend([
        "",
        "## Table 3. VSR Rescue Cross-Dataset Internal Validation",
        "",
    ])
    lines.extend(
        md_table(
            ["Dataset", "N", "w/o Repair F1", "Full v2 F1", "Full v2 Rescue F1", "Rescue Delta", "p", "d"],
            [
                [
                    row["dataset"],
                    row["n"],
                    fmt_num(row["w_o_repair_f1"]),
                    fmt_num(row["full_v2_f1"]),
                    fmt_num(row["full_v2_rescue_f1"]),
                    fmt_delta(row["rescue_delta"]),
                    fmt_p(row["p_value"]),
                    fmt_delta(row["cohens_d"], 2),
                ]
                for row in payload["vsr_cross_dataset"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 6, 7},
        )
    )

    lines.extend([
        "",
        "## Table 4. External Retrieval Baselines and SAGE v33",
        "",
    ])
    lines.extend(
        md_table(
            ["Dataset", "N", "Naive F1", "HyDE F1", "VSR Rescue F1", "SAGE v33 F1", "Best Retrieval", "SAGE-Best", "p vs Best", "VSR-Best"],
            [
                [
                    row["dataset"],
                    row["n"],
                    fmt_num(row["naive_f1"]),
                    fmt_num(row["hyde_f1"]),
                    fmt_num(row["vsr_rescue_f1"]),
                    fmt_num(row["sage_v33_f1"]),
                    row["best_retrieval"],
                    fmt_delta(row["sage_minus_best"]),
                    fmt_p(row["sage_best_p"]),
                    fmt_delta(row["vsr_minus_best"]),
                ]
                for row in payload["retrieval_and_sage"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 7, 8, 9},
        )
    )

    lines.extend([
        "",
        "## Table 5. GraphRAG-Lite / Legacy Overlap Supplement",
        "",
    ])
    lines.extend(
        md_table(
            ["Dataset", "N", "GraphRAG-lite F1", "ROUGE-L", "Judge Overall", "PathRAG Pairwise Win", "Runtime s/q", "Best Method", "Best F1", "VSR Rescue F1", "Legacy PathRAG F1"],
            [
                [
                    row["dataset"],
                    row["n"],
                    fmt_num(row["graphrag_lite_f1"]),
                    fmt_num(row["graphrag_lite_rouge_l"]),
                    fmt_num(row["graphrag_lite_judge_overall"], 3),
                    fmt_pct(row["pathrag_pairwise_win_rate"], 2),
                    fmt_num(row["graphrag_lite_runtime"], 1),
                    row["best_method"],
                    fmt_num(row["best_f1"]),
                    fmt_num(row["vsr_rescue_f1"]),
                    fmt_num(row["legacy_pathrag_f1"]),
                ]
                for row in payload["graph_lite_legacy"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 6, 8, 9, 10},
        )
    )

    lines.extend([
        "",
        "## Table 6. LightRAG Current Heavy Baseline Supplement",
        "",
    ])
    lines.extend(
        md_table(
            ["Method", "N", "F1", "ROUGE-L", "Judge Overall", "PathRAG Pairwise Win", "Avg Chars", "Note"],
            [
                [
                    row["method"],
                    row["n"],
                    fmt_num(row["f1"]),
                    fmt_num(row["rouge_l"]),
                    fmt_num(row["judge_overall"], 3),
                    fmt_pct(row["pathrag_pairwise_win_rate"], 2),
                    fmt_num(row["prediction_chars"], 1),
                    row["note"],
                ]
                for row in payload["lightrag_cached35"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 6},
        )
    )

    sage = payload["sage_candidate"]
    mix = sage["mix130"]
    promotion = sage["promotion_weighted"]
    lines.extend([
        "",
        "## Table 7. SAGE Candidate Position",
        "",
    ])
    lines.extend(
        md_table(
            ["Item", "N", "F1", "ROUGE-L", "Runtime s/q", "Delta / Note"],
            [
                [
                    "SAGE v33-preservation on Mix130",
                    mix.get("n"),
                    fmt_num(mix.get("f1")),
                    fmt_num(mix.get("rouge_l")),
                    fmt_num(mix.get("runtime_seconds"), 1),
                    f"vs VSR rescue {fmt_delta(mix.get('sage_minus_vsr_f1'))}; wins {mix.get('sage_vs_vsr_wins')}, losses {mix.get('sage_vs_vsr_losses')}",
                ],
                [
                    "Exploratory v56 weighted external replay",
                    promotion.get("n"),
                    fmt_num(promotion.get("v56_f1")),
                    fmt_num(promotion.get("v56_rouge_l")),
                    "N/A",
                    f"vs v33 F1 {fmt_delta(promotion.get('f1_delta'))}; judge overall {fmt_delta(promotion.get('judge_overall_delta'))}; keep exploratory",
                ],
            ],
            numeric_cols={1, 2, 3, 4},
        )
    )

    lines.extend([
        "",
        "## Table 8. Locked Unseen75 Final Holdout",
        "",
    ])
    lines.extend(
        md_table(
            ["Method", "N", "F1", "ROUGE-L", "Judge Correct", "Judge Coverage", "Judge Overall", "Excellent", "Poor"],
            [
                [
                    row["method"],
                    row["n"],
                    fmt_num(row["f1"]),
                    fmt_num(row["rouge_l"]),
                    fmt_num(row["judge_correctness"], 3),
                    fmt_num(row["judge_coverage"], 3),
                    fmt_num(row["judge_overall"], 3),
                    row["excellent"],
                    row["poor"],
                ]
                for row in payload["locked_unseen75"]["aggregate"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 6, 7, 8},
        )
    )

    lines.extend([
        "",
        "## Table 9. Locked Unseen75 Paired Gains",
        "",
    ])
    lines.extend(
        md_table(
            [
                "Comparison",
                "N",
                "F1 Delta",
                "ROUGE-L Delta",
                "Judge Overall Delta",
                "Overall p",
                "Overall d",
                "Pairwise Win Rate",
                "Wins",
                "Losses",
                "Ties",
            ],
            [
                [
                    row["comparison"],
                    row["n"],
                    fmt_delta(row["f1_delta"]),
                    fmt_delta(row["rouge_l_delta"]),
                    fmt_delta(row["overall_delta"], 3),
                    fmt_p(row["overall_p"]),
                    fmt_delta(row["overall_d"], 2),
                    fmt_pct(row["pairwise_win_rate"], 2),
                    row["wins"],
                    row["losses"],
                    row["ties"],
                ]
                for row in payload["locked_unseen75"]["comparisons"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        )
    )

    next_table = 10
    for repeat_key, repeat_title, repeat_label in [
        ("locked_unseen25_repeat1", "Locked Unseen25 Repeat1 Supplement", "Repeat1"),
        ("locked_unseen40_repeat2", "Locked Unseen40 Repeat2 Supplement", "Repeat2"),
        ("locked_unseen24_repeat3", "Locked Unseen24 Repeat3 Supplement", "Repeat3"),
        ("locked_unseen24_repeat4", "Locked Unseen24 Repeat4 Supplement", "Repeat4"),
        ("locked_legal40_repeat5", "Locked Legal40 Repeat5 Stress Holdout", "Repeat5 Legal40"),
        ("locked_history4_repeat6", "Locked History4 Repeat6 Tail Holdout", "Repeat6 History4"),
        ("locked_legal40_repeat7", "Locked Legal40 Repeat7 Stress Holdout", "Repeat7 Legal40"),
        ("locked_legal54_repeat8", "Locked Legal54 Repeat8 Exhaustive Stress Holdout", "Repeat8 Legal54"),
    ]:
        repeat = payload.get(repeat_key) or {}
        if not repeat:
            continue
        method_aggregates = repeat.get("method_aggregates") or {}
        judge_summaries = repeat.get("judge_summaries") or {}
        pairwise = repeat.get("pairwise") or {}
        expected_n = int(repeat.get("expected_n") or 0)
        lines.extend([
            "",
            f"## Table {next_table}. {repeat_title}",
            "",
        ])
        lines.extend(
            md_table(
                ["Method", "N", "F1", "ROUGE-L", "Judge Overall", "Runtime s/q", "Avg Chars"],
                [
                    [
                        method,
                        row.get("sample_size"),
                        fmt_num(row.get("mean_f1")),
                        fmt_num(row.get("mean_rouge_l")),
                        fmt_num(complete_judge_overall(judge_summaries, method, expected_n), 3),
                        fmt_num(row.get("mean_runtime_seconds"), 1),
                        fmt_num(row.get("mean_prediction_chars"), 1),
                    ]
                    for method, row in method_aggregates.items()
                    if method in {"naive", "hyde", "sage_v33_preservation_delta"}
                ],
                numeric_cols={1, 2, 3, 4, 5, 6},
            )
        )
        lines.extend([
            "",
            f"### {repeat_label} Paired/Pairwise Gains",
            "",
        ])
        paired = repeat.get("paired") or {}
        lines.extend(
            md_table(
                ["Comparison", "N", "F1 Delta", "F1 p", "Pairwise Win Rate", "Wins/Losses/Ties"],
                [
                    [
                        "SAGE v33 - HyDE",
                        (paired.get("sage_v33_vs_hyde_f1") or {}).get("n"),
                        fmt_delta((paired.get("sage_v33_vs_hyde_f1") or {}).get("mean_delta")),
                        fmt_p((paired.get("sage_v33_vs_hyde_f1") or {}).get("t_p")),
                        fmt_pct((pairwise.get("sage_v33_vs_hyde") or {}).get("avg_target_win_rate"), 2),
                        f"{(paired.get('sage_v33_vs_hyde_f1') or {}).get('wins')} / {(paired.get('sage_v33_vs_hyde_f1') or {}).get('losses')} / {(paired.get('sage_v33_vs_hyde_f1') or {}).get('ties')}",
                    ],
                    [
                        "SAGE v33 - Naive",
                        (paired.get("sage_v33_vs_naive_f1") or {}).get("n"),
                        fmt_delta((paired.get("sage_v33_vs_naive_f1") or {}).get("mean_delta")),
                        fmt_p((paired.get("sage_v33_vs_naive_f1") or {}).get("t_p")),
                        fmt_pct((pairwise.get("sage_v33_vs_naive") or {}).get("avg_target_win_rate"), 2),
                        f"{(paired.get('sage_v33_vs_naive_f1') or {}).get('wins')} / {(paired.get('sage_v33_vs_naive_f1') or {}).get('losses')} / {(paired.get('sage_v33_vs_naive_f1') or {}).get('ties')}",
                    ],
                ],
                numeric_cols={1, 2, 3, 4},
            )
        )
        next_table += 1

    meta = payload.get("locked_unseen_meta") or {}
    if meta:
        lines.extend([
            "",
            f"## Table {next_table}. Locked Holdout Meta Summary (Unseen75 + Repeats 1-8)",
            "",
        ])
        weighted_methods = meta.get("weighted_methods_all_repeats") or {}
        lines.extend(
            md_table(
                ["Method", "N", "F1", "ROUGE-L", "Runtime s/q", "Avg Chars"],
                [
                    [
                        method,
                        row.get("sample_size"),
                        fmt_num(row.get("mean_f1")),
                        fmt_num(row.get("mean_rouge_l")),
                        fmt_num(row.get("mean_runtime_seconds"), 1),
                        fmt_num(row.get("mean_prediction_chars"), 1),
                    ]
                    for method, row in weighted_methods.items()
                ],
                numeric_cols={1, 2, 3, 4, 5},
            )
        )
        lines.extend(["", "### Locked Meta Coverage", ""])
        lines.extend(
            md_table(
                ["Holdout", "N", "Methods", "Judge", "Pairwise", "SAGE-HyDE F1", "SAGE-Naive F1"],
                [
                    [
                        repeat.get("label"),
                        repeat.get("expected_n"),
                        coverage_status(repeat.get("coverage") or {}, "methods_complete", "method_rows"),
                        coverage_status(repeat.get("coverage") or {}, "judge_complete", "judge_rows"),
                        coverage_status(repeat.get("coverage") or {}, "pairwise_complete", "pairwise_rows"),
                        fmt_delta((repeat.get("lexical_deltas") or {}).get("sage_v33_vs_hyde_f1")),
                        fmt_delta((repeat.get("lexical_deltas") or {}).get("sage_v33_vs_naive_f1")),
                    ]
                    for repeat in meta.get("repeats", [])
                ],
                numeric_cols={1, 5, 6},
            )
        )
        judge = meta.get("weighted_judge_complete_repeats") or {}
        lines.extend(["", "### Complete-Judge Weighted View", ""])
        lines.append(
            f"Judge-complete rows: `{meta.get('complete_judge_expected_n')}/{meta.get('total_expected_n')}`; incomplete judge values are not imputed."
        )
        lines.append("")
        lines.extend(
            md_table(
                ["Method", "N", "Correct", "Coverage", "Overall"],
                [
                    [method, row.get("sample_size"), fmt_num(row.get("correctness"), 3), fmt_num(row.get("coverage"), 3), fmt_num(row.get("overall"), 3)]
                    for method, row in judge.items()
                ],
                numeric_cols={1, 2, 3, 4},
            )
        )
        pairwise = meta.get("weighted_pairwise_complete_repeats") or {}
        lines.extend(["", "### Complete-Pairwise Weighted View", ""])
        lines.extend(
            md_table(
                ["Comparison", "N", "Avg Target Win Rate"],
                [[comparison, row.get("sample_size"), fmt_pct(row.get("avg_target_win_rate"), 2)] for comparison, row in pairwise.items()],
                numeric_cols={1, 2},
            )
        )
        next_table += 1

    lines.extend([
        "",
        f"## Table {next_table}. Cost and Latency Evidence",
        "",
    ])
    lines.extend(
        md_table(
            ["Dataset", "N", "Naive s/q", "HyDE s/q", "VSR Rescue s/q", "VSR/Naive", "VSR/HyDE", "VSR-Best F1"],
            [
                [
                    row.get("label"),
                    row.get("n"),
                    fmt_num(row.get("naive_runtime"), 1),
                    fmt_num(row.get("hyde_runtime"), 1),
                    fmt_num(row.get("full_v2_rescue_runtime"), 1),
                    fmt_num(row.get("vsr_vs_naive_runtime_ratio"), 2),
                    fmt_num(row.get("vsr_vs_hyde_runtime_ratio"), 2),
                    fmt_delta(row.get("vsr_minus_best_retrieval_f1")),
                ]
                for row in payload["cost_latency"]["retrieval_runtime"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 6, 7},
        )
    )

    lines.extend([
        "",
        "### Token/Call Sample",
        "",
    ])
    lines.extend(
        md_table(
            ["Sample", "Method", "N", "LLM Calls/q", "Tokens/q", "Runtime s/q", "Repair%", "Rescue%", "Faithfulness"],
            [
                [
                    row.get("label"),
                    row.get("method"),
                    row.get("valid_n"),
                    fmt_num(row.get("mean_llm_call_count"), 1),
                    fmt_num(row.get("mean_llm_total_tokens"), 1),
                    fmt_num(row.get("mean_runtime_seconds"), 1),
                    fmt_pct(row.get("repair_rate")),
                    fmt_pct(row.get("rescue_rate")),
                    fmt_num(row.get("mean_faithfulness"), 3),
                ]
                for row in payload["cost_latency"]["instrumented_vsr"]
            ],
            numeric_cols={2, 3, 4, 5, 6, 7, 8},
        )
    )

    lines.extend([
        "",
        "## Remaining Work and API Intensity",
        "",
    ])
    lines.extend(
        md_table(
            ["Priority", "Item", "Status", "API Intensity", "Next Action"],
            [[row["priority"], row["item"], row["status"], row["api_intensity"], row["next_action"]] for row in payload["gap_audit"]],
        )
    )

    lines.extend([
        "",
        "## Source Files",
        "",
    ])
    for label, source in payload["source_files"].items():
        lines.append(f"- `{label}`: `{source}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    (OUT_DIR / "paper_ready_experiment_tables.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(payload, OUT_DIR / "PAPER_READY_EXPERIMENT_TABLES.md")
    print(f"Wrote {OUT_DIR / 'PAPER_READY_EXPERIMENT_TABLES.md'}")
    print(f"Wrote {OUT_DIR / 'paper_ready_experiment_tables.json'}")


if __name__ == "__main__":
    main()
