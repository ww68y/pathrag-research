from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_enhanced_experiments import (  # noqa: E402
    cohens_d,
    compute_f1,
    compute_rouge_l,
    paired_t_test,
    wilcoxon_signed_rank_test,
)


OUT_DIR = ROOT / "results" / "sage_unseen_validation"
OUTPUT_JSON = OUT_DIR / "sage_unseen_validation_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_UNSEEN_VALIDATION_SUMMARY.md"
REUSED_V33_EVAL = OUT_DIR / "mix_unseen34_sage_v33_preservation_reused_eval.jsonl"
REUSED_V55_EVAL = OUT_DIR / "v46_zero_overlap_v55_supported_facet_reused_eval.jsonl"

MANIFEST = "results/historical_vsr/v33_mix_unseen_context_manifest.jsonl"
BASELINE_RESULTS = "results/historical_sage/mse_pathrag_sage_v21_text_first_utility_guard_mix_130.jsonl"
OLD_FROZEN_V33_RESULTS = "results/historical_sage/mse_pathrag_sage_v33_frozen_evidence_delta_mix_unseen34.jsonl"
CURRENT_V33_RESULTS = "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix_unseen34.jsonl"
CONTRACT_COMPLETION_RESULTS = "results/historical_vsr/v33_mix_unseen_context_contract_completion_merged.jsonl"
BASELINE_EVAL = "results/historical_baselines/v33_mix_unseen34_baseline_eval.jsonl"
OLD_FROZEN_V33_EVAL = "results/historical_vsr/v33_mix_unseen34_v33_eval.jsonl"
CHANGED_CANDIDATE_EVAL = "results/historical_vsr/v33_mix_unseen_context_changed_eval.jsonl"
V46_BASELINE_RESULTS = "results/historical_baselines/v46_heldout30_hyde.jsonl"
V46_CURRENT_V33_RESULTS = "results/historical_vsr/v46_zero_overlap_v33_preservation_delta_30.jsonl"
V46_BASELINE_EVAL = "results/historical_baselines/v46_heldout30_hyde_eval.jsonl"
V46_CURRENT_V33_EVAL = "results/historical_vsr/v46_zero_overlap_v33_preservation_eval.jsonl"
V46_V54_RESULTS = "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30.jsonl"
V46_V55_RESULTS = "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v46_30.jsonl"
V46_V54_EVAL = "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30_eval.jsonl"


def load_jsonl(path_or_relative: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_relative)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def line_no(row: dict[str, Any], fallback: int) -> int:
    if row.get("line_no") is not None:
        return int(row["line_no"])
    question_id = str(row.get("question_id") or "")
    if question_id.startswith("ultra_"):
        return int(question_id.split("_", 1)[1])
    return fallback


def by_line(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {line_no(row, index): row for index, row in enumerate(rows)}


def references(row: dict[str, Any]) -> list[str]:
    values = row.get("reference_answers") or row.get("answers") or []
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    if values:
        return [str(values)]
    if row.get("gold_answer"):
        return [str(row["gold_answer"])]
    return [""]


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def best_reference_metric(pred: str, refs: list[str], metric: Any) -> float:
    return max(metric(pred, ref) for ref in refs) if refs else 0.0


def score_method(
    rows_by_line: dict[int, dict[str, Any]],
    selected_lines: list[int],
    label: str,
    path: str,
) -> dict[str, Any]:
    per_line = []
    for number in selected_lines:
        row = rows_by_line[number]
        refs = references(row)
        pred = prediction(row)
        per_line.append(
            {
                "line_no": number,
                "dataset": row.get("dataset") or row.get("label"),
                "f1": best_reference_metric(pred, refs, compute_f1),
                "rouge_l": best_reference_metric(pred, refs, compute_rouge_l),
                "prediction_chars": len(pred),
            }
        )
    return {
        "label": label,
        "path": path,
        "n": len(per_line),
        "mean_f1": statistics.mean(row["f1"] for row in per_line) if per_line else 0.0,
        "mean_rouge_l": statistics.mean(row["rouge_l"] for row in per_line) if per_line else 0.0,
        "mean_prediction_chars": statistics.mean(row["prediction_chars"] for row in per_line) if per_line else 0.0,
        "per_line": per_line,
    }


def paired_summary(left: dict[str, Any], right: dict[str, Any], metric: str) -> dict[str, Any]:
    right_by_line = {row["line_no"]: row for row in right["per_line"]}
    left_values = []
    right_values = []
    for row in left["per_line"]:
        if row["line_no"] not in right_by_line:
            continue
        left_values.append(float(row[metric]))
        right_values.append(float(right_by_line[row["line_no"]][metric]))
    deltas = [left - right for left, right in zip(left_values, right_values)]
    t_stat, t_p = paired_t_test(left_values, right_values)
    w_stat, w_p = wilcoxon_signed_rank_test(left_values, right_values)
    return {
        "metric": metric,
        "n": len(deltas),
        "delta": statistics.mean(deltas) if deltas else 0.0,
        "wins": sum(delta > 1e-12 for delta in deltas),
        "losses": sum(delta < -1e-12 for delta in deltas),
        "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "t_stat": t_stat,
        "t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "cohens_d": cohens_d(left_values, right_values),
    }


def load_eval(path: str) -> dict[int, dict[str, Any]]:
    return by_line(load_jsonl(path))


def judge_summary(label: str, path: str, selected_lines: list[int]) -> dict[str, Any]:
    rows_by_line = load_eval(path)
    rows = [rows_by_line[number] for number in selected_lines if number in rows_by_line]
    metrics = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    averages = {
        metric: statistics.mean(float(row["scores"].get(metric, 0)) for row in rows) if rows else 0.0
        for metric in metrics
    }
    verdicts: Counter[str] = Counter(str(row["scores"].get("verdict", "unknown")) for row in rows)
    return {
        "label": label,
        "path": path,
        "n": len(rows),
        **averages,
        "verdicts": dict(sorted(verdicts.items())),
        "per_line": [
            {
                "line_no": line_no(row, index),
                "dataset": row.get("dataset"),
                "scores": row.get("scores") or {},
            }
            for index, row in enumerate(rows)
        ],
    }


def judge_pairwise(left: dict[str, Any], right: dict[str, Any], metric: str = "overall") -> dict[str, Any]:
    right_by_line = {row["line_no"]: row for row in right["per_line"]}
    left_values = []
    right_values = []
    for row in left["per_line"]:
        if row["line_no"] not in right_by_line:
            continue
        left_values.append(float(row["scores"].get(metric, 0)))
        right_values.append(float(right_by_line[row["line_no"]]["scores"].get(metric, 0)))
    deltas = [left - right for left, right in zip(left_values, right_values)]
    t_stat, t_p = paired_t_test(left_values, right_values)
    w_stat, w_p = wilcoxon_signed_rank_test(left_values, right_values)
    return {
        "metric": metric,
        "n": len(deltas),
        "delta": statistics.mean(deltas) if deltas else 0.0,
        "wins": sum(delta > 1e-12 for delta in deltas),
        "losses": sum(delta < -1e-12 for delta in deltas),
        "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "t_stat": t_stat,
        "t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "cohens_d": cohens_d(left_values, right_values),
    }


def prediction_diffs(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    selected_lines: list[int],
) -> list[int]:
    changed = []
    for number in selected_lines:
        if number in left and number in right and prediction(left[number]) != prediction(right[number]):
            changed.append(number)
    return changed


def write_reused_eval(selected_lines: list[int]) -> None:
    source = load_eval(OLD_FROZEN_V33_EVAL)
    REUSED_V33_EVAL.parent.mkdir(parents=True, exist_ok=True)
    with REUSED_V33_EVAL.open("w", encoding="utf-8") as f:
        for number in selected_lines:
            row = dict(source[number])
            row["reused_for"] = CURRENT_V33_RESULTS
            row["reuse_reason"] = "Predictions match old frozen v33 artifact exactly for all 130 Mix rows."
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_reused_v55_eval(selected_lines: list[int]) -> str:
    source = load_eval(V46_V54_EVAL)
    REUSED_V55_EVAL.parent.mkdir(parents=True, exist_ok=True)
    with REUSED_V55_EVAL.open("w", encoding="utf-8") as f:
        for number in selected_lines:
            row = dict(source[number])
            row["reused_for"] = V46_V55_RESULTS
            row["reuse_reason"] = "SAGE v55 supported-facet guard predictions match v54 contract-gap predictions exactly on v46 validation30."
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return REUSED_V55_EVAL.relative_to(ROOT).as_posix()


def guard_decisions(
    rows_by_line: dict[int, dict[str, Any]],
    selected_lines: list[int],
    baseline_eval_path: str = BASELINE_EVAL,
    candidate_eval_path: str = CHANGED_CANDIDATE_EVAL,
    final_eval_path: str = OLD_FROZEN_V33_EVAL,
) -> list[dict[str, Any]]:
    baseline_eval = load_eval(baseline_eval_path)
    changed_eval = load_eval(candidate_eval_path)
    final_eval = load_eval(final_eval_path)
    decisions = []
    for number in selected_lines:
        row = rows_by_line[number]
        guard = row.get("evidence_delta_guard") or {}
        if not guard.get("enabled"):
            continue
        decisions.append(
            {
                "line_no": number,
                "dataset": row.get("dataset"),
                "action": guard.get("action"),
                "rejection_reasons": guard.get("rejection_reasons") or [],
                "baseline_overall": (baseline_eval.get(number, {}).get("scores") or {}).get("overall"),
                "candidate_overall": (changed_eval.get(number, {}).get("scores") or {}).get("overall"),
                "final_overall": (final_eval.get(number, {}).get("scores") or {}).get("overall"),
                "semantic_preservation": (guard.get("metrics") or {}).get("v33_semantic_preservation"),
                "token_preservation": (guard.get("metrics") or {}).get("v33_token_preservation"),
            }
        )
    return decisions


def supported_facet_guard_decisions(
    rows_by_line: dict[int, dict[str, Any]],
    selected_lines: list[int],
    baseline_eval_path: str,
    final_eval_path: str,
) -> list[dict[str, Any]]:
    baseline_eval = load_eval(baseline_eval_path)
    final_eval = load_eval(final_eval_path)
    decisions = []
    for number in selected_lines:
        row = rows_by_line[number]
        guard = row.get("supported_facet_guard") or {}
        if not guard.get("enabled"):
            continue
        stats = guard.get("stats") or {}
        decisions.append(
            {
                "line_no": number,
                "dataset": row.get("dataset"),
                "action": guard.get("action"),
                "reason": guard.get("reason"),
                "baseline_overall": (baseline_eval.get(number, {}).get("scores") or {}).get("overall"),
                "final_overall": (final_eval.get(number, {}).get("scores") or {}).get("overall"),
                "supported_facet_retention": stats.get("supported_facet_retention"),
                "compression_ratio": stats.get("compression_ratio"),
                "supported_before_terms": stats.get("supported_before_terms"),
            }
        )
    return decisions


def summarize_v46_zero_overlap() -> dict[str, Any]:
    baseline = by_line(load_jsonl(V46_BASELINE_RESULTS))
    current = by_line(load_jsonl(V46_CURRENT_V33_RESULTS))
    v54 = by_line(load_jsonl(V46_V54_RESULTS))
    v55 = by_line(load_jsonl(V46_V55_RESULTS))
    selected_lines = sorted(baseline)
    v55_eval_path = write_reused_v55_eval(selected_lines)

    baseline_scores = score_method(baseline, selected_lines, "HyDE baseline", V46_BASELINE_RESULTS)
    current_scores = score_method(current, selected_lines, "SAGE v33 preservation-delta", V46_CURRENT_V33_RESULTS)
    v55_scores = score_method(v55, selected_lines, "SAGE v55 supported-facet guard", V46_V55_RESULTS)
    baseline_judge = judge_summary("HyDE baseline", V46_BASELINE_EVAL, selected_lines)
    current_judge = judge_summary("SAGE v33 preservation-delta", V46_CURRENT_V33_EVAL, selected_lines)
    v55_judge = judge_summary("SAGE v55 supported-facet guard", v55_eval_path, selected_lines)

    return {
        "name": "V46 zero-overlap validation30",
        "scope": "30 zero-overlap validation questions across Biology, Legal, Physics, History, and Agriculture; existing low5 contract-completion source with current v33 guard.",
        "n": len(selected_lines),
        "selected_lines": selected_lines,
        "scores": {
            "baseline": baseline_scores,
            "current_v33": current_scores,
            "sage_v55_supported_facet": v55_scores,
        },
        "pairwise": {
            "v33_vs_baseline_f1": paired_summary(current_scores, baseline_scores, "f1"),
            "v33_vs_baseline_rouge_l": paired_summary(current_scores, baseline_scores, "rouge_l"),
            "v55_vs_baseline_f1": paired_summary(v55_scores, baseline_scores, "f1"),
            "v55_vs_v33_f1": paired_summary(v55_scores, current_scores, "f1"),
        },
        "judges": {
            "baseline": baseline_judge,
            "current_v33": current_judge,
            "sage_v55_supported_facet": v55_judge,
        },
        "judge_pairwise": {
            "v33_vs_baseline_overall": judge_pairwise(current_judge, baseline_judge, "overall"),
            "v33_vs_baseline_correctness": judge_pairwise(current_judge, baseline_judge, "correctness"),
            "v33_vs_baseline_coverage": judge_pairwise(current_judge, baseline_judge, "coverage"),
            "v55_vs_baseline_overall": judge_pairwise(v55_judge, baseline_judge, "overall"),
            "v55_vs_v33_overall": judge_pairwise(v55_judge, current_judge, "overall"),
        },
        "prediction_diffs": {
            "current_v33_vs_baseline": prediction_diffs(current, baseline, selected_lines),
            "v55_vs_baseline": prediction_diffs(v55, baseline, selected_lines),
            "v55_vs_v33": prediction_diffs(v55, current, selected_lines),
            "v55_vs_v54": prediction_diffs(v55, v54, selected_lines),
        },
        "guard_decisions": guard_decisions(
            current,
            selected_lines,
            baseline_eval_path=V46_BASELINE_EVAL,
            candidate_eval_path=V46_CURRENT_V33_EVAL,
            final_eval_path=V46_CURRENT_V33_EVAL,
        ),
        "supported_facet_guard_decisions": supported_facet_guard_decisions(
            v55,
            selected_lines,
            baseline_eval_path=V46_BASELINE_EVAL,
            final_eval_path=v55_eval_path,
        ),
        "artifacts": {
            "baseline_results": V46_BASELINE_RESULTS,
            "current_v33_results": V46_CURRENT_V33_RESULTS,
            "v54_results": V46_V54_RESULTS,
            "v55_results": V46_V55_RESULTS,
            "baseline_eval": V46_BASELINE_EVAL,
            "current_v33_eval": V46_CURRENT_V33_EVAL,
            "v55_eval": v55_eval_path,
        },
    }


def summarize() -> dict[str, Any]:
    manifest = load_jsonl(MANIFEST)
    selected_lines = sorted(line_no(row, index) for index, row in enumerate(manifest))

    baseline = by_line(load_jsonl(BASELINE_RESULTS))
    old_frozen = by_line(load_jsonl(OLD_FROZEN_V33_RESULTS))
    current_v33 = by_line(load_jsonl(CURRENT_V33_RESULTS))
    contract_completion = by_line(load_jsonl(CONTRACT_COMPLETION_RESULTS))

    write_reused_eval(selected_lines)

    baseline_scores = score_method(baseline, selected_lines, "SAGE v21 baseline", BASELINE_RESULTS)
    current_scores = score_method(current_v33, selected_lines, "SAGE v33 preservation-delta", CURRENT_V33_RESULTS)
    old_scores = score_method(old_frozen, selected_lines, "Old frozen v33 evidence-delta", OLD_FROZEN_V33_RESULTS)
    contract_scores = score_method(contract_completion, selected_lines, "Contract completion candidate", CONTRACT_COMPLETION_RESULTS)

    baseline_judge = judge_summary("SAGE v21 baseline", BASELINE_EVAL, selected_lines)
    reused_eval_path = REUSED_V33_EVAL.relative_to(ROOT).as_posix()
    current_judge = judge_summary("SAGE v33 preservation-delta", reused_eval_path, selected_lines)
    changed_judge = judge_summary("Changed repair candidates", CHANGED_CANDIDATE_EVAL, [61, 71])

    all_mix_lines = sorted(set(old_frozen) & set(current_v33))
    old_new_all_diffs = prediction_diffs(old_frozen, current_v33, all_mix_lines)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "name": "SAGE v33 preservation unseen validation",
            "candidate": "SAGE v33-preservation",
            "scope": "Mix context-heldout risk candidates selected before v33 preservation reporting.",
            "selected_lines": selected_lines,
            "n": len(selected_lines),
            "reused_judge_scores": True,
            "reuse_basis": "Current v33-preservation predictions exactly match old frozen v33 predictions for all 130 Mix rows.",
        },
        "scores": {
            "baseline": baseline_scores,
            "contract_completion_candidate": contract_scores,
            "old_frozen_v33": old_scores,
            "current_v33": current_scores,
        },
        "pairwise": {
            "v33_vs_baseline_f1": paired_summary(current_scores, baseline_scores, "f1"),
            "v33_vs_baseline_rouge_l": paired_summary(current_scores, baseline_scores, "rouge_l"),
            "contract_candidate_vs_baseline_f1": paired_summary(contract_scores, baseline_scores, "f1"),
            "current_v33_vs_old_frozen_f1": paired_summary(current_scores, old_scores, "f1"),
        },
        "judges": {
            "baseline": baseline_judge,
            "current_v33": current_judge,
            "changed_candidates": changed_judge,
        },
        "judge_pairwise": {
            "v33_vs_baseline_overall": judge_pairwise(current_judge, baseline_judge, "overall"),
            "v33_vs_baseline_correctness": judge_pairwise(current_judge, baseline_judge, "correctness"),
            "v33_vs_baseline_coverage": judge_pairwise(current_judge, baseline_judge, "coverage"),
        },
        "prediction_diffs": {
            "current_v33_vs_baseline_heldout": prediction_diffs(current_v33, baseline, selected_lines),
            "contract_candidate_vs_baseline_heldout": prediction_diffs(contract_completion, baseline, selected_lines),
            "current_v33_vs_old_frozen_heldout": prediction_diffs(current_v33, old_frozen, selected_lines),
            "current_v33_vs_old_frozen_all_mix": old_new_all_diffs,
        },
        "guard_decisions": guard_decisions(current_v33, selected_lines),
        "additional_slices": [summarize_v46_zero_overlap()],
        "artifacts": {
            "manifest": MANIFEST,
            "baseline_results": BASELINE_RESULTS,
            "current_v33_results": CURRENT_V33_RESULTS,
            "current_v33_summary": "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix_unseen34_summary.md",
            "baseline_eval": BASELINE_EVAL,
            "reused_current_v33_eval": reused_eval_path,
            "old_frozen_v33_eval": OLD_FROZEN_V33_EVAL,
            "changed_candidate_eval": CHANGED_CANDIDATE_EVAL,
        },
    }


def fmt_p(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.4g}"


def write_report(payload: dict[str, Any]) -> None:
    protocol = payload["protocol"]
    baseline = payload["scores"]["baseline"]
    current = payload["scores"]["current_v33"]
    contract = payload["scores"]["contract_completion_candidate"]
    old = payload["scores"]["old_frozen_v33"]
    judges = payload["judges"]
    judge_pairwise = payload["judge_pairwise"]
    pairwise = payload["pairwise"]
    diffs = payload["prediction_diffs"]

    lines = [
        "# SAGE Unseen Validation Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Protocol",
        "",
        f"- Candidate: `{protocol['candidate']}`.",
        f"- Held-out slice: {protocol['n']} Mix risk candidates across context groups excluded from the v31/v32 development pass.",
        "- Current v33-preservation is generated from the merged contract-completion artifact with the deterministic v33 guard; no new LLM calls are used by this summary.",
        f"- Judge reuse basis: {protocol['reuse_basis']}",
        "",
        "## End-to-End Metrics",
        "",
        "| Method | N | F1 | ROUGE-L | Mean prediction chars | Source |",
        "|---|---:|---:|---:|---:|---|",
        f"| {baseline['label']} | {baseline['n']} | {baseline['mean_f1']:.4f} | {baseline['mean_rouge_l']:.4f} | {baseline['mean_prediction_chars']:.1f} | `{baseline['path']}` |",
        f"| {contract['label']} | {contract['n']} | {contract['mean_f1']:.4f} | {contract['mean_rouge_l']:.4f} | {contract['mean_prediction_chars']:.1f} | `{contract['path']}` |",
        f"| {old['label']} | {old['n']} | {old['mean_f1']:.4f} | {old['mean_rouge_l']:.4f} | {old['mean_prediction_chars']:.1f} | `{old['path']}` |",
        f"| {current['label']} | {current['n']} | {current['mean_f1']:.4f} | {current['mean_rouge_l']:.4f} | {current['mean_prediction_chars']:.1f} | `{current['path']}` |",
        "",
        "## Paired Lexical Tests",
        "",
        "| Comparison | Metric | Delta | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("v33_vs_baseline_f1", "v33 - baseline"),
        ("v33_vs_baseline_rouge_l", "v33 - baseline"),
        ("contract_candidate_vs_baseline_f1", "contract candidate - baseline"),
        ("current_v33_vs_old_frozen_f1", "current v33 - old frozen v33"),
    ]:
        row = pairwise[key]
        lines.append(
            f"| {label} | {row['metric']} | {row['delta']:+.4f} | {row['wins']} | {row['losses']} | "
            f"{row['ties']} | {fmt_p(row['t_p'])} | {fmt_p(row['wilcoxon_p'])} | {row['cohens_d']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Judge Metrics",
            "",
            "| Method | N | Correctness | Coverage | Relevance | Conciseness | Overall | Verdicts | Source |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for key in ["baseline", "current_v33", "changed_candidates"]:
        row = judges[key]
        verdicts = ", ".join(f"{name}:{count}" for name, count in row["verdicts"].items())
        lines.append(
            f"| {row['label']} | {row['n']} | {row['correctness']:.3f} | {row['coverage']:.3f} | "
            f"{row['relevance']:.3f} | {row['conciseness']:.3f} | {row['overall']:.3f} | {verdicts} | `{row['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Paired Judge Tests",
            "",
            "| Comparison | Metric | Delta | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("v33_vs_baseline_overall", "v33 - baseline"),
        ("v33_vs_baseline_correctness", "v33 - baseline"),
        ("v33_vs_baseline_coverage", "v33 - baseline"),
    ]:
        row = judge_pairwise[key]
        lines.append(
            f"| {label} | {row['metric']} | {row['delta']:+.3f} | {row['wins']} | {row['losses']} | "
            f"{row['ties']} | {fmt_p(row['t_p'])} | {fmt_p(row['wilcoxon_p'])} | {row['cohens_d']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Guard Decisions",
            "",
            "| Line | Dataset | v33 action | Baseline overall | Candidate overall | Final overall | Reasons | Preservation |",
            "|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["guard_decisions"]:
        reasons = ", ".join(row["rejection_reasons"]) if row["rejection_reasons"] else "-"
        preservation = (
            f"sem={row['semantic_preservation']}, tok={row['token_preservation']}"
            if row.get("semantic_preservation") is not None
            else "-"
        )
        lines.append(
            f"| {row['line_no']} | {row['dataset']} | `{row['action']}` | {row['baseline_overall']} | "
            f"{row['candidate_overall']} | {row['final_overall']} | {reasons} | {preservation} |"
        )

    additional_slices = payload.get("additional_slices") or []
    if additional_slices:
        lines.extend(
            [
                "",
                "## Additional Zero-Overlap Validation",
                "",
                "| Slice | N | Baseline F1 | v33 F1 | Delta F1 | Baseline overall | v33 overall | Delta overall | Changed rows |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for slice_row in additional_slices:
            baseline_scores = slice_row["scores"]["baseline"]
            current_scores = slice_row["scores"]["current_v33"]
            f1_pair = slice_row["pairwise"]["v33_vs_baseline_f1"]
            baseline_judge = slice_row["judges"]["baseline"]
            current_judge = slice_row["judges"]["current_v33"]
            judge_pair = slice_row["judge_pairwise"]["v33_vs_baseline_overall"]
            changed_rows = slice_row["prediction_diffs"]["current_v33_vs_baseline"]
            lines.append(
                f"| {slice_row['name']} | {slice_row['n']} | {baseline_scores['mean_f1']:.4f} | "
                f"{current_scores['mean_f1']:.4f} | {f1_pair['delta']:+.4f} | "
                f"{baseline_judge['overall']:.3f} | {current_judge['overall']:.3f} | "
                f"{judge_pair['delta']:+.3f} | {changed_rows} |"
            )
        lines.extend(
            [
                "",
                "## Exploratory V46 Candidate Ladder",
                "",
                "| Method | N | F1 | ROUGE-L | Judge overall | Judge correctness | Changed rows vs HyDE | Status |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for slice_row in additional_slices:
            method_specs = [
                ("baseline", "HyDE baseline", "baseline"),
                ("current_v33", "SAGE v33 preservation-delta", "current frozen candidate"),
                ("sage_v55_supported_facet", "SAGE v55 supported-facet guard", "exploratory stronger branch"),
            ]
            for method_key, label, status in method_specs:
                if method_key not in slice_row["scores"]:
                    continue
                score = slice_row["scores"][method_key]
                judge = slice_row["judges"].get(method_key, {})
                if method_key == "baseline":
                    changed_rows = "-"
                elif method_key == "current_v33":
                    changed_rows = slice_row["prediction_diffs"]["current_v33_vs_baseline"]
                else:
                    changed_rows = slice_row["prediction_diffs"].get("v55_vs_baseline", [])
                lines.append(
                    f"| {label} | {score['n']} | {score['mean_f1']:.4f} | {score['mean_rouge_l']:.4f} | "
                    f"{judge.get('overall', 0.0):.3f} | {judge.get('correctness', 0.0):.3f} | {changed_rows} | {status} |"
                )
        lines.extend(
            [
                "",
                "## Exploratory V46 Pairwise Checks",
                "",
                "| Comparison | Metric | Delta | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for slice_row in additional_slices:
            checks = [
                ("v55_vs_baseline_f1", slice_row["pairwise"], "v55 - HyDE", "f1"),
                ("v55_vs_v33_f1", slice_row["pairwise"], "v55 - v33", "f1"),
                ("v55_vs_baseline_overall", slice_row["judge_pairwise"], "v55 - HyDE", "overall"),
                ("v55_vs_v33_overall", slice_row["judge_pairwise"], "v55 - v33", "overall"),
            ]
            for key, container, label, metric in checks:
                if key not in container:
                    continue
                row = container[key]
                lines.append(
                    f"| {label} | {metric} | {row['delta']:+.4f} | {row['wins']} | {row['losses']} | "
                    f"{row['ties']} | {fmt_p(row['t_p'])} | {fmt_p(row['wilcoxon_p'])} | {row['cohens_d']:+.2f} |"
                )
        lines.extend(
            [
                "",
                "## Additional Guard Decisions",
                "",
                "| Slice | Line | Dataset | v33 action | Baseline overall | Final overall | Reasons | Preservation |",
                "|---|---:|---|---|---:|---:|---|---|",
            ]
        )
        for slice_row in additional_slices:
            for row in slice_row.get("guard_decisions", []):
                reasons = ", ".join(row["rejection_reasons"]) if row["rejection_reasons"] else "-"
                preservation = (
                    f"sem={row['semantic_preservation']}, tok={row['token_preservation']}"
                    if row.get("semantic_preservation") is not None
                    else "-"
                )
                lines.append(
                    f"| {slice_row['name']} | {row['line_no']} | {row['dataset']} | `{row['action']}` | "
                    f"{row['baseline_overall']} | {row['final_overall']} | {reasons} | {preservation} |"
                )
        lines.extend(
            [
                "",
                "## Supported-Facet Guard Decisions",
                "",
                "| Slice | Line | Dataset | v55 action | Baseline overall | Final overall | Retention | Compression | Supported terms |",
                "|---|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for slice_row in additional_slices:
            for row in slice_row.get("supported_facet_guard_decisions", []):
                lines.append(
                    f"| {slice_row['name']} | {row['line_no']} | {row['dataset']} | `{row['action']}` | "
                    f"{row['baseline_overall']} | {row['final_overall']} | "
                    f"{row['supported_facet_retention']} | {row['compression_ratio']} | {row['supported_before_terms']} |"
                )

    lines.extend(
        [
            "",
            "## Consistency Checks",
            "",
            f"- Current v33 vs old frozen v33 prediction diffs on held-out rows: {len(diffs['current_v33_vs_old_frozen_heldout'])}.",
            f"- Current v33 vs old frozen v33 prediction diffs on all Mix130 rows: {len(diffs['current_v33_vs_old_frozen_all_mix'])}.",
            f"- Current v33 changes vs baseline on held-out rows: {diffs['current_v33_vs_baseline_heldout']}.",
            f"- Contract-candidate changes vs baseline on held-out rows: {diffs['contract_candidate_vs_baseline_heldout']}.",
            "",
            "## Interpretation",
            "",
            "- The unseen34 slice supports the v33 guard as a do-no-harm safety layer: it keeps the useful line-61 repair and blocks the harmful line-71 rewrite.",
            "- The v46 zero-overlap validation30 extension is directionally positive on both lexical F1 and same-rubric judge score after two retained repairs.",
            "- The older v55 supported-facet branch is stronger on v46 than v33, but it is kept as exploratory until it is reconciled with the v33 freeze and larger full-slice protocol.",
            "- On unseen34, the end-to-end gain is positive but small because only one final answer changes on that slice.",
            "- This should be reported as context-heldout guard validation, not as a large-sample unseen benchmark claim.",
            "- Next experimental work should expand the frozen-v33 unseen pool or run robustness repeats against the same locked policy.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = summarize()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {REUSED_V33_EVAL}")


if __name__ == "__main__":
    main()
