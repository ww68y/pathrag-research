#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize the formal online full_v2_rescue runner result."""
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
    compute_contains,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    paired_t_test,
    wilcoxon_signed_rank_test,
)
from experiments.run_repair_v2_postverify import (  # noqa: E402
    detect_abstention_language,
    detect_meta_audit_leak,
)


ALIGNED_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
SOURCE_ABLATION_DIR = ROOT_DIR / "results" / "enhanced_ablation"
RESCUE_FILE = ALIGNED_DIR / "full_v2_rescue_aligned_n130.json"
METHOD_FILES = {
    "w/o_all": SOURCE_ABLATION_DIR / "w_o_all_n130.json",
    "w/o_repair": ALIGNED_DIR / "w_o_repair_aligned_n130.json",
    "full_v1": ALIGNED_DIR / "full_aligned_n130.json",
    "full_v2_existing": ALIGNED_DIR / "full_v2_aligned_n130.json",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["question_id"]: row for row in rows}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    return {
        "sample_size": len(rows),
        "mean_f1": mean([m["f1"] for m in metrics]),
        "mean_rouge_l": mean([m["rouge_l"] for m in metrics]),
        "mean_exact_match": mean([m.get("exact_match", 0.0) for m in metrics]),
        "mean_contains": mean([m.get("contains", 0.0) for m in metrics]),
        "mean_faithfulness": mean([m.get("faithfulness", 0.0) for m in metrics]),
        "mean_runtime_seconds": mean([m.get("runtime_seconds", 0.0) for m in metrics]),
        "repair_rate": mean([1.0 if m.get("repair_triggered") else 0.0 for m in metrics]),
        "rescue_applied_rate": mean([1.0 if m.get("abstention_rescue_applied") else 0.0 for m in metrics]),
        "rescue_applied_count": sum(1 for m in metrics if m.get("abstention_rescue_applied")),
        "meta_audit_leak_rate": mean(
            [
                1.0 if detect_meta_audit_leak(row.get("prediction", "")) else 0.0
                for row in rows
            ]
        ),
        "abstention_language_rate": mean(
            [
                1.0 if detect_abstention_language(row.get("prediction", "")) else 0.0
                for row in rows
            ]
        ),
    }


def synthetic_same_run_baseline(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_rows = []
    for row in rows:
        answer = row.get("trace", {}).get("baseline_full_v2_answer")
        if not answer:
            continue
        gold = row["gold_answer"]
        trace = row.get("trace", {})
        baseline_rows.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "gold_answer": gold,
                "prediction": answer,
                "method": "same_run_full_v2_baseline",
                "domain": row.get("domain", ""),
                "dataset": row.get("dataset", "ultradomain"),
                "metrics": {
                    "f1": compute_f1(answer, gold),
                    "rouge_l": compute_rouge_l(answer, gold),
                    "exact_match": compute_exact_match(answer, gold),
                    "contains": compute_contains(answer, gold),
                    "faithfulness": trace.get("baseline_full_v2_faithfulness", trace.get("faithfulness", 0.0)),
                    "repair_triggered": trace.get("baseline_full_v2_repair_triggered", trace.get("repair_triggered", False)),
                    "abstention_language": trace.get(
                        "baseline_full_v2_abstention_language",
                        detect_abstention_language(answer),
                    ),
                    "meta_audit_leak": detect_meta_audit_leak(answer),
                    "runtime_seconds": 0.0,
                },
            }
        )
    return baseline_rows


def compare(left_rows: List[Dict[str, Any]], right_rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    left_by_id = by_id(left_rows)
    right_by_id = by_id(right_rows)
    ids = [qid for qid in left_by_id if qid in right_by_id]
    left = [float(left_by_id[qid]["metrics"][metric]) for qid in ids]
    right = [float(right_by_id[qid]["metrics"][metric]) for qid in ids]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "metric": metric,
        "n": len(ids),
        "mean_left": mean(left),
        "mean_right": mean(right),
        "mean_diff": mean(diffs),
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs),
        "improved": sum(1 for d in diffs if d > 1e-9),
        "worse": sum(1 for d in diffs if d < -1e-9),
        "same": sum(1 for d in diffs if abs(d) <= 1e-9),
    }


def aligned_subset(rows: List[Dict[str, Any]], ids: List[str]) -> List[Dict[str, Any]]:
    rows_by_id = by_id(rows)
    return [rows_by_id[qid] for qid in ids if qid in rows_by_id]


def main() -> None:
    rescue_payload = load_json(RESCUE_FILE)
    rescue_rows_all = rescue_payload["results"]
    valid_rows = [row for row in rescue_rows_all if "error" not in row]
    error_rows = [row for row in rescue_rows_all if "error" in row]
    valid_ids = [row["question_id"] for row in valid_rows]
    same_run_baseline = synthetic_same_run_baseline(valid_rows)
    methods = {
        "same_run_full_v2_baseline": same_run_baseline,
        "full_v2_rescue": valid_rows,
    }
    for name, path in METHOD_FILES.items():
        if path.exists():
            methods[name] = aligned_subset(load_json(path)["results"], valid_ids)

    comparisons = {
        baseline: {
            metric: compare(valid_rows, rows, metric)
            for metric in ["f1", "rouge_l", "faithfulness"]
        }
        for baseline, rows in methods.items()
        if baseline != "full_v2_rescue" and rows
    }
    rescue_only = [row for row in valid_rows if row["metrics"].get("abstention_rescue_applied")]
    rescue_only_baseline = aligned_subset(same_run_baseline, [row["question_id"] for row in rescue_only])
    rescue_only_comparison = {
        metric: compare(rescue_only, rescue_only_baseline, metric)
        for metric in ["f1", "rouge_l", "faithfulness"]
    }
    final_abstention_rows = [row for row in valid_rows if detect_abstention_language(row.get("prediction", ""))]
    low_faith_rows = [row for row in valid_rows if row["metrics"].get("faithfulness", 1.0) < 0.9]

    output_json = ALIGNED_DIR / f"full_v2_rescue_formal_valid_n{len(valid_rows)}.json"
    output_md = ALIGNED_DIR / f"FULL_V2_RESCUE_FORMAL_VALID_N{len(valid_rows)}_SUMMARY.md"
    payload = {
        "experiment": "full_v2_rescue_formal_online",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": str(RESCUE_FILE.relative_to(ROOT_DIR)),
        "valid_sample_size": len(valid_rows),
        "error_count": len(error_rows),
        "error_ids": [row["question_id"] for row in error_rows],
        "aggregated": {name: aggregate(rows) for name, rows in methods.items()},
        "comparisons": comparisons,
        "rescue_only_comparison_vs_same_run_baseline": rescue_only_comparison,
        "quality_flags": {
            "final_abstention_count": len(final_abstention_rows),
            "final_abstention_ids": [row["question_id"] for row in final_abstention_rows],
            "low_faithfulness_count": len(low_faith_rows),
            "low_faithfulness_ids": [row["question_id"] for row in low_faith_rows],
            "meta_audit_leak_count": sum(
                1 for row in valid_rows if detect_meta_audit_leak(row.get("prediction", ""))
            ),
        },
        "results": valid_rows,
    }
    write_json(output_json, payload)

    agg = payload["aggregated"]
    lines = [
        "# Full V2 Rescue Formal Online Summary",
        "",
        f"> Generated: {payload['created_at']}",
        f"> Source: `{payload['source_file']}`",
        f"> Valid rows: {len(valid_rows)}; error rows: {len(error_rows)}",
        "",
        "## Aggregate",
        "",
        "| Method | Rows | F1 | ROUGE-L | Faithfulness | Repair% | Rescue% | Abstain% | MetaLeak% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["w/o_all", "w/o_repair", "full_v1", "full_v2_existing", "same_run_full_v2_baseline", "full_v2_rescue"]:
        if name not in agg:
            continue
        a = agg[name]
        lines.append(
            f"| {name} | {a['sample_size']} | {a['mean_f1']:.4f} | {a['mean_rouge_l']:.4f} | "
            f"{a['mean_faithfulness']:.4f} | {a['repair_rate']:.1%} | {a['rescue_applied_rate']:.1%} | "
            f"{a['abstention_language_rate']:.1%} | {a['meta_audit_leak_rate']:.1%} |"
        )

    lines.extend([
        "",
        "## Paired Comparisons",
        "",
        "| Baseline | Metric | n | Mean Diff | t-p | Wilcoxon-p | Cohen's d | Improved/Worse/Same | 95% Bootstrap CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for baseline in ["same_run_full_v2_baseline", "full_v2_existing", "full_v1", "w/o_repair", "w/o_all"]:
        if baseline not in comparisons:
            continue
        for metric in ["f1", "rouge_l", "faithfulness"]:
            row = comparisons[baseline][metric]
            ci = row["bootstrap_ci"]
            lines.append(
                f"| {baseline} | {metric} | {row['n']} | {row['mean_diff']:+.4f} | "
                f"{row['t_p_value']:.4g} | {row['wilcoxon_p_value']:.4g} | {row['cohens_d']:+.2f} | "
                f"{row['improved']} / {row['worse']} / {row['same']} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )

    lines.extend([
        "",
        "## Rescue-Only Effect",
        "",
        "| Metric | n | Mean Diff vs Same-Run Baseline | t-p | Wilcoxon-p | Cohen's d | Improved/Worse/Same | 95% Bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for metric in ["f1", "rouge_l", "faithfulness"]:
        row = rescue_only_comparison[metric]
        ci = row["bootstrap_ci"]
        lines.append(
            f"| {metric} | {row['n']} | {row['mean_diff']:+.4f} | {row['t_p_value']:.4g} | "
            f"{row['wilcoxon_p_value']:.4g} | {row['cohens_d']:+.2f} | "
            f"{row['improved']} / {row['worse']} / {row['same']} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )

    lines.extend([
        "",
        "## Quality Flags",
        "",
        "| Flag | Count | IDs |",
        "|---|---:|---|",
        f"| Error rows | {len(error_rows)} | {', '.join(payload['error_ids']) or '-'} |",
        f"| Final abstention rows | {payload['quality_flags']['final_abstention_count']} | {', '.join(payload['quality_flags']['final_abstention_ids']) or '-'} |",
        f"| Low faithfulness rows (<0.9) | {payload['quality_flags']['low_faithfulness_count']} | {', '.join(payload['quality_flags']['low_faithfulness_ids']) or '-'} |",
        f"| Meta-audit leak rows | {payload['quality_flags']['meta_audit_leak_count']} | - |",
    ])
    if error_rows:
        lines.extend([
            "",
            "## Resume Command",
            "",
            "```powershell",
            "python experiments\\run_aligned_ablation_rerun.py --configs full_v2_rescue --retry-errors --question-timeout 360 --max-retries 3",
            "python experiments\\summarize_full_v2_rescue_formal.py",
            "```",
        ])

    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print(
        f"valid_n={len(valid_rows)} rescue_f1={agg['full_v2_rescue']['mean_f1']:.4f} "
        f"same_run_delta={comparisons['same_run_full_v2_baseline']['f1']['mean_diff']:+.4f} "
        f"errors={len(error_rows)}"
    )


if __name__ == "__main__":
    main()
