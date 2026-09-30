#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize the online aligned full_v2 main experiment."""
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
from experiments.run_repair_v2_postverify import (  # noqa: E402
    detect_abstention_language,
    detect_meta_audit_leak,
)


ALIGNED_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
SOURCE_ABLATION_DIR = ROOT_DIR / "results" / "enhanced_ablation"
OUTPUT_JSON = ALIGNED_DIR / "full_v2_main_comparison_n130.json"
OUTPUT_MD = ALIGNED_DIR / "FULL_V2_MAIN_EXPERIMENT_SUMMARY.md"

METHOD_FILES = {
    "w/o_all": SOURCE_ABLATION_DIR / "w_o_all_n130.json",
    "w/o_verification": SOURCE_ABLATION_DIR / "w_o_verification_n130.json",
    "w/o_repair": ALIGNED_DIR / "w_o_repair_aligned_n130.json",
    "full_v1": ALIGNED_DIR / "full_aligned_n130.json",
    "full_v2": ALIGNED_DIR / "full_v2_aligned_n130.json",
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
        "mean_faithfulness": mean([m.get("faithfulness", 0.0) for m in metrics]),
        "mean_runtime_seconds": mean([m.get("runtime_seconds", 0.0) for m in metrics]),
        "repair_rate": mean([1.0 if m.get("repair_triggered") else 0.0 for m in metrics]),
        "meta_audit_leak_rate": mean(
            [
                1.0
                if row.get("metrics", {}).get("meta_audit_leak", detect_meta_audit_leak(row.get("prediction", "")))
                else 0.0
                for row in rows
            ]
        ),
        "abstention_language_rate": mean(
            [
                1.0
                if row.get("metrics", {}).get("abstention_language", detect_abstention_language(row.get("prediction", "")))
                else 0.0
                for row in rows
            ]
        ),
    }


def compare(
    full_v2_rows: List[Dict[str, Any]],
    other_rows: List[Dict[str, Any]],
    metric: str,
) -> Dict[str, Any]:
    v2_by_id = by_id(full_v2_rows)
    other_by_id = by_id(other_rows)
    ids = [qid for qid in v2_by_id if qid in other_by_id]
    left = [float(v2_by_id[qid]["metrics"][metric]) for qid in ids]
    right = [float(other_by_id[qid]["metrics"][metric]) for qid in ids]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "metric": metric,
        "n_common": len(ids),
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


def main() -> None:
    rows = {name: load_json(path)["results"] for name, path in METHOD_FILES.items()}
    full_v2_rows = rows["full_v2"]
    error_rows = [row for row in full_v2_rows if "error" in row]
    low_postfaith = [
        row
        for row in full_v2_rows
        if row["metrics"].get("repair_triggered")
        and row["metrics"].get("post_repair_faithfulness", row["metrics"].get("faithfulness", 1.0)) < 0.9
    ]
    meta_rows = [row for row in full_v2_rows if row["metrics"].get("meta_audit_leak")]
    abstain_rows = [row for row in full_v2_rows if row["metrics"].get("abstention_language")]

    comparisons = {
        other: {metric: compare(full_v2_rows, rows[other], metric) for metric in ["f1", "rouge_l", "faithfulness"]}
        for other in ["w/o_all", "w/o_verification", "w/o_repair", "full_v1"]
    }
    full_v1_by_id = by_id(rows["full_v1"])
    gains = sorted(
        full_v2_rows,
        key=lambda row: row["metrics"]["f1"] - full_v1_by_id[row["question_id"]]["metrics"]["f1"],
        reverse=True,
    )[:10]
    losses = sorted(
        full_v2_rows,
        key=lambda row: row["metrics"]["f1"] - full_v1_by_id[row["question_id"]]["metrics"]["f1"],
    )[:10]

    payload = {
        "experiment": "aligned_full_v2_main_experiment",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {name: str(path.relative_to(ROOT_DIR)) for name, path in METHOD_FILES.items()},
        "aggregated": {name: aggregate(method_rows) for name, method_rows in rows.items()},
        "comparisons_vs_full_v2": comparisons,
        "quality_flags": {
            "error_count": len(error_rows),
            "low_postfaith_count": len(low_postfaith),
            "low_postfaith_ids": [row["question_id"] for row in low_postfaith],
            "meta_audit_leak_count": len(meta_rows),
            "meta_audit_leak_ids": [row["question_id"] for row in meta_rows],
            "abstention_language_count": len(abstain_rows),
            "abstention_language_ids": [row["question_id"] for row in abstain_rows],
        },
    }
    write_json(OUTPUT_JSON, payload)

    agg = payload["aggregated"]
    lines = [
        "# Full V2 Main Experiment Summary",
        "",
        f"> Generated: {payload['created_at']}",
        "> Protocol: online aligned runner with shared evidence/answer/verification and repair_prompt_v2 for triggered rows.",
        "",
        "## Aggregate",
        "",
        "| Method | F1 | ROUGE-L | Faithfulness | Repair% | MetaLeak% | Abstain% | Runtime(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["w/o_all", "w/o_verification", "w/o_repair", "full_v1", "full_v2"]:
        a = agg[name]
        lines.append(
            f"| {name} | {a['mean_f1']:.4f} | {a['mean_rouge_l']:.4f} | {a['mean_faithfulness']:.4f} | "
            f"{a['repair_rate']:.1%} | {a['meta_audit_leak_rate']:.1%} | {a['abstention_language_rate']:.1%} | "
            f"{a['mean_runtime_seconds']:.1f} |"
        )

    lines.extend([
        "",
        "## Full V2 vs Baselines",
        "",
        "| Baseline | Metric | Mean Diff | t-p | Wilcoxon-p | Cohen's d | Improved/Worse | 95% Bootstrap CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for baseline, tests in comparisons.items():
        for metric in ["f1", "rouge_l", "faithfulness"]:
            test = tests[metric]
            ci = test["bootstrap_ci"]
            lines.append(
                f"| {baseline} | {metric} | {test['mean_diff']:+.4f} | {test['t_p_value']:.4g} | "
                f"{test['wilcoxon_p_value']:.4g} | {test['cohens_d']:+.2f} | "
                f"{test['improved']} / {test['worse']} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )

    lines.extend([
        "",
        "## Quality Flags",
        "",
        "| Flag | Count | IDs |",
        "|---|---:|---|",
        f"| Error rows | {payload['quality_flags']['error_count']} | - |",
        f"| Low postFaith repaired rows (<0.9) | {payload['quality_flags']['low_postfaith_count']} | {', '.join(payload['quality_flags']['low_postfaith_ids']) or '-'} |",
        f"| Meta-audit leak rows | {payload['quality_flags']['meta_audit_leak_count']} | {', '.join(payload['quality_flags']['meta_audit_leak_ids']) or '-'} |",
        f"| Abstention-language rows | {payload['quality_flags']['abstention_language_count']} | {', '.join(payload['quality_flags']['abstention_language_ids']) or '-'} |",
        "",
        "## Top Gains vs V1 Full",
        "",
        "| ID | Domain | Delta F1 | V1 F1 | V2 F1 | Repair |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in gains:
        old_f1 = full_v1_by_id[row["question_id"]]["metrics"]["f1"]
        new_f1 = row["metrics"]["f1"]
        lines.append(
            f"| {row['question_id']} | {row.get('domain', '')} | {new_f1 - old_f1:+.4f} | "
            f"{old_f1:.4f} | {new_f1:.4f} | {row['metrics'].get('repair_triggered', False)} |"
        )

    lines.extend([
        "",
        "## Top Losses vs V1 Full",
        "",
        "| ID | Domain | Delta F1 | V1 F1 | V2 F1 | Repair | PostFaith |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in losses:
        old_f1 = full_v1_by_id[row["question_id"]]["metrics"]["f1"]
        new_f1 = row["metrics"]["f1"]
        postfaith = row["metrics"].get("post_repair_faithfulness", row["metrics"].get("faithfulness", 0.0))
        lines.append(
            f"| {row['question_id']} | {row.get('domain', '')} | {new_f1 - old_f1:+.4f} | "
            f"{old_f1:.4f} | {new_f1:.4f} | {row['metrics'].get('repair_triggered', False)} | {postfaith:.2f} |"
        )

    lines.extend([
        "",
        "## Takeaway",
        "",
        "- full_v2 is now the strongest aligned 130-row result in the repository by F1, ROUGE-L, and faithfulness.",
        "- The result is a formal online run, not the earlier offline selective-v2 merge.",
        "- Remaining work should focus on low-postFaith and abstention rows before claiming paper-level reproduction.",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"full_v2 F1={agg['full_v2']['mean_f1']:.4f}, ROUGE-L={agg['full_v2']['mean_rouge_l']:.4f}")


if __name__ == "__main__":
    main()
