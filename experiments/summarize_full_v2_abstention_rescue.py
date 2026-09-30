#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an offline 130-row hybrid: full_v2 plus long-context rescue for abstentions."""
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_enhanced_experiments import (  # noqa: E402
    aggregate_results,
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
FULL_V2_FILE = ALIGNED_DIR / "full_v2_aligned_n130.json"
FULL_V1_FILE = ALIGNED_DIR / "full_aligned_n130.json"
NO_REPAIR_FILE = ALIGNED_DIR / "w_o_repair_aligned_n130.json"
W_O_ALL_FILE = SOURCE_ABLATION_DIR / "w_o_all_n130.json"
RESCUE_FILES = [
    ALIGNED_DIR / "full_v2_context_rescue_audit-actions_n16_ctx40000.json",
    ALIGNED_DIR / "full_v2_context_rescue_ids_n2_ctx40000.json",
]
OUTPUT_JSON = ALIGNED_DIR / "full_v2_abstention_rescue_hybrid_n130.json"
OUTPUT_MD = ALIGNED_DIR / "FULL_V2_ABSTENTION_RESCUE_HYBRID_SUMMARY.md"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["question_id"]: row for row in rows}


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def load_rescue_rows() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in RESCUE_FILES:
        if not path.exists():
            continue
        for row in load_json(path).get("results", []):
            if "error" not in row:
                rows[row["question_id"]] = row
    return rows


def hybrid_row(base_row: Dict[str, Any], rescue_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if rescue_row is None or not base_row["metrics"].get("abstention_language"):
        row = json.loads(json.dumps(base_row, ensure_ascii=False))
        row["method"] = "full_v2_abstention_rescue_hybrid"
        row["source_branch"] = "full_v2"
        row["metrics"]["rescue_applied"] = False
        row["metrics"]["baseline_full_v2_f1"] = base_row["metrics"]["f1"]
        row["metrics"]["delta_vs_full_v2_f1"] = 0.0
        return row

    row = json.loads(json.dumps(rescue_row, ensure_ascii=False))
    row["method"] = "full_v2_abstention_rescue_hybrid"
    row["source_branch"] = "long_context_rescue_abstention"
    row["baseline_full_v2_prediction"] = base_row.get("prediction", "")
    row["metrics"]["rescue_applied"] = True
    row["metrics"]["baseline_full_v2_f1"] = base_row["metrics"]["f1"]
    row["metrics"]["delta_vs_full_v2_f1"] = row["metrics"]["f1"] - base_row["metrics"]["f1"]
    return row


def compare(left_rows: List[Dict[str, Any]], right_rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    left_by_id = by_id(left_rows)
    right_by_id = by_id(right_rows)
    ids = [qid for qid in left_by_id if qid in right_by_id]
    left = [left_by_id[qid]["metrics"][metric] for qid in ids]
    right = [right_by_id[qid]["metrics"][metric] for qid in ids]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "metric": metric,
        "n": len(ids),
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
    full_v2_payload = load_json(FULL_V2_FILE)
    full_v2_rows = full_v2_payload["results"]
    rescue_by_id = load_rescue_rows()
    missing = [
        row["question_id"]
        for row in full_v2_rows
        if row["metrics"].get("abstention_language") and row["question_id"] not in rescue_by_id
    ]
    if missing:
        raise RuntimeError(f"Missing rescue rows for abstention ids: {missing}")

    hybrid_rows = [hybrid_row(row, rescue_by_id.get(row["question_id"])) for row in full_v2_rows]
    rescue_rows = [row for row in hybrid_rows if row["metrics"].get("rescue_applied")]
    methods = {
        "hybrid": hybrid_rows,
        "full_v2": full_v2_rows,
        "full_v1": load_json(FULL_V1_FILE)["results"],
        "w/o_repair": load_json(NO_REPAIR_FILE)["results"],
        "w/o_all": load_json(W_O_ALL_FILE)["results"],
    }
    aggregate = {name: aggregate_results(rows, name) for name, rows in methods.items()}
    for name, rows in methods.items():
        aggregate[name]["abstention_language_rate"] = mean(
            [
                1.0
                if row["metrics"].get("abstention_language", detect_abstention_language(row.get("prediction", "")))
                else 0.0
                for row in rows
            ]
        )
        aggregate[name]["meta_audit_leak_rate"] = mean(
            [
                1.0
                if row["metrics"].get("meta_audit_leak", detect_meta_audit_leak(row.get("prediction", "")))
                else 0.0
                for row in rows
            ]
        )
    aggregate["hybrid"]["rescue_applied_count"] = len(rescue_rows)
    aggregate["hybrid"]["mean_rescue_delta_f1"] = mean(
        [row["metrics"]["delta_vs_full_v2_f1"] for row in rescue_rows]
    )

    comparisons = {
        baseline: {
            metric: compare(hybrid_rows, rows, metric)
            for metric in ["f1", "rouge_l", "faithfulness"]
        }
        for baseline, rows in methods.items()
        if baseline != "hybrid"
    }
    payload = {
        "experiment": "full_v2_abstention_rescue_hybrid",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "baseline": str(FULL_V2_FILE.relative_to(ROOT_DIR)),
            "gate": "if full_v2 answer contains abstention-language, replace with 40k-context rescue result",
            "rescue_sources": [str(path.relative_to(ROOT_DIR)) for path in RESCUE_FILES],
            "note": "Offline hybrid from actual rescue reruns; deployment-plausible gate because it uses answer text, not gold labels.",
        },
        "aggregated": aggregate,
        "comparisons": comparisons,
        "results": hybrid_rows,
    }
    write_json(OUTPUT_JSON, payload)

    lines = [
        "# Full V2 Abstention-Rescue Hybrid",
        "",
        f"> Generated: {payload['created_at']}",
        "> Protocol: replace only full_v2 rows that contain abstention-language with actual 40k-context rescue outputs.",
        "",
        "## Aggregate",
        "",
        "| Method | F1 | ROUGE-L | Faithfulness | Repair% | Abstain% | MetaLeak% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["w/o_all", "w/o_repair", "full_v1", "full_v2", "hybrid"]:
        agg = aggregate[name]
        lines.append(
            f"| {name} | {agg['mean_f1']:.4f} | {agg['mean_rouge_l']:.4f} | {agg['mean_faithfulness']:.4f} | "
            f"{agg['repair_rate']:.1%} | {agg['abstention_language_rate']:.1%} | {agg['meta_audit_leak_rate']:.1%} |"
        )

    lines.extend([
        "",
        "## Hybrid vs Baselines",
        "",
        "| Baseline | Metric | Mean Diff | t-p | Wilcoxon-p | Cohen's d | Improved/Worse/Same | 95% Bootstrap CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for baseline in ["full_v2", "full_v1", "w/o_repair", "w/o_all"]:
        for metric in ["f1", "rouge_l", "faithfulness"]:
            row = comparisons[baseline][metric]
            ci = row["bootstrap_ci"]
            lines.append(
                f"| {baseline} | {metric} | {row['mean_diff']:+.4f} | {row['t_p_value']:.4g} | "
                f"{row['wilcoxon_p_value']:.4g} | {row['cohens_d']:+.2f} | "
                f"{row['improved']} / {row['worse']} / {row['same']} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )

    lines.extend([
        "",
        "## Rescued Abstention Rows",
        "",
        "| ID | Domain | Baseline F1 | Rescue F1 | Delta | Faith | Abstain After Rescue |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(rescue_rows, key=lambda r: r["metrics"]["delta_vs_full_v2_f1"], reverse=True):
        m = row["metrics"]
        lines.append(
            f"| {row['question_id']} | {row.get('domain', '')} | {m['baseline_full_v2_f1']:.4f} | "
            f"{m['f1']:.4f} | {m['delta_vs_full_v2_f1']:+.4f} | {m['faithfulness']:.2f} | "
            f"{m.get('abstention_language', False)} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This is not yet a single online runner result; it is an offline hybrid assembled from actual rescue reruns.",
        "- The gate itself is deployment-plausible because it depends only on the generated answer containing abstention-language.",
        "- The next implementation step is to add this fallback to the runner and rerun a small smoke test before a full 130 pass.",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(
        f"hybrid F1={aggregate['hybrid']['mean_f1']:.4f} "
        f"delta_vs_full_v2={comparisons['full_v2']['f1']['mean_diff']:+.4f} "
        f"rescue_applied={len(rescue_rows)}"
    )


if __name__ == "__main__":
    main()
