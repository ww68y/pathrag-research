#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare secondary-gate candidates using existing full_v2_rescue diagnostics."""
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List


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
CURRENT_FILE = ALIGNED_DIR / "full_v2_rescue_aligned_n130.json"
DIAGNOSTIC_FILE = ALIGNED_DIR / "full_v2_context_rescue_ids_n10_ctx40000.json"
QUALITY_AUDIT_FILE = ALIGNED_DIR / "full_v2_rescue_quality_audit_flags_n10.json"
OUTPUT_JSON = ALIGNED_DIR / "full_v2_rescue_gate_candidates_n130.json"
OUTPUT_MD = ALIGNED_DIR / "FULL_V2_RESCUE_GATE_CANDIDATES_SUMMARY.md"


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
    return {row["question_id"]: row for row in rows if "error" not in row}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    return {
        "sample_size": len(rows),
        "mean_f1": mean([m["f1"] for m in metrics]),
        "mean_rouge_l": mean([m["rouge_l"] for m in metrics]),
        "mean_faithfulness": mean([m.get("faithfulness", 0.0) for m in metrics]),
        "repair_rate": mean([1.0 if m.get("repair_triggered") else 0.0 for m in metrics]),
        "primary_rescue_rate": mean([1.0 if m.get("abstention_rescue_applied") else 0.0 for m in metrics]),
        "secondary_rescue_rate": mean([1.0 if m.get("secondary_rescue_applied") else 0.0 for m in metrics]),
        "abstention_language_rate": mean([1.0 if detect_abstention_language(row.get("prediction", "")) else 0.0 for row in rows]),
        "meta_audit_leak_rate": mean([1.0 if detect_meta_audit_leak(row.get("prediction", "")) else 0.0 for row in rows]),
        "low_faithfulness_count": sum(1 for m in metrics if m.get("faithfulness", 1.0) < 0.9),
    }


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


def apply_diagnostic(row: Dict[str, Any], diagnostic_row: Dict[str, Any], policy: str) -> Dict[str, Any]:
    updated = deepcopy(row)
    updated["prediction"] = diagnostic_row["prediction"]
    updated["method"] = f"full_v2_rescue_{policy}"
    for key in [
        "f1",
        "rouge_l",
        "exact_match",
        "contains",
        "faithfulness",
        "coverage",
        "repair_triggered",
        "pre_repair_faithfulness",
        "pre_repair_coverage",
        "post_repair_faithfulness",
        "post_repair_coverage",
        "meta_audit_leak",
        "abstention_language",
    ]:
        if key in diagnostic_row.get("metrics", {}):
            updated["metrics"][key] = diagnostic_row["metrics"][key]
    updated["metrics"]["abstention_language"] = detect_abstention_language(updated.get("prediction", ""))
    updated["metrics"]["meta_audit_leak"] = detect_meta_audit_leak(updated.get("prediction", ""))
    updated["metrics"]["secondary_rescue_applied"] = True
    updated["metrics"]["secondary_rescue_policy"] = policy
    updated["trace"]["secondary_rescue_source"] = str(DIAGNOSTIC_FILE.relative_to(ROOT_DIR))
    updated["trace"]["secondary_rescue_original_prediction"] = row.get("prediction", "")
    return updated


def build_candidate(
    current_rows: List[Dict[str, Any]],
    diagnostic_by_id: Dict[str, Dict[str, Any]],
    policy: str,
    accept: Callable[[Dict[str, Any], Dict[str, Any]], bool],
) -> List[Dict[str, Any]]:
    rows = []
    for row in current_rows:
        diagnostic_row = diagnostic_by_id.get(row["question_id"])
        if diagnostic_row and accept(row, diagnostic_row):
            rows.append(apply_diagnostic(row, diagnostic_row, policy))
        else:
            kept = deepcopy(row)
            kept["metrics"].setdefault("secondary_rescue_applied", False)
            rows.append(kept)
    return rows


def main() -> None:
    current_rows = [row for row in load_json(CURRENT_FILE)["results"] if "error" not in row]
    diagnostic_rows = [row for row in load_json(DIAGNOSTIC_FILE)["results"] if "error" not in row]
    diagnostic_by_id = by_id(diagnostic_rows)
    audit_rows = load_json(QUALITY_AUDIT_FILE)["results"] if QUALITY_AUDIT_FILE.exists() else []
    audit_by_id = by_id(audit_rows)

    def diagnostic_not_abstinent(diagnostic_row: Dict[str, Any]) -> bool:
        return not detect_abstention_language(diagnostic_row.get("prediction", ""))

    def diagnostic_high_conf(diagnostic_row: Dict[str, Any], faith: float = 0.9, coverage: float = 0.8) -> bool:
        metrics = diagnostic_row["metrics"]
        return (
            metrics.get("faithfulness", 0.0) >= faith
            and metrics.get("coverage", 0.0) >= coverage
            and diagnostic_not_abstinent(diagnostic_row)
        )

    candidates = {
        "current_full_v2_rescue": current_rows,
        "force_all_diagnostic_flags": build_candidate(
            current_rows,
            diagnostic_by_id,
            "force_all_diagnostic_flags",
            lambda _row, _diag: True,
        ),
        "current_abstention_high_conf_diagnostic": build_candidate(
            current_rows,
            diagnostic_by_id,
            "current_abstention_high_conf_diagnostic",
            lambda row, diag: detect_abstention_language(row.get("prediction", "")) and diagnostic_high_conf(diag),
        ),
        "low_faith_high_conf_diagnostic": build_candidate(
            current_rows,
            diagnostic_by_id,
            "low_faith_high_conf_diagnostic",
            lambda row, diag: row["metrics"].get("faithfulness", 1.0) < 0.9 and diagnostic_high_conf(diag),
        ),
        "audit_recommended_high_conf_diagnostic": build_candidate(
            current_rows,
            diagnostic_by_id,
            "audit_recommended_high_conf_diagnostic",
            lambda row, diag: audit_by_id.get(row["question_id"], {})
            .get("audit", {})
            .get("recommended_action")
            in {"rerun_with_more_context", "improve_evidence_extraction", "improve_answer_coverage"}
            and diagnostic_high_conf(diag),
        ),
        "oracle_positive_diagnostic_upper_bound": build_candidate(
            current_rows,
            diagnostic_by_id,
            "oracle_positive_diagnostic_upper_bound",
            lambda row, diag: diag["metrics"].get("f1", 0.0) > row["metrics"].get("f1", 0.0),
        ),
    }

    aggregated = {name: aggregate(rows) for name, rows in candidates.items()}
    comparisons = {
        name: {metric: compare(rows, current_rows, metric) for metric in ["f1", "rouge_l", "faithfulness"]}
        for name, rows in candidates.items()
        if name != "current_full_v2_rescue"
    }
    applied_ids = {
        name: [row["question_id"] for row in rows if row["metrics"].get("secondary_rescue_applied")]
        for name, rows in candidates.items()
    }
    diagnostic_table = []
    current_by_id = by_id(current_rows)
    for row in diagnostic_rows:
        qid = row["question_id"]
        current = current_by_id[qid]
        diagnostic_table.append(
            {
                "question_id": qid,
                "current_f1": current["metrics"]["f1"],
                "diagnostic_f1": row["metrics"]["f1"],
                "delta_vs_current": row["metrics"]["f1"] - current["metrics"]["f1"],
                "diagnostic_faithfulness": row["metrics"].get("faithfulness", 0.0),
                "diagnostic_coverage": row["metrics"].get("coverage", 0.0),
                "current_abstention": detect_abstention_language(current.get("prediction", "")),
                "diagnostic_abstention": detect_abstention_language(row.get("prediction", "")),
                "audit_issue": audit_by_id.get(qid, {}).get("audit", {}).get("issue_type", ""),
                "audit_action": audit_by_id.get(qid, {}).get("audit", {}).get("recommended_action", ""),
            }
        )

    payload = {
        "experiment": "full_v2_rescue_secondary_gate_candidates",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_source": str(CURRENT_FILE.relative_to(ROOT_DIR)),
        "diagnostic_source": str(DIAGNOSTIC_FILE.relative_to(ROOT_DIR)),
        "quality_audit_source": str(QUALITY_AUDIT_FILE.relative_to(ROOT_DIR)),
        "note": "Oracle candidate uses gold F1 and is an upper bound, not a deployable policy.",
        "aggregated": aggregated,
        "comparisons_vs_current_full_v2_rescue": comparisons,
        "secondary_rescue_applied_ids": applied_ids,
        "diagnostic_rows": diagnostic_table,
    }
    write_json(OUTPUT_JSON, payload)

    lines = [
        "# Full V2 Rescue Secondary Gate Candidates",
        "",
        f"> Generated: {payload['created_at']}",
        f"> Current source: `{payload['current_source']}`",
        f"> Diagnostic source: `{payload['diagnostic_source']}`",
        "",
        "## Experiment Type",
        "",
        "- `w/o_*` comparisons are strict ablation experiments: modules are removed to estimate contribution.",
        "- `full_v2_rescue` is an enhancement/variant experiment: a new abstention-triggered rescue module is added to the best pipeline.",
        "- quality audits and forced 40k runs are diagnostic experiments: they identify residual failure modes and gate design risk.",
        "",
        "## Candidate Results",
        "",
        "| Candidate | F1 | ROUGE-L | Faith | Secondary Rescue% | Abstain% | LowFaith Count | Applied IDs |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in [
        "current_full_v2_rescue",
        "force_all_diagnostic_flags",
        "current_abstention_high_conf_diagnostic",
        "low_faith_high_conf_diagnostic",
        "audit_recommended_high_conf_diagnostic",
        "oracle_positive_diagnostic_upper_bound",
    ]:
        agg = aggregated[name]
        ids = ", ".join(applied_ids[name]) or "-"
        lines.append(
            f"| {name} | {agg['mean_f1']:.4f} | {agg['mean_rouge_l']:.4f} | {agg['mean_faithfulness']:.4f} | "
            f"{agg['secondary_rescue_rate']:.1%} | {agg['abstention_language_rate']:.1%} | "
            f"{agg['low_faithfulness_count']} | {ids} |"
        )

    lines.extend([
        "",
        "## Comparisons vs Current",
        "",
        "| Candidate | Metric | Mean Diff | t-p | Wilcoxon-p | Cohen's d | Improved/Worse/Same | 95% Bootstrap CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in [
        "force_all_diagnostic_flags",
        "current_abstention_high_conf_diagnostic",
        "low_faith_high_conf_diagnostic",
        "audit_recommended_high_conf_diagnostic",
        "oracle_positive_diagnostic_upper_bound",
    ]:
        for metric in ["f1", "rouge_l", "faithfulness"]:
            row = comparisons[name][metric]
            ci = row["bootstrap_ci"]
            lines.append(
                f"| {name} | {metric} | {row['mean_diff']:+.4f} | {row['t_p_value']:.4g} | "
                f"{row['wilcoxon_p_value']:.4g} | {row['cohens_d']:+.2f} | "
                f"{row['improved']} / {row['worse']} / {row['same']} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )

    lines.extend([
        "",
        "## Diagnostic Rows",
        "",
        "| ID | Current F1 | Diagnostic F1 | Delta | Faith | Coverage | Current Abstain | Diagnostic Abstain | Audit Issue | Action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in sorted(diagnostic_table, key=lambda item: item["delta_vs_current"], reverse=True):
        lines.append(
            f"| {row['question_id']} | {row['current_f1']:.4f} | {row['diagnostic_f1']:.4f} | "
            f"{row['delta_vs_current']:+.4f} | {row['diagnostic_faithfulness']:.2f} | "
            f"{row['diagnostic_coverage']:.2f} | {row['current_abstention']} | {row['diagnostic_abstention']} | "
            f"{row['audit_issue']} | {row['audit_action']} |"
        )

    lines.extend([
        "",
        "## Takeaway",
        "",
        "- The current abstention-triggered rescue is already the clean deployable variant with strong rescue-only effect.",
        "- A naive second-stage forced rescue over all residual flags is not robust enough because several rows lose F1.",
        "- The oracle upper bound shows there is still recoverable headroom, but converting it into a deployable gate needs better confidence features than verifier faithfulness alone.",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(
        f"current_f1={aggregated['current_full_v2_rescue']['mean_f1']:.4f} "
        f"oracle_upper_bound_f1={aggregated['oracle_positive_diagnostic_upper_bound']['mean_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
