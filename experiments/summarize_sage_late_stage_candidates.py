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


OUT_DIR = ROOT / "results" / "sage_late_stage_candidates"
OUTPUT_JSON = OUT_DIR / "sage_late_stage_candidates_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_LATE_STAGE_CANDIDATES_SUMMARY.md"


SLICES: list[dict[str, Any]] = [
    {
        "name": "V40 frozen lazy heldout12",
        "n": 12,
        "scope": "frozen lazy heldout; v56 has one fusion recovery over v55",
        "methods": [
            {
                "key": "hyde",
                "label": "HyDE baseline",
                "status": "baseline",
                "path": "results/historical_baselines/v40_heldout_hyde12.jsonl",
                "eval": "results/historical_baselines/v40_heldout_hyde12_eval.jsonl",
            },
            {
                "key": "v54",
                "label": "SAGE v54 contract-gap repair",
                "status": "late-stage repair",
                "path": "results/historical_sage/mse_pathrag_sage_v54_frozen_contract_gap_v40_12.jsonl",
                "eval": "results/historical_sage/mse_pathrag_sage_v54_frozen_contract_gap_v40_12_routed_eval.jsonl",
                "pairwise": "results/historical_pairwise/v54_frozen_v40_vs_hyde_pairwise.jsonl",
            },
            {
                "key": "v55",
                "label": "SAGE v55 supported-facet guard",
                "status": "late-stage guard",
                "path": "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v40_12.jsonl",
                "eval": "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v40_12_routed_eval.jsonl",
                "pairwise": "results/historical_pairwise/v55_supported_facet_v40_vs_hyde_pairwise.jsonl",
            },
            {
                "key": "v56",
                "label": "SAGE v56 supported-facet fusion",
                "status": "late-stage fusion",
                "path": "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v40_12.jsonl",
                "eval": "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v40_12_routed_eval.jsonl",
                "pairwise": "results/historical_pairwise/v56_supported_facet_fusion_v40_vs_hyde_pairwise.jsonl",
            },
        ],
    },
    {
        "name": "V42 verifier lazy heldout12",
        "n": 12,
        "scope": "frozen verifier-lazy heldout; v56 has no trigger and matches v55",
        "methods": [
            {
                "key": "hyde",
                "label": "HyDE baseline",
                "status": "baseline",
                "path": "results/historical_baselines/v42_heldout_hyde12.jsonl",
                "eval": "results/historical_baselines/v42_heldout_hyde12_eval.jsonl",
            },
            {
                "key": "v55",
                "label": "SAGE v55 supported-facet guard",
                "status": "late-stage guard",
                "path": "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v42_12.jsonl",
                "eval": "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v42_12_routed_eval.jsonl",
                "pairwise": "results/historical_pairwise/v55_supported_facet_v42_vs_hyde_pairwise.jsonl",
            },
            {
                "key": "v56",
                "label": "SAGE v56 supported-facet fusion",
                "status": "late-stage fusion; prediction-identical to v55 on this slice",
                "path": "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v42_12.jsonl",
                "eval_reuse_from": "v55",
                "pairwise_reuse_from": "v55",
            },
        ],
    },
    {
        "name": "V46 zero-overlap validation30",
        "n": 30,
        "scope": "zero-overlap validation; v56 has no trigger and matches v55/v54",
        "methods": [
            {
                "key": "hyde",
                "label": "HyDE baseline",
                "status": "baseline",
                "path": "results/historical_baselines/v46_heldout30_hyde.jsonl",
                "eval": "results/historical_baselines/v46_heldout30_hyde_eval.jsonl",
            },
            {
                "key": "v33",
                "label": "SAGE v33 preservation-delta",
                "status": "current frozen candidate",
                "path": "results/historical_vsr/v46_zero_overlap_v33_preservation_delta_30.jsonl",
                "eval": "results/historical_vsr/v46_zero_overlap_v33_preservation_eval.jsonl",
            },
            {
                "key": "v54",
                "label": "SAGE v54 contract-gap repair",
                "status": "late-stage repair",
                "path": "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30.jsonl",
                "eval": "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30_eval.jsonl",
                "pairwise": "results/historical_pairwise/v46_v54_contract_gap_vs_hyde_pairwise_merged.jsonl",
            },
            {
                "key": "v55",
                "label": "SAGE v55 supported-facet guard",
                "status": "late-stage guard; prediction-identical to v54 on this slice",
                "path": "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v46_30.jsonl",
                "eval": "results/historical_vsr/v46_zero_overlap_v55_supported_facet_reused_eval.jsonl",
                "pairwise_reuse_from": "v54",
            },
            {
                "key": "v56",
                "label": "SAGE v56 supported-facet fusion",
                "status": "late-stage fusion; prediction-identical to v55/v54 on this slice",
                "path": "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v46_30.jsonl",
                "eval_reuse_from": "v55",
                "pairwise_reuse_from": "v54",
            },
        ],
    },
]


def load_jsonl(path_or_relative: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_relative)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def score_result(path: str, label: str, status: str) -> dict[str, Any]:
    rows = load_jsonl(path)
    per_line = []
    for index, row in enumerate(rows):
        pred = prediction(row)
        refs = references(row)
        per_line.append(
            {
                "line_no": line_no(row, index),
                "dataset": row.get("dataset") or row.get("label"),
                "f1": max(compute_f1(pred, ref) for ref in refs),
                "rouge_l": max(compute_rouge_l(pred, ref) for ref in refs),
                "prediction_chars": len(pred),
            }
        )
    return {
        "label": label,
        "status": status,
        "path": path,
        "n": len(per_line),
        "mean_f1": statistics.mean(row["f1"] for row in per_line) if per_line else 0.0,
        "mean_rouge_l": statistics.mean(row["rouge_l"] for row in per_line) if per_line else 0.0,
        "mean_prediction_chars": statistics.mean(row["prediction_chars"] for row in per_line) if per_line else 0.0,
        "per_line": per_line,
    }


def judge_summary(path: str) -> dict[str, Any]:
    rows = load_jsonl(path)
    metrics = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    averages = {
        metric: statistics.mean(float(row.get("scores", {}).get(metric, 0)) for row in rows) if rows else 0.0
        for metric in metrics
    }
    verdicts = Counter(str(row.get("scores", {}).get("verdict", "unknown")) for row in rows)
    return {
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


def paired(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    right_by_line = {row["line_no"]: row for row in right_rows}
    left_values = []
    right_values = []
    for row in left_rows:
        right = right_by_line.get(row["line_no"])
        if not right:
            continue
        left_values.append(float(row[metric]))
        right_values.append(float(right[metric]))
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


def paired_judge(left: dict[str, Any], right: dict[str, Any], metric: str = "overall") -> dict[str, Any]:
    right_by_line = {row["line_no"]: row for row in right["per_line"]}
    left_values = []
    right_values = []
    for row in left["per_line"]:
        right_row = right_by_line.get(row["line_no"])
        if not right_row:
            continue
        left_values.append(float(row["scores"].get(metric, 0)))
        right_values.append(float(right_row["scores"].get(metric, 0)))
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


def prediction_diffs(left_path: str, right_path: str) -> list[int]:
    left = by_line(load_jsonl(left_path))
    right = by_line(load_jsonl(right_path))
    return [
        number
        for number in sorted(set(left) & set(right))
        if prediction(left[number]) != prediction(right[number])
    ]


def pairwise_win_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    rows = load_jsonl(path)
    dims = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]
    dim_stats = {}
    rates = []
    for dim in dims:
        pathrag = 0
        baseline = 0
        ties = 0
        for row in rows:
            winner = (row.get("winners") or {}).get(dim)
            if winner == "pathrag":
                pathrag += 1
            elif winner == "baseline":
                baseline += 1
            else:
                ties += 1
        n = pathrag + baseline + ties
        rate = (pathrag + 0.5 * ties) / n if n else 0.0
        rates.append(rate)
        dim_stats[dim] = {"pathrag": pathrag, "baseline": baseline, "ties": ties, "win_rate": rate}
    return {
        "path": path,
        "n": len(rows),
        "average_win_rate": statistics.mean(rates) if rates else 0.0,
        "dimensions": dim_stats,
    }


def method_eval_path(method: dict[str, Any], methods_by_key: dict[str, dict[str, Any]]) -> str | None:
    if method.get("eval"):
        return str(method["eval"])
    reuse = method.get("eval_reuse_from")
    if reuse:
        return method_eval_path(methods_by_key[reuse], methods_by_key)
    return None


def method_pairwise_path(method: dict[str, Any], methods_by_key: dict[str, dict[str, Any]]) -> str | None:
    if method.get("pairwise"):
        return str(method["pairwise"])
    reuse = method.get("pairwise_reuse_from")
    if reuse:
        return method_pairwise_path(methods_by_key[reuse], methods_by_key)
    return None


def summarize_slice(spec: dict[str, Any]) -> dict[str, Any]:
    methods_by_key = {method["key"]: method for method in spec["methods"]}
    baseline_spec = methods_by_key["hyde"]
    baseline_score = score_result(baseline_spec["path"], baseline_spec["label"], baseline_spec["status"])
    baseline_eval_path = method_eval_path(baseline_spec, methods_by_key)
    if not baseline_eval_path:
        raise ValueError(f"Missing baseline eval for {spec['name']}")
    baseline_judge = judge_summary(baseline_eval_path)

    methods = {}
    for method in spec["methods"]:
        score = score_result(method["path"], method["label"], method["status"])
        eval_path = method_eval_path(method, methods_by_key)
        judge = judge_summary(eval_path) if eval_path else None
        pairwise_path = method_pairwise_path(method, methods_by_key)
        methods[method["key"]] = {
            **score,
            "eval": judge,
            "eval_path": eval_path,
            "pairwise_win": pairwise_win_summary(pairwise_path),
            "eval_reused_from": method.get("eval_reuse_from"),
            "pairwise_reused_from": method.get("pairwise_reuse_from"),
        }

    for key, method in methods.items():
        if key == "hyde":
            method["vs_hyde"] = None
            continue
        method["vs_hyde"] = {
            "f1": paired(method["per_line"], baseline_score["per_line"], "f1"),
            "rouge_l": paired(method["per_line"], baseline_score["per_line"], "rouge_l"),
            "judge_overall": paired_judge(method["eval"], baseline_judge, "overall") if method.get("eval") else None,
            "prediction_diffs": prediction_diffs(method["path"], baseline_spec["path"]),
        }

    order = [method["key"] for method in spec["methods"]]
    for prev_key, key in zip(order, order[1:]):
        left = methods[key]
        right = methods[prev_key]
        left["vs_previous"] = {
            "previous_key": prev_key,
            "previous_label": right["label"],
            "f1": paired(left["per_line"], right["per_line"], "f1"),
            "judge_overall": paired_judge(left["eval"], right["eval"], "overall")
            if left.get("eval") and right.get("eval")
            else None,
            "prediction_diffs": prediction_diffs(left["path"], right["path"]),
        }

    return {
        "name": spec["name"],
        "expected_n": spec["n"],
        "scope": spec["scope"],
        "complete": all(method["n"] == spec["n"] for method in methods.values()),
        "methods": methods,
    }


def summarize() -> dict[str, Any]:
    slices = [summarize_slice(spec) for spec in SLICES]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "name": "SAGE late-stage candidate ladder",
            "scope": "offline consolidation of existing v54/v55/v56 heldout artifacts; no new LLM calls",
            "main_candidate": "SAGE v33-preservation remains the freeze candidate until late-stage branches are replayed on Mix130/full-slice protocol",
        },
        "slices": slices,
    }


def fmt_p(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.4g}"


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# SAGE Late-Stage Candidate Ladder",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Protocol",
        "",
        f"- Scope: {payload['protocol']['scope']}.",
        f"- Main-candidate rule: {payload['protocol']['main_candidate']}.",
        "- v54/v55/v56 are treated as late-stage exploratory branches unless they pass the same frozen replay and full-slice checks as v33.",
        "",
        "## Slice Summary",
        "",
        "| Slice | Complete | Method | N | F1 | ROUGE-L | Judge overall | Correctness | Coverage | Pairwise win vs HyDE | Changed rows vs HyDE | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for slice_row in payload["slices"]:
        for method in slice_row["methods"].values():
            judge = method.get("eval") or {}
            pairwise_win = method.get("pairwise_win") or {}
            vs_hyde = method.get("vs_hyde") or {}
            diffs = vs_hyde.get("prediction_diffs", "-") if vs_hyde else "-"
            win_rate = pairwise_win.get("average_win_rate")
            win_text = "-" if win_rate is None else f"{win_rate:.2%}"
            lines.append(
                f"| {slice_row['name']} | {slice_row['complete']} | {method['label']} | {method['n']} | "
                f"{method['mean_f1']:.4f} | {method['mean_rouge_l']:.4f} | "
                f"{judge.get('overall', 0.0):.3f} | {judge.get('correctness', 0.0):.3f} | "
                f"{judge.get('coverage', 0.0):.3f} | {win_text} | {diffs} | {method['status']} |"
            )

    lines.extend(
        [
            "",
            "## Pairwise vs HyDE",
            "",
            "| Slice | Method | Metric | Delta | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for slice_row in payload["slices"]:
        for key, method in slice_row["methods"].items():
            if key == "hyde" or not method.get("vs_hyde"):
                continue
            for metric_key, label in [("f1", "F1"), ("judge_overall", "Judge overall")]:
                stats = method["vs_hyde"].get(metric_key)
                if not stats:
                    continue
                lines.append(
                    f"| {slice_row['name']} | {method['label']} | {label} | {stats['delta']:+.4f} | "
                    f"{stats['wins']} | {stats['losses']} | {stats['ties']} | {fmt_p(stats['t_p'])} | "
                    f"{fmt_p(stats['wilcoxon_p'])} | {stats['cohens_d']:+.2f} |"
                )

    lines.extend(
        [
            "",
            "## Stepwise Deltas",
            "",
            "| Slice | Step | Delta F1 vs previous | Delta judge overall vs previous | Prediction diffs vs previous |",
            "|---|---|---:|---:|---|",
        ]
    )
    for slice_row in payload["slices"]:
        for method in slice_row["methods"].values():
            prev = method.get("vs_previous")
            if not prev:
                continue
            judge_delta = prev["judge_overall"]["delta"] if prev.get("judge_overall") else 0.0
            lines.append(
                f"| {slice_row['name']} | {method['label']} vs {prev['previous_label']} | "
                f"{prev['f1']['delta']:+.4f} | {judge_delta:+.4f} | {prev['prediction_diffs']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- v56 is the best late-stage heldout branch on the small v40/v42/v46 collection, but its evidence is not yet on the same footing as v33's Mix130/full-slice reproducibility package.",
            "- v46 is the strongest late-stage signal: v55/v56 reach F1=0.3627 and judge overall=4.300 versus HyDE F1=0.3126 and judge overall=3.600.",
            "- v40 shows why v56 is useful: v55 prevents an over-compressed repair, and v56 recovers a better fused answer for line 11.",
            "- v42 and v46 have no v56 trigger, so v56 is prediction-identical to v55 on those slices.",
            "- Next promotion test: replay v54->v55->v56 on Mix130 or the 120-row external full-slice ladder before considering v56 as the main paper method.",
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


if __name__ == "__main__":
    main()
