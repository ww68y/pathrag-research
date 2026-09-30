from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from datetime import datetime
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


OUT_DIR = ROOT / "results" / "sage_promotion_mix130"
OUTPUT_JSON = OUT_DIR / "sage_promotion_mix130_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_PROMOTION_MIX130_SUMMARY.md"

TARGET_LINES = [
    2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26,
    27, 28, 29, 30, 32, 33, 37, 40, 41, 42, 43, 44, 46, 47, 51, 54, 55, 56, 60, 65, 66,
    68, 69, 70, 71, 72, 74, 75, 77, 78, 80, 84, 86, 87, 89, 90, 91, 94, 95, 96, 97, 98,
    99, 100, 103, 106, 109, 110, 111, 112, 116, 117, 118, 119, 121, 123, 125, 127, 128, 129,
]

METHODS = [
    {
        "key": "v33",
        "label": "SAGE v33 preservation-delta",
        "path": "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl",
        "eval": "results/historical_sage/mse_pathrag_sage_v33_frozen_evidence_delta_mix130_eval.jsonl",
        "status": "current freeze candidate",
    },
    {
        "key": "v54",
        "label": "SAGE v54 contract-gap repair candidate82",
        "path": "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_candidate82.jsonl",
        "status": "promotion batch repair",
    },
    {
        "key": "v55",
        "label": "SAGE v55 supported-facet guard candidate82",
        "path": "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_candidate82.jsonl",
        "status": "promotion batch guard",
    },
    {
        "key": "v56",
        "label": "SAGE v56 supported-facet fusion candidate82",
        "path": "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82.jsonl",
        "eval": "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82_routed_eval.jsonl",
        "changed_eval": "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82_changed_eval.jsonl",
        "status": "promotion batch fusion; not full-protocol validated",
    },
]


def load_jsonl(path_or_relative: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_relative)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def line_no(row: dict[str, Any], fallback: int) -> int:
    if row.get("line_no") is not None:
        return int(row["line_no"])
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


def score_rows(path: str) -> dict[str, Any]:
    rows = load_jsonl(path)
    per_line = []
    for index, row in enumerate(rows):
        refs = references(row)
        pred = prediction(row)
        per_line.append(
            {
                "line_no": line_no(row, index),
                "dataset": row.get("dataset") or row.get("label"),
                "f1": max(compute_f1(pred, ref) for ref in refs),
                "rouge_l": max(compute_rouge_l(pred, ref) for ref in refs),
                "prediction_chars": len(pred),
            }
        )
    target = [row for row in per_line if row["line_no"] in TARGET_LINES]
    return {
        "n": len(per_line),
        "target_n": len(target),
        "mean_f1": statistics.mean(row["f1"] for row in per_line) if per_line else 0.0,
        "mean_rouge_l": statistics.mean(row["rouge_l"] for row in per_line) if per_line else 0.0,
        "target_mean_f1": statistics.mean(row["f1"] for row in target) if target else 0.0,
        "target_mean_rouge_l": statistics.mean(row["rouge_l"] for row in target) if target else 0.0,
        "mean_prediction_chars": statistics.mean(row["prediction_chars"] for row in per_line) if per_line else 0.0,
        "per_line": per_line,
    }


def judge_rows(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    eval_path = ROOT / path
    if not eval_path.exists():
        return None
    rows = load_jsonl(path)
    metrics = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    per_line = []
    for index, row in enumerate(rows):
        scores = row.get("scores") or {}
        per_line.append(
            {
                "line_no": line_no(row, index),
                "dataset": row.get("dataset"),
                "scores": scores,
                "verdict": scores.get("verdict"),
                "reason": scores.get("short_reason", ""),
            }
        )
    return {
        "n": len(per_line),
        **{
            metric: statistics.mean(float(row["scores"].get(metric, 0)) for row in per_line) if per_line else 0.0
            for metric in metrics
        },
        "verdicts": dict(sorted(Counter(str(row["verdict"]) for row in per_line).items())),
        "per_line": per_line,
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


def judge_paired(left: dict[str, Any], right: dict[str, Any], metric: str) -> dict[str, Any]:
    left_rows = [
        {"line_no": row["line_no"], metric: float((row.get("scores") or {}).get(metric, 0))}
        for row in left.get("per_line", [])
    ]
    right_rows = [
        {"line_no": row["line_no"], metric: float((row.get("scores") or {}).get(metric, 0))}
        for row in right.get("per_line", [])
    ]
    return paired(left_rows, right_rows, metric)


def changed_lines(left_path: str, right_path: str) -> list[int]:
    left = by_line(load_jsonl(left_path))
    right = by_line(load_jsonl(right_path))
    changed = []
    for line, row in sorted(left.items()):
        if prediction(row).strip() != prediction(right.get(line, {})).strip():
            changed.append(line)
    return changed


def agent_action_counts(path: str, field: str) -> dict[str, int]:
    rows = load_jsonl(path)
    counts = Counter()
    for row in rows:
        agent = row.get(field) or {}
        counts[str(agent.get("action", "not_applicable"))] += 1
    return dict(sorted(counts.items()))


def build_summary() -> dict[str, Any]:
    method_scores = {}
    for method in METHODS:
        score = score_rows(method["path"])
        score.update(
            {
                "key": method["key"],
                "label": method["label"],
                "path": method["path"],
                "status": method["status"],
                "judge": judge_rows(method.get("eval")),
            }
        )
        method_scores[method["key"]] = score

    v33 = method_scores["v33"]
    v56 = method_scores["v56"]
    target_v33 = [row for row in v33["per_line"] if row["line_no"] in TARGET_LINES]
    target_v56 = [row for row in v56["per_line"] if row["line_no"] in TARGET_LINES]
    changed = changed_lines(METHODS[0]["path"], METHODS[-1]["path"])
    changed_v33 = [row for row in v33["per_line"] if row["line_no"] in changed]
    changed_v56 = [row for row in v56["per_line"] if row["line_no"] in changed]

    v33_judge = v33.get("judge") or {}
    v56_judge = v56.get("judge") or {}
    changed_eval = judge_rows(METHODS[-1].get("changed_eval")) or {"per_line": []}
    v33_judge_by_line = {row["line_no"]: row for row in v33_judge.get("per_line", [])}
    changed_judge_rows = []
    for row in changed_eval.get("per_line", []):
        base = v33_judge_by_line.get(row["line_no"], {"scores": {}})
        changed_judge_rows.append(
            {
                "line_no": row["line_no"],
                "dataset": row.get("dataset"),
                "v33_overall": float((base.get("scores") or {}).get("overall", 0)),
                "v56_overall": float((row.get("scores") or {}).get("overall", 0)),
                "delta_overall": float((row.get("scores") or {}).get("overall", 0))
                - float((base.get("scores") or {}).get("overall", 0)),
                "v33_verdict": (base.get("scores") or {}).get("verdict"),
                "v56_verdict": (row.get("scores") or {}).get("verdict"),
                "reason": row.get("reason", ""),
            }
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_lines": TARGET_LINES,
        "method_scores": method_scores,
        "actions": {
            "v54": agent_action_counts(METHODS[1]["path"], "query_expanded_contract_gap_repair"),
            "v55": agent_action_counts(METHODS[2]["path"], "supported_facet_guard"),
            "v56": agent_action_counts(METHODS[3]["path"], "supported_facet_fusion"),
        },
        "changed_lines_vs_v33": changed,
        "lexical_pairwise": {
            "v56_vs_v33_full_f1": paired(v56["per_line"], v33["per_line"], "f1"),
            "v56_vs_v33_full_rouge_l": paired(v56["per_line"], v33["per_line"], "rouge_l"),
            "v56_vs_v33_target_f1": paired(target_v56, target_v33, "f1"),
            "v56_vs_v33_target_rouge_l": paired(target_v56, target_v33, "rouge_l"),
            "v56_vs_v33_changed_f1": paired(changed_v56, changed_v33, "f1"),
            "v56_vs_v33_changed_rouge_l": paired(changed_v56, changed_v33, "rouge_l"),
        },
        "judge_pairwise": {
            "v56_vs_v33_full_overall": judge_paired(v56_judge, v33_judge, "overall") if v56_judge and v33_judge else None,
            "v56_vs_v33_full_coverage": judge_paired(v56_judge, v33_judge, "coverage") if v56_judge and v33_judge else None,
        },
        "changed_judge_rows": changed_judge_rows,
        "risk_notes": [
            "Line 22 decreases by one judge-overall point versus v33 and should be inspected before any promotion claim.",
            "Line 47 improves from v33 but remains poor under the judge, so v54/v55/v56 still need a semantic-anchor guard.",
            "This covers all 82 Mix130 v54 candidates, but external full-slice promotion replay is still not complete.",
        ],
    }


def write_markdown(summary: dict[str, Any]) -> None:
    scores = summary["method_scores"]
    lines = [
        "# SAGE Mix130 Promotion Replay Summary\n\n",
        f"> Generated: {summary['generated_at']}\n\n",
        "## Protocol\n\n",
        "- Scope: complete promotion replay of v54->v55->v56 on all 82 Mix130 candidate rows selected from the v33 freeze candidate.\n",
        "- Target rows: all v54-eligible Mix130 rows under the contract type / length routing rule.\n",
        "- Runtime inputs: question, current prediction, and source evidence only; reference answers and judge scores are not used by v54/v55/v56 routing.\n",
        "- Judge protocol: only changed v56 rows were newly judged; unchanged rows reuse existing v33 judgments by exact prediction match.\n",
        "- Status: promotion evidence only; v33 remains the main freeze candidate until all 82 candidates or the full protocol replay is complete.\n\n",
        "## Score Summary\n\n",
        "| Method | N | Target N | F1 | ROUGE-L | Target F1 | Target ROUGE-L | Judge overall | Judge coverage | Status |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for key in ["v33", "v54", "v55", "v56"]:
        row = scores[key]
        judge = row.get("judge") or {}
        judge_overall = f"{judge.get('overall', 0):.4f}" if judge else "-"
        judge_coverage = f"{judge.get('coverage', 0):.4f}" if judge else "-"
        lines.append(
            f"| {row['label']} | {row['n']} | {row['target_n']} | {row['mean_f1']:.4f} | "
            f"{row['mean_rouge_l']:.4f} | {row['target_mean_f1']:.4f} | {row['target_mean_rouge_l']:.4f} | "
            f"{judge_overall} | {judge_coverage} | {row['status']} |\n"
        )

    lines.extend([
        "\n## Agent Actions\n\n",
        "| Stage | Action | Count |\n",
        "|---|---|---:|\n",
    ])
    for stage, counts in summary["actions"].items():
        for action, count in counts.items():
            lines.append(f"| {stage} | {action} | {count} |\n")

    full_f1 = summary["lexical_pairwise"]["v56_vs_v33_full_f1"]
    target_f1 = summary["lexical_pairwise"]["v56_vs_v33_target_f1"]
    judge_overall = summary["judge_pairwise"]["v56_vs_v33_full_overall"]
    lines.extend(
        [
            "\n## v56 vs v33\n\n",
            "| Comparison | Metric | Delta | Wins | Losses | Ties | t-p | Wilcoxon-p | d |\n",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
            f"| Full Mix130 | F1 | {full_f1['delta']:+.4f} | {full_f1['wins']} | {full_f1['losses']} | {full_f1['ties']} | {full_f1['t_p']:.4g} | {full_f1['wilcoxon_p']:.4g} | {full_f1['cohens_d']:+.2f} |\n",
            f"| Target 82 | F1 | {target_f1['delta']:+.4f} | {target_f1['wins']} | {target_f1['losses']} | {target_f1['ties']} | {target_f1['t_p']:.4g} | {target_f1['wilcoxon_p']:.4g} | {target_f1['cohens_d']:+.2f} |\n",
        ]
    )
    if judge_overall:
        lines.append(
            f"| Full Mix130 routed judge | Overall | {judge_overall['delta']:+.4f} | {judge_overall['wins']} | {judge_overall['losses']} | {judge_overall['ties']} | {judge_overall['t_p']:.4g} | {judge_overall['wilcoxon_p']:.4g} | {judge_overall['cohens_d']:+.2f} |\n"
        )

    lines.extend(
        [
            "\n## Changed Rows\n\n",
            f"- Final v56 changed rows vs v33: {summary['changed_lines_vs_v33']}\n\n",
            "| Line | Dataset | v33 overall | v56 overall | Delta | v33 verdict | v56 verdict | Note |\n",
            "|---:|---|---:|---:|---:|---|---|---|\n",
        ]
    )
    for row in summary["changed_judge_rows"]:
        note = str(row.get("reason", "")).replace("|", "/")
        lines.append(
            f"| {row['line_no']} | {row.get('dataset')} | {row['v33_overall']:.0f} | {row['v56_overall']:.0f} | "
            f"{row['delta_overall']:+.0f} | {row.get('v33_verdict')} | {row.get('v56_verdict')} | {note} |\n"
        )

    lines.extend(["\n## Interpretation\n\n"])
    lines.append(
        f"- v56 improves full Mix130 lexical F1 from {scores['v33']['mean_f1']:.4f} to {scores['v56']['mean_f1']:.4f} and routed judge overall from {(scores['v33']['judge'] or {}).get('overall', 0):.4f} to {(scores['v56']['judge'] or {}).get('overall', 0):.4f} on this complete candidate82 replay.\n"
    )
    lines.append(
        "- The positive signal is concentrated in changed rows with low v33 judge scores; line 42 is especially important because it upgrades the only v33-active Mix130 row from poor to excellent.\n"
    )
    lines.append(
        "- The branch is not ready to replace v33: several changed rows regress under judge, line 47 remains poor, and external full-slice promotion replay is not complete.\n"
    )
    lines.append(
        "- Next step: inspect regression rows and either add a reference-free safety guard or keep v56 as exploratory while moving to external full-slice promotion replay.\n"
    )

    OUTPUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary)
    print(f"Saved summary JSON to {OUTPUT_JSON}")
    print(f"Saved summary markdown to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
