from __future__ import annotations

import json
import statistics
import sys
import time
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


OUT_DIR = ROOT / "results" / "sage_external_probe"
OUTPUT_JSON = OUT_DIR / "sage_external_vs_baselines_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_EXTERNAL_VS_BASELINES_SUMMARY.md"

SLICES = [
    {
        "dataset": "Legal full context",
        "n": 20,
        "naive": "results/legal_fullcontext_20/naive_aligned_n20.jsonl",
        "hyde": "results/legal_fullcontext_20/hyde_aligned_n20.jsonl",
        "vsr_rescue": "results/enhanced_ablation_aligned/legal_fullcontext_20/full_v2_rescue_aligned_n20.json",
        "sage_v33": "results/historical_datasets/legal_hyde_evidence_delta_v33_cumulative.jsonl",
    },
    {
        "dataset": "Biology full context",
        "n": 50,
        "naive": "results/biology_fullcontext_50/naive_aligned_n50.jsonl",
        "hyde": "results/biology_fullcontext_50/hyde_aligned_n50.jsonl",
        "vsr_rescue": "results/enhanced_ablation_aligned/biology_fullcontext_50/full_v2_rescue_aligned_n50.json",
        "sage_v33": "results/historical_datasets/biology_hyde_evidence_delta_v33_cumulative.jsonl",
    },
    {
        "dataset": "Domain expansion windowed",
        "n": 50,
        "naive": "results/domain_expansion_windowed_50/naive_aligned_n50.jsonl",
        "hyde": "results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl",
        "vsr_rescue": "results/enhanced_ablation_aligned/domain_expansion_windowed_50/full_v2_rescue_aligned_n50.json",
        "sage_v33": "results/historical_datasets/domain_hyde_evidence_delta_v33_cumulative.jsonl",
    },
]

METHOD_NAMES = {
    "naive": "Naive RAG",
    "hyde": "HyDE",
    "vsr_rescue": "VSR full_v2_rescue",
    "sage_v33": "SAGE v33-preservation",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_rows(path_or_relative: str) -> list[dict[str, Any]]:
    path = ROOT / path_or_relative
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []


def line_no(row: dict[str, Any], fallback: int) -> int:
    if row.get("line_no") is not None:
        return int(row["line_no"])
    question_id = str(row.get("question_id") or "")
    if question_id.startswith("ultra_"):
        return int(question_id.split("_", 1)[1])
    return fallback


def rows_by_line(path_or_relative: str) -> dict[int, dict[str, Any]]:
    return {line_no(row, index): row for index, row in enumerate(load_rows(path_or_relative))}


def gold_answer(row: dict[str, Any]) -> str:
    if row.get("gold_answer"):
        return str(row.get("gold_answer"))
    answers = row.get("reference_answers") or row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def score_rows(rows: dict[int, dict[str, Any]], gold_rows: dict[int, dict[str, Any]], line_nos: list[int]) -> dict[str, Any]:
    f1 = [compute_f1(prediction(rows[line]), gold_answer(gold_rows[line])) for line in line_nos]
    rouge = [compute_rouge_l(prediction(rows[line]), gold_answer(gold_rows[line])) for line in line_nos]
    return {
        "n": len(line_nos),
        "f1": statistics.mean(f1) if f1 else 0.0,
        "rouge_l": statistics.mean(rouge) if rouge else 0.0,
        "f1_by_line": dict(zip(line_nos, f1)),
        "rouge_l_by_line": dict(zip(line_nos, rouge)),
    }


def pairwise(left: dict[str, Any], right: dict[str, Any], metric: str = "f1") -> dict[str, Any]:
    left_values = [left[f"{metric}_by_line"][line] for line in sorted(left[f"{metric}_by_line"])]
    right_values = [right[f"{metric}_by_line"][line] for line in sorted(right[f"{metric}_by_line"])]
    deltas = [x - y for x, y in zip(left_values, right_values)]
    t_stat, t_p = paired_t_test(left_values, right_values)
    w_stat, w_p = wilcoxon_signed_rank_test(left_values, right_values)
    return {
        "metric": metric,
        "n": len(deltas),
        "delta": statistics.mean(deltas) if deltas else 0.0,
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "t_stat": t_stat,
        "t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "cohens_d": cohens_d(left_values, right_values),
    }


def summarize_slice(spec: dict[str, Any]) -> dict[str, Any]:
    method_rows = {method: rows_by_line(spec[method]) for method in METHOD_NAMES}
    common_lines = sorted(set.intersection(*(set(rows) for rows in method_rows.values())))
    scores = {
        method: score_rows(rows, method_rows["hyde"], common_lines)
        for method, rows in method_rows.items()
    }
    best_retrieval_key = max(["naive", "hyde"], key=lambda method: scores[method]["f1"])
    return {
        "dataset": spec["dataset"],
        "n": len(common_lines),
        "methods": scores,
        "best_retrieval": best_retrieval_key,
        "sage_vs_best_retrieval": pairwise(scores["sage_v33"], scores[best_retrieval_key], "f1"),
        "sage_vs_hyde": pairwise(scores["sage_v33"], scores["hyde"], "f1"),
        "sage_vs_vsr_rescue": pairwise(scores["sage_v33"], scores["vsr_rescue"], "f1"),
    }


def summarize() -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slices": [summarize_slice(spec) for spec in SLICES],
    }


def pct(value: float) -> str:
    return f"{value:+.4f}"


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# SAGE External vs Baselines Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | N | Naive F1 | HyDE F1 | VSR rescue F1 | SAGE v33-preservation F1 | Best retrieval | SAGE - best retrieval | SAGE - VSR rescue |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["slices"]:
        methods = row["methods"]
        best_key = row["best_retrieval"]
        lines.append(
            f"| {row['dataset']} | {row['n']} | "
            f"{methods['naive']['f1']:.4f} | {methods['hyde']['f1']:.4f} | "
            f"{methods['vsr_rescue']['f1']:.4f} | {methods['sage_v33']['f1']:.4f} | "
            f"{METHOD_NAMES[best_key]} | {pct(row['sage_vs_best_retrieval']['delta'])} | "
            f"{pct(row['sage_vs_vsr_rescue']['delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Tests",
            "",
            "| Dataset | Comparison | Delta F1 | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["slices"]:
        for label, stats in [
            (f"SAGE - best retrieval ({METHOD_NAMES[row['best_retrieval']]})", row["sage_vs_best_retrieval"]),
            ("SAGE - HyDE", row["sage_vs_hyde"]),
            ("SAGE - VSR rescue", row["sage_vs_vsr_rescue"]),
        ]:
            lines.append(
                f"| {row['dataset']} | {label} | {stats['delta']:+.4f} | {stats['wins']} | {stats['losses']} | {stats['ties']} | "
                f"{stats['t_p']:.4g} | {stats['wilcoxon_p']:.4g} | {stats['cohens_d']:+.2f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- SAGE v33-preservation numerically beats the best retrieval baseline on Domain and Legal, but those best-retrieval deltas are not statistically significant in the current small external slices.",
            "- It ties HyDE on Biology by reverting the only harmful Biology repair, so Biology is safety-neutral rather than positive.",
            "- Against VSR full_v2_rescue, SAGE v33-preservation is higher on all three external slices under lexical F1; Domain-vs-VSR is statistically stronger than Legal-vs-VSR in this snapshot.",
            "- These are still risk-pool SAGE-on-HyDE outputs, not a final locked all-benchmark protocol; use them to justify the next frozen validation run.",
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
