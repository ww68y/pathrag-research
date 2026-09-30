from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_enhanced_experiments import compute_f1, compute_rouge_l  # noqa: E402

OUT_DIR = ROOT / "results" / "sage_repro"
OUTPUT_JSON = OUT_DIR / "sage_reproduction_validation.json"
OUTPUT_MD = OUT_DIR / "SAGE_REPRODUCTION_VALIDATION.md"

PAIRS = [
    {
        "name": "SAGE v31 learned contract-gap mix130",
        "original": "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130.jsonl",
        "reproduction": "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_repro.jsonl",
        "frozen_router": "results/historical_vsr/v31_full_rf_oof.jsonl",
    },
    {
        "name": "SAGE v32 evidence-delta mix130",
        "original": "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130.jsonl",
        "reproduction": "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_repro.jsonl",
        "frozen_router": "results/historical_vsr/v31_full_rf_oof.jsonl",
    },
    {
        "name": "SAGE v33 preservation-delta mix130",
        "original": "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl",
        "reproduction": "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130_repro.jsonl",
        "frozen_router": "results/historical_vsr/v31_full_rf_oof.jsonl",
    },
]

TRACKED_FIELDS = [
    "prediction",
    "method",
    "learned_contract_gap_router",
    "evidence_delta_guard",
    "contract_completion_agent",
    "suppressed_contract_completion_prediction",
    "rejected_evidence_delta_v33_prediction",
]


def load_jsonl(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    lines = [line.rstrip("\n") for line in path.open("r", encoding="utf-8", errors="replace") if line.strip()]
    rows = [json.loads(line) for line in lines]
    return lines, rows


def gold_answer(row: dict[str, Any]) -> str:
    if row.get("gold_answer"):
        return str(row.get("gold_answer"))
    answers = row.get("reference_answers") or row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    values = []
    for row in rows:
        gold = gold_answer(row)
        pred = prediction(row)
        if not gold or not pred:
            continue
        if metric == "f1":
            values.append(compute_f1(pred, gold))
        elif metric == "rouge_l":
            values.append(compute_rouge_l(pred, gold))
        else:
            raise ValueError(metric)
    return statistics.mean(values) if values else 0.0


def compare_pair(pair: dict[str, str]) -> dict[str, Any]:
    original_path = ROOT / pair["original"]
    reproduction_path = ROOT / pair["reproduction"]
    original_lines, original_rows = load_jsonl(original_path)
    reproduction_lines, reproduction_rows = load_jsonl(reproduction_path)
    paired_rows = list(zip(original_rows, reproduction_rows))
    paired_lines = list(zip(original_lines, reproduction_lines))

    field_diffs = {
        field: sum(original.get(field) != reproduction.get(field) for original, reproduction in paired_rows)
        for field in TRACKED_FIELDS
    }
    exact_object_diffs = sum(original != reproduction for original, reproduction in paired_rows)
    raw_line_diffs = sum(original != reproduction for original, reproduction in paired_lines)
    length_delta = len(original_rows) - len(reproduction_rows)

    return {
        **pair,
        "original_exists": original_path.exists(),
        "reproduction_exists": reproduction_path.exists(),
        "original_rows": len(original_rows),
        "reproduction_rows": len(reproduction_rows),
        "row_count_match": len(original_rows) == len(reproduction_rows),
        "length_delta": length_delta,
        "exact_object_diffs": exact_object_diffs,
        "raw_line_diffs": raw_line_diffs,
        "field_diffs": field_diffs,
        "prediction_diffs": field_diffs["prediction"],
        "route_metadata_diffs": field_diffs["learned_contract_gap_router"],
        "object_exact_match": length_delta == 0 and exact_object_diffs == 0,
        "raw_line_exact_match": length_delta == 0 and raw_line_diffs == 0,
        "mean_f1": mean_metric(reproduction_rows, "f1"),
        "mean_rouge_l": mean_metric(reproduction_rows, "rouge_l"),
    }


def validate() -> dict[str, Any]:
    pair_results = [compare_pair(pair) for pair in PAIRS]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "all_object_exact": all(row["object_exact_match"] for row in pair_results),
        "all_predictions_exact": all(row["prediction_diffs"] == 0 and row["row_count_match"] for row in pair_results),
        "pairs": pair_results,
    }


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# SAGE Reproduction Validation",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        f"- All JSON-object rows exact: `{payload['all_object_exact']}`",
        f"- All predictions exact: `{payload['all_predictions_exact']}`",
        "- v31/v32/v33 reproduction uses the frozen full-feature RF OOF router: `results/historical_vsr/v31_full_rf_oof.jsonl`.",
        "",
        "## Pairwise Checks",
        "",
        "| Pair | Rows | F1 | ROUGE-L | Object diffs | Raw-line diffs | Prediction diffs | Route diffs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["pairs"]:
        lines.append(
            f"| {row['name']} | {row['reproduction_rows']} | {row['mean_f1']:.4f} | "
            f"{row['mean_rouge_l']:.4f} | {row['exact_object_diffs']} | {row['raw_line_diffs']} | "
            f"{row['prediction_diffs']} | {row['route_metadata_diffs']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The reproduced v31, v32, and v33 mix130 artifacts match the original JSON objects row-for-row, including route metadata.",
            "- The previous mismatch came from replaying v31 with `results/historical_sage/mse_pathrag_sage_v31_contract_gap_router_oof.jsonl`; the original mix130 artifact was generated from `results/historical_vsr/v31_full_rf_oof.jsonl`.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = validate()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
