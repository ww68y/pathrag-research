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

from experiments.run_enhanced_experiments import compute_f1, compute_rouge_l  # noqa: E402

OUT_DIR = ROOT / "results" / "sage_vs_baselines"
OUTPUT_JSON = OUT_DIR / "sage_vs_baselines_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_VS_BASELINES_SUMMARY.md"

MIX130_METHODS = [
    {
        "method": "SAGE v33 preservation-delta",
        "path": "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl",
        "protocol": "existing SAGE adaptive-routing replay plus v33 preservation guard",
    },
    {
        "method": "SAGE v32 evidence-delta",
        "path": "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130.jsonl",
        "protocol": "existing SAGE adaptive-routing replay",
    },
    {
        "method": "SAGE v31 learned contract-gap",
        "path": "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130.jsonl",
        "protocol": "existing SAGE adaptive-routing replay",
    },
    {
        "method": "SAGE v21 text-first utility guard",
        "path": "results/historical_sage/mse_pathrag_sage_v21_text_first_utility_guard_mix_130.jsonl",
        "protocol": "existing SAGE adaptive-routing replay",
    },
    {
        "method": "SAGE v33 frozen evidence-delta",
        "path": "results/historical_sage/mse_pathrag_sage_v33_frozen_evidence_delta_mix_unseen34.jsonl",
        "protocol": "existing SAGE frozen-policy replay",
    },
    {
        "method": "Aligned VSR full_v2_rescue",
        "path": "results/enhanced_ablation_aligned/full_v2_rescue_aligned_n130.json",
        "protocol": "current aligned VSR ablation",
    },
    {
        "method": "Aligned VSR full_v2",
        "path": "results/enhanced_ablation_aligned/full_v2_aligned_n130.json",
        "protocol": "current aligned VSR ablation",
    },
    {
        "method": "Aligned VSR w/o_repair",
        "path": "results/enhanced_ablation_aligned/w_o_repair_aligned_n130.json",
        "protocol": "current aligned VSR ablation",
    },
    {
        "method": "Historical VSR-PathRAG",
        "path": "results/enhanced/vsr_pathrag_n130.json",
        "protocol": "legacy mix130 diagnostic",
    },
    {
        "method": "Historical HyDE",
        "path": "results/enhanced/hyde_n130.json",
        "protocol": "legacy mix130 diagnostic",
    },
    {
        "method": "Historical Naive RAG",
        "path": "results/enhanced/naive_rag_n130.json",
        "protocol": "legacy mix130 diagnostic",
    },
    {
        "method": "Historical GraphRAG-sim",
        "path": "results/enhanced/graphrag_sim_n130.json",
        "protocol": "legacy mix130 diagnostic",
    },
]

JUDGE_FILES = [
    {
        "method": "SAGE v31 learned contract-gap",
        "path": "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_eval.jsonl",
    },
    {
        "method": "SAGE v32 evidence-delta",
        "path": "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_eval.jsonl",
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    return []


def gold_answer(row: dict[str, Any]) -> str:
    if row.get("gold_answer"):
        return str(row.get("gold_answer"))
    answers = row.get("reference_answers") or row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def runtime(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    if metrics.get("runtime_seconds") is not None:
        return float(metrics.get("runtime_seconds") or 0.0)
    return float(row.get("query_seconds", 0.0) or 0.0) + float(row.get("insert_seconds", 0.0) or 0.0)


def line_no(row: dict[str, Any], fallback: int) -> int:
    if row.get("line_no") is not None:
        return int(row["line_no"])
    question_id = str(row.get("question_id") or "")
    if question_id.startswith("ultra_"):
        return int(question_id.split("_", 1)[1])
    return fallback


def task_label(row: dict[str, Any]) -> str:
    return str(row.get("domain") or row.get("dataset") or row.get("label") or "unknown")


def aggregate_method(spec: dict[str, str]) -> dict[str, Any]:
    path = ROOT / spec["path"]
    rows = [row for row in load_result_rows(path) if gold_answer(row) and prediction(row)]
    f1s = [compute_f1(prediction(row), gold_answer(row)) for row in rows]
    rouges = [compute_rouge_l(prediction(row), gold_answer(row)) for row in rows]
    runtimes = [runtime(row) for row in rows]
    return {
        **spec,
        "exists": path.exists(),
        "n": len(rows),
        "mean_f1": statistics.mean(f1s) if f1s else 0.0,
        "mean_rouge_l": statistics.mean(rouges) if rouges else 0.0,
        "mean_runtime_seconds": statistics.mean(runtimes) if runtimes else 0.0,
    }


def load_aligned_retrieval_summary() -> list[dict[str, Any]]:
    path = ROOT / "results" / "retrieval_baselines" / "retrieval_baseline_aligned_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for dataset in data.get("datasets", {}).values():
        aggregated = dataset.get("aggregated", {})
        retrieval_candidates = [
            {**aggregated[name], "method_name": name}
            for name in ["naive", "hyde"]
            if name in aggregated
        ]
        best_retrieval = max(retrieval_candidates, key=lambda item: item.get("mean_f1", 0.0))
        rescue = aggregated.get("full_v2_rescue", {})
        rows.append(
            {
                "dataset": dataset.get("name", "unknown"),
                "n": dataset.get("n", rescue.get("sample_size", 0)),
                "best_retrieval_f1": best_retrieval.get("mean_f1", 0.0),
                "best_retrieval_method": best_retrieval.get("method_name", "retrieval"),
                "best_retrieval_runtime": best_retrieval.get("mean_runtime_seconds", 0.0),
                "vsr_rescue_f1": rescue.get("mean_f1", 0.0),
                "vsr_rescue_runtime": rescue.get("mean_runtime_seconds", 0.0),
                "delta_vsr_minus_best_retrieval": rescue.get("mean_f1", 0.0) - best_retrieval.get("mean_f1", 0.0),
            }
        )
    return rows


def judge_summary(spec: dict[str, str]) -> dict[str, Any]:
    path = ROOT / spec["path"]
    rows = load_jsonl(path)
    metrics = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    summary = {
        metric: statistics.mean(float(row.get("scores", {}).get(metric, 0.0) or 0.0) for row in rows)
        for metric in metrics
    }
    verdicts = Counter(str(row.get("scores", {}).get("verdict") or "unknown") for row in rows)
    return {**spec, "n": len(rows), **summary, "verdicts": dict(verdicts)}


def paired_sage_vsr() -> dict[str, Any]:
    sage_rows = load_result_rows(ROOT / "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl")
    vsr_rows = load_result_rows(ROOT / "results/enhanced_ablation_aligned/full_v2_rescue_aligned_n130.json")
    sage_by_line = {line_no(row, idx): row for idx, row in enumerate(sage_rows)}
    deltas = []
    by_task: dict[str, list[dict[str, float]]] = {}
    for idx, vsr_row in enumerate(vsr_rows):
        key = line_no(vsr_row, idx)
        sage_row = sage_by_line.get(key)
        if not sage_row:
            continue
        sage_f1 = compute_f1(prediction(sage_row), gold_answer(sage_row))
        vsr_f1 = compute_f1(prediction(vsr_row), gold_answer(vsr_row))
        delta = sage_f1 - vsr_f1
        deltas.append(delta)
        by_task.setdefault(task_label(vsr_row), []).append({"sage_f1": sage_f1, "vsr_f1": vsr_f1, "delta": delta})
    task_rows = []
    for task, values in sorted(by_task.items()):
        task_rows.append(
            {
                "task": task,
                "n": len(values),
                "sage_f1": statistics.mean(item["sage_f1"] for item in values),
                "vsr_rescue_f1": statistics.mean(item["vsr_f1"] for item in values),
                "delta": statistics.mean(item["delta"] for item in values),
            }
        )
    return {
        "n": len(deltas),
        "mean_delta": statistics.mean(deltas) if deltas else 0.0,
        "wins": sum(delta > 1e-12 for delta in deltas),
        "losses": sum(delta < -1e-12 for delta in deltas),
        "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "by_task": task_rows,
    }


def load_reproduction_validation() -> dict[str, Any]:
    path = ROOT / "results" / "sage_repro" / "sage_reproduction_validation.json"
    if not path.exists():
        return {"all_object_exact": False, "all_predictions_exact": False, "pairs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize() -> dict[str, Any]:
    methods = [aggregate_method(spec) for spec in MIX130_METHODS]
    methods.sort(key=lambda row: (row["n"], row["mean_f1"]), reverse=True)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mix130_methods": methods,
        "sage_vsr_pairwise": paired_sage_vsr(),
        "aligned_retrieval_slices": load_aligned_retrieval_summary(),
        "sage_judge": [judge_summary(spec) for spec in JUDGE_FILES],
        "sage_reproduction": load_reproduction_validation(),
    }


def write_report(payload: dict[str, Any]) -> None:
    pairwise = payload["sage_vsr_pairwise"]
    repro = payload["sage_reproduction"]
    lines = [
        "# SAGE vs Baselines Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Decision View",
        "",
        "- SAGE v31/v32/v33 are the strongest current large-N lexical-F1 candidates on mix130, with v33 adding the preservation safety guard while leaving mix130 predictions unchanged.",
        "- The next API-heavy work should validate the frozen SAGE-style method on external slices or additional unseen contexts, not expand the negative retrieved-top5 VSR branch.",
        f"- Reproduction status: object-exact={repro.get('all_object_exact')}, prediction-exact={repro.get('all_predictions_exact')}.",
        "",
        "## Mix130 Lexical Results",
        "",
        "| Method | Protocol | n | F1 | ROUGE-L | Runtime s/q | Source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["mix130_methods"]:
        lines.append(
            f"| {row['method']} | {row['protocol']} | {row['n']} | {row['mean_f1']:.4f} | "
            f"{row['mean_rouge_l']:.4f} | {row['mean_runtime_seconds']:.1f} | `{row['path']}` |"
        )
    lines.extend(
        [
            "",
            "## Paired Mix130 Check",
            "",
            "| Comparison | n | Mean F1 delta | Wins | Losses | Ties |",
            "|---|---:|---:|---:|---:|---:|",
            f"| SAGE v33 - aligned VSR full_v2_rescue | {pairwise['n']} | {pairwise['mean_delta']:+.4f} | {pairwise['wins']} | {pairwise['losses']} | {pairwise['ties']} |",
            "",
            "| Task family | n | SAGE v33 F1 | VSR rescue F1 | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(pairwise["by_task"], key=lambda item: item["delta"], reverse=True):
        lines.append(
            f"| {row['task']} | {row['n']} | {row['sage_f1']:.4f} | "
            f"{row['vsr_rescue_f1']:.4f} | {row['delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## SAGE Judge Snapshot",
            "",
            "| Method | n | Correctness | Coverage | Relevance | Conciseness | Overall | Verdicts |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["sage_judge"]:
        verdicts = ", ".join(f"{key}:{value}" for key, value in sorted(row["verdicts"].items()))
        lines.append(
            f"| {row['method']} | {row['n']} | {row['correctness']:.3f} | {row['coverage']:.3f} | "
            f"{row['relevance']:.3f} | {row['conciseness']:.3f} | {row['overall']:.3f} | {verdicts} |"
        )
    lines.extend(
        [
            "",
            "## External Aligned Retrieval Status",
            "",
            "| Dataset | n | Best retrieval | Best retrieval F1 | VSR rescue F1 | Delta | Best retrieval runtime | VSR runtime |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["aligned_retrieval_slices"]:
        lines.append(
            f"| {row['dataset']} | {row['n']} | {row['best_retrieval_method']} | "
            f"{row['best_retrieval_f1']:.4f} | {row['vsr_rescue_f1']:.4f} | "
            f"{row['delta_vsr_minus_best_retrieval']:+.4f} | {row['best_retrieval_runtime']:.1f} | "
            f"{row['vsr_rescue_runtime']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Next Run",
            "",
            "- Freeze SAGE v33-preservation as the candidate routing-and-guard policy for the next validation branch.",
            "- Run it on a locked full-slice protocol with the same lexical and judge metrics before using it as the paper headline method.",
            "- Keep VSR full_v2_rescue in the ablation table because it has strong paired internal gains, but do not claim broad F1 superiority over Naive/HyDE from the current aligned evidence.",
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
