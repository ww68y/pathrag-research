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
    compute_contains,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
)

OUT_DIR = ROOT / "results" / "legacy_baseline_overlap"
OUTPUT_JSON = OUT_DIR / "legacy_baseline_overlap_summary.json"
OUTPUT_MD = OUT_DIR / "LEGACY_BASELINE_OVERLAP_SUMMARY.md"

DATASETS = [
    {
        "name": "UltraDomain mix",
        "slug": "ultradomain_mix",
        "n": 130,
        "legacy": {
            "legacy_pathrag": ROOT / "results" / "ultradomain_mix_repro_current.jsonl",
            "legacy_direct_context": ROOT / "results" / "direct_context_baseline_current.jsonl",
            "legacy_graphrag_lite": ROOT / "results" / "ultradomain_mix_graphrag_lite.jsonl",
            "sage_v33_preservation": ROOT / "results" / "mse_pathrag_sage_v33_preservation_delta_mix130.jsonl",
        },
        "aligned": ROOT / "results" / "enhanced_ablation_aligned",
    },
    {
        "name": "Legal full context",
        "slug": "legal_fullcontext_20",
        "n": 20,
        "legacy": {
            "legacy_pathrag": ROOT / "results" / "legal_fullcontext_20_pathrag.jsonl",
            "legacy_naive": ROOT / "results" / "legal_fullcontext_20_naive_rag.jsonl",
            "legacy_hyde": ROOT / "results" / "legal_fullcontext_20_hyde_rag.jsonl",
            "legacy_graphrag_lite": ROOT / "results" / "legal_fullcontext_20_graphrag_lite.jsonl",
            "sage_v18_cost_aware_prerouter": ROOT / "results" / "mse_pathrag_sage_v18_cost_aware_prerouter_legal_20.jsonl",
        },
        "aligned": ROOT / "results" / "enhanced_ablation_aligned" / "legal_fullcontext_20",
    },
    {
        "name": "Biology full context",
        "slug": "biology_fullcontext_50",
        "n": 50,
        "legacy": {
            "legacy_pathrag": ROOT / "results" / "biology_fullcontext_50_pathrag.jsonl",
            "legacy_naive": ROOT / "results" / "biology_fullcontext_50_naive_rag.jsonl",
            "legacy_hyde": ROOT / "results" / "biology_fullcontext_50_hyde_rag.jsonl",
            "legacy_graphrag_lite": ROOT / "results" / "biology_fullcontext_50_graphrag_lite.jsonl",
        },
        "aligned": ROOT / "results" / "enhanced_ablation_aligned" / "biology_fullcontext_50",
    },
    {
        "name": "Domain expansion windowed",
        "slug": "domain_expansion_windowed_50",
        "n": 50,
        "legacy": {
            "legacy_pathrag": ROOT / "results" / "domain_expansion_windowed_50_pathrag.jsonl",
            "legacy_naive": ROOT / "results" / "domain_expansion_windowed_50_naive_rag.jsonl",
            "legacy_hyde": ROOT / "results" / "domain_expansion_windowed_50_hyde_rag.jsonl",
            "legacy_graphrag_lite": ROOT / "results" / "domain_expansion_windowed_50_graphrag_lite.jsonl",
        },
        "aligned": ROOT / "results" / "enhanced_ablation_aligned" / "domain_expansion_windowed_50",
    },
]

ALIGNED_METHOD_FILES = {
    "aligned_w/o_repair": "w_o_repair_aligned_n{n}.json",
    "aligned_full_v2": "full_v2_aligned_n{n}.json",
    "aligned_full_v2_rescue": "full_v2_rescue_aligned_n{n}.json",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_aligned(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [row for row in data.get("results", []) if not row.get("error")]


def gold_answer(row: dict[str, Any]) -> str:
    if row.get("gold_answer"):
        return str(row.get("gold_answer"))
    answers = row.get("reference_answers") or row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    prediction = str(row.get("prediction") or row.get("answer") or "")
    gold = gold_answer(row)
    runtime = float(row.get("query_seconds", 0.0) or 0.0) + float(row.get("insert_seconds", 0.0) or 0.0)
    if not runtime:
        runtime = float((row.get("metrics") or {}).get("runtime_seconds", 0.0) or 0.0)
    return {
        "f1": compute_f1(prediction, gold),
        "rouge_l": compute_rouge_l(prediction, gold),
        "exact_match": compute_exact_match(prediction, gold),
        "contains": compute_contains(prediction, gold),
        "runtime_seconds": runtime,
        "insert_seconds": float(row.get("insert_seconds", 0.0) or 0.0),
        "query_seconds": float(row.get("query_seconds", 0.0) or 0.0),
        "graph_nodes": float(row.get("graph_nodes", 0.0) or 0.0),
        "graph_edges": float(row.get("graph_edges", 0.0) or 0.0),
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def aggregate(method: str, rows: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    metrics = [row_metrics(row) for row in rows if not row.get("error")]
    return {
        "method": method,
        "source": str(source.relative_to(ROOT)),
        "sample_size": len(metrics),
        "mean_f1": mean([m["f1"] for m in metrics]),
        "mean_rouge_l": mean([m["rouge_l"] for m in metrics]),
        "mean_em": mean([m["exact_match"] for m in metrics]),
        "mean_contains": mean([m["contains"] for m in metrics]),
        "mean_runtime_seconds": mean([m["runtime_seconds"] for m in metrics]),
        "mean_insert_seconds": mean([m["insert_seconds"] for m in metrics]),
        "mean_query_seconds": mean([m["query_seconds"] for m in metrics]),
        "mean_graph_nodes": mean([m["graph_nodes"] for m in metrics]),
        "mean_graph_edges": mean([m["graph_edges"] for m in metrics]),
    }


def summarize() -> dict[str, Any]:
    payload: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": {}}
    for spec in DATASETS:
        methods: dict[str, dict[str, Any]] = {}
        for method, path in spec["legacy"].items():
            rows = load_jsonl(path)
            if rows:
                methods[method] = aggregate(method, rows, path)
        for method, template in ALIGNED_METHOD_FILES.items():
            path = spec["aligned"] / template.format(n=spec["n"])
            rows = load_aligned(path)
            if rows:
                methods[method] = aggregate(method, rows, path)
        best_f1 = max((row["mean_f1"] for row in methods.values()), default=0.0)
        best_methods = [method for method, row in methods.items() if abs(row["mean_f1"] - best_f1) < 1e-12]
        payload["datasets"][spec["slug"]] = {
            "name": spec["name"],
            "n": spec["n"],
            "methods": methods,
            "best_f1": best_f1,
            "best_methods": best_methods,
        }
    return payload


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Legacy Baseline Overlap Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Scope",
        "",
        "- This is an offline lexical re-scoring of existing legacy PathRAG/Naive/HyDE/GraphRAG-style outputs against the current reference answers.",
        "- It does not call an API and should not be mixed with LLM-as-judge scores as if they were the same metric.",
        "- Legacy outputs may have different prompts, retrieval settings, and answer verbosity than the aligned VSR runs, so this is a comparability diagnostic rather than the final main table.",
    ]
    for slug, dataset in payload["datasets"].items():
        lines.extend(
            [
                "",
                f"## {dataset['name']}",
                "",
                "| Method | n | F1 | ROUGE-L | EM | Contains | Runtime s/q | Insert s/q | Query s/q | Graph Nodes |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, agg in sorted(dataset["methods"].items(), key=lambda item: item[1]["mean_f1"], reverse=True):
            lines.append(
                f"| {method} | {agg['sample_size']} | {agg['mean_f1']:.4f} | {agg['mean_rouge_l']:.4f} | "
                f"{agg['mean_em']:.4f} | {agg['mean_contains']:.4f} | {agg['mean_runtime_seconds']:.1f} | "
                f"{agg['mean_insert_seconds']:.1f} | {agg['mean_query_seconds']:.1f} | {agg['mean_graph_nodes']:.1f} |"
            )
        lines.extend(
            [
                "",
                f"Best F1: {', '.join(dataset['best_methods'])} ({dataset['best_f1']:.4f}).",
            ]
        )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "- Existing legacy PathRAG outputs often receive strong LLM-as-judge scores, but lexical F1 can penalize their verbosity and extra details.",
            "- This diagnostic helps separate original-paper comparability from the stricter aligned F1/ROUGE protocol used for VSR ablations.",
            "- Any SCI main-result claim should state which metric family is being used and avoid comparing LLM-as-judge scores directly with lexical overlap scores.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = summarize()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
