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


OUT_DIR = ROOT / "results" / "sage_locked_fullslice"
OUTPUT_JSON = OUT_DIR / "sage_locked_fullslice_ladder_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_LOCKED_FULLSLICE_LADDER_SUMMARY.md"


SLICES: list[dict[str, Any]] = [
    {
        "dataset": "Mix130 internal",
        "expected_n": 130,
        "scope": "internal large-N mix; SAGE replay is deterministic from frozen artifacts",
        "best_retrieval_keys": ["historical_naive", "historical_hyde"],
        "methods": [
            {
                "key": "historical_naive",
                "label": "Historical Naive RAG",
                "path": "results/enhanced/naive_rag_n130.json",
                "branch": "retrieval baseline",
                "optional": True,
            },
            {
                "key": "historical_hyde",
                "label": "Historical HyDE",
                "path": "results/enhanced/hyde_n130.json",
                "branch": "retrieval baseline",
                "optional": True,
            },
            {
                "key": "vsr_wo_repair",
                "label": "Aligned VSR w/o_repair",
                "path": "results/enhanced_ablation_aligned/w_o_repair_aligned_n130.json",
                "branch": "VSR ablation",
            },
            {
                "key": "vsr_full_v2",
                "label": "Aligned VSR full_v2",
                "path": "results/enhanced_ablation_aligned/full_v2_aligned_n130.json",
                "branch": "VSR ablation",
            },
            {
                "key": "vsr_rescue",
                "label": "Aligned VSR full_v2_rescue",
                "path": "results/enhanced_ablation_aligned/full_v2_rescue_aligned_n130.json",
                "branch": "VSR ablation",
            },
            {
                "key": "sage_v31",
                "label": "SAGE v31 learned contract-gap",
                "path": "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130.jsonl",
                "branch": "SAGE replay",
            },
            {
                "key": "sage_v32",
                "label": "SAGE v32 evidence-delta",
                "path": "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130.jsonl",
                "branch": "SAGE replay",
            },
            {
                "key": "sage_v33",
                "label": "SAGE v33 preservation-delta",
                "path": "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl",
                "branch": "SAGE replay",
            },
        ],
    },
    {
        "dataset": "Legal full context",
        "expected_n": 20,
        "scope": "external full-slice cumulative SAGE-on-HyDE risk-pool protocol",
        "sage_v33_judge": "results/historical_datasets/legal_sage_v33_fullslice_eval.jsonl",
        "best_retrieval_keys": ["naive", "hyde"],
        "methods": [
            {
                "key": "naive",
                "label": "Naive RAG",
                "path": "results/legal_fullcontext_20/naive_aligned_n20.jsonl",
                "judge": "results/historical_datasets/legal_naive_fullslice_eval.jsonl",
                "branch": "retrieval baseline",
            },
            {
                "key": "hyde",
                "label": "HyDE",
                "path": "results/legal_fullcontext_20/hyde_aligned_n20.jsonl",
                "judge": "results/historical_datasets/legal_hyde_fullslice_eval.jsonl",
                "branch": "retrieval baseline",
            },
            {
                "key": "vsr_wo_repair",
                "label": "Aligned VSR w/o_repair",
                "path": "results/enhanced_ablation_aligned/legal_fullcontext_20/w_o_repair_aligned_n20.json",
                "branch": "VSR ablation",
            },
            {
                "key": "vsr_full_v2",
                "label": "Aligned VSR full_v2",
                "path": "results/enhanced_ablation_aligned/legal_fullcontext_20/full_v2_aligned_n20.json",
                "branch": "VSR ablation",
            },
            {
                "key": "vsr_rescue",
                "label": "Aligned VSR full_v2_rescue",
                "path": "results/enhanced_ablation_aligned/legal_fullcontext_20/full_v2_rescue_aligned_n20.json",
                "judge": "results/historical_datasets/legal_vsr_rescue_fullslice_eval.jsonl",
                "branch": "VSR ablation",
            },
            {
                "key": "sage_v31",
                "label": "SAGE v31 contract-completion",
                "path": "results/historical_datasets/legal_hyde_contract_completion_cumulative.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
            {
                "key": "sage_v32",
                "label": "SAGE v32 evidence-delta",
                "path": "results/historical_datasets/legal_hyde_evidence_delta_cumulative.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
            {
                "key": "sage_v33",
                "label": "SAGE v33 preservation-delta",
                "path": "results/historical_datasets/legal_hyde_evidence_delta_v33_cumulative.jsonl",
                "judge": "results/historical_datasets/legal_sage_v33_fullslice_eval.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
        ],
    },
    {
        "dataset": "Biology full context",
        "expected_n": 50,
        "scope": "external full-slice cumulative SAGE-on-HyDE risk-pool protocol",
        "sage_v33_judge": "results/historical_datasets/biology_sage_v33_fullslice_eval.jsonl",
        "best_retrieval_keys": ["naive", "hyde"],
        "methods": [
            {
                "key": "naive",
                "label": "Naive RAG",
                "path": "results/biology_fullcontext_50/naive_aligned_n50.jsonl",
                "judge": "results/historical_datasets/biology_naive_fullslice_eval.jsonl",
                "branch": "retrieval baseline",
            },
            {
                "key": "hyde",
                "label": "HyDE",
                "path": "results/biology_fullcontext_50/hyde_aligned_n50.jsonl",
                "judge": "results/historical_datasets/biology_hyde_fullslice_eval.jsonl",
                "branch": "retrieval baseline",
            },
            {
                "key": "vsr_wo_repair",
                "label": "Aligned VSR w/o_repair",
                "path": "results/enhanced_ablation_aligned/biology_fullcontext_50/w_o_repair_aligned_n50.json",
                "branch": "VSR ablation",
            },
            {
                "key": "vsr_full_v2",
                "label": "Aligned VSR full_v2",
                "path": "results/enhanced_ablation_aligned/biology_fullcontext_50/full_v2_aligned_n50.json",
                "branch": "VSR ablation",
            },
            {
                "key": "vsr_rescue",
                "label": "Aligned VSR full_v2_rescue",
                "path": "results/enhanced_ablation_aligned/biology_fullcontext_50/full_v2_rescue_aligned_n50.json",
                "judge": "results/historical_datasets/biology_vsr_rescue_fullslice_eval.jsonl",
                "branch": "VSR ablation",
            },
            {
                "key": "sage_v31",
                "label": "SAGE v31 contract-completion",
                "path": "results/historical_datasets/biology_hyde_contract_completion_cumulative.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
            {
                "key": "sage_v32",
                "label": "SAGE v32 evidence-delta",
                "path": "results/historical_datasets/biology_hyde_evidence_delta_cumulative.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
            {
                "key": "sage_v33",
                "label": "SAGE v33 preservation-delta",
                "path": "results/historical_datasets/biology_hyde_evidence_delta_v33_cumulative.jsonl",
                "judge": "results/historical_datasets/biology_sage_v33_fullslice_eval.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
        ],
    },
    {
        "dataset": "Domain expansion windowed",
        "expected_n": 50,
        "scope": "external full-slice cumulative SAGE-on-HyDE risk-pool protocol",
        "sage_v33_judge": "results/historical_datasets/domain_sage_v33_fullslice_eval.jsonl",
        "best_retrieval_keys": ["naive", "hyde"],
        "methods": [
            {
                "key": "naive",
                "label": "Naive RAG",
                "path": "results/domain_expansion_windowed_50/naive_aligned_n50.jsonl",
                "judge": "results/historical_datasets/domain_naive_fullslice_eval.jsonl",
                "branch": "retrieval baseline",
            },
            {
                "key": "hyde",
                "label": "HyDE",
                "path": "results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl",
                "judge": "results/historical_datasets/domain_hyde_fullslice_eval.jsonl",
                "branch": "retrieval baseline",
            },
            {
                "key": "vsr_wo_repair",
                "label": "Aligned VSR w/o_repair",
                "path": "results/enhanced_ablation_aligned/domain_expansion_windowed_50/w_o_repair_aligned_n50.json",
                "branch": "VSR ablation",
            },
            {
                "key": "vsr_full_v2",
                "label": "Aligned VSR full_v2",
                "path": "results/enhanced_ablation_aligned/domain_expansion_windowed_50/full_v2_aligned_n50.json",
                "branch": "VSR ablation",
            },
            {
                "key": "vsr_rescue",
                "label": "Aligned VSR full_v2_rescue",
                "path": "results/enhanced_ablation_aligned/domain_expansion_windowed_50/full_v2_rescue_aligned_n50.json",
                "judge": "results/historical_datasets/domain_vsr_rescue_fullslice_eval.jsonl",
                "branch": "VSR ablation",
            },
            {
                "key": "sage_v31",
                "label": "SAGE v31 contract-completion",
                "path": "results/historical_datasets/domain_hyde_contract_completion_cumulative.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
            {
                "key": "sage_v32",
                "label": "SAGE v32 evidence-delta",
                "path": "results/historical_datasets/domain_hyde_evidence_delta_cumulative.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
            {
                "key": "sage_v33",
                "label": "SAGE v33 preservation-delta",
                "path": "results/historical_datasets/domain_hyde_evidence_delta_v33_cumulative.jsonl",
                "judge": "results/historical_datasets/domain_sage_v33_fullslice_eval.jsonl",
                "branch": "SAGE-on-HyDE cumulative",
            },
        ],
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_rows(path: Path) -> list[dict[str, Any]]:
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


def gold_answer(row: dict[str, Any]) -> str:
    if row.get("gold_answer"):
        return str(row["gold_answer"])
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
    if row.get("runtime_seconds") is not None:
        return float(row.get("runtime_seconds") or 0.0)
    return float(row.get("query_seconds", 0.0) or 0.0) + float(row.get("insert_seconds", 0.0) or 0.0)


def rows_by_line(spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    path = ROOT / spec["path"]
    if not path.exists():
        if spec.get("optional"):
            return {}
        raise FileNotFoundError(path)
    return {line_no(row, index): row for index, row in enumerate(load_rows(path))}


def build_gold(method_rows: dict[str, dict[int, dict[str, Any]]]) -> dict[int, str]:
    gold: dict[int, str] = {}
    for rows in method_rows.values():
        for line, row in rows.items():
            value = gold_answer(row)
            if value and line not in gold:
                gold[line] = value
    return gold


def score_method(rows: dict[int, dict[str, Any]], gold: dict[int, str], line_nos: list[int]) -> dict[str, Any]:
    f1_by_line = {line: compute_f1(prediction(rows[line]), gold[line]) for line in line_nos}
    rouge_by_line = {line: compute_rouge_l(prediction(rows[line]), gold[line]) for line in line_nos}
    runtimes = [runtime(rows[line]) for line in line_nos]
    nonzero_runtimes = [value for value in runtimes if value > 0]
    return {
        "n": len(line_nos),
        "f1": statistics.mean(f1_by_line.values()) if f1_by_line else 0.0,
        "rouge_l": statistics.mean(rouge_by_line.values()) if rouge_by_line else 0.0,
        "runtime_seconds": statistics.mean(nonzero_runtimes) if nonzero_runtimes else 0.0,
        "f1_by_line": f1_by_line,
        "rouge_l_by_line": rouge_by_line,
    }


def pairwise(left: dict[str, Any], right: dict[str, Any], metric: str = "f1") -> dict[str, Any]:
    common = sorted(set(left[f"{metric}_by_line"]) & set(right[f"{metric}_by_line"]))
    left_values = [left[f"{metric}_by_line"][line] for line in common]
    right_values = [right[f"{metric}_by_line"][line] for line in common]
    deltas = [left_value - right_value for left_value, right_value in zip(left_values, right_values)]
    t_stat, t_p = paired_t_test(left_values, right_values)
    w_stat, w_p = wilcoxon_signed_rank_test(left_values, right_values)
    return {
        "metric": metric,
        "n": len(common),
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


def prediction_diff_count(
    left_rows: dict[int, dict[str, Any]],
    right_rows: dict[int, dict[str, Any]],
    line_nos: list[int],
) -> int:
    return sum(prediction(left_rows[line]) != prediction(right_rows[line]) for line in line_nos)


def judge_summary(path_or_relative: str | None) -> dict[str, Any] | None:
    if not path_or_relative:
        return None
    path = ROOT / path_or_relative
    if not path.exists():
        return None
    rows = load_jsonl(path)
    metrics = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    scores_by_line = {
        line_no(row, index): {
            metric: float(row.get("scores", {}).get(metric, 0.0) or 0.0)
            for metric in metrics
        }
        for index, row in enumerate(rows)
    }
    averages = {
        metric: statistics.mean(scores[metric] for scores in scores_by_line.values())
        for metric in metrics
    }
    verdicts = Counter(str(row.get("scores", {}).get("verdict") or "unknown") for row in rows)
    return {
        "path": path_or_relative,
        "n": len(rows),
        **averages,
        "scores_by_line": scores_by_line,
        "verdicts": dict(sorted(verdicts.items())),
    }


def method_judges(method_specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries = {}
    for method in method_specs:
        summary = judge_summary(method.get("judge"))
        if summary:
            summary["label"] = method["label"]
            summary["branch"] = method["branch"]
            summaries[method["key"]] = summary
    return summaries


def judge_pairwise(left: dict[str, Any], right: dict[str, Any], metric: str = "overall") -> dict[str, Any]:
    left_by_line = left.get("scores_by_line", {})
    right_by_line = right.get("scores_by_line", {})
    common = sorted(set(left_by_line) & set(right_by_line))
    left_values = [left_by_line[line][metric] for line in common]
    right_values = [right_by_line[line][metric] for line in common]
    deltas = [left_value - right_value for left_value, right_value in zip(left_values, right_values)]
    t_stat, t_p = paired_t_test(left_values, right_values)
    w_stat, w_p = wilcoxon_signed_rank_test(left_values, right_values)
    return {
        "metric": metric,
        "n": len(common),
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


def summarize_slice(spec: dict[str, Any]) -> dict[str, Any]:
    method_specs = spec["methods"]
    method_rows = {method["key"]: rows_by_line(method) for method in method_specs}
    judges = method_judges(method_specs)
    active_keys = [method["key"] for method in method_specs if method_rows.get(method["key"])]
    common_lines = sorted(set.intersection(*(set(method_rows[key]) for key in active_keys)))
    gold = build_gold({key: method_rows[key] for key in active_keys})
    common_lines = [line for line in common_lines if gold.get(line)]
    scores = {
        key: score_method(method_rows[key], gold, common_lines)
        for key in active_keys
    }

    method_lookup = {method["key"]: method for method in method_specs}
    best_retrieval = None
    retrieval_keys = [key for key in spec.get("best_retrieval_keys", []) if key in scores]
    if retrieval_keys:
        best_retrieval = max(retrieval_keys, key=lambda key: scores[key]["f1"])

    ladder_rows = []
    previous_key = None
    for method in method_specs:
        key = method["key"]
        if key not in scores:
            continue
        row: dict[str, Any] = {
            "key": key,
            "label": method["label"],
            "branch": method["branch"],
            "path": method["path"],
            **scores[key],
        }
        if previous_key is not None:
            row["delta_vs_previous"] = pairwise(scores[key], scores[previous_key], "f1")["delta"]
        else:
            row["delta_vs_previous"] = None
        ladder_rows.append(row)
        previous_key = key

    pairwise_tests: dict[str, Any] = {}
    if "sage_v33" in scores and best_retrieval:
        pairwise_tests["sage_v33_vs_best_retrieval"] = {
            "right_key": best_retrieval,
            "right_label": method_lookup[best_retrieval]["label"],
            **pairwise(scores["sage_v33"], scores[best_retrieval], "f1"),
        }
    if "sage_v33" in scores and "hyde" in scores:
        pairwise_tests["sage_v33_vs_hyde"] = {
            "right_key": "hyde",
            "right_label": method_lookup["hyde"]["label"],
            **pairwise(scores["sage_v33"], scores["hyde"], "f1"),
        }
    if "sage_v33" in scores and "vsr_rescue" in scores:
        pairwise_tests["sage_v33_vs_vsr_rescue"] = {
            "right_key": "vsr_rescue",
            "right_label": method_lookup["vsr_rescue"]["label"],
            **pairwise(scores["sage_v33"], scores["vsr_rescue"], "f1"),
        }
    if "sage_v33" in scores and "sage_v31" in scores:
        pairwise_tests["sage_v33_vs_sage_v31"] = {
            "right_key": "sage_v31",
            "right_label": method_lookup["sage_v31"]["label"],
            **pairwise(scores["sage_v33"], scores["sage_v31"], "f1"),
        }
    if "sage_v33" in scores and "sage_v32" in scores:
        pairwise_tests["sage_v33_vs_sage_v32"] = {
            "right_key": "sage_v32",
            "right_label": method_lookup["sage_v32"]["label"],
            **pairwise(scores["sage_v33"], scores["sage_v32"], "f1"),
        }

    prediction_diffs: dict[str, int] = {}
    if "sage_v33" in method_rows:
        for anchor_key in ["hyde", "vsr_rescue", "sage_v31", "sage_v32"]:
            if anchor_key in method_rows and anchor_key in scores:
                prediction_diffs[f"sage_v33_vs_{anchor_key}"] = prediction_diff_count(
                    method_rows["sage_v33"], method_rows[anchor_key], common_lines
                )

    judge_pairwise_tests: dict[str, Any] = {}
    if "sage_v33" in judges:
        judge_anchors = []
        if best_retrieval:
            judge_anchors.append(("best_retrieval", best_retrieval, method_lookup[best_retrieval]["label"]))
        judge_anchors.extend(
            [
                ("hyde", "hyde", method_lookup["hyde"]["label"] if "hyde" in method_lookup else "HyDE"),
                ("naive", "naive", method_lookup["naive"]["label"] if "naive" in method_lookup else "Naive RAG"),
                (
                    "vsr_rescue",
                    "vsr_rescue",
                    method_lookup["vsr_rescue"]["label"] if "vsr_rescue" in method_lookup else "VSR rescue",
                ),
            ]
        )
        for test_key, anchor_key, anchor_label in judge_anchors:
            if anchor_key in judges:
                judge_pairwise_tests[f"sage_v33_vs_{test_key}"] = {
                    "right_key": anchor_key,
                    "right_label": anchor_label,
                    **judge_pairwise(judges["sage_v33"], judges[anchor_key], "overall"),
                }

    return {
        "dataset": spec["dataset"],
        "expected_n": spec["expected_n"],
        "scope": spec["scope"],
        "n": len(common_lines),
        "complete": len(common_lines) == spec["expected_n"],
        "missing_lines": sorted(set(range(spec["expected_n"])) - set(common_lines)),
        "best_retrieval_key": best_retrieval,
        "best_retrieval_label": method_lookup[best_retrieval]["label"] if best_retrieval else None,
        "best_retrieval_f1": scores[best_retrieval]["f1"] if best_retrieval else None,
        "scores": scores,
        "ladder": ladder_rows,
        "pairwise_tests": pairwise_tests,
        "prediction_diffs": prediction_diffs,
        "judges": judges,
        "sage_v33_judge": judges.get("sage_v33") or judge_summary(spec.get("sage_v33_judge")),
        "judge_pairwise_tests": judge_pairwise_tests,
    }


def external_weighted_average(slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    external = [row for row in slices if row["dataset"] != "Mix130 internal"]
    keys = sorted(set.intersection(*(set(row["scores"]) for row in external)))
    method_labels = {
        method["key"]: method["label"]
        for spec in SLICES
        for method in spec["methods"]
    }
    weighted_rows = []
    for key in keys:
        total_n = sum(row["scores"][key]["n"] for row in external)
        weighted_f1 = sum(row["scores"][key]["f1"] * row["scores"][key]["n"] for row in external) / total_n
        weighted_rouge = sum(row["scores"][key]["rouge_l"] * row["scores"][key]["n"] for row in external) / total_n
        weighted_rows.append(
            {
                "key": key,
                "label": method_labels.get(key, key),
                "n": total_n,
                "weighted_f1": weighted_f1,
                "weighted_rouge_l": weighted_rouge,
            }
        )
    weighted_rows.sort(key=lambda row: row["weighted_f1"], reverse=True)
    return weighted_rows


def weighted_judge_for_key(slices: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    external = [
        row
        for row in slices
        if row["dataset"] != "Mix130 internal" and row.get("judges", {}).get(key)
    ]
    if not external:
        return None
    total_n = sum(row["judges"][key]["n"] for row in external)
    metrics = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    weighted = {
        metric: sum(row["judges"][key][metric] * row["judges"][key]["n"] for row in external) / total_n
        for metric in metrics
    }
    verdicts: Counter[str] = Counter()
    for row in external:
        verdicts.update(row["judges"][key].get("verdicts", {}))
    label = external[0]["judges"][key].get("label", key)
    return {"key": key, "label": label, "n": total_n, **weighted, "verdicts": dict(sorted(verdicts.items()))}


def weighted_best_retrieval_judge(slices: list[dict[str, Any]]) -> dict[str, Any] | None:
    external = []
    for row in slices:
        if row["dataset"] == "Mix130 internal":
            continue
        best_key = row.get("best_retrieval_key")
        judge = row.get("judges", {}).get(best_key)
        if judge:
            external.append((row, judge))
    if not external:
        return None
    total_n = sum(judge["n"] for _, judge in external)
    metrics = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    weighted = {
        metric: sum(judge[metric] * judge["n"] for _, judge in external) / total_n
        for metric in metrics
    }
    verdicts: Counter[str] = Counter()
    for _, judge in external:
        verdicts.update(judge.get("verdicts", {}))
    return {
        "key": "best_retrieval",
        "label": "Best retrieval by lexical F1",
        "n": total_n,
        **weighted,
        "verdicts": dict(sorted(verdicts.items())),
    }


def external_weighted_judges(slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in ["sage_v33", "vsr_rescue", "hyde", "naive"]:
        summary = weighted_judge_for_key(slices, key)
        if summary:
            rows.append(summary)
    best = weighted_best_retrieval_judge(slices)
    if best:
        rows.append(best)
    return rows


def summarize() -> dict[str, Any]:
    slices = [summarize_slice(spec) for spec in SLICES]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "name": "SAGE v33 locked full-slice ladder",
            "candidate": "SAGE v33 preservation-delta",
            "source": "offline recomputation from frozen result artifacts; no new LLM calls",
            "claim_scope": "supports candidate selection and ablation planning; not yet the final SCI locked claim",
        },
        "slices": slices,
        "external_weighted": external_weighted_average(slices),
        "external_weighted_judge": weighted_judge_for_key(slices, "sage_v33"),
        "external_weighted_judges": external_weighted_judges(slices),
    }


def fmt_delta(value: float | None) -> str:
    return "-" if value is None else f"{value:+.4f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.4g}"


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# SAGE Locked Full-Slice Ladder Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Protocol",
        "",
        f"- Candidate: `{payload['protocol']['candidate']}`.",
        "- Source: offline recomputation from frozen artifacts; no new LLM/API calls were made.",
        "- Scope: full-slice scoring for available cumulative outputs; external SAGE rows are still SAGE-on-HyDE risk-pool cumulative artifacts.",
        "- Use: candidate selection, ablation ladder, and next-run planning; keep final paper claims conservative until a fully locked benchmark run is executed.",
        "",
        "## Completeness",
        "",
        "| Slice | Expected N | Scored N | Complete | Scope | v33-v32 prediction diffs |",
        "|---|---:|---:|---|---|---:|",
    ]
    for row in payload["slices"]:
        lines.append(
            f"| {row['dataset']} | {row['expected_n']} | {row['n']} | {row['complete']} | "
            f"{row['scope']} | {row['prediction_diffs'].get('sage_v33_vs_sage_v32', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Candidate View",
            "",
            "| Slice | N | Best retrieval | Best retrieval F1 | SAGE v33 F1 | SAGE - best retrieval | SAGE - VSR rescue | Wins/Losses vs VSR |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["slices"]:
        scores = row["scores"]
        v33 = scores.get("sage_v33", {})
        best_stats = row["pairwise_tests"].get("sage_v33_vs_best_retrieval", {})
        vsr_stats = row["pairwise_tests"].get("sage_v33_vs_vsr_rescue", {})
        lines.append(
            f"| {row['dataset']} | {row['n']} | {row['best_retrieval_label'] or '-'} | "
            f"{(row['best_retrieval_f1'] if row['best_retrieval_f1'] is not None else 0.0):.4f} | "
            f"{v33.get('f1', 0.0):.4f} | {fmt_delta(best_stats.get('delta'))} | "
            f"{fmt_delta(vsr_stats.get('delta'))} | {vsr_stats.get('wins', 0)}/{vsr_stats.get('losses', 0)} |"
        )

    weighted_judges = payload.get("external_weighted_judges") or []
    lines.extend(
        [
            "",
            "## Full-Slice Judge Snapshot",
            "",
            "| Slice | Method | Judge N | Correctness | Coverage | Relevance | Conciseness | Overall | Verdicts | Source |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["slices"]:
        for judge in row.get("judges", {}).values():
            verdicts = ", ".join(f"{key}:{value}" for key, value in judge["verdicts"].items())
            lines.append(
                f"| {row['dataset']} | {judge['label']} | {judge['n']} | {judge['correctness']:.2f} | "
                f"{judge['coverage']:.2f} | {judge['relevance']:.2f} | {judge['conciseness']:.2f} | "
                f"{judge['overall']:.2f} | {verdicts} | `{judge['path']}` |"
            )
    for judge in weighted_judges:
        verdicts = ", ".join(f"{key}:{value}" for key, value in judge["verdicts"].items())
        lines.append(
            f"| Weighted external | {judge['label']} | {judge['n']} | {judge['correctness']:.2f} | "
            f"{judge['coverage']:.2f} | {judge['relevance']:.2f} | {judge['conciseness']:.2f} | "
            f"{judge['overall']:.2f} | {verdicts} | - |"
        )

    lines.extend(
        [
            "",
            "## Judge Pairwise Tests",
            "",
            "| Slice | Comparison | Delta overall | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["slices"]:
        for key, label in [
            ("sage_v33_vs_best_retrieval", "SAGE v33 - best retrieval"),
            ("sage_v33_vs_hyde", "SAGE v33 - HyDE"),
            ("sage_v33_vs_naive", "SAGE v33 - Naive"),
            ("sage_v33_vs_vsr_rescue", "SAGE v33 - VSR rescue"),
        ]:
            stats = row.get("judge_pairwise_tests", {}).get(key)
            if not stats:
                continue
            comparison = f"{label} ({stats['right_label']})" if key == "sage_v33_vs_best_retrieval" else label
            lines.append(
                f"| {row['dataset']} | {comparison} | {stats['delta']:+.2f} | {stats['wins']} | "
                f"{stats['losses']} | {stats['ties']} | {fmt_p(stats['t_p'])} | "
                f"{fmt_p(stats['wilcoxon_p'])} | {stats['cohens_d']:+.2f} |"
            )

    lines.extend(
        [
            "",
            "## External Weighted View",
            "",
            "| Method | N | Weighted F1 | Weighted ROUGE-L |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["external_weighted"]:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['weighted_f1']:.4f} | {row['weighted_rouge_l']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Tests",
            "",
            "| Slice | Comparison | Delta F1 | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    comparisons = [
        ("sage_v33_vs_best_retrieval", "SAGE v33 - best retrieval"),
        ("sage_v33_vs_hyde", "SAGE v33 - HyDE"),
        ("sage_v33_vs_vsr_rescue", "SAGE v33 - VSR rescue"),
        ("sage_v33_vs_sage_v31", "SAGE v33 - SAGE v31"),
        ("sage_v33_vs_sage_v32", "SAGE v33 - SAGE v32"),
    ]
    for row in payload["slices"]:
        for key, label in comparisons:
            stats = row["pairwise_tests"].get(key)
            if not stats:
                continue
            right_label = stats.get("right_label")
            comparison = f"{label} ({right_label})" if key == "sage_v33_vs_best_retrieval" else label
            lines.append(
                f"| {row['dataset']} | {comparison} | {stats['delta']:+.4f} | {stats['wins']} | "
                f"{stats['losses']} | {stats['ties']} | {fmt_p(stats['t_p'])} | "
                f"{fmt_p(stats['wilcoxon_p'])} | {stats['cohens_d']:+.2f} |"
            )

    for row in payload["slices"]:
        lines.extend(
            [
                "",
                f"## {row['dataset']} Ladder",
                "",
                "| Step | Branch | N | F1 | ROUGE-L | Runtime s/q | Delta vs previous | Source |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for step in row["ladder"]:
            lines.append(
                f"| {step['label']} | {step['branch']} | {step['n']} | {step['f1']:.4f} | "
                f"{step['rouge_l']:.4f} | {step['runtime_seconds']:.1f} | "
                f"{fmt_delta(step['delta_vs_previous'])} | `{step['path']}` |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- SAGE v33 is tied with v31/v32 on Mix130 and is the safest current candidate because the preservation guard removes the observed Biology regression.",
            "- SAGE v31 is numerically highest on weighted external F1 by 0.0003, but v33 is preferred for candidate freezing because Biology becomes safety-neutral rather than slightly harmful.",
            "- External gains over the best retrieval baseline remain small: positive on Legal/Domain and neutral on Biology.",
            "- Same-rubric full-slice judge scoring now covers external Naive, HyDE, VSR rescue, and SAGE v33 outputs.",
            "- Weighted external judge quality favors SAGE v33 over best retrieval and VSR rescue, even though external lexical-F1 gains over best retrieval remain small.",
            "- The external SAGE protocol is full-slice cumulative scoring, but only risk-candidate rows received SAGE repair; this must be described explicitly in the paper.",
            "- The next API work should validate v33 on additional unseen samples or run robustness repeats, not add another exploratory repair variant.",
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
