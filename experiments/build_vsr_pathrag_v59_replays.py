import json
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.build_vsr_pathrag_v53_selective_replay import (
    METRICS,
    keyed,
    load_jsonl,
    metric_means,
    write_jsonl,
)
from experiments.build_vsr_pathrag_v54_pairwise_replay import DIMENSIONS


def build_selective(results_dir: Path) -> dict:
    base = keyed(
        load_jsonl(
            results_dir / "vsr_pathrag_v58_selective_replay_30.jsonl"
        )
    )
    changed = keyed(
        load_jsonl(
            results_dir / "vsr_pathrag_v59_lien_release_case.jsonl"
        )
    )
    base_eval = keyed(
        load_jsonl(
            results_dir
            / "vsr_pathrag_v58_selective_replay_30_eval.jsonl"
        )
    )
    changed_eval = keyed(
        load_jsonl(
            results_dir
            / "vsr_pathrag_v59_lien_release_case_eval.jsonl"
        )
    )
    if set(base) != set(range(30)) or set(changed) != {10}:
        raise ValueError("v59 selective inputs are inconsistent")
    rows = [changed.get(i, base[i]) for i in range(30)]
    evaluations = [
        changed_eval.get(i, base_eval[i]) for i in range(30)
    ]
    write_jsonl(
        results_dir / "vsr_pathrag_v59_selective_replay_30.jsonl",
        rows,
    )
    write_jsonl(
        results_dir / "vsr_pathrag_v59_selective_replay_30_eval.jsonl",
        evaluations,
    )
    means = metric_means(evaluations)
    base_means = metric_means(base_eval)
    manifest = {
        "protocol": "deterministic_gate_selective_replay",
        "agent_version": "vsr-pathrag-v59",
        "sample_count": 30,
        "audited_trigger_count": 1,
        "actual_v59_rerun_count": 1,
        "reused_v58_nontrigger_count": 29,
        "triggered_line_nos": [10],
        "means": means,
        "v58_means": base_means,
        "mean_deltas": {
            metric: round(means[metric] - base_means[metric], 3)
            for metric in METRICS
        },
        "verdict_counts": dict(
            Counter(row["scores"]["verdict"] for row in evaluations)
        ),
        "runtime_reference_answer_used": False,
        "full_stochastic_rerun": False,
        "ablation": False,
    }
    (
        results_dir
        / "vsr_pathrag_v59_selective_replay_manifest.json"
    ).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_pairwise(results_dir: Path) -> dict:
    base = keyed(
        load_jsonl(
            results_dir
            / "vsr_pathrag_v58_vs_hyde_pairwise_replay.jsonl"
        )
    )
    changed = keyed(
        load_jsonl(
            results_dir
            / "vsr_pathrag_v59_vs_hyde_changed_pairwise.jsonl"
        )
    )
    if set(base) != set(range(30)) or set(changed) != {10}:
        raise ValueError("v59 pairwise inputs are inconsistent")
    rows = [changed.get(i, base[i]) for i in range(30)]
    write_jsonl(
        results_dir
        / "vsr_pathrag_v59_vs_hyde_pairwise_replay.jsonl",
        rows,
    )
    counts = {}
    rates = {}
    for dimension in DIMENSIONS:
        counter = Counter(row["winners"][dimension] for row in rows)
        counts[dimension] = {
            "pathrag": counter["pathrag"],
            "baseline": counter["baseline"],
            "tie": counter["tie"],
        }
        rates[dimension] = round(
            (counter["pathrag"] + counter["tie"] * 0.5)
            / len(rows)
            * 100,
            2,
        )
    manifest = {
        "protocol": "changed_answer_pairwise_replay",
        "agent_version": "vsr-pathrag-v59",
        "sample_count": 30,
        "rejudged_line_nos": [10],
        "reused_v58_pairwise_count": 29,
        "counts": counts,
        "win_rates": rates,
        "average_pathrag_win_rate": round(
            statistics.mean(rates.values()), 2
        ),
        "runtime_reference_answer_used": False,
        "judge_reference_used": True,
        "full_pairwise_rerun": False,
        "ablation": False,
    }
    (
        results_dir
        / "vsr_pathrag_v59_vs_hyde_pairwise_replay_manifest.json"
    ).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    results_dir = ROOT_DIR / "results"
    selective = build_selective(results_dir)
    pairwise = build_pairwise(results_dir)
    print(
        f"overall={selective['means']['overall']:.3f} "
        f"pairwise={pairwise['average_pathrag_win_rate']:.2f}%"
    )


if __name__ == "__main__":
    main()
