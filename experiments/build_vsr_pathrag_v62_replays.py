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


def main() -> None:
    out = ROOT_DIR / "results"
    base = keyed(
        load_jsonl(out / "vsr_pathrag_v61_selective_replay_30.jsonl")
    )
    changed = keyed(
        load_jsonl(out / "vsr_pathrag_v62_grazing_intensity_case.jsonl")
    )
    base_eval = keyed(
        load_jsonl(
            out / "vsr_pathrag_v61_selective_replay_30_eval.jsonl"
        )
    )
    changed_eval = keyed(
        load_jsonl(
            out / "vsr_pathrag_v62_grazing_intensity_case_eval.jsonl"
        )
    )
    if set(base) != set(range(30)) or set(changed) != {25}:
        raise ValueError("v62 selective inputs are inconsistent")
    rows = [changed.get(i, base[i]) for i in range(30)]
    evaluations = [
        changed_eval.get(i, base_eval[i]) for i in range(30)
    ]
    write_jsonl(out / "vsr_pathrag_v62_selective_replay_30.jsonl", rows)
    write_jsonl(
        out / "vsr_pathrag_v62_selective_replay_30_eval.jsonl",
        evaluations,
    )
    means = metric_means(evaluations)
    base_means = metric_means(base_eval)
    selective_manifest = {
        "protocol": "deterministic_gate_selective_replay",
        "agent_version": "vsr-pathrag-v62",
        "sample_count": 30,
        "audited_trigger_count": 1,
        "actual_v62_rerun_count": 1,
        "reused_v61_nontrigger_count": 29,
        "triggered_line_nos": [25],
        "means": means,
        "v61_means": base_means,
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
        out / "vsr_pathrag_v62_selective_replay_manifest.json"
    ).write_text(
        json.dumps(selective_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    base_pairwise = keyed(
        load_jsonl(
            out / "vsr_pathrag_v61_vs_hyde_pairwise_replay.jsonl"
        )
    )
    changed_pairwise = keyed(
        load_jsonl(
            out / "vsr_pathrag_v62_vs_hyde_changed_pairwise.jsonl"
        )
    )
    if set(base_pairwise) != set(range(30)) or set(
        changed_pairwise
    ) != {25}:
        raise ValueError("v62 pairwise inputs are inconsistent")
    pairwise_rows = [
        changed_pairwise.get(i, base_pairwise[i]) for i in range(30)
    ]
    write_jsonl(
        out / "vsr_pathrag_v62_vs_hyde_pairwise_replay.jsonl",
        pairwise_rows,
    )
    counts = {}
    rates = {}
    for dimension in DIMENSIONS:
        counter = Counter(
            row["winners"][dimension] for row in pairwise_rows
        )
        counts[dimension] = {
            "pathrag": counter["pathrag"],
            "baseline": counter["baseline"],
            "tie": counter["tie"],
        }
        rates[dimension] = round(
            (counter["pathrag"] + counter["tie"] * 0.5)
            / len(pairwise_rows)
            * 100,
            2,
        )
    pairwise_manifest = {
        "protocol": "changed_answer_pairwise_replay",
        "agent_version": "vsr-pathrag-v62",
        "sample_count": 30,
        "rejudged_line_nos": [25],
        "reused_v61_pairwise_count": 29,
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
        out / "vsr_pathrag_v62_vs_hyde_pairwise_replay_manifest.json"
    ).write_text(
        json.dumps(pairwise_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"overall={means['overall']:.3f} "
        f"pairwise={pairwise_manifest['average_pathrag_win_rate']:.2f}%"
    )


if __name__ == "__main__":
    main()
