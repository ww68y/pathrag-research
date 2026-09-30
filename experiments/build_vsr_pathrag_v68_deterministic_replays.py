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
        load_jsonl(out / "vsr_pathrag_v66_selective_replay_30.jsonl")
    )
    changed = keyed(
        load_jsonl(
            out
            / "vsr_pathrag_v68_darkness_role_deterministic_case.jsonl"
        )
    )
    base_eval = keyed(
        load_jsonl(
            out / "vsr_pathrag_v66_selective_replay_30_eval.jsonl"
        )
    )
    changed_eval = keyed(
        load_jsonl(
            out
            / "vsr_pathrag_v68_darkness_role_deterministic_case_eval.jsonl"
        )
    )
    if set(base) != set(range(30)) or set(changed) != {17}:
        raise ValueError("v68 selective inputs are inconsistent")
    rows = [changed.get(i, base[i]) for i in range(30)]
    evaluations = [
        changed_eval.get(i, base_eval[i]) for i in range(30)
    ]
    write_jsonl(out / "vsr_pathrag_v68_selective_replay_30.jsonl", rows)
    write_jsonl(
        out / "vsr_pathrag_v68_selective_replay_30_eval.jsonl",
        evaluations,
    )
    means = metric_means(evaluations)
    base_means = metric_means(base_eval)
    manifest = {
        "protocol": "frozen_v66_deterministic_postrepair_replay",
        "agent_version": "vsr-pathrag-v68",
        "sample_count": 30,
        "audited_trigger_count": 1,
        "deterministic_v68_postrepair_count": 1,
        "actual_v68_agent_rerun_count": 0,
        "actual_v68_agent_rerun_error": (
            "TimeoutError at 600 seconds"
        ),
        "reused_v66_nontrigger_count": 29,
        "triggered_line_nos": [17],
        "means": means,
        "v66_means": base_means,
        "mean_deltas": {
            metric: round(means[metric] - base_means[metric], 3)
            for metric in METRICS
        },
        "verdict_counts": dict(
            Counter(row["scores"]["verdict"] for row in evaluations)
        ),
        "runtime_reference_answer_used": False,
        "full_agent_rerun_completed": False,
        "full_stochastic_rerun": False,
        "provisional": True,
        "ablation": False,
    }
    (
        out / "vsr_pathrag_v68_selective_replay_manifest.json"
    ).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    base_pairwise = keyed(
        load_jsonl(
            out / "vsr_pathrag_v66_vs_hyde_pairwise_replay.jsonl"
        )
    )
    changed_pairwise = keyed(
        load_jsonl(
            out / "vsr_pathrag_v68_vs_hyde_deterministic_pairwise.jsonl"
        )
    )
    if set(base_pairwise) != set(range(30)) or set(
        changed_pairwise
    ) != {17}:
        raise ValueError("v68 pairwise inputs are inconsistent")
    pairwise_rows = [
        changed_pairwise.get(i, base_pairwise[i]) for i in range(30)
    ]
    write_jsonl(
        out / "vsr_pathrag_v68_vs_hyde_pairwise_replay.jsonl",
        pairwise_rows,
    )
    counts = {}
    rates = {}
    for dimension in DIMENSIONS:
        counter = Counter(
            row["winners"][dimension] for row in pairwise_rows
        )
        counts[dimension] = {
            key: counter[key] for key in ("pathrag", "baseline", "tie")
        }
        rates[dimension] = round(
            (counter["pathrag"] + counter["tie"] * 0.5)
            / len(pairwise_rows)
            * 100,
            2,
        )
    pairwise_manifest = {
        "protocol": "frozen_v66_deterministic_postrepair_pairwise_replay",
        "agent_version": "vsr-pathrag-v68",
        "sample_count": 30,
        "rejudged_line_nos": [17],
        "reused_v66_pairwise_count": 29,
        "counts": counts,
        "win_rates": rates,
        "average_pathrag_win_rate": round(
            statistics.mean(rates.values()), 2
        ),
        "runtime_reference_answer_used": False,
        "judge_reference_used": True,
        "full_agent_rerun_completed": False,
        "provisional": True,
        "ablation": False,
    }
    (
        out / "vsr_pathrag_v68_vs_hyde_pairwise_replay_manifest.json"
    ).write_text(
        json.dumps(pairwise_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"overall={means['overall']:.3f} "
        f"pairwise={pairwise_manifest['average_pathrag_win_rate']:.2f}% "
        "provisional=true"
    )


if __name__ == "__main__":
    main()
