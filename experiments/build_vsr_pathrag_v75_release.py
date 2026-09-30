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
    base = keyed(load_jsonl(out / "vsr_pathrag_v74_selective_30.jsonl"))
    changed = keyed(
        load_jsonl(out / "vsr_pathrag_v75_sociolexical_case.jsonl")
    )
    base_eval = keyed(
        load_jsonl(out / "vsr_pathrag_v74_selective_30_eval.jsonl")
    )
    changed_eval = keyed(
        load_jsonl(out / "vsr_pathrag_v75_sociolexical_case_eval.jsonl")
    )
    base_pairwise = keyed(
        load_jsonl(out / "vsr_pathrag_v74_vs_hyde_pairwise_30.jsonl")
    )
    changed_pairwise = keyed(
        load_jsonl(out / "vsr_pathrag_v75_vs_hyde_pairwise.jsonl")
    )
    if set(base) != set(range(30)) or set(changed) != {23}:
        raise ValueError("v75 answer inputs are inconsistent")
    if set(base_eval) != set(range(30)) or set(changed_eval) != {23}:
        raise ValueError("v75 evaluation inputs are inconsistent")
    if set(base_pairwise) != set(range(30)) or set(
        changed_pairwise
    ) != {23}:
        raise ValueError("v75 pairwise inputs are inconsistent")

    real = changed[23]
    if real.get("ok") is not True:
        raise ValueError("v75 changed case is not a successful real run")
    trace = real.get("trace") or {}
    if trace.get("version") != "vsr-pathrag-v75":
        raise ValueError("v75 changed case has wrong version")
    freeze = trace.get("innovation_freeze", {})
    if freeze.get("ablation_started") or freeze.get("frozen"):
        raise ValueError("v75 unexpectedly entered ablation or freeze")
    statuses = trace.get("claim_evidence_graph", {}).get(
        "status_counts",
        {},
    )
    if (
        int(statuses.get("contradicted", 0) or 0)
        or int(statuses.get("unknown", 0) or 0)
    ):
        raise ValueError("v75 changed case failed signed audit")

    rows = [changed.get(i, base[i]) for i in range(30)]
    evaluations = [
        changed_eval.get(i, base_eval[i]) for i in range(30)
    ]
    pairwise_rows = [
        changed_pairwise.get(i, base_pairwise[i]) for i in range(30)
    ]
    write_jsonl(out / "vsr_pathrag_v75_selective_30.jsonl", rows)
    write_jsonl(
        out / "vsr_pathrag_v75_selective_30_eval.jsonl",
        evaluations,
    )
    write_jsonl(
        out / "vsr_pathrag_v75_vs_hyde_pairwise_30.jsonl",
        pairwise_rows,
    )

    means = metric_means(evaluations)
    base_means = metric_means(base_eval)
    manifest = {
        "protocol": "v74_real_lineage_plus_real_v75_selective_promotion",
        "agent_version": "vsr-pathrag-v75",
        "sample_count": 30,
        "real_v75_rerun_line_nos": [23],
        "reused_v74_lineage_count": 29,
        "deterministic_output_count": 0,
        "means": means,
        "v74_means": base_means,
        "mean_deltas": {
            metric: round(means[metric] - base_means[metric], 3)
            for metric in METRICS
        },
        "verdict_counts": dict(
            Counter(row["scores"]["verdict"] for row in evaluations)
        ),
        "poor_count": sum(
            row["scores"]["verdict"] == "poor" for row in evaluations
        ),
        "runtime_reference_answer_used": False,
        "full_real_output_lineage": True,
        "provisional": False,
        "innovation_frozen": False,
        "ablation": False,
    }
    (
        out / "vsr_pathrag_v75_selective_manifest.json"
    ).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
        "protocol": "v74_pairwise_plus_rejudged_real_v75_line23",
        "agent_version": "vsr-pathrag-v75",
        "sample_count": 30,
        "rejudged_line_nos": [23],
        "counts": counts,
        "win_rates": rates,
        "average_pathrag_win_rate": round(
            statistics.mean(rates.values()),
            2,
        ),
        "runtime_reference_answer_used": False,
        "judge_reference_used": True,
        "provisional": False,
        "innovation_frozen": False,
        "ablation": False,
    }
    (
        out / "vsr_pathrag_v75_vs_hyde_pairwise_manifest.json"
    ).write_text(
        json.dumps(pairwise_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"overall={means['overall']:.3f} "
        f"pairwise={pairwise_manifest['average_pathrag_win_rate']:.2f}% "
        f"poor={manifest['poor_count']} ablation=false"
    )


if __name__ == "__main__":
    main()
