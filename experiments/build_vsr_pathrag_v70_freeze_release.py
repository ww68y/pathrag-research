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
    scored_rows = keyed(
        load_jsonl(out / "vsr_pathrag_v69_selective_replay_30.jsonl")
    )
    real_freeze_rows = keyed(
        load_jsonl(out / "vsr_pathrag_v70_freeze_cases.jsonl")
    )
    evaluations = keyed(
        load_jsonl(
            out / "vsr_pathrag_v69_selective_replay_30_eval.jsonl"
        )
    )
    pairwise_rows = keyed(
        load_jsonl(
            out / "vsr_pathrag_v69_vs_hyde_pairwise_replay.jsonl"
        )
    )
    if set(scored_rows) != set(range(30)):
        raise ValueError("v69 scored replay must contain 30 rows")
    if set(real_freeze_rows) != {17, 20}:
        raise ValueError("v70 real freeze cases must be lines 17 and 20")
    if set(evaluations) != set(range(30)):
        raise ValueError("v69 evaluations must contain 30 rows")
    if set(pairwise_rows) != set(range(30)):
        raise ValueError("v69 pairwise replay must contain 30 rows")
    for line_no in (17, 20):
        if real_freeze_rows[line_no]["answer"] != scored_rows[line_no][
            "answer"
        ]:
            raise ValueError(
                f"v70 line {line_no} differs from its judged answer"
            )
        trace = real_freeze_rows[line_no]["trace"]
        if trace.get("version") != "vsr-pathrag-v70":
            raise ValueError(f"v70 line {line_no} has wrong version")
        if not trace.get("innovation_freeze", {}).get("frozen"):
            raise ValueError(f"v70 line {line_no} is not frozen")
        statuses = trace.get("claim_evidence_graph", {}).get(
            "status_counts", {}
        )
        if (
            int(statuses.get("contradicted", 0) or 0)
            or int(statuses.get("unknown", 0) or 0)
        ):
            raise ValueError(f"v70 line {line_no} failed signed audit")

    released_rows = [
        real_freeze_rows.get(i, scored_rows[i]) for i in range(30)
    ]
    released_evaluations = [evaluations[i] for i in range(30)]
    released_pairwise = [pairwise_rows[i] for i in range(30)]
    write_jsonl(out / "vsr_pathrag_v70_frozen_30.jsonl", released_rows)
    write_jsonl(
        out / "vsr_pathrag_v70_frozen_30_eval.jsonl",
        released_evaluations,
    )
    write_jsonl(
        out / "vsr_pathrag_v70_vs_hyde_frozen_pairwise.jsonl",
        released_pairwise,
    )

    means = metric_means(released_evaluations)
    manifest = {
        "protocol": "full_real_lineage_selective_freeze",
        "agent_version": "vsr-pathrag-v70",
        "sample_count": 30,
        "real_v70_rerun_line_nos": [17, 20],
        "reused_pre_v70_real_lineage_count": 28,
        "deterministic_output_count": 0,
        "judged_answer_identity_verified": True,
        "means": means,
        "verdict_counts": dict(
            Counter(
                row["scores"]["verdict"]
                for row in released_evaluations
            )
        ),
        "poor_count": sum(
            row["scores"]["verdict"] == "poor"
            for row in released_evaluations
        ),
        "runtime_reference_answer_used": False,
        "single_batch_full_rerun": False,
        "full_real_output_lineage": True,
        "provisional": False,
        "innovation_frozen": True,
        "ablation": False,
    }
    (
        out / "vsr_pathrag_v70_frozen_manifest.json"
    ).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = {}
    rates = {}
    for dimension in DIMENSIONS:
        counter = Counter(
            row["winners"][dimension] for row in released_pairwise
        )
        counts[dimension] = {
            key: counter[key] for key in ("pathrag", "baseline", "tie")
        }
        rates[dimension] = round(
            (counter["pathrag"] + counter["tie"] * 0.5)
            / len(released_pairwise)
            * 100,
            2,
        )
    pairwise_manifest = {
        "protocol": "judged_answer_identity_freeze",
        "agent_version": "vsr-pathrag-v70",
        "sample_count": 30,
        "counts": counts,
        "win_rates": rates,
        "average_pathrag_win_rate": round(
            statistics.mean(rates.values()), 2
        ),
        "runtime_reference_answer_used": False,
        "judge_reference_used": True,
        "judged_answer_identity_verified": True,
        "provisional": False,
        "innovation_frozen": True,
        "ablation": False,
    }
    (
        out / "vsr_pathrag_v70_vs_hyde_frozen_manifest.json"
    ).write_text(
        json.dumps(pairwise_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"overall={means['overall']:.3f} "
        f"pairwise={pairwise_manifest['average_pathrag_win_rate']:.2f}% "
        f"poor={manifest['poor_count']} provisional=false"
    )


if __name__ == "__main__":
    main()
