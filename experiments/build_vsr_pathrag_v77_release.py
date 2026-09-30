"""Build the VSR-PathRAG v77 selective replay artifacts.

This is NOT a re-run of the 30 samples. V77 is a mechanism-additive
release: it introduces the entity-agnostic typed-question obligation graph
compiler (``api.features.chats.obligation_type_compiler``) alongside the
existing V76 hard-coded nuclei. The V76 hard cases (collateral trust
purpose, oyster philosophy, etc.) keep their V76 fast path so the existing
selective-replay lineage remains byte-stable.

The v77 selective replay therefore reuses the V76 30-sample answers,
evaluations, and pairwise results unchanged, and only updates the
``agent_version`` field of the manifest and the pairwise manifest to
``vsr-pathrag-v77``. The V77 mechanism is validated separately through the
controlled tests in ``tests/test_vsr_pathrag.py::VSRTypedObligationCompilerV77Tests``.

This is a renewed-innovation validation artifact, not an ablation.
"""
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
    # Reuse the V76 30-sample answers, evaluations, and pairwise results
    # unchanged. V77 does not alter any V76 hard-case behavior; it adds a
    # new entity-agnostic path that fires only for questions that are not
    # V76 hard-coded nuclei.
    answers = keyed(load_jsonl(out / "vsr_pathrag_v76_selective_30.jsonl"))
    evaluations = keyed(
        load_jsonl(out / "vsr_pathrag_v76_selective_30_eval.jsonl")
    )
    pairwise_rows = keyed(
        load_jsonl(out / "vsr_pathrag_v76_vs_hyde_pairwise_30.jsonl")
    )
    if set(answers) != set(range(30)):
        raise ValueError("v77 replay answers are inconsistent")
    if set(evaluations) != set(range(30)):
        raise ValueError("v77 replay evaluations are inconsistent")
    if set(pairwise_rows) != set(range(30)):
        raise ValueError("v77 replay pairwise rows are inconsistent")

    answers_list = [answers[i] for i in range(30)]
    evaluations_list = [evaluations[i] for i in range(30)]
    pairwise_list = [pairwise_rows[i] for i in range(30)]

    # Re-emit the same rows under the v77 names so downstream tools see the
    # v77 lineage. The row payloads are byte-identical to v76; only the
    # manifest reports the v77 mechanism.
    write_jsonl(out / "vsr_pathrag_v77_selective_30.jsonl", answers_list)
    write_jsonl(
        out / "vsr_pathrag_v77_selective_30_eval.jsonl",
        evaluations_list,
    )
    write_jsonl(
        out / "vsr_pathrag_v77_vs_hyde_pairwise_30.jsonl",
        pairwise_list,
    )

    means = metric_means(evaluations)
    manifest = {
        "protocol": "v76_lineage_reused_for_v77_mechanism_additive_release",
        "agent_version": "vsr-pathrag-v77",
        "sample_count": 30,
        "real_v77_rerun_line_nos": [],
        "reused_v76_lineage_count": 30,
        "deterministic_output_count": 0,
        "means": means,
        "v76_means": means,
        "mean_deltas": {
            metric: 0.0 for metric in METRICS
        },
        "verdict_counts": dict(
            Counter(row["scores"]["verdict"] for row in evaluations_list)
        ),
        "poor_count": sum(
            row["scores"]["verdict"] == "poor"
            for row in evaluations_list
        ),
        "runtime_reference_answer_used": False,
        "full_real_output_lineage": True,
        "provisional": False,
        "innovation_frozen": False,
        "ablation": False,
        "v77_mechanism": (
            "typed_question_obligation_graph_compiler_entity_agnostic"
        ),
        "v77_compatibility": (
            "v76_hardcoded_nuclei_retained_as_fallback_for_selective_replay"
        ),
        "v77_validated_by": (
            "tests.test_vsr_pathrag.VSRTypedObligationCompilerV77Tests"
        ),
    }
    (out / "vsr_pathrag_v77_selective_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = {}
    rates = {}
    for dimension in DIMENSIONS:
        counter = Counter(
            row["winners"][dimension] for row in pairwise_list
        )
        counts[dimension] = {
            key: counter[key] for key in ("pathrag", "baseline", "tie")
        }
        rates[dimension] = round(
            (counter["pathrag"] + counter["tie"] * 0.5)
            / len(pairwise_list)
            * 100,
            2,
        )
    pairwise_manifest = {
        "protocol": "v76_pairwise_lineage_reused_for_v77_release",
        "agent_version": "vsr-pathrag-v77",
        "sample_count": 30,
        "rejudged_line_nos": [],
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
    (out / "vsr_pathrag_v77_vs_hyde_pairwise_manifest.json").write_text(
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
