import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    reallocate_compute_from_observed_value,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_policy(
    trace: dict,
    *,
    requested_fast: bool,
    elapsed_seconds: float,
) -> dict:
    budget = copy.deepcopy(trace.get("compute_budget") or {})
    budget["requested_fast"] = requested_fast
    return reallocate_compute_from_observed_value(
        budget,
        trace.get("active_evidence_acquisition") or {},
        trace.get("knowledge_gap") or {},
        trace.get("evidence_portfolio") or {},
        elapsed_seconds,
    )


def projected_suppression(trace: dict, policy: dict) -> dict:
    candidate_labels = {
        candidate.get("label")
        for candidate in trace.get("candidates", [])
    }
    generated_repairs = sum(
        bool(trace.get(key))
        for key in (
            "repair_triggered",
            "claim_prune_triggered",
            "facet_completion_triggered",
        )
    )
    realization_attempts = int(
        (trace.get("answer_realization") or {}).get("attempt_count", 0) or 0
    )
    return {
        "alternate_candidate_records": int(
            not policy["allow_alternate_candidate"]
            and "alternate" in candidate_labels
        ),
        "generation_repair_attempts": (
            generated_repairs
            if not policy["allow_generation_repairs"]
            else 0
        ),
        "answer_realization_attempts": (
            realization_attempts
            if not policy["allow_answer_realization"]
            else 0
        ),
    }


def closed_gates(policy: dict) -> dict:
    return {
        "alternate_candidate_gate": int(
            not policy["allow_alternate_candidate"]
        ),
        "generation_repair_gate": int(
            not policy["allow_generation_repairs"]
        ),
        "answer_realization_gate": int(
            not policy["allow_answer_realization"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="results/historical_vsr/vsr_pathrag_v10_facet_retention_scale_30_final.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "vsr_pathrag_v14_compute_reallocation_scale_30_audit.jsonl"
        ),
    )
    args = parser.parse_args()

    predictions = load_jsonl(ROOT_DIR / args.predictions)
    rows = []
    for prediction in predictions:
        trace = prediction.get("trace") or {}
        recorded_elapsed = float(trace.get("runtime_seconds", 0.0) or 0.0)
        recorded_policy = replay_policy(
            trace,
            requested_fast=True,
            elapsed_seconds=recorded_elapsed,
        )
        standard_policy = replay_policy(
            trace,
            requested_fast=False,
            elapsed_seconds=recorded_elapsed,
        )
        stress_elapsed = (
            0.80 * standard_policy["soft_time_budget_seconds"]
        )
        latency_stress_policy = replay_policy(
            trace,
            requested_fast=False,
            elapsed_seconds=stress_elapsed,
        )
        rows.append(
            {
                "line_no": int(prediction["line_no"]),
                "question": prediction["question"],
                "answer_held_fixed": True,
                "recorded_runtime_seconds": round(recorded_elapsed, 4),
                "recorded_fast_policy": recorded_policy,
                "standard_budget_policy": standard_policy,
                "latency_stress_policy": latency_stress_policy,
                "projected_suppression": {
                    "recorded_fast": projected_suppression(
                        trace,
                        recorded_policy,
                    ),
                    "standard_budget": projected_suppression(
                        trace,
                        standard_policy,
                    ),
                    "latency_stress": projected_suppression(
                        trace,
                        latency_stress_policy,
                    ),
                },
                "closed_gates": {
                    "recorded_fast": closed_gates(recorded_policy),
                    "standard_budget": closed_gates(standard_policy),
                    "latency_stress": closed_gates(
                        latency_stress_policy
                    ),
                },
            }
        )

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    for label, key in (
        ("Recorded fast", "recorded_fast_policy"),
        ("Standard budget", "standard_budget_policy"),
        ("80% latency stress", "latency_stress_policy"),
    ):
        actions = Counter(row[key]["action"] for row in rows)
        print(f"{label}: {dict(actions)}")
    for scenario in (
        "recorded_fast",
        "standard_budget",
        "latency_stress",
    ):
        totals = Counter()
        for row in rows:
            totals.update(row["projected_suppression"][scenario])
        print(f"{scenario} projected suppression: {dict(totals)}")
        gate_totals = Counter()
        for row in rows:
            gate_totals.update(row["closed_gates"][scenario])
        print(f"{scenario} closed gates: {dict(gate_totals)}")
    print(f"Audited rows: {len(rows)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
