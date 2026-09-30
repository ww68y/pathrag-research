import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    reallocate_after_claim_verification,
    reallocate_compute_from_observed_value,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay(
    trace: dict,
    *,
    requested_fast: bool,
    latency_stress: bool,
) -> tuple[dict, dict]:
    budget = copy.deepcopy(trace.get("compute_budget") or {})
    budget["requested_fast"] = requested_fast
    elapsed = float(trace.get("runtime_seconds", 0.0) or 0.0)
    initial = reallocate_compute_from_observed_value(
        budget,
        trace.get("active_evidence_acquisition") or {},
        trace.get("knowledge_gap") or {},
        trace.get("evidence_portfolio") or {},
        elapsed,
    )
    if latency_stress:
        elapsed = 0.80 * initial["soft_time_budget_seconds"]
        initial = reallocate_compute_from_observed_value(
            budget,
            trace.get("active_evidence_acquisition") or {},
            trace.get("knowledge_gap") or {},
            trace.get("evidence_portfolio") or {},
            elapsed,
        )
    gap = trace.get("knowledge_gap") or {}
    pre_gap = float(
        gap.get(
            "pre_claim_verification_gap_score",
            gap.get("gap_score", 0.0),
        )
        or 0.0
    )
    post = reallocate_after_claim_verification(
        initial,
        budget,
        trace.get("claim_active_verification") or {},
        gap,
        trace.get("evidence_portfolio") or {},
        elapsed,
        pre_verification_gap_score=pre_gap,
    )
    return initial, post


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
            "vsr_pathrag_v15_post_verification_scale_30_audit.jsonl"
        ),
    )
    args = parser.parse_args()

    rows = []
    for prediction in load_jsonl(ROOT_DIR / args.predictions):
        trace = prediction.get("trace") or {}
        recorded_initial, recorded_post = replay(
            trace,
            requested_fast=True,
            latency_stress=False,
        )
        standard_initial, standard_post = replay(
            trace,
            requested_fast=False,
            latency_stress=False,
        )
        stress_initial, stress_post = replay(
            trace,
            requested_fast=False,
            latency_stress=True,
        )
        rows.append(
            {
                "line_no": int(prediction["line_no"]),
                "question": prediction["question"],
                "answer_held_fixed": True,
                "claim_active_verification": (
                    trace.get("claim_active_verification") or {}
                ),
                "recorded_fast": {
                    "initial": recorded_initial,
                    "post_verification": recorded_post,
                },
                "standard_budget": {
                    "initial": standard_initial,
                    "post_verification": standard_post,
                },
                "latency_stress": {
                    "initial": stress_initial,
                    "post_verification": stress_post,
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

    for scenario in (
        "recorded_fast",
        "standard_budget",
        "latency_stress",
    ):
        actions = Counter(
            row[scenario]["post_verification"]["action"]
            for row in rows
        )
        deltas = Counter(
            row[scenario]["post_verification"]["decision_delta"]
            for row in rows
        )
        transitions = Counter(
            (
                row[scenario]["initial"]["action"],
                row[scenario]["post_verification"]["action"],
            )
            for row in rows
        )
        print(f"{scenario} actions: {dict(actions)}")
        print(f"{scenario} decision deltas: {dict(deltas)}")
        print(f"{scenario} transitions: {dict(transitions)}")

    standard_rows = [
        row["standard_budget"]["post_verification"]
        for row in rows
    ]
    print(
        "standard verified value: "
        f"mean={sum(row['verified_value_score'] for row in standard_rows) / max(1, len(standard_rows)):.4f}, "
        f"positive={sum(row['verified_value_score'] > 0 for row in standard_rows)}"
    )
    print(
        "standard residual need: "
        f"mean={sum(row['residual_verification_need'] for row in standard_rows) / max(1, len(standard_rows)):.4f}, "
        f"high={sum(row['residual_verification_need'] >= 0.45 for row in standard_rows)}"
    )
    print(f"Audited rows: {len(rows)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
