import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.counterevidence_information_memory import (
    append_counterevidence_information_updates,
    load_counterevidence_information_memory,
)


CONTEXT = {
    "task_type": "claim",
    "claim_count": 1,
    "term_hashes": ["hash-bounded"],
}


def negative(family: str, route: str) -> dict:
    return {
        "context": CONTEXT,
        "sample_count": 1,
        "mean_realized_information_value": 0.1,
        "information_m2": 0.0,
        "low_value_streak": 1,
        "negative_only": True,
        "query_family_hashes": [family],
        "retrieval_route_hashes": [route],
        "require_route_diversity": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "results/historical_vsr/vsr_pathrag_v47_bounded_provisional_audit.json"
        ),
    )
    args = parser.parse_args()
    cases = []

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fresh.jsonl"
        append_counterevidence_information_updates(
            [negative("family-a", "route-a")],
            path,
            current_time=1000,
        )
        feedback = append_counterevidence_information_updates(
            [negative("family-b", "route-b")],
            path,
            current_time=1000 + 5 * 86400,
        )
        row = load_counterevidence_information_memory(path)[0]
    cases.append(
        {
            "case": "fresh_diverse_feedback_confirms",
            "observed": {"feedback": feedback, "row": row},
            "expectation_met": bool(
                row["sample_count"] == 2
                and feedback["confirmed_negative_update_count"] == 1
            ),
        }
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "expired.jsonl"
        append_counterevidence_information_updates(
            [negative("family-a", "route-a")],
            path,
            current_time=1000,
        )
        feedback = append_counterevidence_information_updates(
            [negative("family-b", "route-b")],
            path,
            current_time=1000 + 31 * 86400,
        )
        row = load_counterevidence_information_memory(path)[0]
    cases.append(
        {
            "case": "expired_feedback_cannot_confirm",
            "observed": {"feedback": feedback, "row": row},
            "expectation_met": bool(
                feedback["expired_provisional_count"] == 1
                and row["sample_count"] == 0
                and row["provisional_sample_count"] == 1
            ),
        }
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cap.jsonl"
        capped_events = 0
        for index in range(7):
            feedback = append_counterevidence_information_updates(
                [negative("family-a", "route-a")],
                path,
                current_time=1000 + index,
                max_provisional_samples=3,
            )
            capped_events += feedback[
                "capped_provisional_update_count"
            ]
        row = load_counterevidence_information_memory(path)[0]
    cases.append(
        {
            "case": "provisional_samples_are_capped",
            "observed": {
                "capped_events": capped_events,
                "row": row,
            },
            "expectation_met": bool(
                row["provisional_sample_count"] == 3
                and capped_events >= 1
            ),
        }
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "positive.jsonl"
        append_counterevidence_information_updates(
            [negative("family-a", "route-a")],
            path,
            current_time=1000,
        )
        feedback = append_counterevidence_information_updates(
            [
                {
                    **negative("family-b", "route-b"),
                    "mean_realized_information_value": 0.8,
                    "low_value_streak": 0,
                    "negative_only": False,
                }
            ],
            path,
            current_time=1001,
        )
        row = load_counterevidence_information_memory(path)[0]
    cases.append(
        {
            "case": "positive_feedback_clears_provisional",
            "observed": {"feedback": feedback, "row": row},
            "expectation_met": bool(
                row["sample_count"] == 1
                and row["provisional_sample_count"] == 0
                and feedback["immediate_positive_update_count"] == 1
            ),
        }
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "boundary.jsonl"
        append_counterevidence_information_updates(
            [negative("family-a", "route-a")],
            path,
            current_time=1000,
            provisional_ttl_days=30,
        )
        feedback = append_counterevidence_information_updates(
            [negative("family-b", "route-b")],
            path,
            current_time=1000 + 30 * 86400,
            provisional_ttl_days=30,
        )
        row = load_counterevidence_information_memory(path)[0]
    cases.append(
        {
            "case": "ttl_boundary_remains_confirmable",
            "observed": {"feedback": feedback, "row": row},
            "expectation_met": bool(
                feedback["expired_provisional_count"] == 0
                and row["sample_count"] == 2
            ),
        }
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "configured.jsonl"
        feedback = append_counterevidence_information_updates(
            [negative("family-a", "route-a")],
            path,
            current_time=1000,
            provisional_ttl_days=14,
            max_provisional_samples=5,
        )
    cases.append(
        {
            "case": "configured_bounds_are_auditable",
            "observed": feedback,
            "expectation_met": bool(
                feedback["provisional_ttl_days"] == 14
                and feedback["max_provisional_samples"] == 5
            ),
        }
    )

    result = {
        "agent_version": "vsr-pathrag-v47",
        "protocol": "bounded_provisional_information_memory",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(case["expectation_met"]) for case in cases
        ),
        "all_expectations_met": all(
            case["expectation_met"] for case in cases
        ),
        "default_ttl_days": 30,
        "default_max_provisional_samples": 8,
        "reference_answer_used": False,
        "generator_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "cases={case_count} met={expectation_met_count} "
        "ttl={default_ttl_days} cap={default_max_provisional_samples} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
