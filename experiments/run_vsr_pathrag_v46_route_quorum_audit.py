import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    build_counterevidence_retrieval_route_hash,
    run_active_counterevidence_challenge,
)
from api.features.chats.counterevidence_information_memory import (
    append_counterevidence_information_updates,
    load_counterevidence_information_memory,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
)


CONTEXT = {
    "task_type": "claim",
    "claim_count": 1,
    "term_hashes": ["hash-route"],
}


def negative(family: str, route: str) -> dict:
    return {
        "context": CONTEXT,
        "sample_count": 1,
        "mean_realized_information_value": 0.1,
        "low_value_streak": 1,
        "negative_only": True,
        "query_family_hashes": [family],
        "retrieval_route_hashes": [route],
        "require_route_diversity": True,
    }


async def main_async(_args) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "route-quorum.jsonl"
        append_counterevidence_information_updates(
            [negative("family-a", "route-x")],
            path,
        )
        same_route_feedback = (
            append_counterevidence_information_updates(
                [negative("family-b", "route-x")],
                path,
            )
        )
        same_route_row = load_counterevidence_information_memory(
            path
        )[0]
        independent_feedback = (
            append_counterevidence_information_updates(
                [negative("family-c", "route-y")],
                path,
            )
        )
        confirmed_row = load_counterevidence_information_memory(
            path
        )[0]

    same_route_case = {
        "case": "different_queries_same_route_remain_provisional",
        "observed": {
            "feedback": same_route_feedback,
            "row": same_route_row,
        },
        "expectation_met": bool(
            same_route_row["sample_count"] == 0
            and len(
                same_route_row[
                    "provisional_query_family_hashes"
                ]
            )
            == 2
            and len(
                same_route_row[
                    "provisional_retrieval_route_hashes"
                ]
            )
            == 1
        ),
    }
    confirmation_case = {
        "case": "second_route_confirms_negative_quorum",
        "observed": {
            "feedback": independent_feedback,
            "row": confirmed_row,
        },
        "expectation_met": bool(
            confirmed_row["sample_count"] == 3
            and independent_feedback[
                "confirmed_negative_update_count"
            ]
            == 1
        ),
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "same-family.jsonl"
        append_counterevidence_information_updates(
            [negative("family-a", "route-x")],
            path,
        )
        append_counterevidence_information_updates(
            [negative("family-a", "route-y")],
            path,
        )
        same_family_row = load_counterevidence_information_memory(
            path
        )[0]
    same_family_case = {
        "case": "different_routes_same_query_family_remain_provisional",
        "observed": same_family_row,
        "expectation_met": bool(
            same_family_row["sample_count"] == 0
            and len(
                same_family_row[
                    "provisional_retrieval_route_hashes"
                ]
            )
            == 2
            and len(
                same_family_row[
                    "provisional_query_family_hashes"
                ]
            )
            == 1
        ),
    }

    route_hash = build_counterevidence_retrieval_route_hash(
        "local_scan",
        "route-a",
    )
    privacy_case = {
        "case": "route_identity_is_hashed",
        "observed": {"route_hash": route_hash},
        "expectation_met": bool(
            len(route_hash) == 16
            and "route" not in route_hash
            and "local" not in route_hash
        ),
    }

    claim = "The program caused higher sales."

    async def empty_search(_query, _max_results):
        return {
            "rows": [],
            "retrieval_adapter": "adapter-a",
            "retrieval_route_id": "route-a",
            "usage_status": "complete_local_retrieval",
            "usage": {"call_count": 0, "total_tokens": 0},
        }

    _answer, challenge = await run_active_counterevidence_challenge(
        claim,
        build_transaction(
            [claim],
            [{"risk_level": "high", "risk_score": 0.9}],
        ),
        empty_search,
        max_claims=1,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_information_posterior_control=True,
        enable_family_normalized_novelty=True,
        enable_failure_aware_information_admission=True,
        enable_two_stage_negative_memory=True,
        enable_correlated_route_confirmation=True,
        historical_information_priors=[],
        max_global_queries=1,
    )
    emitted = challenge[
        "counterevidence_information_memory_updates"
    ][0]
    scheduler_case = {
        "case": "scheduler_emits_query_and_route_quorum_hashes",
        "observed": emitted,
        "expectation_met": bool(
            emitted["require_route_diversity"]
            and len(emitted["query_family_hashes"]) == 1
            and len(emitted["retrieval_route_hashes"]) == 1
            and len(emitted["retrieval_route_hashes"][0]) == 16
        ),
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scheduler.jsonl"
        feedback = append_counterevidence_information_updates(
            [emitted],
            path,
        )
        scheduler_row = load_counterevidence_information_memory(
            path
        )[0]
    scheduler_persistence_case = {
        "case": "single_scheduler_route_stays_provisional",
        "observed": {
            "feedback": feedback,
            "row": scheduler_row,
        },
        "expectation_met": bool(
            scheduler_row["sample_count"] == 0
            and scheduler_row["provisional_sample_count"] == 1
        ),
    }

    cases = [
        same_route_case,
        confirmation_case,
        same_family_case,
        privacy_case,
        scheduler_case,
        scheduler_persistence_case,
    ]
    return {
        "agent_version": "vsr-pathrag-v46",
        "protocol": "query_and_retrieval_route_negative_quorum",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "same_route_committed_samples": same_route_row[
            "sample_count"
        ],
        "same_family_committed_samples": same_family_row[
            "sample_count"
        ],
        "independent_quorum_committed_samples": confirmed_row[
            "sample_count"
        ],
        "scheduler_provisional_samples": scheduler_row[
            "provisional_sample_count"
        ],
        "reference_answer_used": False,
        "generator_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/historical_vsr/vsr_pathrag_v46_route_quorum_audit.json",
    )
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "cases={case_count} met={expectation_met_count} "
        "same_route={same_route_committed_samples} "
        "same_family={same_family_committed_samples} "
        "independent={independent_quorum_committed_samples} "
        "scheduler_provisional={scheduler_provisional_samples} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
