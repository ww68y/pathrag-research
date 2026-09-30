import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
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
    "term_hashes": ["hash-a"],
}


def update(value: float, family: str, negative: bool) -> dict:
    return {
        "context": CONTEXT,
        "sample_count": 1,
        "mean_realized_information_value": value,
        "information_m2": 0.0,
        "low_value_streak": int(negative),
        "negative_only": negative,
        "query_family_hashes": [family],
    }


async def execute_search(search_func) -> dict:
    claim = "The program caused higher sales."
    _answer, challenge = await run_active_counterevidence_challenge(
        claim,
        build_transaction(
            [claim],
            [{"risk_level": "high", "risk_score": 0.9}],
        ),
        search_func,
        max_claims=1,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_information_posterior_control=True,
        enable_family_normalized_novelty=True,
        enable_failure_aware_information_admission=True,
        enable_two_stage_negative_memory=True,
        historical_information_priors=[],
        max_global_queries=1,
    )
    return challenge


async def main_async(_args) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "two-stage.jsonl"
        first_feedback = append_counterevidence_information_updates(
            [update(0.1, "family-a", True)],
            path,
        )
        first_row = load_counterevidence_information_memory(path)[0]
        same_feedback = append_counterevidence_information_updates(
            [update(0.2, "family-a", True)],
            path,
        )
        same_row = load_counterevidence_information_memory(path)[0]
        confirm_feedback = (
            append_counterevidence_information_updates(
                [update(0.2, "family-b", True)],
                path,
            )
        )
        confirmed_row = load_counterevidence_information_memory(
            path
        )[0]

    first_case = {
        "case": "first_negative_family_remains_provisional",
        "observed": {"feedback": first_feedback, "row": first_row},
        "expectation_met": bool(
            first_row["sample_count"] == 0
            and first_row["provisional_sample_count"] == 1
            and first_feedback["provisional_update_count"] == 1
        ),
    }
    same_case = {
        "case": "same_family_cannot_self_confirm",
        "observed": {"feedback": same_feedback, "row": same_row},
        "expectation_met": bool(
            same_row["sample_count"] == 0
            and same_row["provisional_sample_count"] == 2
            and len(
                same_row["provisional_query_family_hashes"]
            )
            == 1
        ),
    }
    confirm_case = {
        "case": "second_family_confirms_negative_memory",
        "observed": {
            "feedback": confirm_feedback,
            "row": confirmed_row,
        },
        "expectation_met": bool(
            confirmed_row["sample_count"] == 3
            and confirmed_row["provisional_sample_count"] == 0
            and confirm_feedback[
                "confirmed_negative_update_count"
            ]
            == 1
        ),
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recovery.jsonl"
        append_counterevidence_information_updates(
            [update(0.1, "family-a", True)],
            path,
        )
        positive_feedback = (
            append_counterevidence_information_updates(
                [update(0.8, "family-b", False)],
                path,
            )
        )
        positive_row = load_counterevidence_information_memory(path)[0]
    positive_case = {
        "case": "positive_feedback_commits_immediately",
        "observed": {
            "feedback": positive_feedback,
            "row": positive_row,
        },
        "expectation_met": bool(
            positive_row["sample_count"] == 1
            and positive_row["provisional_sample_count"] == 0
            and positive_row["feedback_status"]
            == "committed_positive"
        ),
    }

    async def complete_empty(_query, _max_results):
        return {
            "rows": [],
            "usage_status": "complete_local_retrieval",
            "usage": {"call_count": 0, "total_tokens": 0},
        }

    negative_challenge = await execute_search(complete_empty)
    emitted = negative_challenge[
        "counterevidence_information_memory_updates"
    ][0]
    emission_case = {
        "case": "scheduler_emits_hashed_provisional_feedback",
        "observed": emitted,
        "expectation_met": bool(
            emitted["negative_only"]
            and len(emitted["query_family_hashes"]) == 1
            and len(emitted["query_family_hashes"][0]) == 16
            and "Evidence that" not in json.dumps(emitted)
        ),
    }

    async def positive_search(_query, _max_results):
        return {
            "rows": [
                {
                    "source_id": "positive-1",
                    "document_id": "document-1",
                    "text": "The program caused higher sales.",
                    "source_reliability": 0.5,
                }
            ],
            "usage_status": "complete_local_retrieval",
            "usage": {"call_count": 0, "total_tokens": 0},
        }

    positive_challenge = await execute_search(positive_search)
    positive_emitted = positive_challenge[
        "counterevidence_information_memory_updates"
    ][0]
    scheduler_positive_case = {
        "case": "scheduler_marks_positive_feedback_immediate",
        "observed": positive_emitted,
        "expectation_met": not positive_emitted["negative_only"],
    }

    cases = [
        first_case,
        same_case,
        confirm_case,
        positive_case,
        emission_case,
        scheduler_positive_case,
    ]
    return {
        "agent_version": "vsr-pathrag-v45",
        "protocol": "two_stage_negative_information_memory",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "first_provisional_samples": first_row[
            "provisional_sample_count"
        ],
        "same_family_provisional_samples": same_row[
            "provisional_sample_count"
        ],
        "confirmed_samples": confirmed_row["sample_count"],
        "positive_committed_samples": positive_row["sample_count"],
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
        default=(
            "results/historical_vsr/vsr_pathrag_v45_two_stage_memory_audit.json"
        ),
    )
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "cases={case_count} met={expectation_met_count} "
        "first_provisional={first_provisional_samples} "
        "same_family_provisional={same_family_provisional_samples} "
        "confirmed={confirmed_samples} "
        "positive={positive_committed_samples} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
