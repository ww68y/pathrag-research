import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    build_counterevidence_cost_context,
    build_robust_counterevidence_information_prior,
    run_active_counterevidence_challenge,
)
from api.features.chats.counterevidence_information_memory import (
    append_counterevidence_information_updates,
    load_counterevidence_information_memory,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
)


CLAIMS = [
    "The campaign always drew wider attendance.",
    "The program caused 20 percent higher revenue.",
]
RISKS = [
    {"risk_level": "medium", "risk_score": 0.0},
    {"risk_level": "high", "risk_score": 0.9},
]


async def execute_prior(prior: dict) -> dict:
    queries = []

    async def empty_search(query, _max_results):
        queries.append(query)
        return []

    _answer, challenge = await run_active_counterevidence_challenge(
        " ".join(CLAIMS),
        build_transaction(CLAIMS, RISKS),
        empty_search,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_information_posterior_control=True,
        enable_family_normalized_novelty=True,
        historical_information_priors=[prior],
        max_global_queries=1,
    )
    return {"queries": queries, "challenge": challenge}


async def main_async(_args) -> dict:
    matching_context = build_counterevidence_cost_context(
        "claim",
        [CLAIMS[1]],
    )
    matching_prior = {
        "context": matching_context,
        "sample_count": 20,
        "mean_realized_information_value": 0.05,
        "information_m2": 0.0,
        "low_value_streak": 8,
        "updated_at": int(time.time()),
    }
    matched = await execute_prior(matching_prior)
    matched_challenge = matched["challenge"]
    matched_case = {
        "case": "similar_low_value_prior_reprioritizes_route",
        "observed": {
            "query": matched["queries"][0],
            "applications": matched_challenge[
                "historical_information_prior_application_count"
            ],
            "reprioritizations": matched_challenge[
                "information_reprioritization_count"
            ],
        },
        "expectation_met": bool(
            "campaign" in matched["queries"][0].lower()
            and matched_challenge[
                "historical_information_prior_application_count"
            ]
            == 1
            and matched_challenge[
                "information_reprioritization_count"
            ]
            == 1
        ),
        "challenge": matched_challenge,
    }

    unrelated_prior = {
        **matching_prior,
        "context": build_counterevidence_cost_context(
            "claim",
            ["The orchard harvest increased apple exports."],
        ),
    }
    unrelated = await execute_prior(unrelated_prior)
    unrelated_case = {
        "case": "unrelated_prior_is_similarity_rejected",
        "observed": {
            "query": unrelated["queries"][0],
            "applications": unrelated["challenge"][
                "historical_information_prior_application_count"
            ],
        },
        "expectation_met": bool(
            "20 percent" in unrelated["queries"][0]
            and unrelated["challenge"][
                "historical_information_prior_application_count"
            ]
            == 0
        ),
        "challenge": unrelated["challenge"],
    }

    stale = build_robust_counterevidence_information_prior(
        {
            **matching_prior,
            "updated_at": int(time.time()) - 365 * 86400,
        }
    )
    stale_case = {
        "case": "stale_prior_is_rejected_by_effective_samples",
        "observed": {"robust_prior": stale},
        "expectation_met": stale is None,
    }

    uncertain = build_robust_counterevidence_information_prior(
        {
            **matching_prior,
            "sample_count": 10,
            "mean_realized_information_value": 0.3,
            "information_m2": 0.9,
            "low_value_streak": 3,
        }
    )
    uncertainty_case = {
        "case": "uncertain_prior_uses_optimistic_upper_bound",
        "observed": uncertain,
        "expectation_met": bool(
            uncertain["information_upper_bound"] > 0.3
            and uncertain["effective_sample_count"] > 9.9
        ),
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "information.jsonl"
        append_counterevidence_information_updates(
            [
                {
                    "context": matching_context,
                    "sample_count": 2,
                    "mean_realized_information_value": 0.2,
                    "information_m2": 0.02,
                    "low_value_streak": 2,
                }
            ],
            path,
        )
        feedback = append_counterevidence_information_updates(
            [
                {
                    "context": matching_context,
                    "sample_count": 1,
                    "mean_realized_information_value": 0.8,
                    "information_m2": 0.0,
                    "low_value_streak": 0,
                }
            ],
            path,
        )
        persisted = load_counterevidence_information_memory(path)[0]
    persistence_case = {
        "case": "persistent_memory_merges_and_resets_streak",
        "observed": {
            "feedback": feedback,
            "row": persisted,
        },
        "expectation_met": bool(
            persisted["sample_count"] == 3
            and persisted[
                "mean_realized_information_value"
            ]
            == 0.4
            and persisted["low_value_streak"] == 0
        ),
    }

    updates = matched_challenge[
        "counterevidence_information_memory_updates"
    ]
    privacy_case = {
        "case": "feedback_contains_hashed_context_not_claim_text",
        "observed": updates,
        "expectation_met": bool(
            len(updates) == 1
            and updates[0]["context"]["term_hashes"]
            and "20 percent"
            not in json.dumps(updates).lower()
            and "campaign" not in json.dumps(updates).lower()
        ),
    }

    cases = [
        matched_case,
        unrelated_case,
        stale_case,
        uncertainty_case,
        persistence_case,
        privacy_case,
    ]
    return {
        "agent_version": "vsr-pathrag-v43",
        "protocol": "persistent_information_posterior_memory",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "similar_prior_applications": matched_challenge[
            "historical_information_prior_application_count"
        ],
        "unrelated_prior_applications": unrelated["challenge"][
            "historical_information_prior_application_count"
        ],
        "historical_reprioritization_count": matched_challenge[
            "information_reprioritization_count"
        ],
        "persisted_sample_count": persisted["sample_count"],
        "memory_update_count": len(updates),
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
            "results/historical_vsr/vsr_pathrag_v43_information_memory_audit.json"
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
        "similar={similar_prior_applications} "
        "unrelated={unrelated_prior_applications} "
        "reprioritized={historical_reprioritization_count} "
        "samples={persisted_sample_count} "
        "updates={memory_update_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
