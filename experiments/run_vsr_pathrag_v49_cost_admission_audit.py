import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    assess_counterevidence_cost_admission,
    normalize_counterevidence_search_response,
    run_active_counterevidence_challenge,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
)


CLAIM = "The program caused higher sales."


def assess(response, retrieval_failed=False) -> dict:
    normalized = normalize_counterevidence_search_response(
        response,
        12,
    )
    admission = assess_counterevidence_cost_admission(
        normalized,
        retrieval_failed=retrieval_failed,
        require_complete_usage_contract=True,
    )
    return {
        "normalized": normalized,
        "admission": admission,
    }


async def execute(response) -> dict:
    async def search_func(_query, _max_results):
        if isinstance(response, Exception):
            raise response
        return response

    _answer, challenge = await run_active_counterevidence_challenge(
        CLAIM,
        build_transaction(
            [CLAIM],
            [{"risk_level": "high", "risk_score": 0.9}],
        ),
        search_func,
        max_claims=1,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_observed_cost_control=True,
        enable_information_posterior_control=True,
        enable_family_normalized_novelty=True,
        enable_failure_aware_information_admission=True,
        enable_two_stage_negative_memory=True,
        enable_correlated_route_confirmation=True,
        enable_bounded_provisional_memory=True,
        require_complete_retrieval_usage_contract=True,
        enable_failure_aware_cost_admission=True,
        historical_information_priors=[],
        max_global_queries=1,
    )
    return challenge


async def main_async(_args) -> dict:
    complete_response = {
        "rows": [],
        "latency_ms": 10,
        "retrieval_adapter": "local_scan",
        "usage_status": "complete_local_retrieval",
        "usage": {"call_count": 0, "total_tokens": 0},
    }
    incomplete_response = {
        **complete_response,
        "usage": {"call_count": 0},
    }
    partial_response = {
        **complete_response,
        "usage_status": "partial",
    }
    cases = []
    helper_fixtures = [
        (
            "complete_usage_is_admitted",
            complete_response,
            False,
            True,
            "complete_cost_observation",
        ),
        (
            "missing_usage_is_rejected",
            incomplete_response,
            False,
            False,
            "usage_contract_incomplete",
        ),
        (
            "partial_usage_status_is_rejected",
            partial_response,
            False,
            False,
            "usage_status_incomplete",
        ),
        (
            "retrieval_exception_is_rejected",
            complete_response,
            True,
            False,
            "retrieval_exception",
        ),
    ]
    for name, response, failed, admitted, reason in helper_fixtures:
        observed = assess(response, retrieval_failed=failed)
        cases.append(
            {
                "case": name,
                "observed": observed,
                "expectation_met": bool(
                    observed["admission"]["admitted"] == admitted
                    and observed["admission"]["reason"] == reason
                ),
            }
        )

    complete = await execute(complete_response)
    cases.append(
        {
            "case": "complete_observation_updates_cost_memory",
            "observed": {
                "admitted": complete["cost_memory_admitted_count"],
                "rejected": complete["cost_memory_rejected_count"],
                "profile_count": len(complete["observed_cost_profiles"]),
                "row_count": len(complete["observed_cost_rows"]),
                "update_count": len(
                    complete["counterevidence_cost_memory_updates"]
                ),
            },
            "expectation_met": bool(
                complete["cost_memory_admitted_count"] == 1
                and complete["cost_memory_rejected_count"] == 0
                and len(complete["observed_cost_profiles"]) == 1
                and len(complete["observed_cost_rows"]) == 1
                and len(
                    complete["counterevidence_cost_memory_updates"]
                )
                == 1
            ),
        }
    )

    incomplete = await execute(incomplete_response)
    cases.append(
        {
            "case": "incomplete_observation_cannot_update_cost_memory",
            "observed": {
                "admitted": incomplete["cost_memory_admitted_count"],
                "rejected": incomplete["cost_memory_rejected_count"],
                "profiles": incomplete["observed_cost_profiles"],
                "rows": incomplete["observed_cost_rows"],
                "updates": incomplete[
                    "counterevidence_cost_memory_updates"
                ],
            },
            "expectation_met": bool(
                incomplete["cost_memory_admitted_count"] == 0
                and incomplete["cost_memory_rejected_count"] == 1
                and not incomplete["observed_cost_profiles"]
                and not incomplete["observed_cost_rows"]
                and not incomplete[
                    "counterevidence_cost_memory_updates"
                ]
            ),
        }
    )

    evidence_response = {
        **incomplete_response,
        "rows": [
            {
                "source_id": "counterevidence-1",
                "document_id": "independent-document-1",
                "text": (
                    "An independent audit found no measurable sales "
                    "increase caused by the program."
                ),
                "source_reliability": 0.95,
            }
        ],
    }
    evidence = await execute(evidence_response)
    cases.append(
        {
            "case": "cost_rejection_does_not_discard_current_evidence",
            "observed": {
                "fetched_evidence_count": evidence[
                    "fetched_evidence_count"
                ],
                "cost_rejected_count": evidence[
                    "cost_memory_rejected_count"
                ],
                "cost_update_count": len(
                    evidence["counterevidence_cost_memory_updates"]
                ),
            },
            "expectation_met": bool(
                evidence["fetched_evidence_count"] == 1
                and evidence["cost_memory_rejected_count"] == 1
                and not evidence[
                    "counterevidence_cost_memory_updates"
                ]
            ),
        }
    )

    return {
        "agent_version": "vsr-pathrag-v49",
        "protocol": "failure_aware_cost_memory_admission",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(case["expectation_met"]) for case in cases
        ),
        "all_expectations_met": all(
            case["expectation_met"] for case in cases
        ),
        "complete_admitted_count": complete[
            "cost_memory_admitted_count"
        ],
        "incomplete_rejected_count": incomplete[
            "cost_memory_rejected_count"
        ],
        "evidence_preserved_count": evidence[
            "fetched_evidence_count"
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
        default="results/historical_vsr/vsr_pathrag_v49_cost_admission_audit.json",
    )
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "cases={case_count} met={expectation_met_count} "
        "complete_admitted={complete_admitted_count} "
        "incomplete_rejected={incomplete_rejected_count} "
        "evidence_preserved={evidence_preserved_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
