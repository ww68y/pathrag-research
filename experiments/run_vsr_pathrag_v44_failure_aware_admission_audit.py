import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    assess_counterevidence_information_admission,
    run_active_counterevidence_challenge,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
)


CLAIM = "The program caused higher sales."


async def execute(search_func) -> dict:
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
        enable_information_posterior_control=True,
        enable_family_normalized_novelty=True,
        enable_online_budget_reallocation=True,
        enable_failure_aware_information_admission=True,
        historical_information_priors=[],
        max_global_queries=1,
    )
    return challenge


async def main_async(_args) -> dict:
    direct_inputs = [
        (
            "complete_family_grounded_result_is_admitted",
            {
                "rows": [
                    {
                        "source_id": "chunk-1",
                        "document_id": "document-1",
                        "text": "evidence",
                    }
                ],
                "metadata_source": "structured_response",
                "usage_status": "reported",
            },
            True,
            "complete_family_grounded_observation",
        ),
        (
            "partial_usage_is_rejected",
            {
                "rows": [],
                "metadata_source": "structured_response",
                "usage_status": "partial",
            },
            False,
            "usage_status_incomplete",
        ),
        (
            "legacy_callback_is_rejected",
            {
                "rows": [],
                "metadata_source": "measured_legacy_response",
                "usage_status": "legacy_partial",
            },
            False,
            "legacy_usage_incomplete",
        ),
        (
            "missing_family_lineage_is_rejected",
            {
                "rows": [{"source_id": "chunk-1", "text": "evidence"}],
                "metadata_source": "structured_response",
                "usage_status": "reported",
            },
            False,
            "family_lineage_incomplete",
        ),
    ]
    cases = []
    for name, response, admitted, reason in direct_inputs:
        observed = assess_counterevidence_information_admission(
            response
        )
        cases.append(
            {
                "case": name,
                "observed": observed,
                "expectation_met": bool(
                    observed["admitted"] == admitted
                    and observed["reason"] == reason
                ),
            }
        )

    async def failing_search(_query, _max_results):
        raise RuntimeError("temporary retrieval outage")

    failed = await execute(failing_search)
    failure_case = {
        "case": "retrieval_exception_cannot_poison_memory",
        "observed": {
            "rejected": failed["information_memory_rejected_count"],
            "updates": failed[
                "counterevidence_information_memory_updates"
            ],
            "budget_action": failed[
                "online_budget_adjustment_audit"
            ][0]["action"],
            "reason": failed[
                "information_memory_admission_audit"
            ][0]["reason"],
        },
        "expectation_met": bool(
            failed["information_memory_rejected_count"] == 1
            and not failed[
                "counterevidence_information_memory_updates"
            ]
            and failed["online_budget_adjustment_audit"][0][
                "action"
            ]
            == "hold"
        ),
        "challenge": failed,
    }

    async def complete_empty_search(_query, _max_results):
        return {
            "rows": [],
            "latency_ms": 10,
            "usage": {"call_count": 0, "total_tokens": 0},
            "usage_status": "complete_local_retrieval",
        }

    complete = await execute(complete_empty_search)
    complete_case = {
        "case": "complete_zero_yield_remains_learnable",
        "observed": {
            "admitted": complete[
                "information_memory_admitted_count"
            ],
            "updates": len(
                complete[
                    "counterevidence_information_memory_updates"
                ]
            ),
            "reason": complete[
                "information_memory_admission_audit"
            ][0]["reason"],
        },
        "expectation_met": bool(
            complete["information_memory_admitted_count"] == 1
            and len(
                complete[
                    "counterevidence_information_memory_updates"
                ]
            )
            == 1
        ),
        "challenge": complete,
    }
    cases.extend([failure_case, complete_case])
    return {
        "agent_version": "vsr-pathrag-v44",
        "protocol": "failure_aware_information_memory_admission",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "failure_rejected_count": failed[
            "information_memory_rejected_count"
        ],
        "failure_memory_update_count": len(
            failed["counterevidence_information_memory_updates"]
        ),
        "complete_empty_admitted_count": complete[
            "information_memory_admitted_count"
        ],
        "complete_empty_memory_update_count": len(
            complete["counterevidence_information_memory_updates"]
        ),
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
            "results/historical_vsr/vsr_pathrag_v44_failure_aware_admission_audit.json"
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
        "failure_rejected={failure_rejected_count} "
        "failure_updates={failure_memory_update_count} "
        "empty_admitted={complete_empty_admitted_count} "
        "empty_updates={complete_empty_memory_update_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
