import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    assess_counterevidence_information_admission,
    normalize_counterevidence_search_response,
    run_active_counterevidence_challenge,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
)


def assess(response) -> dict:
    normalized = normalize_counterevidence_search_response(
        response,
        12,
    )
    admission = assess_counterevidence_information_admission(
        normalized,
        require_complete_usage_contract=True,
    )
    return {
        "normalized": normalized,
        "admission": admission,
    }


async def main_async(_args) -> dict:
    complete_response = {
        "rows": [],
        "latency_ms": 10,
        "retrieval_adapter": "local_scan",
        "usage_status": "complete_local_retrieval",
        "usage": {"call_count": 0, "total_tokens": 0},
    }
    fixtures = [
        (
            "complete_contract_is_admitted",
            complete_response,
            True,
            [],
        ),
        (
            "missing_latency_is_rejected",
            {
                key: value
                for key, value in complete_response.items()
                if key != "latency_ms"
            },
            False,
            ["latency_ms"],
        ),
        (
            "missing_adapter_is_rejected",
            {
                key: value
                for key, value in complete_response.items()
                if key != "retrieval_adapter"
            },
            False,
            ["retrieval_adapter"],
        ),
        (
            "missing_usage_status_is_rejected",
            {
                key: value
                for key, value in complete_response.items()
                if key != "usage_status"
            },
            False,
            ["usage_status"],
        ),
        (
            "missing_token_metadata_is_rejected",
            {
                **complete_response,
                "usage": {"call_count": 0},
            },
            False,
            ["usage.total_tokens"],
        ),
    ]
    cases = []
    for name, response, admitted, missing_fields in fixtures:
        observed = assess(response)
        actual_missing = observed["normalized"][
            "usage_contract_missing_fields"
        ]
        cases.append(
            {
                "case": name,
                "observed": observed,
                "expectation_met": bool(
                    observed["admission"]["admitted"] == admitted
                    and all(
                        field in actual_missing
                        for field in missing_fields
                    )
                ),
            }
        )

    claim = "The program caused higher sales."

    async def production_search(_query, _max_results):
        return complete_response

    _answer, challenge = await run_active_counterevidence_challenge(
        claim,
        build_transaction(
            [claim],
            [{"risk_level": "high", "risk_score": 0.9}],
        ),
        production_search,
        max_claims=1,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_information_posterior_control=True,
        enable_family_normalized_novelty=True,
        enable_failure_aware_information_admission=True,
        enable_two_stage_negative_memory=True,
        enable_correlated_route_confirmation=True,
        enable_bounded_provisional_memory=True,
        require_complete_retrieval_usage_contract=True,
        historical_information_priors=[],
        max_global_queries=1,
    )
    audit = challenge["information_memory_admission_audit"][0]
    scheduling_case = {
        "case": "production_shaped_callback_passes_contract",
        "observed": {
            "audit": audit,
            "admitted_count": challenge[
                "information_memory_admitted_count"
            ],
            "rejected_count": challenge[
                "information_memory_rejected_count"
            ],
            "updates": challenge[
                "counterevidence_information_memory_updates"
            ],
        },
        "expectation_met": bool(
            audit["usage_contract_complete"]
            and challenge["information_memory_admitted_count"] == 1
            and challenge["information_memory_rejected_count"] == 0
            and len(
                challenge[
                    "counterevidence_information_memory_updates"
                ]
            )
            == 1
        ),
        "challenge": challenge,
    }
    cases.append(scheduling_case)
    result = {
        "agent_version": "vsr-pathrag-v48",
        "protocol": "complete_retrieval_usage_contract",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(case["expectation_met"]) for case in cases
        ),
        "all_expectations_met": all(
            case["expectation_met"] for case in cases
        ),
        "complete_admitted_count": challenge[
            "information_memory_admitted_count"
        ],
        "complete_rejected_count": challenge[
            "information_memory_rejected_count"
        ],
        "required_contract_fields": [
            "latency_ms",
            "retrieval_adapter",
            "usage_status",
            "usage.call_count",
            "usage.total_tokens",
        ],
        "reference_answer_used": False,
        "generator_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/historical_vsr/vsr_pathrag_v48_usage_contract_audit.json",
    )
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "cases={case_count} met={expectation_met_count} "
        "admitted={complete_admitted_count} "
        "rejected={complete_rejected_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
