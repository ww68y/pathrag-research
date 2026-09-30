import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    counterevidence_independence_family_key,
    run_active_counterevidence_challenge,
)
from experiments.run_vsr_pathrag_v36_cost_memory_audit import (
    build_transaction,
)


CLAIM = "The program caused higher sales."


async def execute_rows(
    batches: list[list[dict]],
    max_queries: int,
) -> dict:
    transaction = build_transaction(
        [CLAIM],
        [{"risk_level": "high", "risk_score": 0.9}],
    )
    call_count = 0

    async def local_search(_query, _max_results):
        nonlocal call_count
        rows = batches[min(call_count, len(batches) - 1)]
        call_count += 1
        return {
            "rows": rows,
            "latency_ms": 10,
            "usage": {"call_count": 0, "total_tokens": 0},
        }

    _answer, challenge = await run_active_counterevidence_challenge(
        CLAIM,
        transaction,
        local_search,
        max_claims=1,
        enable_value_control=True,
        enable_joint_portfolio=True,
        enable_observed_cost_control=True,
        enable_risk_coverage_guard=True,
        enable_adaptive_query_budget=True,
        enable_online_budget_reallocation=True,
        enable_information_posterior_control=True,
        enable_family_normalized_novelty=True,
        max_queries_per_claim=3,
        max_global_queries=max_queries,
    )
    return challenge


async def main_async(_args) -> dict:
    same_document_rows = [
        {
            "source_id": "chunk-1",
            "document_id": "document-1",
            "text": "A weather archive recorded rainfall.",
        },
        {
            "source_id": "chunk-2",
            "document_id": "document-1",
            "text": "A weather archive recorded snowfall.",
        },
    ]
    independent_rows = [
        same_document_rows[0],
        {
            "source_id": "chunk-3",
            "document_id": "document-2",
            "text": "A climate archive recorded snowfall.",
        },
    ]
    same_key = (
        counterevidence_independence_family_key(
            same_document_rows[0],
            "chunk-1",
        )
        == counterevidence_independence_family_key(
            same_document_rows[1],
            "chunk-2",
        )
    )
    key_case = {
        "case": "same_document_chunks_share_family_key",
        "observed": {"same_key": same_key},
        "expectation_met": same_key,
    }

    root_first = counterevidence_independence_family_key(
        {
            "source_id": "chunk-a",
            "document_id": "section-a",
            "root_document_id": "root-1",
        },
        "chunk-a",
    )
    root_second = counterevidence_independence_family_key(
        {
            "source_id": "chunk-b",
            "document_id": "section-b",
            "root_document_id": "root-1",
        },
        "chunk-b",
    )
    root_case = {
        "case": "root_document_precedes_section_identity",
        "observed": {
            "first": root_first,
            "second": root_second,
        },
        "expectation_met": bool(
            root_first == root_second == "ancestor:root-1"
        ),
    }

    duplicate = await execute_rows(
        [same_document_rows],
        1,
    )
    duplicate_observed = duplicate["observed_cost_rows"][0]
    duplicate_case = {
        "case": "duplicate_chunks_count_as_one_new_family",
        "observed": duplicate_observed,
        "expectation_met": bool(
            duplicate_observed["new_unique_count"] == 2
            and duplicate_observed[
                "new_independent_family_count"
            ]
            == 1
            and duplicate_observed["family_unique_yield"] == 0.5
        ),
    }

    independent = await execute_rows(
        [independent_rows],
        1,
    )
    independent_observed = independent[
        "observed_cost_rows"
    ][0]
    independent_case = {
        "case": "independent_documents_keep_full_novelty",
        "observed": independent_observed,
        "expectation_met": bool(
            independent_observed[
                "new_independent_family_count"
            ]
            == 2
            and independent_observed["family_unique_yield"] == 1.0
        ),
    }

    repeated = await execute_rows(
        [
            [
                {
                    **same_document_rows[0],
                    "text": CLAIM,
                    "source_reliability": 0.5,
                }
            ],
            [
                {
                    **same_document_rows[1],
                    "source_id": "chunk-2-round-2",
                    "text": CLAIM,
                    "source_reliability": 0.5,
                }
            ],
        ],
        2,
    )
    repeated_rows = repeated["observed_cost_rows"]
    cross_round_case = {
        "case": "known_family_earns_no_second_round_novelty",
        "observed": repeated_rows,
        "expectation_met": bool(
            len(repeated_rows) == 2
            and repeated_rows[0][
                "new_independent_family_count"
            ]
            == 1
            and repeated_rows[1][
                "new_independent_family_count"
            ]
            == 0
            and repeated_rows[1]["family_unique_yield"] == 0.0
        ),
    }

    fallback_first = counterevidence_independence_family_key(
        {"source_id": "source-1"},
        "source-1",
    )
    fallback_second = counterevidence_independence_family_key(
        {"source_id": "source-2"},
        "source-2",
    )
    fallback_case = {
        "case": "unlinked_sources_remain_distinct_families",
        "observed": {
            "first": fallback_first,
            "second": fallback_second,
        },
        "expectation_met": fallback_first != fallback_second,
    }

    cases = [
        key_case,
        root_case,
        duplicate_case,
        independent_case,
        cross_round_case,
        fallback_case,
    ]
    return {
        "agent_version": "vsr-pathrag-v42",
        "protocol": "provenance_family_normalized_novelty",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(row["expectation_met"]) for row in cases
        ),
        "all_expectations_met": all(
            row["expectation_met"] for row in cases
        ),
        "duplicate_source_count": duplicate_observed[
            "new_unique_count"
        ],
        "duplicate_family_count": duplicate_observed[
            "new_independent_family_count"
        ],
        "independent_family_count": independent_observed[
            "new_independent_family_count"
        ],
        "second_round_family_count": repeated_rows[1][
            "new_independent_family_count"
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
        default=(
            "results/historical_vsr/vsr_pathrag_v42_family_novelty_audit.json"
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
        "duplicate_sources={duplicate_source_count} "
        "duplicate_families={duplicate_family_count} "
        "independent_families={independent_family_count} "
        "second_round_families={second_round_family_count} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
