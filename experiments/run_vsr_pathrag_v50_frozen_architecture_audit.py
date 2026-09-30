import argparse
import inspect
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats import agent_controller, answer_realization


def load_audit(version: int) -> tuple[Path | None, dict]:
    matches = sorted(
        (ROOT_DIR / "results").glob(
            f"vsr_pathrag_v{version}_*audit.json"
        )
    )
    if len(matches) != 1:
        return None, {}
    return matches[0], json.loads(
        matches[0].read_text(encoding="utf-8")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "results/historical_vsr/vsr_pathrag_v50_frozen_architecture_audit.json"
        ),
    )
    args = parser.parse_args()

    audits = {
        version: load_audit(version)
        for version in range(30, 50)
    }
    controller_source = inspect.getsource(agent_controller)
    realization_source = inspect.getsource(answer_realization)
    freeze = agent_controller.V50_INNOVATION_FREEZE
    cases = []

    cases.append(
        {
            "case": "pre_ablation_freeze_is_explicit",
            "observed": {
                "agent_version": "vsr-pathrag-v50",
                "current_agent_version": agent_controller.AGENT_VERSION,
                "freeze": freeze,
            },
            "expectation_met": bool(
                freeze["frozen"]
                and freeze["stage"] == "frozen_pre_ablation"
                and freeze["last_mechanism_version"]
                == "vsr-pathrag-v49"
                and not freeze["ablation_started"]
            ),
        }
    )

    missing_versions = [
        version
        for version, (path, _audit) in audits.items()
        if path is None
    ]
    failed_versions = [
        version
        for version, (_path, audit) in audits.items()
        if not audit.get("all_expectations_met")
    ]
    cases.append(
        {
            "case": "v30_v49_audit_chain_is_complete_and_passing",
            "observed": {
                "audit_count": len(audits) - len(missing_versions),
                "missing_versions": missing_versions,
                "failed_versions": failed_versions,
                "total_historical_case_count": sum(
                    int(audit.get("case_count", 0))
                    for _path, audit in audits.values()
                ),
            },
            "expectation_met": bool(
                not missing_versions and not failed_versions
            ),
        }
    )

    ablation_versions = [
        version
        for version, (_path, audit) in audits.items()
        if audit.get("ablation") is not False
    ]
    cases.append(
        {
            "case": "historical_innovation_audits_are_non_ablation",
            "observed": {
                "ablation_versions": ablation_versions,
            },
            "expectation_met": not ablation_versions,
        }
    )

    required_flags = [
        "enable_counterevidence_risk_coverage_guard=True",
        "enable_adaptive_counterevidence_query_budget=True",
        "enable_online_counterevidence_budget_reallocation=True",
        "enable_counterevidence_information_posterior_control=True",
        "enable_counterevidence_family_normalized_novelty=True",
        (
            "enable_failure_aware_counterevidence_"
            "information_admission=True"
        ),
        "enable_two_stage_counterevidence_negative_memory=True",
        (
            "enable_counterevidence_correlated_"
            "route_confirmation=True"
        ),
        "enable_bounded_counterevidence_provisional_memory=True",
        "require_complete_counterevidence_retrieval_usage=True",
        (
            "enable_failure_aware_counterevidence_"
            "cost_admission=True"
        ),
    ]
    missing_flags = [
        flag for flag in required_flags if flag not in controller_source
    ]
    cases.append(
        {
            "case": "production_wiring_enables_frozen_mechanisms",
            "observed": {
                "required_flag_count": len(required_flags),
                "missing_flags": missing_flags,
            },
            "expectation_met": not missing_flags,
        }
    )

    challenge_controller = "cost_admission_counterevidence_v20"
    transaction_controller = (
        "cost_admission_transactional_rollback_v27"
    )
    cases.append(
        {
            "case": "final_controller_lineage_is_frozen",
            "observed": {
                "challenge_controller": challenge_controller,
                "transaction_controller": transaction_controller,
            },
            "expectation_met": bool(
                challenge_controller in realization_source
                and transaction_controller in realization_source
                and freeze["challenge_controller"]
                == challenge_controller
                and freeze["transaction_controller"]
                == transaction_controller
            ),
        }
    )

    v43 = audits[43][1]
    v46 = audits[46][1]
    cases.append(
        {
            "case": "persistent_information_memory_is_private_and_diverse",
            "observed": {
                "privacy_audit_passed": any(
                    case.get("case")
                    == "feedback_contains_hashed_context_not_claim_text"
                    and case.get("expectation_met")
                    for case in v43.get("cases", [])
                ),
                "same_route_committed_samples": v46.get(
                    "same_route_committed_samples"
                ),
                "same_family_committed_samples": v46.get(
                    "same_family_committed_samples"
                ),
                "independent_quorum_committed_samples": v46.get(
                    "independent_quorum_committed_samples"
                ),
            },
            "expectation_met": bool(
                v43.get("all_expectations_met")
                and v46.get("same_route_committed_samples") == 0
                and v46.get("same_family_committed_samples") == 0
                and v46.get(
                    "independent_quorum_committed_samples"
                )
                == 3
            ),
        }
    )

    v47 = audits[47][1]
    v48 = audits[48][1]
    cases.append(
        {
            "case": "provisional_memory_and_usage_contract_are_bounded",
            "observed": {
                "ttl_days": v47.get("default_ttl_days"),
                "max_provisional_samples": v47.get(
                    "default_max_provisional_samples"
                ),
                "usage_contract_fields": v48.get(
                    "required_contract_fields"
                ),
                "complete_admitted_count": v48.get(
                    "complete_admitted_count"
                ),
            },
            "expectation_met": bool(
                v47.get("default_ttl_days") == 30
                and v47.get("default_max_provisional_samples") == 8
                and len(v48.get("required_contract_fields", [])) == 5
                and v48.get("complete_admitted_count") == 1
            ),
        }
    )

    v49 = audits[49][1]
    cases.append(
        {
            "case": "failed_cost_observations_cannot_pollute_memory",
            "observed": {
                "complete_admitted_count": v49.get(
                    "complete_admitted_count"
                ),
                "incomplete_rejected_count": v49.get(
                    "incomplete_rejected_count"
                ),
                "evidence_preserved_count": v49.get(
                    "evidence_preserved_count"
                ),
            },
            "expectation_met": bool(
                v49.get("complete_admitted_count") == 1
                and v49.get("incomplete_rejected_count") == 1
                and v49.get("evidence_preserved_count") == 1
            ),
        }
    )

    result = {
        "agent_version": "vsr-pathrag-v50",
        "protocol": "frozen_pre_ablation_architecture",
        "case_count": len(cases),
        "expectation_met_count": sum(
            bool(case["expectation_met"]) for case in cases
        ),
        "all_expectations_met": all(
            case["expectation_met"] for case in cases
        ),
        "historical_audit_count": len(audits) - len(missing_versions),
        "historical_case_count": sum(
            int(audit.get("case_count", 0))
            for _path, audit in audits.values()
        ),
        "innovation_architecture_frozen": True,
        "ablation": False,
        "cases": cases,
    }
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "cases={case_count} met={expectation_met_count} "
        "historical_audits={historical_audit_count} "
        "historical_cases={historical_case_count} "
        "frozen={innovation_architecture_frozen} "
        "all={all_expectations_met}".format(**result)
    )


if __name__ == "__main__":
    main()
