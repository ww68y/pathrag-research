import json
from collections import Counter
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.experience_memory import (
    build_experience_episode,
    recall_experiences,
    route_family,
    select_candidate_with_memory,
    write_experience_memory,
)


TRAIN_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v3_crossdomain_suite_12.jsonl"
HELDOUT_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v3_heldout_suite_12.jsonl"
MEMORY_PATH = ROOT_DIR / "data_repro" / "vsr_agent_experience_memory.jsonl"
OUT_JSON = ROOT_DIR / "results" / "vsr_pathrag_v4_memory_train_eval.json"
OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v4_memory_train_eval_summary.md"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def selected_candidate(trace: dict) -> dict | None:
    family = route_family(trace.get("final_policy") or trace.get("selected_profile"))
    candidates = trace.get("candidates") or []
    for candidate in candidates:
        if route_family(candidate.get("label") or candidate.get("profile")) == family:
            return candidate
    return candidates[0] if candidates else None


def synthetic_memory_benchmark() -> dict:
    portfolio_context = {
        "question_type": "process_or_explanation",
        "contract_type": "explanation",
        "relation_intents": ["mechanism", "cause_effect"],
        "target_type": "ANSWER_ENTITY_OR_VALUE",
        "requires_multi_hop": True,
        "gap_bucket": "medium",
        "budget_tier": "medium",
        "conflict_present": False,
        "unresolved_conflict": False,
        "repair_available": False,
        "portfolio_risk": "low",
        "repair_dependency_bucket": "low",
    }
    guard_context = {
        "question_type": "enumeration",
        "contract_type": "structured_coverage",
        "relation_intents": ["enumeration", "role"],
        "target_type": "ANSWER_ENTITY_OR_VALUE",
        "requires_multi_hop": False,
        "gap_bucket": "low",
        "budget_tier": "low",
        "conflict_present": False,
        "unresolved_conflict": False,
        "repair_available": True,
        "portfolio_risk": "medium",
        "repair_dependency_bucket": "medium",
    }
    episodes = []
    for index in range(10):
        episodes.append(
            {
                "episode_id": f"synthetic-portfolio-{index}",
                "context": portfolio_context,
                "outcome": {
                    "route_family": "portfolio",
                    "success": True,
                    "utility": 0.94,
                },
                "candidate_reflections": [
                    {
                        "route_family": "overlay",
                        "failure_tags": ["low_facet_coverage"],
                    }
                ],
            }
        )
        episodes.append(
            {
                "episode_id": f"synthetic-guard-{index}",
                "context": guard_context,
                "outcome": {
                    "route_family": "full_guard",
                    "success": True,
                    "utility": 0.96,
                },
                "candidate_reflections": [
                    {
                        "route_family": "overlay",
                        "failure_tags": ["unsupported_claims", "low_facet_coverage"],
                    }
                ],
            }
        )
    checks = []
    for index in range(10):
        portfolio_recall = recall_experiences(portfolio_context, episodes=episodes)
        checks.append(
            {
                "case_id": f"portfolio-memory-{index}",
                "expected": "portfolio",
                "actual": portfolio_recall.get("recommended_route"),
                "passed": portfolio_recall.get("recommended_route") == "portfolio",
            }
        )
        guard_recall = recall_experiences(guard_context, episodes=episodes)
        checks.append(
            {
                "case_id": f"guard-memory-{index}",
                "expected": "full_guard",
                "actual": guard_recall.get("recommended_route"),
                "passed": guard_recall.get("recommended_route") == "full_guard",
            }
        )
    return {
        "episode_count": len(episodes),
        "checks": checks,
        "passed": sum(row["passed"] for row in checks),
        "total": len(checks),
    }


def main() -> None:
    train_rows = [row for row in load_jsonl(TRAIN_JSONL) if row.get("ok")]
    heldout_rows = [row for row in load_jsonl(HELDOUT_JSONL) if row.get("ok")]
    train_episodes = [build_experience_episode(row["trace"]) for row in train_rows]
    write_experience_memory(train_episodes, MEMORY_PATH)

    evaluation = []
    recommendation_counts = Counter()
    safety_gate_counts = Counter()
    for index, row in enumerate(heldout_rows):
        trace = row["trace"]
        episode = build_experience_episode(trace)
        context = episode["context"]
        recall = recall_experiences(context, episodes=train_episodes)
        deterministic = selected_candidate(trace)
        if deterministic:
            selected, decision = select_candidate_with_memory(
                trace.get("candidates") or [],
                deterministic,
                recall,
            )
            selected_family = route_family(selected.get("label") or selected.get("profile"))
        else:
            decision = {
                "override": False,
                "safety_gate": "no_candidate_available",
                "reason": "trace contains no candidate",
            }
            selected_family = None
        recommendation_counts[str(recall.get("recommended_route") or "none")] += 1
        safety_gate_counts[str(decision.get("safety_gate", "unknown"))] += 1
        evaluation.append(
            {
                "index": index,
                "question_hash": episode["question_hash"],
                "memory_retrieved": recall.get("retrieved_count", 0),
                "recommended_route": recall.get("recommended_route"),
                "memory_confidence": recall.get("confidence", 0),
                "deterministic_route": route_family(
                    deterministic.get("label") or deterministic.get("profile")
                ) if deterministic else None,
                "selected_route": selected_family,
                "decision": decision,
            }
        )

    synthetic = synthetic_memory_benchmark()
    metrics = {
        "train_episodes": len(train_episodes),
        "heldout_samples": len(evaluation),
        "heldout_with_neighbors": sum(row["memory_retrieved"] > 0 for row in evaluation),
        "heldout_with_recommendation": sum(bool(row["recommended_route"]) for row in evaluation),
        "heldout_safe_overrides": sum(bool(row["decision"].get("override")) for row in evaluation),
        "heldout_safety_rejections": sum(
            str(row["decision"].get("reason", "")).startswith("memory_override_rejected")
            for row in evaluation
        ),
        "recommendation_counts": dict(recommendation_counts),
        "safety_gate_counts": dict(safety_gate_counts),
        "synthetic_memory_passed": synthetic["passed"],
        "synthetic_memory_total": synthetic["total"],
        "raw_question_stored": any("question" in episode for episode in train_episodes),
    }
    payload = {
        "method": "safe_episodic_failure_memory_router_v1",
        "memory_path": str(MEMORY_PATH),
        "metrics": metrics,
        "heldout_evaluation": evaluation,
        "synthetic_benchmark": synthetic,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# VSR-PathRAG v4 Experience Memory Train/Eval",
        "",
        f"- Development episodes used for memory: `{metrics['train_episodes']}`",
        f"- Held-out samples: `{metrics['heldout_samples']}`",
        f"- Held-out samples with recalled neighbors: `{metrics['heldout_with_neighbors']}`",
        f"- Held-out samples with a route recommendation: `{metrics['heldout_with_recommendation']}`",
        f"- Safe memory overrides: `{metrics['heldout_safe_overrides']}`",
        f"- Unsafe/degrading overrides rejected: `{metrics['heldout_safety_rejections']}`",
        f"- Recommendation counts: `{metrics['recommendation_counts']}`",
        f"- Safety gate counts: `{metrics['safety_gate_counts']}`",
        f"- Controlled memory routing: `{metrics['synthetic_memory_passed']}/{metrics['synthetic_memory_total']}`",
        f"- Raw question stored: `{metrics['raw_question_stored']}`",
        "",
    ]
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {MEMORY_PATH}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_SUMMARY}")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
