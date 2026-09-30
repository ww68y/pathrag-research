import tempfile
import unittest
from pathlib import Path

from api.features.chats.experience_memory import (
    build_experience_episode,
    context_similarity,
    load_experience_memory,
    recall_experiences,
    select_candidate_with_memory,
    write_experience_memory,
)


def candidate(label: str, facet: float, support: float = 1.0, unsupported: int = 0):
    return {
        "label": label,
        "profile": label,
        "score": 0.75,
        "state": {
            "facet_coverage_ratio": facet,
            "question_term_coverage": 0.75,
            "refusal_like": False,
            "long_answer": False,
        },
        "flags": [],
        "claim_verification": {
            "support_ratio": support,
            "unsupported_count": unsupported,
        },
    }


class ExperienceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.context = {
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

    def test_context_similarity_prefers_matching_structure(self):
        unrelated = dict(self.context)
        unrelated.update(
            {
                "question_type": "precision_lookup",
                "contract_type": "precision_lookup",
                "relation_intents": ["quantity"],
                "budget_tier": "low",
                "requires_multi_hop": False,
            }
        )
        self.assertEqual(context_similarity(self.context, self.context), 1.0)
        self.assertLess(context_similarity(self.context, unrelated), 0.70)

    def test_episode_anonymizes_question_and_records_failure_reflection(self):
        trace = {
            "query_graph": {
                "question": "What confidential mechanism does Aurora use?",
                "relation_intents": ["mechanism"],
                "target_type": "ANSWER_ENTITY_OR_VALUE",
                "requires_multi_hop": True,
            },
            "routing_context": self.context,
            "question_type": "process_or_explanation",
            "answer_contract": {"type": "explanation"},
            "final_policy": "use_risk_aware_portfolio_rescue",
            "claim_verification": {"support_ratio": 1.0, "unsupported_count": 0},
            "facet_coverage": {"coverage_ratio": 0.9},
            "evidence_audit": {"risk_level": "low"},
            "runtime_seconds": 0.4,
            "candidates": [candidate("fast_extractive_rescue", 0.4)],
            "verification_outcome_credit": {
                "strategy_credit": {
                    "claim_core": {
                        "query_count": 2,
                        "mean_delayed_credit": 0.8,
                        "retained_query_count": 2,
                        "revoked_query_count": 0,
                    }
                }
            },
        }
        episode = build_experience_episode(trace)
        serialized = str(episode)
        self.assertNotIn("confidential mechanism", serialized.lower())
        self.assertEqual(len(episode["question_hash"]), 16)
        self.assertIn(
            "low_facet_coverage",
            episode["candidate_reflections"][0]["failure_tags"],
        )
        self.assertEqual(
            episode["verification_strategy_credit"]["claim_core"][
                "mean_delayed_credit"
            ],
            0.8,
        )

    def test_memory_round_trip_and_successful_route_recall(self):
        episode = {
            "episode_id": "episode-test",
            "context": self.context,
            "outcome": {
                "route_family": "portfolio",
                "success": True,
                "utility": 0.95,
            },
            "candidate_reflections": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            write_experience_memory([episode], path)
            loaded = load_experience_memory(path)
        self.assertEqual(len(loaded), 1)
        recall = recall_experiences(self.context, episodes=loaded)
        self.assertEqual(recall["recommended_route"], "portfolio")
        self.assertEqual(recall["retrieved_count"], 1)

    def test_failed_route_is_not_recommended(self):
        failed = {
            "episode_id": "episode-failed",
            "context": self.context,
            "outcome": {
                "route_family": "overlay",
                "success": False,
                "utility": 0.2,
            },
            "candidate_reflections": [],
        }
        recall = recall_experiences(self.context, episodes=[failed])
        self.assertIsNone(recall["recommended_route"])
        self.assertTrue(recall["fallback_required"])

    def test_recall_aggregates_delayed_verification_strategy_credit(self):
        episode = {
            "episode_id": "episode-credit",
            "context": self.context,
            "outcome": {
                "route_family": "portfolio",
                "success": True,
                "utility": 0.9,
            },
            "candidate_reflections": [],
            "verification_strategy_credit": {
                "claim_core": {
                    "query_count": 2,
                    "mean_delayed_credit": 0.75,
                },
                "lexical_pivot": {
                    "query_count": 1,
                    "mean_delayed_credit": -0.1,
                },
            },
        }
        recall = recall_experiences(
            self.context,
            episodes=[episode],
        )
        self.assertEqual(
            recall["verification_strategy_scores"]["claim_core"],
            0.75,
        )
        self.assertEqual(
            recall["verification_strategy_scores"]["lexical_pivot"],
            -0.1,
        )
        self.assertEqual(
            recall["verification_strategy_observations"]["claim_core"],
            2,
        )

    def test_safe_memory_override_accepts_non_degrading_candidate(self):
        deterministic = candidate("fast_extractive_rescue", 0.80)
        portfolio = candidate("risk_aware_portfolio_rescue", 0.82)
        selected, decision = select_candidate_with_memory(
            [deterministic, portfolio],
            deterministic,
            {"recommended_route": "portfolio", "confidence": 0.8},
        )
        self.assertIs(selected, portfolio)
        self.assertTrue(decision["override"])
        self.assertEqual(decision["safety_gate"], "memory_override_accepted")

    def test_low_confidence_memory_cannot_override_current_route(self):
        deterministic = candidate("fast_extractive_rescue", 0.80)
        portfolio = candidate("risk_aware_portfolio_rescue", 0.82)
        selected, decision = select_candidate_with_memory(
            [deterministic, portfolio],
            deterministic,
            {"recommended_route": "portfolio", "confidence": 0.01},
        )
        self.assertIs(selected, deterministic)
        self.assertFalse(decision["override"])
        self.assertEqual(decision["safety_gate"], "low_confidence_memory_rejected")

    def test_safety_gate_rejects_low_coverage_memory_route(self):
        deterministic = candidate("fast_extractive_rescue", 0.90)
        portfolio = candidate("risk_aware_portfolio_rescue", 0.50)
        selected, decision = select_candidate_with_memory(
            [deterministic, portfolio],
            deterministic,
            {"recommended_route": "portfolio", "confidence": 0.8},
        )
        self.assertIs(selected, deterministic)
        self.assertFalse(decision["override"])
        self.assertIn("coverage_guard", decision["reason"])


if __name__ == "__main__":
    unittest.main()
