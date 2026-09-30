import tempfile
import unittest
from pathlib import Path

from api.features.chats.counterevidence_information_memory import (
    append_counterevidence_information_updates,
    load_counterevidence_information_memory,
)


class CounterevidenceInformationMemoryTests(unittest.TestCase):
    def test_expired_provisional_cannot_form_distant_quorum(self):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-expiry"],
        }

        def negative(family, route):
            return {
                "context": context,
                "sample_count": 1,
                "mean_realized_information_value": 0.1,
                "low_value_streak": 1,
                "negative_only": True,
                "query_family_hashes": [family],
                "retrieval_route_hashes": [route],
                "require_route_diversity": True,
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expiry.jsonl"
            append_counterevidence_information_updates(
                [negative("family-a", "route-a")],
                path,
                current_time=1000,
                provisional_ttl_days=30,
            )
            feedback = append_counterevidence_information_updates(
                [negative("family-b", "route-b")],
                path,
                current_time=1000 + 31 * 86400,
                provisional_ttl_days=30,
            )
            row = load_counterevidence_information_memory(path)[0]
        self.assertEqual(feedback["expired_provisional_count"], 1)
        self.assertEqual(row["sample_count"], 0)
        self.assertEqual(row["provisional_sample_count"], 1)
        self.assertEqual(
            row["provisional_query_family_hashes"],
            ["family-b"],
        )

    def test_provisional_sample_count_is_capped(self):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-cap"],
        }
        update = {
            "context": context,
            "sample_count": 1,
            "mean_realized_information_value": 0.1,
            "low_value_streak": 1,
            "negative_only": True,
            "query_family_hashes": ["family-a"],
            "retrieval_route_hashes": ["route-a"],
            "require_route_diversity": True,
        }
        capped_count = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cap.jsonl"
            for index in range(6):
                feedback = (
                    append_counterevidence_information_updates(
                        [update],
                        path,
                        current_time=1000 + index,
                        max_provisional_samples=3,
                    )
                )
                capped_count += feedback[
                    "capped_provisional_update_count"
                ]
            row = load_counterevidence_information_memory(path)[0]
        self.assertEqual(row["provisional_sample_count"], 3)
        self.assertGreater(capped_count, 0)

    def test_route_quorum_blocks_correlated_negative_confirmation(
        self,
    ):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-route-quorum"],
        }

        def negative(family, route):
            return {
                "context": context,
                "sample_count": 1,
                "mean_realized_information_value": 0.1,
                "low_value_streak": 1,
                "negative_only": True,
                "query_family_hashes": [family],
                "retrieval_route_hashes": [route],
                "require_route_diversity": True,
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route-quorum.jsonl"
            append_counterevidence_information_updates(
                [negative("family-a", "route-x")],
                path,
            )
            correlated = (
                append_counterevidence_information_updates(
                    [negative("family-b", "route-x")],
                    path,
                )
            )
            provisional = load_counterevidence_information_memory(
                path
            )[0]
            independent = (
                append_counterevidence_information_updates(
                    [negative("family-c", "route-y")],
                    path,
                )
            )
            confirmed = load_counterevidence_information_memory(
                path
            )[0]
        self.assertEqual(correlated["provisional_update_count"], 1)
        self.assertEqual(provisional["sample_count"], 0)
        self.assertEqual(
            len(provisional["provisional_query_family_hashes"]),
            2,
        )
        self.assertEqual(
            len(
                provisional[
                    "provisional_retrieval_route_hashes"
                ]
            ),
            1,
        )
        self.assertEqual(
            independent["confirmed_negative_update_count"],
            1,
        )
        self.assertEqual(confirmed["sample_count"], 3)

    def test_negative_feedback_requires_two_query_families(self):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-confirm"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "information.jsonl"
            first = append_counterevidence_information_updates(
                [
                    {
                        "context": context,
                        "sample_count": 1,
                        "mean_realized_information_value": 0.1,
                        "low_value_streak": 1,
                        "negative_only": True,
                        "query_family_hashes": ["family-a"],
                    }
                ],
                path,
            )
            provisional = load_counterevidence_information_memory(
                path
            )[0]
            second = append_counterevidence_information_updates(
                [
                    {
                        "context": context,
                        "sample_count": 1,
                        "mean_realized_information_value": 0.2,
                        "low_value_streak": 1,
                        "negative_only": True,
                        "query_family_hashes": ["family-b"],
                    }
                ],
                path,
            )
            confirmed = load_counterevidence_information_memory(
                path
            )[0]
        self.assertEqual(first["provisional_update_count"], 1)
        self.assertEqual(provisional["sample_count"], 0)
        self.assertEqual(provisional["provisional_sample_count"], 1)
        self.assertEqual(
            second["confirmed_negative_update_count"],
            1,
        )
        self.assertEqual(confirmed["sample_count"], 2)
        self.assertEqual(confirmed["provisional_sample_count"], 0)
        self.assertEqual(
            confirmed["feedback_status"],
            "confirmed_negative",
        )

    def test_positive_feedback_commits_and_clears_provisional(self):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-recovery"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "information.jsonl"
            append_counterevidence_information_updates(
                [
                    {
                        "context": context,
                        "sample_count": 1,
                        "mean_realized_information_value": 0.1,
                        "low_value_streak": 1,
                        "negative_only": True,
                        "query_family_hashes": ["family-a"],
                    }
                ],
                path,
            )
            feedback = append_counterevidence_information_updates(
                [
                    {
                        "context": context,
                        "sample_count": 1,
                        "mean_realized_information_value": 0.8,
                        "low_value_streak": 0,
                        "negative_only": False,
                        "query_family_hashes": ["family-b"],
                    }
                ],
                path,
            )
            row = load_counterevidence_information_memory(path)[0]
        self.assertEqual(
            feedback["immediate_positive_update_count"],
            1,
        )
        self.assertEqual(row["sample_count"], 1)
        self.assertEqual(row["provisional_sample_count"], 0)
        self.assertEqual(row["feedback_status"], "committed_positive")

    def test_information_memory_merges_weighted_statistics(self):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-a"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "information.jsonl"
            first = append_counterevidence_information_updates(
                [
                    {
                        "context": context,
                        "sample_count": 2,
                        "mean_realized_information_value": 0.2,
                        "information_m2": 0.02,
                        "low_value_streak": 2,
                    }
                ],
                path,
            )
            second = append_counterevidence_information_updates(
                [
                    {
                        "context": context,
                        "sample_count": 1,
                        "mean_realized_information_value": 0.8,
                        "information_m2": 0.0,
                        "low_value_streak": 0,
                    }
                ],
                path,
            )
            rows = load_counterevidence_information_memory(path)
        self.assertTrue(first["written"])
        self.assertTrue(second["written"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_count"], 3)
        self.assertEqual(
            rows[0]["mean_realized_information_value"],
            0.4,
        )
        self.assertEqual(rows[0]["low_value_streak"], 0)
        self.assertGreater(rows[0]["information_variance"], 0.0)

    def test_information_memory_extends_all_low_value_streak(self):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-b"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "information.jsonl"
            for _index in range(2):
                append_counterevidence_information_updates(
                    [
                        {
                            "context": context,
                            "sample_count": 1,
                            "mean_realized_information_value": 0.1,
                            "low_value_streak": 1,
                        }
                    ],
                    path,
                )
            row = load_counterevidence_information_memory(path)[0]
        self.assertEqual(row["sample_count"], 2)
        self.assertEqual(row["low_value_streak"], 2)


if __name__ == "__main__":
    unittest.main()
