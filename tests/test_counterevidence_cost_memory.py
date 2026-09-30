import tempfile
import unittest
from pathlib import Path

from api.features.chats.counterevidence_cost_memory import (
    append_counterevidence_cost_updates,
    load_counterevidence_cost_memory,
)


class CounterevidenceCostMemoryTests(unittest.TestCase):
    def test_cost_memory_clips_extreme_batch_mean(self):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-outlier"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost-memory.jsonl"
            append_counterevidence_cost_updates(
                [
                    {
                        "context": context,
                        "sample_count": 4,
                        "mean_normalized_cost": 2.0,
                        "mean_yield_score": 0.7,
                        "cost_m2": 0.12,
                        "yield_m2": 0.01,
                    }
                ],
                path,
            )
            feedback = append_counterevidence_cost_updates(
                [
                    {
                        "context": context,
                        "sample_count": 1,
                        "mean_normalized_cost": 100.0,
                        "mean_yield_score": 0.0,
                    }
                ],
                path,
            )
            row = load_counterevidence_cost_memory(path)[0]

        self.assertEqual(feedback["clipped_update_count"], 1)
        self.assertLess(row["mean_normalized_cost"], 3.0)
        self.assertGreater(row["cost_variance"], 0.0)

    def test_cost_memory_round_trip_merges_weighted_profiles(
        self,
    ):
        context = {
            "task_type": "claim",
            "claim_count": 1,
            "term_hashes": ["hash-a", "hash-b"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost-memory.jsonl"
            first = append_counterevidence_cost_updates(
                [
                    {
                        "context": context,
                        "sample_count": 1,
                        "mean_normalized_cost": 2.0,
                        "mean_yield_score": 0.5,
                    }
                ],
                path,
            )
            second = append_counterevidence_cost_updates(
                [
                    {
                        "context": context,
                        "sample_count": 2,
                        "mean_normalized_cost": 5.0,
                        "mean_yield_score": 0.8,
                    }
                ],
                path,
            )
            rows = load_counterevidence_cost_memory(path)

        self.assertTrue(first["written"])
        self.assertTrue(second["written"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_count"], 3)
        self.assertEqual(rows[0]["mean_normalized_cost"], 4.0)
        self.assertEqual(rows[0]["mean_yield_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
