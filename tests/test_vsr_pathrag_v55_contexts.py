import json
import unittest
from pathlib import Path

from experiments.build_multiquery_condensed_subset import split_chunks
from experiments.build_vsr_pathrag_v55_order_preserving_contexts import (
    audit_order_preservation,
    is_ordered_application_question,
    ordered_clause_span,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


class OrderedClauseContextTests(unittest.TestCase):
    def test_ordered_application_question_gate_is_narrow(self):
        self.assertTrue(
            is_ordered_application_question(
                "How are proceeds from enforcement to be applied?"
            )
        )
        self.assertFalse(
            is_ordered_application_question(
                "What actions can the Collateral Trustee take?"
            )
        )

    def test_span_preserves_all_explicit_ordinals(self):
        chunks = [
            "Background.",
            "The following order of application: FIRST, fees; SECOND, debt;",
            "THIRD, providers pro rata; FOURTH, surplus to the grantor.",
            "Next section.",
        ]
        question = "How are proceeds to be applied?"
        span = ordered_clause_span(chunks, question)
        self.assertEqual(span, {1, 2})
        before = audit_order_preservation(chunks, {1}, question)
        after = audit_order_preservation(chunks, span, question)
        self.assertFalse(before["passed"])
        self.assertTrue(after["passed"])

    def test_built_dataset_adds_only_legal_order_chunks(self):
        path = (
            ROOT_DIR
            / "datasets"
            / "UltraDomain"
            / "vsr_v55_order_preserving_validation_30.jsonl"
        )
        if not path.exists():
            self.skipTest("v55 dataset has not been built")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 30)
        additions = {
            row["dataset"]: tuple(
                row.get("order_preserving_added_chunk_indices", [])
            )
            for row in rows
        }
        self.assertEqual(additions["legal"], (34, 35))
        for domain in ("biology", "physics", "history", "agriculture"):
            self.assertEqual(additions[domain], ())


if __name__ == "__main__":
    unittest.main()
