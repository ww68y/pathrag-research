import unittest
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.build_vsr_pathrag_v52_anchor_preserving_contexts import (
    audit_selection,
    definition_subject,
    expand_with_neighbors,
    select_anchor_centers,
)


class AnchorPreservingContextTests(unittest.TestCase):
    def test_definition_subject_extracts_primary_definition_target(self):
        self.assertEqual(
            definition_subject(
                "What is the primary definition of grazing "
                "according to the book?"
            ),
            "grazing",
        )

    def test_anchor_selection_preserves_rare_named_entity(self):
        chunks = [
            "Lawrence Scott Sheets wrote about Leningrad.",
            "General discussion of the KGB and Soviet officers.",
            (
                "Valery was the KGB officer who questioned Sheets and "
                "tried to recruit him."
            ),
        ]
        question = (
            "Describe the role of KGB officer Valery in the context of "
            "Lawrence Scott Sheets's stay in Leningrad."
        )
        centers = select_anchor_centers(chunks, question)
        selected = expand_with_neighbors(set(centers), len(chunks))
        self.assertIn(2, centers)
        self.assertTrue(
            audit_selection(chunks, selected, question)["passed"]
        )

    def test_definition_selection_preserves_explicit_definition(self):
        chunks = [
            "Rational grazing depends on carefully timed rest periods.",
            (
                "What is grazing? The author proposes this definition: "
                "Grazing is the meeting of cow and grass."
            ),
            "The primary objective is to preserve pasture productivity.",
        ]
        question = (
            "What is the primary definition of grazing according to "
            "the book?"
        )
        centers = select_anchor_centers(chunks, question)
        self.assertIn(1, centers)
        self.assertTrue(
            audit_selection(chunks, {1}, question)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
