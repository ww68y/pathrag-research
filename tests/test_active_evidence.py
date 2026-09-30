import unittest

from api.features.chats.active_evidence import (
    admit_active_relations,
    build_evidence_obligations,
    focus_chunk_text_for_query,
    plan_active_acquisition,
    run_active_evidence_acquisition,
)
from api.features.chats.vsr_pathrag import build_query_graph, diagnose_knowledge_gaps


class _VectorStore:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def query(self, query, top_k=5):
        self.queries.append((query, top_k))
        return self.rows[:top_k]


class _GraphStore:
    def __init__(self):
        self.edges = {
            ("AURORA PROJECT", "TEMPERATURE MONITORING"): {
                "keywords": "improve safety through monitoring mechanism",
                "description": (
                    "The Aurora project improved battery safety through real-time "
                    "temperature monitoring."
                ),
                "source_id": "chunk-active",
                "weight": 1.0,
            }
        }

    async def get_edge(self, source, target):
        return self.edges.get((source, target))

    async def get_node(self, _name):
        return None


class _RAG:
    def __init__(self):
        self.chunk_entity_relation_graph = _GraphStore()
        self.relationships_vdb = _VectorStore(
            [
                {
                    "src_id": "AURORA PROJECT",
                    "tgt_id": "TEMPERATURE MONITORING",
                    "distance": 0.93,
                }
            ]
        )
        self.entities_vdb = _VectorStore([])


class ActiveEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.question = "How did the Aurora project improve battery safety?"
        self.base = {
            "entities": [
                {
                    "name": "AURORA PROJECT",
                    "description": "A battery pilot project.",
                    "source_ids": ["chunk-base"],
                }
            ],
            "relationships": [
                {
                    "src": "GREENVOLT",
                    "tgt": "AURORA PROJECT",
                    "keywords": "designed project",
                    "description": "GreenVolt designed the Aurora project.",
                    "source_ids": ["chunk-base"],
                }
            ],
        }
        self.query_graph = build_query_graph(
            self.question,
            "process_or_explanation",
            {"type": "explanation"},
            self.base,
        )
        self.gap = diagnose_knowledge_gaps(self.query_graph, self.base)

    async def test_obligations_and_plan_target_missing_relation(self):
        obligations = build_evidence_obligations(self.query_graph, self.gap, self.base)
        self.assertIn("intent:mechanism", {row["id"] for row in obligations})
        plan = plan_active_acquisition(
            self.question,
            self.query_graph,
            self.gap,
            obligations,
            {"tier": "medium"},
        )
        self.assertTrue(plan["should_run"])
        self.assertGreater(plan["expected_information_value"], 0)
        self.assertIn("AURORA PROJECT", plan["planned_queries"][0]["query"])

    async def test_source_admission_rejects_unprovenanced_relation(self):
        obligation = {
            "id": "intent:mechanism",
            "type": "relation_intent",
            "target": "mechanism",
            "terms": ["mechanism", "through", "monitor"],
        }
        admitted = admit_active_relations(
            [
                {
                    "src": "AURORA PROJECT",
                    "tgt": "TEMPERATURE MONITORING",
                    "description": "Aurora improves safety through a monitoring mechanism.",
                    "score": 0.9,
                    "source_ids": [],
                }
            ],
            obligation,
            [],
        )
        self.assertEqual(admitted["admitted_count"], 0)
        self.assertIn(
            "missing_source_provenance",
            admitted["rejected"][0]["active_admission_reasons"],
        )

    async def test_active_loop_adds_grounded_edge_and_reduces_gap(self):
        rag = _RAG()
        original_count = len(self.base["relationships"])
        expanded, result = await run_active_evidence_acquisition(
            rag,
            self.question,
            self.query_graph,
            self.gap,
            self.base,
            {"tier": "medium"},
        )
        self.assertTrue(result["triggered"])
        self.assertGreaterEqual(result["query_count"], 1)
        self.assertEqual(result["admitted_relation_count"], 1)
        self.assertLess(result["final_gap_score"], result["initial_gap_score"])
        self.assertEqual(len(self.base["relationships"]), original_count)
        self.assertEqual(len(expanded["relationships"]), original_count + 1)
        self.assertTrue(result["base_graph_unchanged"])

    async def test_chunk_focus_selects_relevant_late_passage(self):
        leading = "A film production note with unrelated cast details. " * 70
        text = (
            f"Passage 1: {leading}\n\n"
            "Passage 2: Wale was born in Washington, D.C. His research "
            "examined Nigerian heritage and musical identity."
        )
        focused, audit = focus_chunk_text_for_query(
            text,
            "What did Wale research about Nigerian heritage?",
            max_chars=500,
        )
        self.assertIn("Wale was born", focused)
        self.assertIn("Nigerian heritage", focused)
        self.assertNotIn("unrelated cast details", focused)
        self.assertEqual(audit["strategy"], "query_focused_segments")
        self.assertEqual(audit["selected_segment_indexes"], [1])


if __name__ == "__main__":
    unittest.main()
