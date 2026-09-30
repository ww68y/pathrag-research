import unittest

from api.features.chats.claim_active_verification import (
    apply_diagnosed_claim_correction,
    build_delayed_verification_credit,
    claim_status_transition_audit,
    claim_query_feedback_priority,
    plan_claim_verification_queries,
    locate_unverified_claims,
    run_claim_level_active_verification,
    split_draft_claims,
)


class _VectorStore:
    def __init__(self, rows):
        self.rows = rows

    async def query(self, _query, top_k=5):
        return self.rows[:top_k]


class _GraphStore:
    def __init__(self, edge, target):
        self.edge = edge
        self.target = target

    async def get_edge(self, source, target):
        if (source, target) == ("AURORA", self.target):
            return self.edge
        return None

    async def get_node(self, _name):
        return None


class _RAG:
    def __init__(
        self,
        description,
        source_id="claim-source",
        target="PLASMA SHIELDING",
        keywords="uses plasma shielding prevent overheating safety",
    ):
        self.chunk_entity_relation_graph = _GraphStore(
            {
                "keywords": keywords,
                "description": description,
                "source_id": source_id,
                "weight": 1.0,
            },
            target,
        )
        self.relationships_vdb = _VectorStore(
            [
                {
                    "src_id": "AURORA",
                    "tgt_id": target,
                    "distance": 0.95,
                }
            ]
        )
        self.entities_vdb = _VectorStore([])


class _SequentialVectorStore:
    def __init__(self, batches):
        self.batches = batches
        self.query_count = 0

    async def query(self, _query, top_k=5):
        index = min(self.query_count, len(self.batches) - 1)
        self.query_count += 1
        return self.batches[index][:top_k]


class _BundleGraphStore:
    def __init__(self, edges):
        self.edges = edges

    async def get_edge(self, source, target):
        return self.edges.get((source, target))

    async def get_node(self, _name):
        return None


class _BundleRAG:
    def __init__(self, edges):
        targets = list(edges)
        self.chunk_entity_relation_graph = _BundleGraphStore(edges)
        self.relationships_vdb = _SequentialVectorStore(
            [
                [
                    {
                        "src_id": targets[0][0],
                        "tgt_id": targets[0][1],
                        "distance": 0.95,
                    }
                ],
                [
                    {
                        "src_id": targets[1][0],
                        "tgt_id": targets[1][1],
                        "distance": 0.94,
                    }
                ],
            ]
        )
        self.entities_vdb = _VectorStore([])


class _EntityGraphStore:
    def __init__(self, node):
        self.node = node

    async def get_edge(self, _source, _target):
        return None

    async def get_node(self, name):
        return self.node if name == "PLASMA SHIELDING" else None


class _EntityRAG:
    def __init__(self, description):
        self.chunk_entity_relation_graph = _EntityGraphStore(
            {
                "entity_type": "SAFETY_MECHANISM",
                "description": description,
                "source_id": "entity-source",
            }
        )
        self.relationships_vdb = _VectorStore([])
        self.entities_vdb = _VectorStore(
            [
                {
                    "entity_name": "PLASMA SHIELDING",
                    "distance": 0.96,
                }
            ]
        )


class _ChunkTextStore:
    def __init__(self, text):
        self.text = text

    async def get_by_id(self, source_id):
        if source_id == "chunk-claim":
            return {"content": self.text}
        return None


class _ChunkRAG:
    def __init__(self, text):
        self.chunk_entity_relation_graph = _GraphStore({}, "")
        self.relationships_vdb = _VectorStore([])
        self.entities_vdb = _VectorStore([])
        self.chunks_vdb = _VectorStore(
            [
                {
                    "__id__": "chunk-claim",
                    "distance": 0.98,
                }
            ]
        )
        self.text_chunks = _ChunkTextStore(text)


class ClaimActiveVerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.question = "How does Aurora prevent overheating?"
        self.answer = (
            "Aurora uses plasma shielding to prevent battery overheating during "
            "high-load charging operation in summer."
        )
        self.base = {
            "entities": [],
            "relationships": [
                {
                    "src": "AURORA",
                    "tgt": "BATTERY PROJECT",
                    "keywords": "project description",
                    "description": "Aurora is an experimental battery project.",
                    "source_ids": ["base-source"],
                }
            ],
        }

    def test_transition_audit_allows_new_diagnosis_without_regression(self):
        current = {
            "claims": [
                {"claim_id": "claim-0", "status": "unknown"},
                {"claim_id": "claim-1", "status": "unknown"},
                {"claim_id": "claim-2", "status": "supported"},
            ]
        }
        tentative = {
            "claims": [
                {"claim_id": "claim-0", "status": "supported"},
                {"claim_id": "claim-1", "status": "contradicted"},
                {"claim_id": "claim-2", "status": "supported"},
            ]
        }
        audit = claim_status_transition_audit(current, tentative, 0)
        self.assertTrue(audit["safe_gain"])
        self.assertEqual(audit["status_gain"], 2)
        self.assertEqual(audit["diagnosed_claim_ids"], ["claim-1"])
        self.assertEqual(audit["regressed_claim_ids"], [])

    def test_transition_audit_rejects_supported_claim_regression(self):
        current = {
            "claims": [
                {"claim_id": "claim-0", "status": "unknown"},
                {"claim_id": "claim-1", "status": "supported"},
            ]
        }
        tentative = {
            "claims": [
                {"claim_id": "claim-0", "status": "supported"},
                {"claim_id": "claim-1", "status": "contradicted"},
            ]
        }
        audit = claim_status_transition_audit(current, tentative, 0)
        self.assertFalse(audit["safe_gain"])
        self.assertEqual(audit["regressed_claim_ids"], ["claim-1"])

    def test_delayed_credit_rewards_retained_source_and_supported_claim(self):
        verification = {
            "rounds": [
                {
                    "claim_id": "claim-0",
                    "query_strategy": "claim_core",
                    "counterfactual_gain": 1,
                    "total_seconds": 0.4,
                    "committed_candidates": [
                        {"source_ids": ["source-1"]}
                    ],
                    "status_transitions": [
                        {
                            "claim_id": "claim-0",
                            "claim": (
                                "Aurora uses plasma shielding to prevent "
                                "battery overheating."
                            ),
                            "before": "unknown",
                            "after": "supported",
                        }
                    ],
                }
            ]
        }
        credit = build_delayed_verification_credit(
            verification,
            {"committed": True, "used_source_ids": ["source-1"]},
            {
                "claims": [
                    {
                        "claim_id": "claim-0",
                        "claim": (
                            "Aurora uses plasma shielding to prevent "
                            "battery overheating."
                        ),
                        "status": "supported",
                    }
                ]
            },
            {"risk_level": "low"},
            {"action": "commit"},
            {
                "source_excerpts": [
                    {"source_id": "source-1", "text": "verified"}
                ]
            },
        )
        self.assertEqual(credit["retained_query_count"], 1)
        self.assertEqual(credit["mean_delayed_credit"], 1.0)
        self.assertEqual(
            credit["strategy_credit"]["claim_core"][
                "mean_delayed_credit"
            ],
            1.0,
        )

    def test_delayed_credit_revokes_dropped_unsafe_support(self):
        verification = {
            "rounds": [
                {
                    "claim_id": "claim-0",
                    "query_strategy": "claim_core",
                    "counterfactual_gain": 1,
                    "committed_candidates": [
                        {"source_ids": ["source-1"]}
                    ],
                    "status_transitions": [
                        {
                            "claim_id": "claim-0",
                            "claim": "Aurora uses plasma shielding.",
                            "before": "unknown",
                            "after": "supported",
                        }
                    ],
                }
            ]
        }
        credit = build_delayed_verification_credit(
            verification,
            {"committed": False, "used_source_ids": []},
            {
                "claims": [
                    {
                        "claim_id": "claim-0",
                        "claim": "Aurora uses plasma shielding.",
                        "status": "contradicted",
                    }
                ]
            },
            {"risk_level": "high"},
            {"action": "abstain"},
            {"source_excerpts": []},
        )
        self.assertEqual(credit["revoked_query_count"], 1)
        self.assertLess(credit["mean_delayed_credit"], 0.35)

    def test_delayed_credit_retains_diagnosis_when_false_claim_removed(self):
        verification = {
            "rounds": [
                {
                    "claim_id": "claim-0",
                    "query_strategy": "counterevidence_question",
                    "counterfactual_gain": 1,
                    "committed_candidates": [
                        {"source_ids": ["source-1"]}
                    ],
                    "status_transitions": [
                        {
                            "claim_id": "claim-0",
                            "claim": "Aurora does not use plasma shielding.",
                            "before": "unknown",
                            "after": "contradicted",
                        }
                    ],
                }
            ]
        }
        credit = build_delayed_verification_credit(
            verification,
            {"committed": False, "used_source_ids": []},
            {"claims": []},
            {"risk_level": "low"},
            {"action": "commit"},
            {
                "source_excerpts": [
                    {"source_id": "source-1", "text": "verified"}
                ]
            },
        )
        self.assertEqual(credit["retained_query_count"], 1)
        self.assertTrue(
            credit["query_credits"][0]["transition_outcomes"][0][
                "retained"
            ]
        )

    def test_diagnosis_correction_trims_only_false_meta_clause(self):
        answer = (
            "Wale's communications background represents his early career "
            "in journalism, though the data does not provide specific "
            "details about his professional journalism work."
        )
        verification = {
            "rounds": [
                {
                    "committed_candidates": [
                        {"source_ids": ["wale-source"]}
                    ],
                    "status_transitions": [
                        {
                            "claim_id": "claim-0",
                            "claim": answer,
                            "before": "unknown",
                            "after": "contradicted",
                        }
                    ],
                }
            ]
        }
        corrected, audit = apply_diagnosed_claim_correction(
            (
                "How did Wale's career evolve from journalism to "
                "academia?"
            ),
            answer,
            verification,
            {
                "source_excerpts": [
                    {
                        "source_id": "wale-source",
                        "text": (
                            "Wale worked as a professional journalist before "
                            "entering academia."
                        ),
                    }
                ]
            },
        )
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["trimmed_clause_count"], 1)
        self.assertEqual(audit["removed_sentence_count"], 0)
        self.assertIn("early career in journalism", corrected)
        self.assertNotIn("data does not provide", corrected)
        self.assertEqual(
            audit["corrections"][0]["prohibited_assertion"],
            (
                "though the data does not provide specific details about "
                "his professional journalism work."
            ),
        )

    def test_diagnosis_correction_replaces_false_sentence_from_evidence(self):
        false_claim = (
            "Aurora does not use plasma shielding to prevent battery "
            "overheating."
        )
        answer = (
            "Aurora is a battery platform. "
            f"{false_claim} "
            "It operates during high-load charging."
        )
        verification = {
            "rounds": [
                {
                    "committed_candidates": [
                        {"source_ids": ["aurora-source"]}
                    ],
                    "status_transitions": [
                        {
                            "claim_id": "claim-1",
                            "claim": false_claim,
                            "before": "unknown",
                            "after": "contradicted",
                        }
                    ],
                }
            ]
        }
        corrected, audit = apply_diagnosed_claim_correction(
            "How does Aurora prevent battery overheating?",
            answer,
            verification,
            {
                "source_excerpts": [
                    {
                        "source_id": "aurora-source",
                        "text": (
                            "Aurora uses plasma shielding to prevent battery "
                            "overheating."
                        ),
                    }
                ]
            },
        )
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["grounded_replacement_count"], 1)
        self.assertEqual(audit["removed_sentence_count"], 0)
        self.assertNotIn(false_claim, corrected)
        self.assertIn(
            "Aurora uses plasma shielding to prevent battery overheating",
            corrected,
        )
        self.assertIn("Aurora is a battery platform", corrected)

    def test_diagnosis_correction_fails_closed_when_text_not_located(self):
        answer = "Aurora uses verified thermal monitoring during charging."
        corrected, audit = apply_diagnosed_claim_correction(
            "How does Aurora prevent overheating?",
            answer,
            {
                "rounds": [
                    {
                        "status_transitions": [
                            {
                                "claim_id": "claim-0",
                                "claim": (
                                    "A separate unrelated claim about a "
                                    "different system."
                                ),
                                "after": "contradicted",
                            }
                        ]
                    }
                ]
            },
            {"source_excerpts": []},
        )
        self.assertFalse(audit["accepted"])
        self.assertEqual(
            audit["stop_reason"],
            "diagnosed_claim_text_not_located",
        )
        self.assertEqual(corrected, answer)

    async def test_claim_split_and_unknown_location(self):
        claims = split_draft_claims(self.answer)
        graph, targets = locate_unverified_claims(claims, self.base)
        self.assertEqual(len(claims), 1)
        self.assertEqual(graph["status_counts"]["unknown"], 1)
        self.assertEqual(targets[0]["status"], "unknown")

    async def test_query_plan_sanitizes_meta_claim_and_diversifies(self):
        targets = [
            {
                "claim_index": 0,
                "claim_id": "claim-0",
                "claim": (
                    "Based on the provided data, Mary's objection is not "
                    "explicitly stated."
                ),
                "status": "contradicted",
                "confidence": 0.1,
                "priority": 1.0,
            },
            {
                "claim_index": 1,
                "claim_id": "claim-1",
                "claim": (
                    "Mary objected that the poem contained no human interest."
                ),
                "status": "unknown",
                "confidence": 0.1,
                "priority": 0.75,
            },
        ]
        plans = plan_claim_verification_queries(
            "What was Mary's objection to The Witch of Atlas?",
            targets,
            {"tier": "medium"},
        )
        self.assertEqual(len(plans), 4)
        self.assertEqual(
            [plan["claim_id"] for plan in plans[:2]],
            ["claim-0", "claim-1"],
        )
        meta_queries = [
            plan["query_plan"]["query"].lower()
            for plan in plans
            if plan["claim_id"] == "claim-0"
        ]
        self.assertTrue(
            all("verify this claim" not in query for query in meta_queries)
        )
        self.assertTrue(
            all("provided data" not in query for query in meta_queries)
        )
        self.assertTrue(
            all("not explicitly stated" not in query for query in meta_queries)
        )
        fast_plans = plan_claim_verification_queries(
            "What was Mary's objection to The Witch of Atlas?",
            targets,
            {"tier": "medium", "requested_fast": True},
        )
        self.assertEqual(len(fast_plans), 1)

    async def test_query_feedback_prioritizes_pending_bundle_evidence(self):
        plan = {
            "query_plan": {
                "expected_information_value": 0.8,
            }
        }
        pending_priority = claim_query_feedback_priority(
            plan,
            {
                "attempt_count": 1,
                "no_value_streak": 0,
            },
            pending_evidence_count=1,
        )
        no_value_priority = claim_query_feedback_priority(
            plan,
            {
                "attempt_count": 1,
                "no_value_streak": 1,
            },
            pending_evidence_count=0,
        )
        self.assertGreater(pending_priority, no_value_priority)

    async def test_historical_strategy_credit_changes_query_priority(self):
        plan = {
            "query_plan": {"expected_information_value": 0.5},
            "historical_strategy_credit": 0.8,
        }
        positive = claim_query_feedback_priority(plan, {}, 0)
        plan["historical_strategy_credit"] = -0.8
        negative = claim_query_feedback_priority(plan, {}, 0)
        self.assertGreater(positive, negative)

    async def test_claim_level_retrieval_commits_only_status_improving_evidence(self):
        rag = _RAG(
            self.answer
        )
        expanded, result = await run_claim_level_active_verification(
            rag,
            self.question,
            self.answer,
            self.base,
            {"tier": "medium"},
        )
        self.assertTrue(result["triggered"])
        self.assertEqual(result["resolved_claim_count"], 1)
        self.assertEqual(result["counterfactual_committed_count"], 1)
        self.assertEqual(result["final_status_counts"]["supported"], 1)
        self.assertEqual(len(self.base["relationships"]), 1)
        self.assertEqual(len(expanded["relationships"]), 2)

    async def test_run_records_recalled_strategy_credit(self):
        _expanded, result = await run_claim_level_active_verification(
            _RAG(self.answer),
            self.question,
            self.answer,
            self.base,
            {"tier": "low"},
            strategy_prior={"claim_core": 0.75},
        )
        self.assertEqual(
            result["historical_strategy_credit"]["claim_core"],
            0.75,
        )
        self.assertEqual(
            result["planned_queries"][0][
                "historical_strategy_credit"
            ],
            0.75,
        )
        self.assertEqual(
            result["rounds"][0]["historical_strategy_credit"],
            0.75,
        )

    async def test_counterfactual_gate_rejects_relevant_but_non_resolving_relation(self):
        rag = _RAG(
            "Aurora documented a general safety process during preliminary research.",
            target="SAFETY REPORT",
            keywords="documented preliminary report",
        )
        expanded, result = await run_claim_level_active_verification(
            rag,
            self.question,
            self.answer,
            self.base,
            {"tier": "medium"},
        )
        self.assertEqual(result["resolved_claim_count"], 0)
        self.assertEqual(result["counterfactual_committed_count"], 0)
        self.assertGreaterEqual(result["counterfactual_rejected_count"], 1)
        self.assertEqual(len(expanded["relationships"]), len(self.base["relationships"]))

    async def test_supported_draft_skips_additional_retrieval(self):
        complete = {
            "entities": [],
            "relationships": [
                {
                    "src": "AURORA",
                    "tgt": "PLASMA SHIELDING",
                    "keywords": "uses plasma shielding prevent overheating",
                    "description": self.answer,
                    "source_ids": ["complete-source"],
                }
            ],
        }
        _expanded, result = await run_claim_level_active_verification(
            _RAG(self.answer),
            self.question,
            self.answer,
            complete,
            {"tier": "medium"},
        )
        self.assertFalse(result["triggered"])
        self.assertEqual(result["query_count"], 0)
        self.assertEqual(result["stop_reason"], "no_unverified_claims")

    async def test_entity_evidence_can_resolve_unknown_claim(self):
        expanded, result = await run_claim_level_active_verification(
            _EntityRAG(self.answer),
            self.question,
            self.answer,
            self.base,
            {"tier": "low"},
        )
        self.assertEqual(result["resolved_claim_count"], 1)
        self.assertEqual(result["admitted_entity_count"], 1)
        self.assertEqual(result["counterfactual_committed_count"], 1)
        self.assertEqual(len(expanded["entities"]), 1)

    async def test_chunk_evidence_can_resolve_graph_missing_claim(self):
        expanded, result = await run_claim_level_active_verification(
            _ChunkRAG(self.answer),
            self.question,
            self.answer,
            self.base,
            {"tier": "low"},
        )
        self.assertEqual(result["resolved_claim_count"], 1)
        self.assertEqual(result["admitted_chunk_count"], 1)
        self.assertEqual(result["counterfactual_committed_count"], 1)
        self.assertEqual(
            expanded["source_excerpts"][0]["source_id"],
            "chunk-claim",
        )

    async def test_chunk_counterevidence_diagnoses_false_unknown_claim(self):
        false_answer = (
            "Aurora does not use plasma shielding to prevent battery "
            "overheating during high-load charging operation."
        )
        source_text = (
            "Aurora uses plasma shielding to prevent battery overheating "
            "during high-load charging operation."
        )
        _expanded, result = await run_claim_level_active_verification(
            _ChunkRAG(source_text),
            self.question,
            false_answer,
            self.base,
            {"tier": "low"},
        )
        self.assertEqual(result["resolved_claim_count"], 0)
        self.assertEqual(result["diagnosed_claim_count"], 1)
        self.assertEqual(result["epistemically_resolved_claim_count"], 1)
        self.assertEqual(result["final_status_counts"]["contradicted"], 1)
        self.assertEqual(
            result["rounds"][0]["decision"],
            "commit_claim_diagnosing_counterevidence",
        )

    async def test_cross_query_bundle_commits_only_combined_status_gain(self):
        answer = (
            "Aurora advanced battery platform uses plasma shielding thermal "
            "sensors cooling channels adaptive controls ceramic barriers "
            "redundant alarms during rapid summer charging operations."
        )
        edges = {
            ("AURORA", "PLASMA SHIELDING"): {
                "keywords": "Aurora battery platform plasma shielding",
                "description": "Aurora battery platform plasma shielding.",
                "source_id": "bundle-source-1",
                "weight": 1.0,
            },
            ("AURORA", "REDUNDANT ALARMS"): {
                "keywords": "ceramic barriers redundant alarms",
                "description": "Ceramic barriers redundant alarms.",
                "source_id": "bundle-source-2",
                "weight": 1.0,
            },
        }
        expanded, result = await run_claim_level_active_verification(
            _BundleRAG(edges),
            "How does Aurora protect its battery platform?",
            answer,
            {"entities": [], "relationships": []},
            {"tier": "low"},
        )
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["resolved_claim_count"], 1)
        self.assertEqual(result["counterfactual_committed_count"], 2)
        self.assertEqual(
            result["rounds"][0]["decision"],
            "defer_for_cross_query_evidence_bundle",
        )
        self.assertEqual(
            result["rounds"][1]["decision"],
            "commit_claim_improving_evidence",
        )
        self.assertEqual(len(expanded["relationships"]), 2)
        self.assertGreaterEqual(result["runtime_seconds"], 0)
        self.assertIn("retrieved_candidates", result["rounds"][0])
        self.assertIn("feedback_after_query", result["rounds"][0])

    async def test_high_tier_stops_third_variant_after_two_no_gain_queries(self):
        rag = _RAG(
            "Aurora documented a general safety process during research.",
            target="SAFETY REPORT",
            keywords="documented preliminary report",
        )
        _expanded, result = await run_claim_level_active_verification(
            rag,
            self.question,
            self.answer,
            self.base,
            {"tier": "high"},
        )
        self.assertEqual(result["query_plan_count"], 3)
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["low_value_stopped_query_count"], 1)
        self.assertEqual(
            result["low_value_stopped_claim_ids"],
            ["claim-0"],
        )


if __name__ == "__main__":
    unittest.main()
