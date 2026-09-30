import json
import unittest

from api.features.chats.answer_realization import (
    apply_counterevidence_information_profile,
    apply_transactional_correction_rollback,
    assess_counterevidence_information_admission,
    assess_counterevidence_cost_admission,
    audit_correction_source_certificates,
    audit_prohibited_claim_relapse,
    audit_realized_answer,
    audit_supported_facet_retention,
    build_counterfactual_bundle_stability_certificate,
    build_counterevidence_cost_context,
    build_robust_counterevidence_cost_prior,
    build_robust_counterevidence_information_prior,
    build_counterevidence_queries,
    build_counterevidence_retrieval_route_hash,
    build_fused_counterevidence_query,
    build_correction_persistence_constraints,
    build_evidence_obligations,
    build_facet_activation_policy,
    build_realization_dependency_graph,
    build_realization_prompt,
    build_supported_facet_plan,
    build_supported_facets,
    classify_supported_facet_query,
    classify_claim_evidence_relation,
    cluster_counterevidence_claims,
    counterevidence_independence_family_key,
    estimate_claim_relation_risk,
    estimate_counterevidence_query_value,
    materialize_correction_source_chunks,
    normalize_counterevidence_search_response,
    parse_realization_response,
    plan_adaptive_counterevidence_query_budget,
    prepare_retrieved_excerpts,
    question_requests_event,
    rank_source_excerpts,
    realize_answer,
    run_active_counterevidence_challenge,
    select_obligation_aware_claim_bundle,
    update_online_counterevidence_query_budget,
)


def _joint_counterevidence_transaction():
    claims = [
        "The program caused higher sales.",
        "The program caused greater revenue.",
    ]
    nodes = [
        {
            "node_id": f"claim-{index}",
            "claim_index": index - 1,
            "text": claim,
            "evidence_support": [
                {
                    "evidence_id": f"S{index}",
                    "family_ids": [
                        f"support-document-{index}"
                    ],
                }
            ],
        }
        for index, claim in enumerate(claims, start=1)
    ]
    transaction = {
        "dependency_graph": {
            "nodes": nodes,
            "edges": [],
            "kept_node_ids": ["claim-1", "claim-2"],
        },
        "claim_bundle_selection": {
            "final_selected_node_ids": [
                "claim-1",
                "claim-2",
            ]
        },
        "counterfactual_stability_certificate": {
            "semantic_family_quorum_node_ids": [
                "claim-1",
                "claim-2",
            ],
            "final_retained_node_ids": [
                "claim-1",
                "claim-2",
            ],
            "provenance_certificates": [
                {
                    "node_id": f"claim-{index}",
                    "claim_relation_risk": {
                        "risk_level": "medium",
                        "risk_score": 0.46,
                    },
                    "support_families": [
                        {
                            "family_ids": [
                                f"support-document-{index}"
                            ]
                        }
                    ],
                }
                for index in [1, 2]
            ],
        },
        "rollback_source_ids": [],
        "actions": [],
    }
    return claims, nodes, transaction


class AnswerRealizationTests(unittest.TestCase):
    def test_cost_admission_requires_trustworthy_usage(self):
        rejected = assess_counterevidence_cost_admission(
            {
                "usage_contract_complete": False,
                "usage_contract_missing_fields": [
                    "usage.total_tokens"
                ],
            },
            require_complete_usage_contract=True,
        )
        admitted = assess_counterevidence_cost_admission(
            {
                "usage_contract_complete": True,
                "usage_contract_missing_fields": [],
            },
            require_complete_usage_contract=True,
        )
        failed = assess_counterevidence_cost_admission(
            {"usage_contract_complete": True},
            retrieval_failed=True,
            require_complete_usage_contract=True,
        )
        partial = assess_counterevidence_cost_admission(
            {
                "usage_contract_complete": True,
                "usage_contract_missing_fields": [],
                "usage_status": "partial",
            },
            require_complete_usage_contract=True,
        )
        self.assertFalse(rejected["admitted"])
        self.assertTrue(admitted["admitted"])
        self.assertEqual(failed["reason"], "retrieval_exception")
        self.assertEqual(
            partial["reason"],
            "usage_status_incomplete",
        )

    def test_retrieval_usage_contract_requires_explicit_fields(self):
        complete = normalize_counterevidence_search_response(
            {
                "rows": [],
                "latency_ms": 10,
                "retrieval_adapter": "local_scan",
                "usage_status": "complete_local_retrieval",
                "usage": {
                    "call_count": 0,
                    "total_tokens": 0,
                },
            },
            11,
        )
        incomplete = normalize_counterevidence_search_response(
            {
                "rows": [],
                "usage": {"call_count": 0},
            },
            11,
        )
        complete_admission = (
            assess_counterevidence_information_admission(
                complete,
                require_complete_usage_contract=True,
            )
        )
        incomplete_admission = (
            assess_counterevidence_information_admission(
                incomplete,
                require_complete_usage_contract=True,
            )
        )
        self.assertTrue(complete["usage_contract_complete"])
        self.assertTrue(complete_admission["admitted"])
        self.assertFalse(incomplete["usage_contract_complete"])
        self.assertEqual(
            incomplete_admission["reason"],
            "usage_contract_incomplete",
        )
        self.assertIn(
            "usage.total_tokens",
            incomplete[
                "usage_contract_missing_fields"
            ],
        )

    def test_retrieval_route_hash_is_stable_and_private(self):
        first = build_counterevidence_retrieval_route_hash(
            "local_scan",
            "route-a",
        )
        second = build_counterevidence_retrieval_route_hash(
            "local_scan",
            "route-a",
        )
        other = build_counterevidence_retrieval_route_hash(
            "local_scan",
            "route-b",
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 16)
        self.assertNotIn("route", first)

    def test_information_admission_distinguishes_complete_and_partial(
        self,
    ):
        complete_empty = assess_counterevidence_information_admission(
            {
                "rows": [],
                "metadata_source": "structured_response",
                "usage_status": "complete_local_retrieval",
            }
        )
        partial = assess_counterevidence_information_admission(
            {
                "rows": [],
                "metadata_source": "structured_response",
                "usage_status": "partial",
            }
        )
        missing_lineage = assess_counterevidence_information_admission(
            {
                "rows": [{"source_id": "chunk-1", "text": "x"}],
                "metadata_source": "structured_response",
                "usage_status": "reported",
            }
        )
        self.assertTrue(complete_empty["admitted"])
        self.assertFalse(partial["admitted"])
        self.assertEqual(
            missing_lineage["reason"],
            "family_lineage_incomplete",
        )

    def test_robust_information_prior_uses_upper_bound_and_recency(
        self,
    ):
        prior = {
            "sample_count": 10,
            "mean_realized_information_value": 0.3,
            "information_m2": 0.9,
            "low_value_streak": 4,
            "updated_at": 1000,
        }
        robust = build_robust_counterevidence_information_prior(
            prior,
            current_time=1000,
        )
        stale = build_robust_counterevidence_information_prior(
            prior,
            current_time=1000 + 365 * 86400,
        )
        self.assertGreater(
            robust["information_upper_bound"],
            0.3,
        )
        self.assertEqual(
            robust["effective_low_value_streak"],
            4,
        )
        self.assertIsNone(stale)

    def test_independence_family_key_collapses_document_chunks(
        self,
    ):
        first = counterevidence_independence_family_key(
            {
                "source_id": "chunk-1",
                "document_id": "document-1",
            },
            "chunk-1",
        )
        second = counterevidence_independence_family_key(
            {
                "source_id": "chunk-2",
                "document_id": "document-1",
            },
            "chunk-2",
        )
        other = counterevidence_independence_family_key(
            {
                "source_id": "chunk-3",
                "document_id": "document-2",
            },
            "chunk-3",
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(first, "ancestor:document-1")

    def test_information_profile_penalizes_repeated_low_value_route(
        self,
    ):
        task = {
            "member_node_ids": ["claim-1"],
            "adjusted_expected_value": 1.0,
            "value_per_cost": 1.0,
        }
        single = apply_counterevidence_information_profile(
            task,
            {
                "claim-1": {
                    "sample_count": 1,
                    "mean_realized_information_value": 0.3,
                    "low_value_streak": 1,
                }
            },
            True,
        )
        fused = apply_counterevidence_information_profile(
            {
                **task,
                "member_node_ids": ["claim-1", "claim-2"],
            },
            {
                "claim-1": {
                    "sample_count": 1,
                    "mean_realized_information_value": 0.3,
                    "low_value_streak": 1,
                }
            },
            True,
        )
        self.assertLess(
            single["information_value_multiplier"],
            0.6,
        )
        self.assertGreater(
            fused["information_value_multiplier"],
            single["information_value_multiplier"],
        )
        self.assertEqual(
            single["information_profile_source"],
            "claim_information_posterior",
        )

    def test_online_budget_releases_and_revokes_with_coverage_floor(
        self,
    ):
        released = update_online_counterevidence_query_budget(
            current_query_budget=2,
            hard_query_budget=4,
            executed_query_count=1,
            unresolved_unqueried_count=1,
            realized_information_value=0.8,
            selected_task_remains_unresolved=True,
        )
        revoked = update_online_counterevidence_query_budget(
            current_query_budget=4,
            hard_query_budget=4,
            executed_query_count=1,
            unresolved_unqueried_count=1,
            realized_information_value=0.2,
            selected_task_remains_unresolved=True,
        )
        floor_held = update_online_counterevidence_query_budget(
            current_query_budget=2,
            hard_query_budget=4,
            executed_query_count=1,
            unresolved_unqueried_count=1,
            realized_information_value=0.2,
            selected_task_remains_unresolved=True,
        )
        self.assertEqual(released["action"], "release")
        self.assertEqual(released["updated_query_budget"], 3)
        self.assertEqual(revoked["action"], "revoke")
        self.assertEqual(revoked["updated_query_budget"], 3)
        self.assertEqual(floor_held["action"], "hold")
        self.assertEqual(floor_held["updated_query_budget"], 2)

    def test_adaptive_budget_keeps_coverage_floor_for_medium_risk(
        self,
    ):
        claims, nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        plan = plan_adaptive_counterevidence_query_budget(
            ["claim-1", "claim-2"],
            {row["node_id"]: row for row in nodes},
            {
                row["node_id"]: row
                for row in transaction[
                    "counterfactual_stability_certificate"
                ]["provenance_certificates"]
            },
            hard_query_budget=4,
        )
        self.assertEqual(plan["coverage_floor"], 2)
        self.assertEqual(plan["effective_query_budget"], 2)
        self.assertEqual(plan["budget_saved_count"], 2)
        self.assertEqual(
            [row["yield_source"] for row in plan["claim_plans"]],
            [
                "unseen_route_exploration",
                "unseen_route_exploration",
            ],
        )

    def test_adaptive_budget_uses_conservative_historical_yield(
        self,
    ):
        claims = [
            "The program caused higher sales.",
            "The program caused greater revenue.",
        ]
        nodes = {
            f"claim-{index}": {
                "node_id": f"claim-{index}",
                "text": claim,
            }
            for index, claim in enumerate(claims, start=1)
        }
        certificates = {
            node_id: {
                "node_id": node_id,
                "claim_relation_risk": {
                    "risk_level": "high",
                    "risk_score": 0.9,
                },
            }
            for node_id in nodes
        }
        low_yield_priors = [
            {
                "context": build_counterevidence_cost_context(
                    "claim",
                    [claim],
                ),
                "sample_count": 10,
                "mean_normalized_cost": 2.0,
                "mean_yield_score": 0.0,
            }
            for claim in claims
        ]
        plan = plan_adaptive_counterevidence_query_budget(
            list(nodes),
            nodes,
            certificates,
            hard_query_budget=4,
            historical_cost_priors=low_yield_priors,
        )
        self.assertEqual(plan["coverage_floor"], 2)
        self.assertEqual(plan["risk_extension_query_count"], 1)
        self.assertEqual(plan["effective_query_budget"], 3)
        self.assertTrue(
            all(
                row["yield_source"]
                == "historical_lower_confidence_bound"
                for row in plan["claim_plans"]
            )
        )

    def test_robust_cost_prior_uses_uncertainty_and_recency(
        self,
    ):
        prior = {
            "context": {
                "task_type": "claim",
                "claim_count": 1,
                "term_hashes": ["hash-a"],
            },
            "sample_count": 10,
            "mean_normalized_cost": 4.0,
            "mean_yield_score": 0.7,
            "cost_m2": 18.0,
            "yield_m2": 0.18,
            "updated_at": 1000,
        }
        robust = build_robust_counterevidence_cost_prior(
            prior,
            current_time=1000,
        )
        stale = build_robust_counterevidence_cost_prior(
            prior,
            current_time=1000 + 365 * 86400,
        )
        self.assertGreater(robust["cost_upper_bound"], 4.0)
        self.assertLess(robust["yield_lower_bound"], 0.7)
        self.assertIsNone(stale)

    def test_clusters_related_claims_and_builds_unique_anchors(
        self,
    ):
        claims, nodes, _transaction = (
            _joint_counterevidence_transaction()
        )
        clusters = cluster_counterevidence_claims(
            ["claim-1", "claim-2"],
            {row["node_id"]: row for row in nodes},
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            clusters[0]["node_ids"],
            ["claim-1", "claim-2"],
        )
        self.assertIn(
            "sale",
            clusters[0]["required_anchor_terms"]["claim-1"],
        )
        self.assertIn(
            "revenue",
            clusters[0]["required_anchor_terms"]["claim-2"],
        )
        query = build_fused_counterevidence_query(claims)
        self.assertIn("[C1]", query)
        self.assertIn("[C2]", query)

    def test_rank_source_excerpts_prefers_question_specific_passage(self):
        context = (
            "Passage 1:\nUnrelated information about a city.\n\n"
            "Passage 2:\nMary objected that the poem lacked human interest and affection."
        )
        rows = rank_source_excerpts("What was Mary's objection to the poem?", context)
        self.assertTrue(rows)
        self.assertIn("lacked human interest", rows[0]["text"])

    def test_rank_source_excerpts_includes_continuing_neighbor(self):
        context = (
            "The standard likelihood approach requires evaluation of the "
            + "model parameters and observed interactions. " * 20
            + "\n\n"
            "partition function, whose computation scales exponentially with system size.\n\n"
            "An unrelated conclusion."
        )
        rows = rank_source_excerpts(
            "What challenge affects the standard likelihood approach?",
            context,
        )
        self.assertTrue(any("partition function" in row["text"] for row in rows))
        self.assertTrue(
            any(row["selection_reason"].startswith("neighbor_of_") for row in rows)
        )

    def test_parse_json_code_fence(self):
        parsed = parse_realization_response(
            "```json\n" + json.dumps({"answer": "A direct answer.", "used_source_ids": ["S2"]}) + "\n```"
        )
        self.assertEqual(parsed["answer"], "A direct answer.")
        self.assertEqual(parsed["used_source_ids"], ["S2"])

    def test_audit_accepts_direct_supported_answer(self):
        audit = audit_realized_answer(
            "Why did the actor change?",
            "The actor changed because scheduling constraints prevented the original performer from continuing.",
            {"entities": [], "relationships": []},
            [{"source_id": "S1", "text": "Scheduling constraints prevented the original performer from continuing."}],
            used_evidence_ids=["S1"],
        )
        self.assertTrue(audit["accepted"])

    def test_audit_rejects_graph_dump(self):
        audit = audit_realized_answer(
            "What connects A and B?",
            "A -> B: They are connected by the film.",
            {"entities": [], "relationships": [{"src": "A", "tgt": "B", "description": "film connection"}]},
            [],
            used_evidence_ids=["G1"],
        )
        self.assertFalse(audit["accepted"])
        self.assertTrue(audit["evidence_dump"])

    def test_audit_rejects_unsupported_number(self):
        audit = audit_realized_answer(
            "How many events did the rider win?",
            "The rider won 36 events during the season.",
            {"entities": [], "relationships": []},
            [{"source_id": "S1", "text": "The rider won five events during the season."}],
            used_evidence_ids=["S1"],
        )
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["unsupported_numbers"], ["36"])

    def test_prohibited_claim_audit_detects_same_polarity_paraphrase(self):
        audit = audit_prohibited_claim_relapse(
            (
                "Available information does not provide details about his "
                "work as a journalist."
            ),
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                    "source_ids": ["S1"],
                }
            ],
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["relapse_count"], 1)
        self.assertEqual(audit["relapses"][0]["claim_id"], "claim-3")

    def test_correction_constraints_use_rejected_suffix(self):
        constraints = build_correction_persistence_constraints(
            {
                "accepted": True,
                "corrections": [
                    {
                        "claim_id": "claim-3",
                        "prohibited_assertion": (
                            "though the data does not describe his work."
                        ),
                        "original_preview": (
                            "His journalism was an early career phase, "
                            "though the data does not describe his work."
                        ),
                        "source_ids": ["S1"],
                    }
                ],
            }
        )
        self.assertEqual(len(constraints), 1)
        self.assertEqual(
            constraints[0]["assertion"],
            "though the data does not describe his work.",
        )
        self.assertEqual(constraints[0]["source_ids"], ["S1"])

    def test_correction_constraints_recover_suffix_from_v19_trace(self):
        constraints = build_correction_persistence_constraints(
            {
                "accepted": True,
                "corrections": [
                    {
                        "claim_id": "claim-3",
                        "action": "trim_diagnosed_meta_clause",
                        "original_preview": (
                            "His journalism was an early career phase, "
                            "though the data does not describe his work."
                        ),
                        "corrected_preview": (
                            "His journalism was an early career phase."
                        ),
                    }
                ],
            }
        )
        self.assertEqual(
            constraints[0]["assertion"],
            "though the data does not describe his work.",
        )

    def test_prohibited_claim_audit_allows_supported_opposite_polarity(self):
        audit = audit_prohibited_claim_relapse(
            (
                "The source provides specific details about his "
                "professional journalism work."
            ),
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                }
            ],
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["relapse_count"], 0)

    def test_lineage_audit_detects_distant_implicit_absence_rewrite(self):
        audit = audit_prohibited_claim_relapse(
            "His reporting career remains undocumented.",
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                }
            ],
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["relapses"][0]["match_mode"],
            "semantic_family",
        )
        self.assertEqual(
            audit["relapses"][0]["assertion_family"],
            "evidence_absence",
        )
        self.assertEqual(
            audit["relapses"][0]["anchor_overlap"],
            ["career", "journalist"],
        )

    def test_lineage_audit_detects_assertion_split_across_sentences(self):
        audit = audit_prohibited_claim_relapse(
            (
                "His career began in reporting. "
                "The records remain silent on the specifics."
            ),
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                }
            ],
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["relapses"][0]["match_mode"],
            "discourse_window",
        )
        self.assertEqual(audit["relapses"][0]["window_size"], 2)
        self.assertEqual(audit["cross_sentence_match_count"], 1)

    def test_lineage_audit_allows_unrelated_absence_assertion(self):
        audit = audit_prohibited_claim_relapse(
            "The records remain silent on her medical history.",
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                }
            ],
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["relapse_count"], 0)

    def test_lineage_audit_requires_complete_topic_anchor(self):
        audit = audit_prohibited_claim_relapse(
            (
                "The data does not contain specific details about his "
                "academic career trajectory."
            ),
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                }
            ],
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["relapse_count"], 0)

    def test_lineage_audit_preserves_generic_factual_negation_guard(self):
        assertion = (
            "Aurora does not use plasma shielding to prevent overheating."
        )
        audit = audit_prohibited_claim_relapse(
            assertion,
            [{"claim_id": "claim-1", "assertion": assertion}],
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["relapses"][0]["match_mode"],
            "lexical_overlap",
        )
        self.assertEqual(
            audit["relapses"][0]["assertion_family"],
            "generic",
        )

    def test_prohibited_claim_audit_scans_late_answer_claims(self):
        prefix = " ".join(
            f"Verified background fact number {index}."
            for index in range(13)
        )
        audit = audit_prohibited_claim_relapse(
            (
                f"{prefix} The records do not provide details about his "
                "professional journalism work."
            ),
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                }
            ],
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["relapse_count"], 1)

    def test_realized_answer_audit_rejects_corrected_claim_relapse(self):
        answer = (
            "The records do not provide specific details about his "
            "professional journalism work."
        )
        audit = audit_realized_answer(
            "What details exist about his journalism work?",
            answer,
            {"entities": [], "relationships": []},
            [{"source_id": "S1", "text": answer, "score": 0.9}],
            used_evidence_ids=["S1"],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                }
            ],
        )
        self.assertFalse(audit["accepted"])
        self.assertFalse(audit["correction_persistence"]["passed"])

    def test_realized_answer_audit_requires_bound_correction_source(self):
        answer = (
            "Adebanwi worked as a freelance reporter, writer, journalist "
            "and editor before joining academia."
        )
        audit = audit_realized_answer(
            "What journalism work did Adebanwi do?",
            answer,
            {"entities": [], "relationships": []},
            [
                {
                    "source_id": "correction-source",
                    "text": answer,
                    "score": 0.9,
                },
                {
                    "source_id": "ordinary-source",
                    "text": answer,
                    "score": 0.9,
                },
            ],
            used_evidence_ids=["ordinary-source"],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
        )
        self.assertFalse(audit["accepted"])
        self.assertTrue(audit["correction_persistence"]["passed"])
        self.assertFalse(
            audit["correction_source_certificates"]["passed"]
        )

    def test_realization_prompt_carries_corrected_claim_constraint(self):
        _system, prompt = build_realization_prompt(
            "What was his journalism work?",
            "He began his career in journalism.",
            {"entities": [], "relationships": []},
            [],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": "The data does not describe his journalism work.",
                }
            ],
        )
        self.assertIn("Previously corrected assertions", prompt)
        self.assertIn("[claim-3]", prompt)
        self.assertIn("Never repeat, paraphrase, or split", prompt)

    def test_event_ambiguity_requires_distinct_supported_events(self):
        question = (
            "What significant event involving Abd al-Muttalib and the Kaaba occurred, "
            "and how did it impact the region?"
        )
        evidence = {
            "entities": [
                {
                    "name": "ABD AL-MUTTALIB",
                    "description": "He rediscovered the Zamzam Well and defended the Kaaba.",
                }
            ],
            "relationships": [
                {
                    "src": "ABD AL-MUTTALIB",
                    "tgt": "YEAR OF THE ELEPHANT",
                    "description": "He defended the Kaaba during the Year of the Elephant.",
                }
            ],
        }
        obligations = build_evidence_obligations(question, evidence, [])
        self.assertEqual(len(obligations), 2)
        audit = audit_realized_answer(
            question,
            (
                "During the Year of the Elephant, Abd al-Muttalib defended the Kaaba "
                "from attack and preserved its religious standing."
            ),
            evidence,
            [],
            used_evidence_ids=["G1", "G2"],
            obligations=obligations,
        )
        self.assertFalse(audit["accepted"])
        self.assertTrue(audit["obligation_gate_required"])
        self.assertLess(audit["evidence_obligations"]["coverage_ratio"], 1.0)

    def test_event_detection_task_is_not_treated_as_event_ambiguity(self):
        self.assertFalse(
            question_requests_event(
                "What challenges does the human-AI loop address in event detection?"
            )
        )

    def test_prepare_retrieved_excerpts_preserves_runtime_source_ids(self):
        excerpts = prepare_retrieved_excerpts(
            [{"source_id": "chunk-abc", "content": "Grounded source text.", "score": 0.7}]
        )
        self.assertEqual(excerpts[0]["source_id"], "chunk-abc")
        self.assertEqual(excerpts[0]["text"], "Grounded source text.")

    def test_materializes_committed_correction_source_first(self):
        constraints = [
            {
                "claim_id": "claim-3",
                "assertion": "The data does not describe his work.",
                "source_ids": ["correction-source"],
            }
        ]
        chunks, audit = materialize_correction_source_chunks(
            [{"source_id": "ordinary-source", "text": "Other evidence."}],
            {
                "source_excerpts": [
                    {
                        "source_id": "correction-source",
                        "text": "Direct correction evidence.",
                    }
                ]
            },
            constraints,
        )
        self.assertEqual(chunks[0]["source_id"], "correction-source")
        self.assertEqual(audit["materialized_count"], 1)
        self.assertEqual(audit["missing_source_ids"], [])

    def test_materialization_refreshes_stale_same_id_chunk(self):
        chunks, audit = materialize_correction_source_chunks(
            [
                {
                    "source_id": "correction-source",
                    "text": "Long chunk prefix without the correction fact.",
                }
            ],
            {
                "source_excerpts": [
                    {
                        "source_id": "correction-source",
                        "text": "Focused committed correction evidence.",
                    }
                ]
            },
            [
                {
                    "assertion": "The data does not describe his work.",
                    "source_ids": ["correction-source"],
                }
            ],
        )
        self.assertEqual(
            chunks[0]["text"],
            "Focused committed correction evidence.",
        )
        self.assertEqual(audit["refreshed_existing_count"], 1)
        self.assertEqual(
            audit["refreshed_source_ids"],
            ["correction-source"],
        )

    def test_source_certificate_accepts_cited_correction_source(self):
        answer = (
            "Adebanwi worked as a freelance reporter, writer, journalist "
            "and editor before joining academia."
        )
        audit = audit_correction_source_certificates(
            answer,
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
            {"correction-source": answer},
            ["correction-source"],
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["certificate_required_count"], 1)
        self.assertEqual(audit["certified_count"], 1)
        self.assertEqual(
            audit["certificates"][0]["reason"],
            "source_bound_correction_certified",
        )

    def test_source_certificate_accepts_abstract_grounded_inversion(self):
        answer = (
            "His communications background represents his early career "
            "phase in journalism."
        )
        source = (
            "He worked as a reporter, journalist and editor before he "
            "joined academia."
        )
        audit = audit_correction_source_certificates(
            answer,
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
            {"correction-source": source},
            ["correction-source"],
        )
        self.assertTrue(audit["passed"])
        self.assertGreaterEqual(
            audit["certificates"][0]["best_source_support"][
                "support_score"
            ],
            0.60,
        )

    def test_source_certificate_rejects_unbound_general_citation(self):
        answer = (
            "Adebanwi worked as a freelance reporter, writer, journalist "
            "and editor before joining academia."
        )
        audit = audit_correction_source_certificates(
            answer,
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
            {
                "correction-source": answer,
                "ordinary-source": answer,
            },
            ["ordinary-source"],
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["uncertified_count"], 1)
        self.assertEqual(
            audit["certificates"][0]["reason"],
            "correction_source_not_cited",
        )

    def test_source_certificate_rejects_weak_same_source_support(self):
        answer = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        audit = audit_correction_source_certificates(
            answer,
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
            {
                "correction-source": (
                    "Adebanwi earned a degree in political science."
                )
            },
            ["correction-source"],
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["certificates"][0]["reason"],
            "correction_source_support_below_threshold",
        )

    def test_source_certificate_audits_all_opposite_claims(self):
        supported = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        unsupported = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        audit = audit_correction_source_certificates(
            f"{supported} {unsupported}",
            [
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
            {"correction-source": supported},
            ["correction-source"],
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["certificate_required_count"], 2)
        self.assertEqual(audit["certified_count"], 1)
        self.assertEqual(audit["uncertified_count"], 1)

    def test_transactional_rollback_replaces_only_unsafe_claim(self):
        safe_claim = (
            "Adebanwi taught political science after entering academia."
        )
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        constraints = [
            {
                "claim_id": "claim-3",
                "assertion": (
                    "The data does not provide details about his "
                    "professional journalism work."
                ),
                "source_ids": ["correction-source"],
            }
        ]
        generated = f"{safe_claim} {unsafe_claim}"
        generated_audit = {
            "correction_persistence": audit_prohibited_claim_relapse(
                generated,
                constraints,
            ),
            "correction_source_certificates": (
                audit_correction_source_certificates(
                    generated,
                    constraints,
                    {"correction-source": fallback_claim},
                    ["correction-source"],
                )
            ),
        }
        patched, result = apply_transactional_correction_rollback(
            generated,
            fallback_claim,
            constraints,
            generated_audit,
        )
        self.assertTrue(result["triggered"])
        self.assertTrue(result["generated_answer_changed"])
        self.assertEqual(result["replaced_span_count"], 1)
        self.assertEqual(result["preserved_generated_claim_count"], 1)
        self.assertIn(safe_claim, patched)
        self.assertIn(fallback_claim, patched)
        self.assertNotIn("Mars expedition", patched)

    def test_dependency_graph_detects_high_confidence_reference(self):
        graph = build_realization_dependency_graph(
            "How did Adebanwi move into academia?",
            (
                "Adebanwi worked as a journalist. "
                "This experience preceded his move into academia. "
                "He later taught political science."
            ),
            {
                "claims": [
                    {
                        "claim": "Adebanwi worked as a journalist.",
                        "support_score": 0.8,
                    }
                ]
            },
        )
        self.assertEqual(graph["node_count"], 3)
        self.assertEqual(graph["edge_count"], 1)
        self.assertEqual(
            graph["edges"][0]["dependency_type"],
            "anaphoric_reference",
        )
        self.assertEqual(
            graph["edges"][0]["parent_node_id"],
            "claim-1",
        )
        self.assertEqual(
            graph["edges"][0]["child_node_id"],
            "claim-2",
        )

    def test_audit_records_claim_supporting_evidence_ids(self):
        audit = audit_realized_answer(
            "How did Adebanwi move from journalism into academia?",
            (
                "Adebanwi worked as a journalist and editor. "
                "He later joined the University of Ibadan as a lecturer."
            ),
            {},
            [
                {
                    "source_id": "journalism-source",
                    "text": (
                        "Adebanwi worked as a journalist and editor "
                        "for newspapers."
                    ),
                    "score": 0.9,
                },
                {
                    "source_id": "academic-source",
                    "text": (
                        "Adebanwi joined the University of Ibadan "
                        "as a lecturer."
                    ),
                    "score": 0.9,
                },
            ],
            used_evidence_ids=[
                "journalism-source",
                "academic-source",
            ],
            supported_facets=[],
            facet_gate_mode="skip",
        )
        self.assertEqual(
            audit["claims"][0]["supporting_evidence_ids"],
            ["journalism-source"],
        )
        self.assertEqual(
            audit["claims"][1]["supporting_evidence_ids"],
            ["academic-source"],
        )

    def test_dependency_graph_resolves_distant_concept_antecedent(self):
        graph = build_realization_dependency_graph(
            "How did Adebanwi move from journalism into academia?",
            (
                "Adebanwi worked as a television journalist. "
                "The University of Ibadan is located in Nigeria. "
                "This reporting experience made Adebanwi famous."
            ),
            {
                "claims": [
                    {
                        "claim": (
                            "Adebanwi worked as a television "
                            "journalist."
                        ),
                        "support_score": 0.1,
                        "supporting_evidence_ids": [],
                    },
                    {
                        "claim": (
                            "The University of Ibadan is located "
                            "in Nigeria."
                        ),
                        "support_score": 0.8,
                        "supporting_evidence_ids": ["academic-source"],
                    },
                    {
                        "claim": (
                            "This reporting experience made "
                            "Adebanwi famous."
                        ),
                        "support_score": 0.1,
                        "supporting_evidence_ids": [],
                    },
                ]
            },
        )
        self.assertEqual(graph["distant_edge_count"], 1)
        self.assertEqual(
            graph["edges"][0]["parent_node_id"],
            "claim-1",
        )
        self.assertEqual(
            graph["edges"][0]["child_node_id"],
            "claim-3",
        )
        self.assertEqual(
            graph["edges"][0]["resolution_mode"],
            "evidence_conditioned_distant",
        )
        self.assertIn(
            "career",
            graph["edges"][0]["concept_overlap"],
        )
        self.assertTrue(
            graph["edges"][0]["counterfactual_stable"]
        )

    def test_dependency_graph_uses_shared_evidence_for_antecedent(self):
        graph = build_realization_dependency_graph(
            "What improved safety?",
            (
                "The pilot began in 2020. "
                "A separate committee met in Oslo. "
                "This result improved battery safety."
            ),
            {
                "claims": [
                    {
                        "claim": "The pilot began in 2020.",
                        "support_score": 0.8,
                        "supporting_evidence_ids": ["pilot-source"],
                    },
                    {
                        "claim": (
                            "A separate committee met in Oslo."
                        ),
                        "support_score": 0.8,
                        "supporting_evidence_ids": [
                            "committee-source"
                        ],
                    },
                    {
                        "claim": (
                            "This result improved battery safety."
                        ),
                        "support_score": 0.8,
                        "supporting_evidence_ids": ["pilot-source"],
                    },
                ]
            },
        )
        edge = graph["edges"][0]
        self.assertEqual(edge["parent_node_id"], "claim-1")
        self.assertEqual(edge["antecedent_distance"], 2)
        self.assertEqual(
            edge["shared_evidence_ids"],
            ["pilot-source"],
        )
        self.assertFalse(edge["counterfactual_stable"])
        self.assertEqual(edge["selection_margin"], 0.075)

    def test_bundle_optimizer_maximizes_dependency_closure(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A weak parent premise.",
                    "support_score": 0.1,
                    "supporting_evidence_ids": [],
                    "net_utility": -0.2,
                },
                {
                    "node_id": "claim-2",
                    "claim_index": 1,
                    "text": "This supported consequence matters.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["S1"],
                    "net_utility": 0.5,
                },
                {
                    "node_id": "claim-3",
                    "claim_index": 2,
                    "text": "Unrelated unsupported noise.",
                    "support_score": 0.0,
                    "supporting_evidence_ids": [],
                    "net_utility": -0.3,
                },
                {
                    "node_id": "claim-4",
                    "claim_index": 3,
                    "text": "The critical obligation detail.",
                    "support_score": 0.16,
                    "supporting_evidence_ids": ["S2"],
                    "net_utility": -0.05,
                },
            ],
            "edges": [
                {
                    "parent_node_id": "claim-1",
                    "child_node_id": "claim-2",
                }
            ],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {
                "claim-1",
                "claim-2",
                "claim-3",
                "claim-4",
            },
            obligations=[
                {
                    "obligation_id": "O1",
                    "evidence_id": "S2",
                    "required_terms": [
                        "critical",
                        "obligation",
                        "detail",
                    ],
                }
            ],
        )
        self.assertEqual(
            selection["selected_node_ids"],
            ["claim-1", "claim-2", "claim-4"],
        )
        self.assertEqual(
            selection["pruned_node_ids"],
            ["claim-3"],
        )
        self.assertEqual(
            selection["mandatory_node_ids"],
            ["claim-4"],
        )
        self.assertTrue(selection["coherence_passed"])

    def test_stability_certificate_prunes_boundary_selection(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A marginal supported detail.",
                    "support_score": 0.2,
                    "supporting_evidence_ids": ["S1"],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
            )
        )
        self.assertEqual(
            certificate["unstable_selected_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["stability_pruned_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            [],
        )
        self.assertTrue(certificate["passed"])

    def test_stability_certificate_protects_strong_source_claim(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A strongly verified detail.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["S1"],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
            )
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["stability_pruned_node_ids"],
            [],
        )

    def test_stability_certificate_protects_evidence_bound_obligation(
        self,
    ):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "The required source detail.",
                    "support_score": 0.2,
                    "supporting_evidence_ids": ["S1"],
                    "net_utility": -0.02,
                }
            ],
            "edges": [],
        }
        obligations = [
            {
                "obligation_id": "O1",
                "evidence_id": "S1",
                "required_terms": ["required", "source", "detail"],
            }
        ]
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
            obligations=obligations,
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                obligations=obligations,
            )
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            [],
        )
        self.assertEqual(
            certificate["stability_pruned_node_ids"],
            [],
        )
        self.assertEqual(
            certificate["stable_selected_node_ids"],
            ["claim-1"],
        )

    def test_stability_certificate_prunes_ambiguous_dependency(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A stable parent.",
                    "support_score": 0.3,
                    "supporting_evidence_ids": ["S1"],
                    "net_utility": 0.2,
                },
                {
                    "node_id": "claim-2",
                    "claim_index": 1,
                    "text": "This ambiguous consequence follows.",
                    "support_score": 0.3,
                    "supporting_evidence_ids": ["S1"],
                    "net_utility": 0.2,
                },
            ],
            "edges": [
                {
                    "parent_node_id": "claim-1",
                    "child_node_id": "claim-2",
                    "counterfactual_stable": False,
                }
            ],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1", "claim-2"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
            )
        )
        self.assertEqual(
            certificate["unstable_dependency_node_ids"],
            ["claim-2"],
        )
        self.assertEqual(
            certificate["stability_pruned_node_ids"],
            ["claim-2"],
        )

    def test_audit_records_graph_and_text_provenance(self):
        answer = "Dana directed the film school."
        audit = audit_realized_answer(
            "What did Dana direct?",
            answer,
            {
                "entities": [
                    {
                        "name": "Dana",
                        "description": "Dana directed the film school.",
                        "source_ids": ["graph-origin"],
                    }
                ],
                "relationships": [],
            },
            [
                {
                    "source_id": "text-evidence",
                    "origin_ids": ["document-origin"],
                    "text": answer,
                    "score": 0.9,
                }
            ],
            used_evidence_ids=["G1", "text-evidence"],
            supported_facets=[],
            facet_gate_mode="skip",
        )
        evidence_support = {
            row["evidence_id"]: row
            for row in audit["claims"][0]["evidence_support"]
        }
        self.assertEqual(
            evidence_support["G1"]["evidence_type"],
            "graph",
        )
        self.assertEqual(
            evidence_support["G1"]["origin_ids"],
            ["graph-origin"],
        )
        self.assertEqual(
            evidence_support["text-evidence"]["evidence_type"],
            "text",
        )
        self.assertEqual(
            evidence_support["text-evidence"]["origin_ids"],
            ["document-origin"],
        )

    def test_diversity_certificate_rejects_single_source(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "An unstable strong claim.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["S1"],
                    "evidence_support": [
                        {
                            "evidence_id": "S1",
                            "evidence_type": "text",
                            "origin_ids": ["doc-1"],
                            "support_score": 0.8,
                        }
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
            )
        )
        self.assertEqual(
            certificate["single_source_rejected_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            [],
        )
        self.assertEqual(
            certificate["stability_pruned_node_ids"],
            ["claim-1"],
        )

    def test_diversity_certificate_accepts_independent_sources(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "An independently verified claim.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["S1", "S2"],
                    "evidence_support": [
                        {
                            "evidence_id": "S1",
                            "evidence_type": "text",
                            "origin_ids": ["doc-1"],
                            "support_score": 0.8,
                        },
                        {
                            "evidence_id": "S2",
                            "evidence_type": "text",
                            "origin_ids": ["doc-2"],
                            "support_score": 0.8,
                        },
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
            )
        )
        self.assertEqual(
            certificate["independent_source_quorum_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["stability_pruned_node_ids"],
            [],
        )

    def test_diversity_certificate_rejects_duplicate_origin(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A duplicated-source claim.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["S1", "S2"],
                    "evidence_support": [
                        {
                            "evidence_id": evidence_id,
                            "evidence_type": "text",
                            "origin_ids": ["same-document"],
                            "support_score": 0.8,
                        }
                        for evidence_id in ["S1", "S2"]
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
            )
        )
        self.assertEqual(
            certificate["independent_source_quorum_node_ids"],
            [],
        )
        self.assertEqual(
            certificate["single_source_rejected_node_ids"],
            ["claim-1"],
        )

    def test_diversity_certificate_accepts_graph_text_agreement(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A graph and text verified claim.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["G1", "S1"],
                    "evidence_support": [
                        {
                            "evidence_id": "G1",
                            "evidence_type": "graph",
                            "origin_ids": ["doc-1"],
                            "support_score": 0.8,
                        },
                        {
                            "evidence_id": "S1",
                            "evidence_type": "text",
                            "origin_ids": ["doc-1"],
                            "support_score": 0.8,
                        },
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
            )
        )
        self.assertEqual(
            certificate["graph_text_agreement_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            ["claim-1"],
        )

    def test_transactional_removal_cascades_dangling_dependency(self):
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        dependent_claim = (
            "This achievement made him internationally famous."
        )
        independent_claim = (
            "Adebanwi later taught political science."
        )
        constraints = [
            {
                "claim_id": "claim-3",
                "assertion": (
                    "The data does not provide details about his "
                    "professional journalism work."
                ),
                "source_ids": ["correction-source"],
            }
        ]
        generated = (
            f"{fallback_claim} {unsafe_claim} "
            f"{dependent_claim} {independent_claim}"
        )
        generated_audit = {
            "claims": [],
            "correction_persistence": audit_prohibited_claim_relapse(
                generated,
                constraints,
            ),
            "correction_source_certificates": (
                audit_correction_source_certificates(
                    generated,
                    constraints,
                    {"correction-source": fallback_claim},
                    ["correction-source"],
                )
            ),
        }
        patched, result = apply_transactional_correction_rollback(
            generated,
            fallback_claim,
            constraints,
            generated_audit,
        )
        self.assertNotIn("Mars expedition", patched)
        self.assertNotIn("internationally famous", patched)
        self.assertIn(independent_claim, patched)
        self.assertEqual(
            result["dependency_removed_span_count"],
            1,
        )
        self.assertEqual(
            result["dependency_graph"]["cascade_removed_node_ids"],
            ["claim-3"],
        )
        self.assertTrue(
            result["dependency_graph"]["coherence_passed"]
        )

    def test_transactional_replacement_rebinds_dependency(self):
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        dependent_claim = (
            "This experience preceded his move into academia."
        )
        constraints = [
            {
                "claim_id": "claim-3",
                "assertion": (
                    "The data does not provide details about his "
                    "professional journalism work."
                ),
                "source_ids": ["correction-source"],
            }
        ]
        generated = f"{unsafe_claim} {dependent_claim}"
        generated_audit = {
            "claims": [],
            "correction_persistence": audit_prohibited_claim_relapse(
                generated,
                constraints,
            ),
            "correction_source_certificates": (
                audit_correction_source_certificates(
                    generated,
                    constraints,
                    {"correction-source": fallback_claim},
                    ["correction-source"],
                )
            ),
        }
        patched, result = apply_transactional_correction_rollback(
            generated,
            fallback_claim,
            constraints,
            generated_audit,
        )
        self.assertIn(fallback_claim, patched)
        self.assertIn(dependent_claim, patched)
        self.assertEqual(result["dependency_rebound_count"], 1)
        self.assertEqual(
            result["dependency_graph"]["rebound_node_ids"],
            ["claim-2"],
        )
        self.assertTrue(
            result["dependency_graph"]["coherence_passed"]
        )

    def test_transactional_removal_cascades_multihop_causal_chain(self):
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        generated = (
            f"{fallback_claim} {unsafe_claim} "
            "This achievement made him famous. "
            "Therefore, he received an interplanetary award."
        )
        constraints = [
            {
                "claim_id": "claim-3",
                "assertion": (
                    "The data does not provide details about his "
                    "professional journalism work."
                ),
                "source_ids": ["correction-source"],
            }
        ]
        generated_audit = {
            "claims": [],
            "correction_persistence": audit_prohibited_claim_relapse(
                generated,
                constraints,
            ),
            "correction_source_certificates": (
                audit_correction_source_certificates(
                    generated,
                    constraints,
                    {"correction-source": fallback_claim},
                    ["correction-source"],
                )
            ),
        }
        patched, result = apply_transactional_correction_rollback(
            generated,
            fallback_claim,
            constraints,
            generated_audit,
        )
        self.assertNotIn("famous", patched)
        self.assertNotIn("interplanetary award", patched)
        self.assertEqual(
            result["dependency_removed_span_count"],
            2,
        )
        self.assertEqual(
            result["dependency_graph"]["cascade_removed_node_ids"],
            ["claim-3", "claim-4"],
        )

    def test_source_certificate_is_not_required_without_inversion(self):
        assertion = (
            "The data does not provide details about his professional "
            "journalism work."
        )
        audit = audit_correction_source_certificates(
            assertion,
            [
                {
                    "claim_id": "claim-3",
                    "assertion": assertion,
                    "source_ids": ["correction-source"],
                }
            ],
            {"correction-source": "Grounded source."},
            ["correction-source"],
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["certificate_required_count"], 0)

    def test_supported_facets_keep_coordinated_objections_separate(self):
        question = 'What is Mary\'s objection to "The Witch of Atlas"?'
        excerpts = [
            {
                "source_id": "S1",
                "text": (
                    "Mary condemned the poem because it contained no human interest "
                    "and did not tell a story."
                ),
                "score": 0.9,
            }
        ]
        facets = build_supported_facets(question, {}, excerpts)
        self.assertGreaterEqual(len(facets), 2)
        incomplete = audit_supported_facet_retention(
            "Mary objected that the poem contained no human interest.",
            facets,
        )
        complete = audit_supported_facet_retention(
            (
                "Mary objected that the poem contained no human interest "
                "and did not tell a story."
            ),
            facets,
        )
        self.assertLess(incomplete["coverage_ratio"], 1.0)
        self.assertEqual(complete["coverage_ratio"], 1.0)

    def test_supported_facet_audit_accepts_light_morphology_changes(self):
        question = 'What is Mary\'s objection to "The Witch of Atlas"?'
        excerpts = [
            {
                "source_id": "S1",
                "text": (
                    "Mary condemned the poem because it contained no human interest "
                    "and did not tell a story."
                ),
                "score": 0.9,
            }
        ]
        facets = build_supported_facets(question, {}, excerpts)
        audit = audit_supported_facet_retention(
            (
                "Mary's objection is that the poem contains no human interest; "
                "she condemns it because it tells no story."
            ),
            facets,
        )
        self.assertEqual(audit["coverage_ratio"], 1.0)

    def test_supported_facet_gate_ignores_non_reason_questions(self):
        facets = build_supported_facets(
            "Where was the agreement signed?",
            {},
            [
                {
                    "source_id": "S1",
                    "text": (
                        "The agreement was signed in Paris after months of negotiation "
                        "and later revised in London."
                    ),
                    "score": 0.9,
                }
            ],
        )
        self.assertEqual(facets, [])

    def test_facet_query_planner_activates_explicit_multi_aspect_question(self):
        plan = classify_supported_facet_query(
            "What are Dana Blankstein-Cohen's contributions to film education?"
        )
        self.assertTrue(plan["enabled"])
        self.assertEqual(plan["query_family"], "contribution")
        self.assertEqual(plan["activation_reason"], "explicit_plural_request")

    def test_generalized_facet_plan_keeps_candidate_supported_contributions(self):
        question = (
            "What are Dana Blankstein-Cohen's contributions to film and "
            "television education?"
        )
        candidate = (
            "Dana Blankstein-Cohen founded a municipal film department and "
            "was appointed director of the Sam Spiegel Film School."
        )
        excerpts = [
            {
                "source_id": "S1",
                "text": (
                    "Dana Blankstein-Cohen founded and directed the film department "
                    "at the Kfar Saba municipality. Dana Blankstein-Cohen was appointed "
                    "director of the Sam Spiegel Film and Television School."
                ),
                "score": 0.9,
            },
            {
                "source_id": "S2",
                "text": "Levin directed many episodic television productions.",
                "score": 0.8,
            },
        ]
        plan = build_supported_facet_plan(
            question,
            {},
            excerpts,
            candidate_answer=candidate,
        )
        self.assertEqual(plan["stop_reason"], "high_confidence_facets_planned")
        self.assertGreaterEqual(plan["facet_count"], 2)
        self.assertTrue(all(row["evidence_id"] == "S1" for row in plan["facets"]))
        self.assertTrue(all(row["confidence"] >= 0.68 for row in plan["facets"]))

    def test_generalized_facet_plan_abstains_without_candidate_support(self):
        plan = build_supported_facet_plan(
            "What are the key findings from the regression analysis?",
            {},
            [
                {
                    "source_id": "S1",
                    "text": (
                        "The analysis found that short answers benefit most from "
                        "pre-training."
                    ),
                    "score": 0.9,
                }
            ],
            candidate_answer="The graph contains dataset and model entities.",
        )
        self.assertTrue(plan["enabled"])
        self.assertEqual(plan["facets"], [])
        self.assertEqual(plan["stop_reason"], "no_high_confidence_supported_facets")

    def test_facet_activation_policy_enforces_high_omission_risk(self):
        policy = build_facet_activation_policy(
            "The graph evidence directly supports several unrelated entities.",
            {
                "expects_multiple": False,
                "facets": [
                    {
                        "required_terms": ["human", "interest"],
                        "confidence": 0.9,
                    },
                    {
                        "required_terms": ["tell", "story"],
                        "confidence": 0.9,
                    },
                ],
            },
        )
        self.assertEqual(policy["action"], "enforce")
        self.assertTrue(policy["hard_gate_required"])
        self.assertGreaterEqual(policy["omission_risk"], 0.68)

    def test_facet_activation_policy_monitors_covered_multi_aspect_fact(self):
        policy = build_facet_activation_policy(
            (
                "The graph evidence directly supports this facet:\n"
                "- Dana was appointed director of the Sam Spiegel School."
            ),
            {
                "expects_multiple": True,
                "facets": [
                    {
                        "required_terms": ["director", "spiegel", "school"],
                        "confidence": 0.9,
                    }
                ],
            },
        )
        self.assertEqual(policy["action"], "monitor")
        self.assertFalse(policy["hard_gate_required"])

    def test_facet_activation_policy_skips_empty_plan(self):
        policy = build_facet_activation_policy(
            "A concise candidate answer.",
            {"expects_multiple": False, "facets": []},
        )
        self.assertEqual(policy["action"], "skip")
        self.assertEqual(policy["omission_risk"], 0.0)
        self.assertFalse(policy["hard_gate_required"])

    def test_facet_gate_monitor_does_not_force_extra_repair(self):
        question = "What contributions did Dana make?"
        answer = "Dana directed the film school."
        evidence = [
            {
                "source_id": "S1",
                "text": (
                    "Dana directed the film school and launched an Arabic-language "
                    "preparatory program."
                ),
                "score": 0.9,
            }
        ]
        facets = [
            {
                "facet_id": "F1",
                "evidence_id": "S1",
                "text": "Dana launched an Arabic-language preparatory program.",
                "required_terms": ["arabic-language", "preparatory", "program"],
                "confidence": 0.9,
            }
        ]
        monitored = audit_realized_answer(
            question,
            answer,
            {},
            evidence,
            used_evidence_ids=["S1"],
            supported_facets=facets,
            facet_gate_mode="monitor",
        )
        enforced = audit_realized_answer(
            question,
            answer,
            {},
            evidence,
            used_evidence_ids=["S1"],
            supported_facets=facets,
            facet_gate_mode="enforce",
        )
        self.assertTrue(monitored["accepted"])
        self.assertFalse(monitored["facet_gate_required"])
        self.assertFalse(enforced["accepted"])
        self.assertTrue(enforced["facet_gate_required"])


    def test_audit_resolves_graph_and_text_to_same_document_family(
        self,
    ):
        claim = "Dana directed the film school."
        audit = audit_realized_answer(
            "What did Dana direct?",
            claim,
            {
                "entities": [
                    {
                        "name": "Dana",
                        "description": claim,
                        "source_ids": ["chunk-1"],
                    }
                ],
                "relationships": [],
            },
            [
                {
                    "source_id": "chunk-1",
                    "document_id": "document-root",
                    "text": claim,
                    "score": 0.9,
                }
            ],
            used_evidence_ids=["G1", "chunk-1"],
            supported_facets=[],
            facet_gate_mode="skip",
        )
        support = {
            row["evidence_id"]: row
            for row in audit["claims"][0]["evidence_support"]
        }
        self.assertIn(
            "document-root",
            support["G1"]["family_ids"],
        )
        self.assertIn(
            "document-root",
            support["chunk-1"]["family_ids"],
        )

    def test_lineage_certificate_collapses_derived_chunks(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A derived-chunk claim.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["S1", "S2"],
                    "evidence_support": [
                        {
                            "evidence_id": evidence_id,
                            "evidence_type": "text",
                            "origin_ids": [evidence_id],
                            "family_ids": [
                                evidence_id,
                                "shared-document",
                            ],
                            "stance": "supports",
                            "support_score": 0.8,
                        }
                        for evidence_id in ["S1", "S2"]
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
                require_lineage_independence=True,
            )
        )
        self.assertEqual(
            certificate["correlated_quorum_rejected_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["correlated_evidence_collapsed_count"],
            1,
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            [],
        )

    def test_lineage_certificate_rejects_same_origin_graph_text(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A graph-derived claim.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["G1", "S1"],
                    "evidence_support": [
                        {
                            "evidence_id": "G1",
                            "evidence_type": "graph",
                            "family_ids": ["document-root"],
                            "stance": "supports",
                            "support_score": 0.8,
                        },
                        {
                            "evidence_id": "S1",
                            "evidence_type": "text",
                            "family_ids": ["document-root"],
                            "stance": "supports",
                            "support_score": 0.8,
                        },
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
                require_lineage_independence=True,
            )
        )
        self.assertEqual(
            certificate["graph_text_agreement_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["lineage_family_quorum_node_ids"],
            [],
        )
        self.assertEqual(
            certificate["stability_pruned_node_ids"],
            ["claim-1"],
        )

    def test_lineage_certificate_accepts_independent_families(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "An independently verified claim.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["S1", "S2"],
                    "evidence_support": [
                        {
                            "evidence_id": f"S{index}",
                            "evidence_type": "text",
                            "family_ids": [f"document-{index}"],
                            "stance": "supports",
                            "support_score": 0.8,
                        }
                        for index in [1, 2]
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
                require_lineage_independence=True,
            )
        )
        self.assertEqual(
            certificate["lineage_family_quorum_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            ["claim-1"],
        )

    def test_lineage_certificate_rejects_conflicting_family(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "A disputed claim.",
                    "support_score": 0.8,
                    "supporting_evidence_ids": ["S1", "S2", "S3"],
                    "evidence_support": [
                        {
                            "evidence_id": f"S{index}",
                            "evidence_type": "text",
                            "family_ids": [f"document-{index}"],
                            "stance": (
                                "contradicts"
                                if index == 3
                                else "supports"
                            ),
                            "support_score": 0.8,
                        }
                        for index in [1, 2, 3]
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
                require_lineage_independence=True,
            )
        )
        self.assertEqual(
            certificate[
                "conflicting_support_rejected_node_ids"
            ],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            [],
        )

    def test_semantic_relation_classifies_entailment_neutral_and_conflict(
        self,
    ):
        claim = "Dana directed the film school."
        self.assertEqual(
            classify_claim_evidence_relation(
                claim,
                claim,
            )["relation"],
            "entails",
        )
        self.assertEqual(
            classify_claim_evidence_relation(
                claim,
                "Dana attended the film school.",
            )["relation"],
            "neutral",
        )
        self.assertEqual(
            classify_claim_evidence_relation(
                claim,
                "Dana did not direct the film school.",
            )["relation"],
            "contradicts",
        )
        self.assertTrue(
            classify_claim_evidence_relation(
                "Dana joined the university in 2009.",
                "Dana joined the university in 2019.",
            )["numeric_conflict"]
        )
        self.assertTrue(
            classify_claim_evidence_relation(
                "Dana joined the university.",
                "Dana left the university.",
            )["predicate_conflict"]
        )

    def test_audit_records_neutral_relation_and_reliability(self):
        claim = "Dana directed the film school."
        audit = audit_realized_answer(
            "What did Dana direct?",
            claim,
            {"entities": [], "relationships": []},
            [
                {
                    "source_id": "S1",
                    "document_id": "document-1",
                    "source_reliability": 0.91,
                    "text": "Dana attended the film school.",
                    "score": 0.9,
                }
            ],
            used_evidence_ids=["S1"],
            supported_facets=[],
            facet_gate_mode="skip",
        )
        relation = audit["claims"][0]["evidence_support"][0]
        self.assertEqual(
            relation["semantic_relation"],
            "neutral",
        )
        self.assertEqual(
            relation["source_reliability"],
            0.91,
        )

    def test_semantic_certificate_rejects_neutral_overlap(self):
        graph = {
            "nodes": [
                {
                    "node_id": "claim-1",
                    "claim_index": 0,
                    "text": "Dana directed the film school.",
                    "support_score": 0.9,
                    "supporting_evidence_ids": ["S1", "S2"],
                    "evidence_support": [
                        {
                            "evidence_id": "S1",
                            "evidence_type": "text",
                            "family_ids": ["document-1"],
                            "semantic_relation": "entails",
                            "relation_confidence": 0.9,
                            "source_reliability": 0.8,
                            "support_score": 0.9,
                        },
                        {
                            "evidence_id": "S2",
                            "evidence_type": "text",
                            "family_ids": ["document-2"],
                            "semantic_relation": "neutral",
                            "relation_confidence": 0.8,
                            "source_reliability": 0.8,
                            "support_score": 0.8,
                        },
                    ],
                    "net_utility": 0.03,
                }
            ],
            "edges": [],
        }
        selection = select_obligation_aware_claim_bundle(
            graph,
            {"claim-1"},
        )
        certificate = (
            build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
                require_lineage_independence=True,
                require_semantic_relation_calibration=True,
            )
        )
        self.assertEqual(
            certificate["neutral_relation_rejected_node_ids"],
            ["claim-1"],
        )
        self.assertEqual(
            certificate["fallback_protected_node_ids"],
            [],
        )

    def test_semantic_certificate_adapts_to_high_claim_risk(self):
        self.assertEqual(
            estimate_claim_relation_risk(
                "Dana joined the university in 2009."
            )["risk_level"],
            "high",
        )

        def certificate(reliability):
            graph = {
                "nodes": [
                    {
                        "node_id": "claim-1",
                        "claim_index": 0,
                        "text": (
                            "Dana joined the university in 2009."
                        ),
                        "support_score": 0.9,
                        "supporting_evidence_ids": ["S1", "S2"],
                        "evidence_support": [
                            {
                                "evidence_id": f"S{index}",
                                "evidence_type": "text",
                                "family_ids": [
                                    f"document-{index}"
                                ],
                                "semantic_relation": "entails",
                                "relation_confidence": 1.0,
                                "source_reliability": reliability,
                                "support_score": 0.9,
                            }
                            for index in [1, 2]
                        ],
                        "net_utility": 0.03,
                    }
                ],
                "edges": [],
            }
            selection = select_obligation_aware_claim_bundle(
                graph,
                {"claim-1"},
            )
            return build_counterfactual_bundle_stability_certificate(
                graph,
                selection,
                require_provenance_diversity=True,
                require_lineage_independence=True,
                require_semantic_relation_calibration=True,
            )

        default_sources = certificate(0.72)
        authoritative_sources = certificate(0.90)
        self.assertEqual(
            default_sources[
                "high_risk_quorum_rejected_node_ids"
            ],
            ["claim-1"],
        )
        self.assertEqual(
            default_sources["fallback_protected_node_ids"],
            [],
        )
        self.assertEqual(
            authoritative_sources[
                "semantic_family_quorum_node_ids"
            ],
            ["claim-1"],
        )
        self.assertEqual(
            authoritative_sources["fallback_protected_node_ids"],
            ["claim-1"],
        )

    def test_counterevidence_queries_are_bounded_and_claim_targeted(
        self,
    ):
        queries = build_counterevidence_queries(
            "Dana joined the university in 2009.",
            "When did Dana join?",
            max_queries=2,
        )
        self.assertEqual(len(queries), 2)
        self.assertTrue(
            all("Dana joined" in query for query in queries)
        )

    def test_counterevidence_query_value_supports_early_stopping(
        self,
    ):
        risk = {"risk_score": 0.46}
        initial = estimate_counterevidence_query_value(
            risk,
            0,
        )
        decisive = estimate_counterevidence_query_value(
            risk,
            1,
            contradiction_found=True,
        )
        diminishing = estimate_counterevidence_query_value(
            risk,
            2,
            previous_new_unique_count=0,
        )
        self.assertTrue(initial["execute"])
        self.assertFalse(decisive["execute"])
        self.assertEqual(
            decisive["reason"],
            "decisive_contradiction",
        )
        self.assertFalse(diminishing["execute"])


class AnswerRealizationAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_cost_admission_blocks_incomplete_profile_update(
        self,
    ):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )

        async def incomplete_search(_query, _max_results):
            return {
                "rows": [],
                "latency_ms": 10,
                "retrieval_adapter": "local_scan",
                "usage_status": "complete_local_retrieval",
                "usage": {"call_count": 0},
            }

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                claims[0],
                transaction,
                incomplete_search,
                max_claims=1,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_observed_cost_control=True,
                require_complete_retrieval_usage_contract=True,
                enable_failure_aware_cost_admission=True,
                max_global_queries=1,
            )
        )
        self.assertEqual(challenge["cost_memory_rejected_count"], 1)
        self.assertEqual(challenge["observed_cost_rows"], [])
        self.assertEqual(challenge["observed_cost_profiles"], {})
        self.assertEqual(
            challenge["counterevidence_cost_memory_updates"],
            [],
        )

    async def test_failure_aware_admission_rejects_exception_feedback(
        self,
    ):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )

        async def failing_search(_query, _max_results):
            raise RuntimeError("temporary retrieval outage")

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                claims[0],
                transaction,
                failing_search,
                max_claims=1,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_information_posterior_control=True,
                enable_family_normalized_novelty=True,
                enable_online_budget_reallocation=True,
                enable_failure_aware_information_admission=True,
                historical_information_priors=[],
                max_global_queries=1,
            )
        )
        self.assertEqual(
            challenge["information_memory_rejected_count"],
            1,
        )
        self.assertEqual(
            challenge["counterevidence_information_memory_updates"],
            [],
        )
        self.assertEqual(
            challenge["information_memory_admission_audit"][0][
                "reason"
            ],
            "retrieval_exception",
        )
        self.assertEqual(
            challenge["online_budget_adjustment_audit"][0][
                "action"
            ],
            "hold",
        )

    async def test_failure_aware_admission_learns_complete_empty_result(
        self,
    ):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )

        async def complete_empty_search(_query, _max_results):
            return {
                "rows": [],
                "latency_ms": 10,
                "usage": {"call_count": 0, "total_tokens": 0},
                "usage_status": "complete_local_retrieval",
            }

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                claims[0],
                transaction,
                complete_empty_search,
                max_claims=1,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_information_posterior_control=True,
                enable_family_normalized_novelty=True,
                enable_failure_aware_information_admission=True,
                historical_information_priors=[],
                max_global_queries=1,
            )
        )
        self.assertEqual(
            challenge["information_memory_admitted_count"],
            1,
        )
        self.assertEqual(
            len(
                challenge[
                    "counterevidence_information_memory_updates"
                ]
            ),
            1,
        )
        self.assertEqual(
            challenge["information_memory_admission_audit"][0][
                "reason"
            ],
            "complete_zero_yield_observation",
        )

    async def test_historical_information_prior_reprioritizes_route(
        self,
    ):
        claims = [
            "The campaign always drew wider attendance.",
            "The program caused 20 percent higher revenue.",
        ]
        _base_claims, nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        for index, node in enumerate(nodes):
            node["text"] = claims[index]
        certificates = transaction[
            "counterfactual_stability_certificate"
        ]["provenance_certificates"]
        certificates[0]["claim_relation_risk"] = {
            "risk_level": "medium",
            "risk_score": 0.0,
        }
        certificates[1]["claim_relation_risk"] = {
            "risk_level": "high",
            "risk_score": 0.9,
        }
        prior = {
            "context": build_counterevidence_cost_context(
                "claim",
                [claims[1]],
            ),
            "sample_count": 20,
            "mean_realized_information_value": 0.05,
            "information_m2": 0.0,
            "low_value_streak": 8,
        }
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            return []

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_information_posterior_control=True,
                enable_family_normalized_novelty=True,
                historical_information_priors=[prior],
                max_global_queries=1,
            )
        )
        self.assertIn("campaign", queries[0].lower())
        self.assertEqual(
            challenge[
                "historical_information_prior_application_count"
            ],
            1,
        )
        first = challenge["information_selection_audit"][0]
        self.assertEqual(
            first["information_profile_source"],
            "unseen_claim",
        )
        self.assertTrue(first["reprioritized"])

    async def test_family_normalized_novelty_collapses_duplicate_chunks(
        self,
    ):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )

        async def search_func(_query, _max_results):
            return {
                "rows": [
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
                ],
                "latency_ms": 10,
                "usage": {"call_count": 0, "total_tokens": 0},
            }

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                claims[0],
                transaction,
                search_func,
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
                max_global_queries=1,
            )
        )
        observed = challenge["observed_cost_rows"][0]
        self.assertEqual(observed["new_unique_count"], 2)
        self.assertEqual(
            observed["new_independent_family_count"],
            1,
        )
        self.assertEqual(observed["source_unique_yield"], 1.0)
        self.assertEqual(observed["family_unique_yield"], 0.5)
        self.assertEqual(
            observed["novelty_source"],
            "independent_provenance_family",
        )
        self.assertEqual(
            challenge["observed_independent_family_count"],
            1,
        )

    async def test_family_normalized_novelty_preserves_independent_docs(
        self,
    ):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )

        async def search_func(_query, _max_results):
            return [
                {
                    "source_id": "chunk-1",
                    "document_id": "document-1",
                    "text": "A weather archive recorded rainfall.",
                },
                {
                    "source_id": "chunk-2",
                    "document_id": "document-2",
                    "text": "A climate archive recorded snowfall.",
                },
            ]

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                claims[0],
                transaction,
                search_func,
                max_claims=1,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_observed_cost_control=True,
                enable_family_normalized_novelty=True,
                max_global_queries=1,
            )
        )
        observed = challenge["observed_cost_rows"][0]
        self.assertEqual(
            observed["new_independent_family_count"],
            2,
        )
        self.assertEqual(observed["family_unique_yield"], 1.0)

    async def test_information_posterior_reprioritizes_low_value_route(
        self,
    ):
        claims = [
            "The campaign always drew wider attendance.",
            "The program caused 20 percent higher revenue.",
        ]
        _base_claims, nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        for index, node in enumerate(nodes):
            node["text"] = claims[index]
        certificates = transaction[
            "counterfactual_stability_certificate"
        ]["provenance_certificates"]
        certificates[0]["claim_relation_risk"] = {
            "risk_level": "medium",
            "risk_score": 0.0,
        }
        certificates[1]["claim_relation_risk"] = {
            "risk_level": "high",
            "risk_score": 0.9,
        }
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            return [
                {
                    "source_id": f"irrelevant-{len(queries)}",
                    "document_id": "weather-record",
                    "text": "A weather archive recorded rainfall.",
                    "source_reliability": 0.9,
                }
            ]

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_risk_coverage_guard=True,
                enable_adaptive_query_budget=True,
                enable_information_posterior_control=True,
                max_queries_per_claim=3,
                max_global_queries=3,
            )
        )
        self.assertIn("20 percent", queries[0])
        self.assertIn("campaign", queries[1].lower())
        self.assertGreaterEqual(
            challenge["information_reprioritization_count"],
            1,
        )
        second = challenge["information_selection_audit"][1]
        self.assertEqual(
            second["cost_selected_node_id"],
            "claim-2",
        )
        self.assertEqual(
            second["information_selected_node_id"],
            "claim-1",
        )
        self.assertTrue(second["reprioritized"])
        self.assertEqual(
            challenge["coverage_reserve_intervention_count"],
            0,
        )

    async def test_online_budget_expands_for_relevant_unresolved_routes(
        self,
    ):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            claim = (
                claims[0]
                if "higher sales" in query
                else claims[1]
            )
            return [
                {
                    "source_id": f"relevant-{len(queries)}",
                    "document_id": f"document-{len(queries)}",
                    "text": claim,
                    "source_reliability": 0.5,
                }
            ]

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_risk_coverage_guard=True,
                enable_adaptive_query_budget=True,
                enable_online_budget_reallocation=True,
                max_queries_per_claim=3,
                max_global_queries=4,
            )
        )
        self.assertEqual(
            challenge["initial_effective_global_query_budget"],
            2,
        )
        self.assertGreaterEqual(
            challenge["online_budget_release_count"],
            1,
        )
        self.assertGreater(
            challenge["effective_global_query_budget"],
            challenge["initial_effective_global_query_budget"],
        )
        self.assertEqual(
            challenge["covered_risky_claim_count"],
            2,
        )

    async def test_online_budget_revokes_low_value_extension(
        self,
    ):
        claims = [
            "The campaign always drew wider attendance.",
            "The program caused 20 percent higher revenue.",
        ]
        _base_claims, nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        for index, node in enumerate(nodes):
            node["text"] = claims[index]
        certificates = transaction[
            "counterfactual_stability_certificate"
        ]["provenance_certificates"]
        for certificate in certificates:
            certificate["claim_relation_risk"] = {
                "risk_level": "high",
                "risk_score": 0.9,
            }

        async def search_func(query, _max_results):
            return [
                {
                    "source_id": str(hash(query)),
                    "document_id": "weather-record",
                    "text": "A weather archive recorded rainfall.",
                    "source_reliability": 0.9,
                }
            ]

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_risk_coverage_guard=True,
                enable_adaptive_query_budget=True,
                enable_online_budget_reallocation=True,
                max_queries_per_claim=3,
                max_global_queries=4,
            )
        )
        self.assertEqual(
            challenge["initial_effective_global_query_budget"],
            4,
        )
        self.assertEqual(
            challenge["online_budget_revocation_count"],
            2,
        )
        self.assertEqual(
            challenge["effective_global_query_budget"],
            2,
        )
        self.assertEqual(challenge["online_budget_saved_count"], 2)
        self.assertEqual(
            challenge["covered_risky_claim_count"],
            2,
        )

    async def test_risk_coverage_guard_prevents_budget_starvation(
        self,
    ):
        claims = [
            "The campaign always drew wider attendance.",
            "The program caused 20 percent higher revenue.",
        ]
        _base_claims, nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        for index, node in enumerate(nodes):
            node["text"] = claims[index]
        certificates = transaction[
            "counterfactual_stability_certificate"
        ]["provenance_certificates"]
        certificates[0]["claim_relation_risk"] = {
            "risk_level": "medium",
            "risk_score": 0.0,
        }
        certificates[1]["claim_relation_risk"] = {
            "risk_level": "high",
            "risk_score": 0.9,
        }
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            return [
                {
                    "source_id": f"irrelevant-{len(queries)}",
                    "document_id": "weather-record",
                    "text": "A weather archive recorded rainfall.",
                    "source_reliability": 0.9,
                }
            ]

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_risk_coverage_guard=True,
                max_queries_per_claim=3,
                max_global_queries=2,
            )
        )

        self.assertIn("20 percent", queries[0])
        self.assertIn("campaign", queries[1].lower())
        self.assertEqual(
            challenge["coverage_reserve_intervention_count"],
            1,
        )
        self.assertEqual(
            challenge["covered_risky_claim_count"],
            2,
        )
        self.assertEqual(
            challenge["uncovered_risky_claim_count"],
            0,
        )
        second_round = challenge[
            "coverage_reserve_selection_audit"
        ][1]
        self.assertEqual(
            second_round["normal_selected_node_id"],
            "claim-2",
        )
        self.assertEqual(
            second_round["selected_node_id"],
            "claim-1",
        )
        self.assertTrue(second_round["intervened"])

    async def test_fused_query_counts_all_members_as_covered(
        self,
    ):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )

        async def search_func(_query, _max_results):
            return []

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_query_fusion=True,
                enable_risk_coverage_guard=True,
                max_queries_per_claim=3,
                max_global_queries=1,
            )
        )

        self.assertEqual(challenge["fused_query_count"], 1)
        self.assertEqual(
            challenge["covered_risky_node_ids"],
            ["claim-1", "claim-2"],
        )
        self.assertEqual(
            challenge["uncovered_risky_claim_count"],
            0,
        )
        self.assertEqual(
            challenge["coverage_reserve_intervention_count"],
            0,
        )

    async def test_historical_cost_prior_is_similarity_gated(
        self,
    ):
        claims = [
            "The campaign always drew wider attendance.",
            "The program caused 20 percent higher revenue.",
        ]

        def make_transaction():
            _claims, nodes, transaction = (
                _joint_counterevidence_transaction()
            )
            for index, node in enumerate(nodes):
                node["text"] = claims[index]
            certificates = transaction[
                "counterfactual_stability_certificate"
            ]["provenance_certificates"]
            certificates[0]["claim_relation_risk"] = {
                "risk_level": "medium",
                "risk_score": 0.46,
            }
            certificates[1]["claim_relation_risk"] = {
                "risk_level": "high",
                "risk_score": 0.9,
            }
            return transaction

        matching_prior = {
            "context": build_counterevidence_cost_context(
                "claim",
                [claims[1]],
            ),
            "sample_count": 10,
            "mean_normalized_cost": 20.0,
            "mean_yield_score": 0.0,
        }
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            return {
                "rows": [],
                "latency_ms": 10,
                "usage": {
                    "call_count": 0,
                    "total_tokens": 0,
                },
            }

        _answer, matched = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                make_transaction(),
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_observed_cost_control=True,
                historical_cost_priors=[matching_prior],
                max_queries_per_claim=3,
                max_global_queries=1,
            )
        )
        self.assertIn("campaign", queries[0].lower())
        self.assertEqual(
            matched[
                "historical_cost_prior_task_application_count"
            ],
            1,
        )
        self.assertEqual(
            matched["observed_cost_reprioritization_count"],
            1,
        )
        self.assertNotIn(
            "campaign",
            json.dumps(
                matched[
                    "counterevidence_cost_memory_updates"
                ]
            ).lower(),
        )

        queries.clear()
        unrelated_prior = {
            **matching_prior,
            "context": build_counterevidence_cost_context(
                "claim",
                ["The orchard harvest increased apple exports."],
            ),
        }
        _answer, rejected = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                make_transaction(),
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_observed_cost_control=True,
                historical_cost_priors=[unrelated_prior],
                max_queries_per_claim=3,
                max_global_queries=1,
            )
        )
        self.assertIn("20 percent", queries[0])
        self.assertEqual(
            rejected[
                "historical_cost_prior_task_application_count"
            ],
            0,
        )
        self.assertEqual(
            rejected["observed_cost_reprioritization_count"],
            0,
        )

    async def test_observed_cost_reprioritizes_expensive_claim(
        self,
    ):
        claims = [
            "The campaign always drew wider attendance.",
            "The program caused 20 percent higher revenue.",
        ]
        _base_claims, nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        for index, node in enumerate(nodes):
            node["text"] = claims[index]
        certificates = transaction[
            "counterfactual_stability_certificate"
        ]["provenance_certificates"]
        certificates[0]["claim_relation_risk"] = {
            "risk_level": "medium",
            "risk_score": 0.0,
        }
        certificates[1]["claim_relation_risk"] = {
            "risk_level": "high",
            "risk_score": 0.9,
        }
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            if "20 percent" in query:
                return {
                    "rows": [
                        {
                            "source_id": "expensive-irrelevant",
                            "document_id": "weather-document",
                            "text": (
                                "A weather archive recorded rainfall."
                            ),
                            "source_reliability": 0.9,
                        }
                    ],
                    "latency_ms": 8000,
                    "usage": {
                        "call_count": 2,
                        "total_tokens": 4000,
                    },
                }
            return {
                "rows": [],
                "latency_ms": 100,
                "usage": {
                    "call_count": 1,
                    "total_tokens": 100,
                },
            }

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_observed_cost_control=True,
                max_queries_per_claim=3,
                max_global_queries=2,
            )
        )
        self.assertEqual(len(queries), 2)
        self.assertIn("20 percent", queries[0])
        self.assertIn("campaign", queries[1].lower())
        self.assertEqual(
            challenge["observed_cost_reprioritization_count"],
            1,
        )
        self.assertEqual(
            challenge["observed_provider_call_count"],
            3,
        )
        self.assertEqual(
            challenge["observed_provider_token_count"],
            4100,
        )
        self.assertEqual(
            challenge["observed_query_latency_ms"],
            8100.0,
        )
        self.assertEqual(
            challenge["observed_cost_profile_update_count"],
            2,
        )
        self.assertTrue(
            challenge["observed_selection_audit"][1][
                "reprioritized"
            ]
        )

    async def test_partial_fused_coverage_runs_residual_query(self):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            if "previous multi-claim search" in query.lower():
                return [
                    {
                        "source_id": "residual-revenue",
                        "document_id": "revenue-document",
                        "text": claims[1],
                        "source_reliability": 0.9,
                    }
                ]
            return [
                {
                    "source_id": "partial-sales",
                    "document_id": "sales-document",
                    "text": claims[0],
                    "source_reliability": 0.9,
                }
            ]

        answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_query_fusion=True,
                enable_residual_decomposition=True,
                max_queries_per_claim=3,
            )
        )
        self.assertEqual(len(queries), 2)
        self.assertIn("[C1]", queries[0])
        self.assertIn("[R1]", queries[1])
        self.assertEqual(
            challenge["partial_coverage_cluster_count"],
            1,
        )
        self.assertEqual(challenge["residual_query_count"], 1)
        self.assertEqual(challenge["residual_claim_count"], 1)
        self.assertEqual(
            challenge["residual_resolution_count"],
            1,
        )
        self.assertEqual(
            challenge["passed_node_ids"],
            ["claim-1", "claim-2"],
        )
        self.assertEqual(answer, " ".join(claims))

    async def test_residual_query_accepts_evidence_grounded_alias(
        self,
    ):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )

        async def search_func(query, _max_results):
            if "previous multi-claim search" in query.lower():
                return [
                    {
                        "source_id": "residual-earnings",
                        "document_id": "earnings-document",
                        "text": (
                            "The program caused greater earnings."
                        ),
                        "source_reliability": 0.9,
                    }
                ]
            return [
                {
                    "source_id": "partial-sales",
                    "document_id": "sales-document",
                    "text": claims[0],
                    "source_reliability": 0.9,
                }
            ]

        _answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_query_fusion=True,
                enable_residual_decomposition=True,
                max_queries_per_claim=3,
            )
        )
        revenue_case = next(
            row
            for row in challenge["cases"]
            if row["node_id"] == "claim-2"
        )
        alias_evidence = next(
            row
            for row in revenue_case["evidence"]
            if row["evidence_id"] == "residual-earnings"
        )
        self.assertEqual(revenue_case["status"], "passed")
        self.assertTrue(alias_evidence["alias_anchor_hits"])
        self.assertGreater(
            alias_evidence["effective_lexical_coverage"],
            alias_evidence["raw_lexical_coverage"],
        )
        self.assertEqual(
            challenge["residual_resolution_count"],
            1,
        )
        self.assertEqual(challenge["alias_coverage_count"], 1)

    async def test_fused_query_covers_two_related_claims(self):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            return [
                {
                    "source_id": "fused-counter-check",
                    "document_id": "independent-document",
                    "text": (
                        "The program caused higher sales and greater "
                        "revenue."
                    ),
                    "source_reliability": 0.9,
                }
            ]

        answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_query_fusion=True,
                max_queries_per_claim=3,
            )
        )
        self.assertEqual(len(queries), 1)
        self.assertIn("[C1]", queries[0])
        self.assertEqual(challenge["fused_query_count"], 1)
        self.assertEqual(challenge["fused_cluster_count"], 1)
        self.assertEqual(
            challenge["fused_claim_coverage_count"],
            2,
        )
        self.assertEqual(
            challenge["cross_claim_reuse_count"],
            1,
        )
        self.assertEqual(
            challenge["passed_node_ids"],
            ["claim-1", "claim-2"],
        )
        self.assertEqual(answer, " ".join(claims))

    async def test_fused_query_rejects_generic_shared_wording(self):
        claims, _nodes, transaction = (
            _joint_counterevidence_transaction()
        )

        async def search_func(_query, _max_results):
            return [
                {
                    "source_id": "generic-counter-check",
                    "document_id": "independent-document",
                    "text": "The program caused results.",
                    "source_reliability": 0.9,
                }
            ]

        answer, challenge = (
            await run_active_counterevidence_challenge(
                " ".join(claims),
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                enable_query_fusion=True,
                max_queries_per_claim=3,
                max_global_queries=1,
            )
        )
        self.assertEqual(challenge["fused_query_count"], 1)
        self.assertEqual(
            challenge["generic_reuse_rejected_count"],
            2,
        )
        self.assertEqual(challenge["shared_evidence_count"], 0)
        self.assertEqual(
            challenge["inconclusive_node_ids"],
            ["claim-1", "claim-2"],
        )
        self.assertEqual(
            challenge["pruned_node_ids"],
            ["claim-1", "claim-2"],
        )
        self.assertEqual(answer, " ".join(claims))

    async def test_joint_counterevidence_reuses_one_result_across_claims(
        self,
    ):
        claim_1 = "The program caused higher sales."
        claim_2 = "The program caused greater revenue."
        nodes = [
            {
                "node_id": f"claim-{index}",
                "claim_index": index - 1,
                "text": claim,
                "evidence_support": [
                    {
                        "evidence_id": f"S{index}",
                        "family_ids": [f"support-document-{index}"],
                    }
                ],
            }
            for index, claim in enumerate(
                [claim_1, claim_2],
                start=1,
            )
        ]
        transaction = {
            "dependency_graph": {
                "nodes": nodes,
                "edges": [],
                "kept_node_ids": ["claim-1", "claim-2"],
            },
            "claim_bundle_selection": {
                "final_selected_node_ids": [
                    "claim-1",
                    "claim-2",
                ]
            },
            "counterfactual_stability_certificate": {
                "semantic_family_quorum_node_ids": [
                    "claim-1",
                    "claim-2",
                ],
                "final_retained_node_ids": [
                    "claim-1",
                    "claim-2",
                ],
                "provenance_certificates": [
                    {
                        "node_id": f"claim-{index}",
                        "claim_relation_risk": {
                            "risk_level": "medium",
                            "risk_score": 0.46,
                        },
                        "support_families": [
                            {
                                "family_ids": [
                                    f"support-document-{index}"
                                ]
                            }
                        ],
                    }
                    for index in [1, 2]
                ],
            },
            "rollback_source_ids": [],
            "actions": [],
        }
        queries = []

        async def search_func(query, _max_results):
            queries.append(query)
            return [
                {
                    "source_id": "shared-counter-check",
                    "document_id": "independent-document",
                    "text": (
                        "The program caused higher sales and greater "
                        "revenue."
                    ),
                    "source_reliability": 0.9,
                }
            ]

        answer, challenge = (
            await run_active_counterevidence_challenge(
                f"{claim_1} {claim_2}",
                transaction,
                search_func,
                enable_value_control=True,
                enable_joint_portfolio=True,
                max_queries_per_claim=3,
            )
        )
        self.assertEqual(len(queries), 1)
        self.assertEqual(challenge["query_count"], 1)
        self.assertEqual(
            challenge["cross_claim_reuse_count"],
            1,
        )
        self.assertEqual(challenge["shared_evidence_count"], 1)
        self.assertEqual(
            challenge["passed_node_ids"],
            ["claim-1", "claim-2"],
        )
        self.assertEqual(challenge["pruned_node_ids"], [])
        self.assertEqual(answer, f"{claim_1} {claim_2}")

    async def test_runtime_voi_counterevidence_early_stops(self):
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        distractor = (
            "A comet made him internationally famous near Nemerax amid "
            "zephyr quuxium cobalt amber prism lunar vortex nebula "
            "solstice titanium orchard cipher delta quartz aurora "
            "harbor matrix kinetic falcon granite meadow plasma "
            "radian summit."
        )
        dependent = (
            "This experience made him internationally famous."
        )
        generated = f"{unsafe_claim} {distractor} {dependent}"
        constraints = [
            {
                "claim_id": "claim-3",
                "assertion": (
                    "The data does not provide details about his "
                    "professional journalism work."
                ),
                "source_ids": ["correction-source"],
            }
        ]
        sources = [
            {
                "source_id": "correction-source",
                "document_id": "career-document",
                "text": fallback_claim,
                "score": 0.9,
            },
            {
                "source_id": "support-1",
                "document_id": "document-1",
                "text": dependent,
                "score": 0.9,
            },
            {
                "source_id": "support-2",
                "document_id": "document-2",
                "text": dependent,
                "score": 0.9,
            },
        ]

        async def run_with_results(challenge_rows):
            async def llm_func(_prompt, system_prompt=None):
                return json.dumps(
                    {
                        "answer": generated,
                        "used_source_ids": [
                            row["source_id"] for row in sources
                        ],
                        "insufficient": False,
                    }
                )

            async def search_func(_query, max_results):
                return challenge_rows[:max_results]

            _answer, result = await realize_answer(
                llm_func,
                (
                    "How did Adebanwi move from journalism into "
                    "academia?"
                ),
                fallback_claim,
                {"entities": [], "relationships": []},
                sources,
                prohibited_assertions=constraints,
                counterevidence_search_func=search_func,
                enable_counterevidence_value_control=True,
            )
            return result["transactional_correction_rollback"]

        contradiction = await run_with_results(
            [
                {
                    "source_id": "counter-1",
                    "document_id": "counter-document",
                    "text": (
                        "This experience did not make him "
                        "internationally famous."
                    ),
                    "source_reliability": 0.9,
                }
            ]
        )
        confirmation = await run_with_results(
            [
                {
                    "source_id": "confirm-1",
                    "document_id": "confirm-document",
                    "text": dependent,
                    "source_reliability": 0.9,
                }
            ]
        )
        empty = await run_with_results([])
        self.assertEqual(
            contradiction["counterevidence_query_count"],
            1,
        )
        self.assertEqual(
            contradiction["counterevidence_decisive_stop_count"],
            1,
        )
        self.assertEqual(
            confirmation["counterevidence_query_count"],
            1,
        )
        self.assertEqual(
            confirmation["counterevidence_saved_query_count"],
            2,
        )
        self.assertEqual(
            empty["counterevidence_query_count"],
            2,
        )
        self.assertEqual(
            empty[
                "counterevidence_skipped_low_value_query_count"
            ],
            1,
        )

    async def test_runtime_counterevidence_challenge_outcomes(self):
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        distractor = (
            "A comet made him internationally famous near Nemerax amid "
            "zephyr quuxium cobalt amber prism lunar vortex nebula "
            "solstice titanium orchard cipher delta quartz aurora "
            "harbor matrix kinetic falcon granite meadow plasma "
            "radian summit."
        )
        dependent = (
            "This experience made him internationally famous."
        )
        generated = f"{unsafe_claim} {distractor} {dependent}"
        constraints = [
            {
                "claim_id": "claim-3",
                "assertion": (
                    "The data does not provide details about his "
                    "professional journalism work."
                ),
                "source_ids": ["correction-source"],
            }
        ]
        sources = [
            {
                "source_id": "correction-source",
                "document_id": "career-document",
                "text": fallback_claim,
                "score": 0.9,
            },
            {
                "source_id": "support-1",
                "document_id": "document-1",
                "text": dependent,
                "score": 0.9,
            },
            {
                "source_id": "support-2",
                "document_id": "document-2",
                "text": dependent,
                "score": 0.9,
            },
        ]

        async def run_with_challenge(challenge_row):
            async def llm_func(_prompt, system_prompt=None):
                return json.dumps(
                    {
                        "answer": generated,
                        "used_source_ids": [
                            row["source_id"] for row in sources
                        ],
                        "insufficient": False,
                    }
                )

            async def search_func(_query, _max_results):
                return [challenge_row]

            return await realize_answer(
                llm_func,
                (
                    "How did Adebanwi move from journalism into "
                    "academia?"
                ),
                fallback_claim,
                {"entities": [], "relationships": []},
                sources,
                prohibited_assertions=constraints,
                counterevidence_search_func=search_func,
            )

        contradicted_answer, contradicted_result = (
            await run_with_challenge(
                {
                    "source_id": "challenge-1",
                    "document_id": "counter-document",
                    "text": (
                        "This experience did not make him "
                        "internationally famous."
                    ),
                    "source_reliability": 0.9,
                }
            )
        )
        passed_answer, passed_result = await run_with_challenge(
            {
                "source_id": "challenge-2",
                "document_id": "independent-document",
                "text": dependent,
                "source_reliability": 0.9,
            }
        )
        inconclusive_answer, inconclusive_result = (
            await run_with_challenge(
                {
                    "source_id": "support-1",
                    "document_id": "document-1",
                    "text": dependent,
                    "source_reliability": 0.9,
                }
            )
        )
        self.assertNotIn(dependent, contradicted_answer)
        self.assertIn(dependent, passed_answer)
        self.assertNotIn(dependent, inconclusive_answer)
        contradicted = contradicted_result[
            "transactional_correction_rollback"
        ]
        passed = passed_result[
            "transactional_correction_rollback"
        ]
        inconclusive = inconclusive_result[
            "transactional_correction_rollback"
        ]
        self.assertEqual(
            contradicted["counterevidence_contradiction_count"],
            1,
        )
        self.assertEqual(
            contradicted[
                "active_counterevidence_pruned_claim_count"
            ],
            1,
        )
        self.assertEqual(
            passed["active_counterevidence_pruned_claim_count"],
            0,
        )
        self.assertEqual(
            inconclusive[
                "active_counterevidence_pruned_claim_count"
            ],
            1,
        )
        self.assertLessEqual(
            contradicted["counterevidence_query_count"],
            2,
        )

    async def test_runtime_requires_diverse_sources_for_unstable_claim(
        self,
    ):
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        distractor = (
            "A comet made him internationally famous near Nemerax amid "
            "zephyr quuxium cobalt amber prism lunar vortex nebula "
            "solstice titanium orchard cipher delta quartz aurora "
            "harbor matrix kinetic falcon granite meadow plasma "
            "radian summit."
        )
        dependent = (
            "This experience made him internationally famous."
        )
        generated = f"{unsafe_claim} {distractor} {dependent}"
        constraints = [
            {
                "claim_id": "claim-3",
                "assertion": (
                    "The data does not provide details about his "
                    "professional journalism work."
                ),
                "source_ids": ["correction-source"],
            }
        ]

        async def run_with_sources(source_count):
            source_rows = [
                {
                    "source_id": "correction-source",
                    "origin_ids": ["career-document"],
                    "text": fallback_claim,
                    "score": 0.9,
                }
            ] + [
                {
                    "source_id": f"support-{index}",
                    "origin_ids": [f"independent-{index}"],
                    "text": dependent,
                    "score": 0.9,
                }
                for index in range(1, source_count + 1)
            ]

            async def llm_func(_prompt, system_prompt=None):
                return json.dumps(
                    {
                        "answer": generated,
                        "used_source_ids": [
                            row["source_id"] for row in source_rows
                        ],
                        "insufficient": False,
                    }
                )

            return await realize_answer(
                llm_func,
                (
                    "How did Adebanwi move from journalism into "
                    "academia?"
                ),
                fallback_claim,
                {"entities": [], "relationships": []},
                source_rows,
                prohibited_assertions=constraints,
            )

        single_answer, single_result = await run_with_sources(1)
        diverse_answer, diverse_result = await run_with_sources(2)
        self.assertEqual(single_answer, fallback_claim)
        self.assertNotIn(dependent, single_answer)
        self.assertIn(dependent, diverse_answer)
        self.assertEqual(
            single_result["transactional_correction_rollback"][
                "single_source_stability_rejected_count"
            ],
            1,
        )
        self.assertEqual(
            diverse_result["transactional_correction_rollback"][
                "diverse_provenance_protected_count"
            ],
            1,
        )
        self.assertEqual(
            single_result["transactional_correction_rollback"][
                "dropped_stale_evidence_count"
            ],
            1,
        )
        self.assertEqual(
            diverse_result["transactional_correction_rollback"][
                "dropped_stale_evidence_count"
            ],
            0,
        )
        self.assertEqual(single_result["attempt_count"], 1)
        self.assertEqual(diverse_result["attempt_count"], 1)

    async def test_stability_prunes_marginal_supported_claim(self):
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        marginal_claim = (
            "Newspapers and magazines accompanied zephyr quuxium vortex "
            "lunar cobalt amber prism nebula."
        )

        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": f"{unsafe_claim} {marginal_claim}",
                    "used_source_ids": ["correction-source"],
                    "insufficient": False,
                }
            )

        answer, result = await realize_answer(
            llm_func,
            "How did Adebanwi move from journalism into academia?",
            fallback_claim,
            {"entities": [], "relationships": []},
            [
                {
                    "source_id": "correction-source",
                    "text": (
                        f"{fallback_claim} Adebanwi worked for "
                        "newspapers and magazines."
                    ),
                    "score": 0.9,
                }
            ],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
        )
        self.assertTrue(result["committed"])
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(answer, fallback_claim)
        transaction = result["transactional_correction_rollback"]
        self.assertEqual(
            transaction["bundle_pruned_claim_count"],
            0,
        )
        self.assertEqual(
            transaction["stability_pruned_claim_count"],
            1,
        )
        certificate = result["attempts"][0][
            "transactional_rollback"
        ]["counterfactual_stability_certificate"]
        self.assertEqual(
            certificate["unstable_selected_node_ids"],
            ["claim-2"],
        )
        self.assertEqual(
            certificate["final_retained_node_ids"],
            [],
        )

    async def test_global_bundle_prunes_unsupported_independent_claim(self):
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        unsupported_claim = (
            "Zyphron calibrated quuxium vortices beneath Nemerax."
        )

        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": (
                        f"{unsafe_claim} {unsupported_claim}"
                    ),
                    "used_source_ids": ["correction-source"],
                    "insufficient": False,
                }
            )

        answer, result = await realize_answer(
            llm_func,
            "How did Adebanwi move from journalism into academia?",
            fallback_claim,
            {"entities": [], "relationships": []},
            [
                {
                    "source_id": "correction-source",
                    "text": fallback_claim,
                    "score": 0.9,
                }
            ],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
        )
        self.assertTrue(result["committed"])
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(answer, fallback_claim)
        transaction = result["transactional_correction_rollback"]
        self.assertEqual(
            transaction["bundle_pruned_claim_count"],
            1,
        )
        selection = result["attempts"][0][
            "transactional_rollback"
        ]["claim_bundle_selection"]
        self.assertEqual(
            selection["pruned_node_ids"],
            ["claim-2"],
        )
        self.assertTrue(selection["coherence_passed"])

    async def test_dependency_rebind_commits_through_full_audit(self):
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        dependent_claim = (
            "This experience came before he joined the University of "
            "Ibadan's Department of Political Science."
        )

        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": f"{unsafe_claim} {dependent_claim}",
                    "used_source_ids": ["correction-source"],
                    "insufficient": False,
                }
            )

        answer, result = await realize_answer(
            llm_func,
            "How did Adebanwi move from journalism into academia?",
            fallback_claim,
            {"entities": [], "relationships": []},
            [
                {
                    "source_id": "correction-source",
                    "text": (
                        f"{fallback_claim} {dependent_claim}"
                    ),
                    "score": 0.9,
                }
            ],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
        )
        self.assertTrue(result["committed"])
        self.assertEqual(result["attempt_count"], 1)
        self.assertIn(fallback_claim, answer)
        self.assertIn(dependent_claim, answer)
        transaction = result["transactional_correction_rollback"]
        self.assertTrue(transaction["committed"])
        self.assertEqual(transaction["dependency_rebound_count"], 1)
        attempt_transaction = result["attempts"][0][
            "transactional_rollback"
        ]
        self.assertEqual(
            attempt_transaction["dependency_graph"][
                "rebound_node_ids"
            ],
            ["claim-2"],
        )

    async def test_transactional_rollback_commits_safe_generation(self):
        safe_claim = (
            "Adebanwi taught political science after entering academia."
        )
        unsafe_claim = (
            "Adebanwi worked as a television journalist during a Mars "
            "expedition."
        )
        fallback_claim = (
            "Adebanwi worked as a freelance journalist before academia."
        )
        generated = f"{safe_claim} {unsafe_claim}"
        source = (
            f"{safe_claim} {fallback_claim}"
        )

        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": generated,
                    "used_source_ids": ["correction-source"],
                    "insufficient": False,
                }
            )

        answer, result = await realize_answer(
            llm_func,
            "What did Adebanwi do in journalism and academia?",
            fallback_claim,
            {"entities": [], "relationships": []},
            [
                {
                    "source_id": "correction-source",
                    "text": source,
                    "score": 0.9,
                }
            ],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
        )
        self.assertTrue(result["committed"])
        self.assertEqual(result["attempt_count"], 1)
        self.assertIn(safe_claim, answer)
        self.assertIn(fallback_claim, answer)
        self.assertNotIn("Mars expedition", answer)
        transaction = result["transactional_correction_rollback"]
        self.assertTrue(transaction["committed"])
        self.assertTrue(transaction["avoided_full_answer_fallback"])
        self.assertEqual(transaction["preserved_generated_claim_count"], 1)
        self.assertEqual(
            result["stop_reason"],
            "transactional_correction_rollback_committed",
        )

    async def test_repeated_relapse_retains_pre_realization_answer(self):
        rejected = (
            "The records do not provide specific details about his "
            "professional journalism work."
        )

        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": rejected,
                    "used_source_ids": ["S1"],
                    "insufficient": False,
                }
            )

        fallback = "He began his career as a professional journalist."
        answer, result = await realize_answer(
            llm_func,
            "How did his career evolve from journalism?",
            fallback,
            {"entities": [], "relationships": []},
            [{"source_id": "S1", "text": rejected, "score": 0.9}],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide specific details about "
                        "his professional journalism work."
                    ),
                }
            ],
        )
        self.assertEqual(answer, fallback)
        self.assertFalse(result["committed"])
        self.assertTrue(result["fallback_answer_retained"])
        self.assertTrue(
            result["correction_lineage"]["fallback_due_to_relapse"]
        )
        self.assertEqual(
            result["correction_lineage"]["relapse_attempt_count"],
            2,
        )
        self.assertEqual(
            result["correction_lineage"]["match_mode_counts"],
            {"lexical_overlap": 2},
        )
        self.assertEqual(
            result["stop_reason"],
            "realization_rejected_correction_relapse",
        )

    async def test_uncertified_inversion_retains_corrected_answer(self):
        generated = (
            "Adebanwi worked as a freelance reporter, writer, journalist "
            "and editor before joining academia."
        )

        async def llm_func(_prompt, system_prompt=None):
            return json.dumps(
                {
                    "answer": generated,
                    "used_source_ids": ["ordinary-source"],
                    "insufficient": False,
                }
            )

        fallback = "His communications background began his early career."
        answer, result = await realize_answer(
            llm_func,
            "What journalism work did Adebanwi do?",
            fallback,
            {"entities": [], "relationships": []},
            [
                {
                    "source_id": "correction-source",
                    "text": generated,
                    "score": 0.9,
                },
                {
                    "source_id": "ordinary-source",
                    "text": generated,
                    "score": 0.9,
                },
            ],
            prohibited_assertions=[
                {
                    "claim_id": "claim-3",
                    "assertion": (
                        "The data does not provide details about his "
                        "professional journalism work."
                    ),
                    "source_ids": ["correction-source"],
                }
            ],
        )
        self.assertEqual(answer, fallback)
        self.assertFalse(result["committed"])
        self.assertTrue(result["fallback_answer_retained"])
        self.assertTrue(
            result["correction_source_certificates"][
                "fallback_due_to_uncertified_inversion"
            ]
        )
        self.assertEqual(
            result["correction_source_certificates"]["uncertified_count"],
            2,
        )
        self.assertEqual(
            result["stop_reason"],
            "realization_rejected_uncertified_correction_inversion",
        )


if __name__ == "__main__":
    unittest.main()
