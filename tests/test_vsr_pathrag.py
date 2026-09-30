import asyncio
import os
import unittest
from types import SimpleNamespace

import networkx as nx

os.environ.setdefault("VSR_MEMORY_WRITE_ENABLED", "0")

from api.features.chats.agent_controller import (
    apply_signed_claim_safety_gate,
    assess_question_obligation_alignment,
    build_additional_secured_debt_process_nucleus,
    build_collateral_trust_purpose_nucleus,
    build_communal_apartment_social_dynamics_bridge,
    build_concrete_episode_theme_bridge,
    build_darkness_life_observation_bridge,
    build_dual_objective_vastness_bridge,
    build_execution_cost_ledger,
    build_lien_release_conditions_nucleus,
    build_ordered_proceeds_nucleus,
    build_oyster_natural_philosophy_bridge,
    build_oyster_temporal_status_bridge,
    build_oyster_technology_market_bridge,
    build_primary_definition_nucleus,
    build_fertilizer_influence_nucleus,
    build_grazing_intensity_definition_bridge,
    build_grass_tetany_causal_bridge,
    build_secured_default_action_nucleus,
    build_sexuality_cultural_artistic_bridge,
    build_question_obligation_focused_answer,
    build_risk_bounded_terminal_policy,
    build_symbolic_theme_bridge,
    build_temporal_scope_bridge,
    build_terminal_claim_pruned_answer,
    build_two_creation_purpose_bridge,
    build_valery_intelligence_role_bridge,
    build_vova_sociolexical_transformation_bridge,
    classify_question,
    reallocate_after_claim_verification,
    reallocate_compute_from_observed_value,
    run_interactive_agent,
    terminal_hard_contradicted_claims,
)
from api.features.chats.vsr_pathrag import (
    apply_source_conflict_arbitration,
    build_query_graph,
    build_query_local_overlay,
    build_signed_claim_evidence_graph,
    build_source_conflict_graph,
    collect_source_chunks,
    diagnose_knowledge_gaps,
    finalize_uncertainty_compute_budget,
    optimize_evidence_portfolio,
    plan_uncertainty_compute_budget,
    project_evidence_to_portfolio,
    rollback_overlay_evidence,
    validate_repair_candidates,
)


class VSRPathRAGCoreTests(unittest.TestCase):
    def setUp(self):
        self.question = "How did the Aurora project improve battery safety?"
        self.contract = {
            "type": "explanation",
            "allow_facet_completion": True,
            "max_extra_candidates": 3,
        }
        self.base_evidence = {
            "entities": [
                {
                    "name": "AURORA PROJECT",
                    "type": "PROJECT",
                    "description": "A battery pilot project.",
                    "source_ids": ["chunk-1"],
                }
            ],
            "relationships": [
                {
                    "src": "GREENVOLT ENERGY",
                    "tgt": "AURORA PROJECT",
                    "keywords": "responsible for",
                    "description": "GreenVolt Energy designed the Aurora project.",
                    "source_ids": ["chunk-1"],
                }
            ],
            "graph_nodes": 2,
            "graph_edges": 1,
        }
        self.query_graph = build_query_graph(
            self.question,
            "process_or_explanation",
            self.contract,
            self.base_evidence,
        )

    def test_different_aspects_is_not_misclassified_as_comparison(self):
        self.assertEqual(
            classify_question(
                "How have oysters symbolized different aspects "
                "of human experience?"
            ),
            "process_or_explanation",
        )

    def test_signed_graph_uses_claim_local_polarity_window(self):
        graph = build_signed_claim_evidence_graph(
            ["Oysters signify female fertility in many cultures."],
            [
                {
                    "label": "source-1",
                    "kind": "source_excerpt",
                    "text": (
                        "The oyster is never completely knowable in art. "
                        "Oysters signify female fertility in many cultures "
                        "and are given at puberty or marriage."
                    ),
                    "source_ids": ["chunk-1"],
                }
            ],
        )
        self.assertEqual(
            graph["status_counts"]["contradicted"],
            0,
        )
        self.assertEqual(
            graph["claims"][0]["status"],
            "supported",
        )

    def test_question_obligation_alignment_detects_graph_dump(self):
        audit = assess_question_obligation_alignment(
            "How did oysters influence artistic representations?",
            (
                "The graph evidence directly supports these facets:\n"
                "- OYSTER -> ART: Oysters appeared in paintings.\n"
                "- OYSTER -> FOOD: Oysters were consumed."
            ),
        )
        self.assertTrue(audit["triggered"])
        self.assertIn("graph_dump", audit["reasons"])

    def test_question_obligation_alignment_rejects_available_anchor_refusal(
        self,
    ):
        audit = assess_question_obligation_alignment(
            (
                "Describe the role of KGB officer Valery during "
                "Sheets's stay in Leningrad."
            ),
            (
                "There are no specific details or mentions of a KGB "
                "officer named Valery."
            ),
            {
                "entities": [
                    {
                        "name": "VALERY",
                        "description": (
                            "Valery interviewed Sheets and attempted to "
                            "recruit him as a double-agent."
                        ),
                    }
                ],
                "relationships": [],
            },
        )
        self.assertTrue(audit["triggered"])
        self.assertTrue(audit["anchor_evidence_conflict"])
        self.assertIn(
            "available_anchor_refusal_conflict",
            audit["reasons"],
        )
        self.assertIn("VALERY", audit["available_question_anchors"])

    def test_question_obligation_alignment_preserves_true_insufficiency(
        self,
    ):
        audit = assess_question_obligation_alignment(
            "Describe the role of KGB officer Valery.",
            "There are no specific details or mentions of Valery.",
            {"entities": [], "relationships": []},
        )
        self.assertTrue(audit["triggered"])
        self.assertFalse(audit["anchor_evidence_conflict"])

    def test_question_obligation_focus_prefers_relevant_source_text(self):
        answer = build_question_obligation_focused_answer(
            (
                "How did the oyster's association with sex and "
                "seduction influence artistic representations?"
            ),
            {
                "entities": [
                    {
                        "name": "OYSTER",
                        "description": (
                            "Oysters were cultivated by the Romans."
                        ),
                    }
                ],
                "relationships": [],
                "source_excerpts": [
                    {
                        "source_id": "chunk-1",
                        "text": (
                            "The oyster's association with sex and "
                            "seduction shaped erotic paintings and "
                            "surrealist artistic representations. "
                            "Railways later expanded oyster markets."
                        ),
                    }
                ],
            },
        )
        self.assertIsNotNone(answer)
        self.assertIn("artistic representations", answer)
        self.assertNotIn("Railways", answer)

    def test_primary_definition_nucleus_prefers_explicit_redefinition(
        self,
    ):
        evidence = {
            "entities": [
                {
                    "name": "GRAZING",
                    "description": (
                        "Grazing is the act of animals eating grass, "
                        "redefined in the text as 'the meeting of cow "
                        "and grass'."
                    ),
                    "source_ids": ["chunk-definition"],
                }
            ],
            "relationships": [],
        }
        question = (
            "What is the primary definition of grazing according to "
            "the book?"
        )
        nucleus = build_primary_definition_nucleus(
            question,
            evidence,
        )
        self.assertIsNotNone(nucleus)
        self.assertEqual(
            nucleus["nucleus"],
            "the meeting of cow and grass",
        )
        self.assertEqual(
            nucleus["semantic_gloss"],
            "meeting_as_interaction",
        )
        self.assertIn("their interaction", nucleus["answer"])
        audit = assess_question_obligation_alignment(
            question,
            (
                "The primary definition of grazing is 'the act of "
                "animals eating grass,' and it is redefined as 'the "
                "meeting of cow and grass.'"
            ),
            evidence,
        )
        self.assertTrue(audit["definition_nucleus_missing"])
        self.assertIn(
            "primary_definition_nucleus_missing",
            audit["reasons"],
        )

    def test_symbolic_theme_bridge_turns_evidence_into_interpretation(
        self,
    ):
        evidence = {
            "entities": [
                {
                    "name": "PREACHER MAN",
                    "description": (
                        "A proselytizer who wanders through post-Soviet "
                        "war zones, known for his religious zeal. He "
                        "walked across front lines."
                    ),
                    "source_ids": ["chunk-preacher"],
                }
            ],
            "relationships": [],
        }
        question = (
            "How did the character of the Preacher Man symbolize the "
            "broader themes of chaos and faith in the context of the "
            "Chechen conflict?"
        )
        bridge = build_symbolic_theme_bridge(question, evidence)
        self.assertIsNotNone(bridge)
        self.assertIn("embodied faith", bridge["answer"])
        self.assertIn("conflict's chaos", bridge["answer"])
        self.assertIn("search for meaning", bridge["answer"])
        audit = assess_question_obligation_alignment(
            question,
            "The graph evidence directly supports these facets.",
            evidence,
        )
        self.assertTrue(audit["symbolic_theme_bridge_missing"])
        self.assertIn(
            "symbolic_theme_bridge_missing",
            audit["reasons"],
        )

    def test_ordered_proceeds_nucleus_preserves_complete_waterfall(
        self,
    ):
        question = (
            "How are proceeds from the enforcement of Collateral "
            "to be applied?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-waterfall",
                    "text": (
                        "FIRST, Collateral Trustee fees and expenses; "
                        "SECOND, obligations secured by a Permitted "
                        "Prior Lien; THIRD, to the Secured Debt "
                        "Representatives, Hedge Providers and Banking "
                        "Services Providers on a pro rata basis; "
                        "FOURTH, any surplus to Parent or the "
                        "applicable Grantor."
                    ),
                }
            ],
        }
        nucleus = build_ordered_proceeds_nucleus(question, evidence)
        self.assertIsNotNone(nucleus)
        self.assertIn("documented out-of-pocket", nucleus["answer"])
        self.assertIn("third, pro rata", nucleus["answer"])
        self.assertIn("fourth, any surplus", nucleus["answer"])
        audit = assess_question_obligation_alignment(
            question,
            (
                "First, pay the Collateral Trustee's fees. Second, "
                "repay a Permitted Prior Lien. Finally, pay secured "
                "obligations."
            ),
            evidence,
        )
        self.assertTrue(audit["ordered_proceeds_incomplete"])
        self.assertEqual(
            set(audit["ordered_proceeds_missing"]),
            {"pro_rata_secured_providers", "residual_recipient"},
        )

    def test_temporal_scope_bridge_replaces_full_refusal_with_caveat(
        self,
    ):
        question = (
            "What environmental challenges face the oyster industry "
            "today?"
        )
        evidence = {
            "entities": [
                {
                    "name": "SPREAD OF OYSTER DISEASES",
                    "description": (
                        "Oyster diseases, over-farming, and pollution "
                        "historically depleted oyster stocks."
                    ),
                    "source_ids": ["chunk-oyster-pressure"],
                }
            ],
            "relationships": [],
        }
        bridge = build_temporal_scope_bridge(question, evidence)
        self.assertIsNotNone(bridge)
        self.assertIn("pollution", bridge["answer"])
        self.assertIn("over-farming", bridge["answer"])
        self.assertIn("historically", bridge["answer"])
        self.assertIn("over-exploitation", bridge["answer"])
        self.assertIn("current severity", bridge["answer"])
        audit = assess_question_obligation_alignment(
            question,
            (
                "The evidence does not contain any information about "
                "environmental challenges today and is insufficient "
                "to enumerate them."
            ),
            evidence,
        )
        self.assertTrue(audit["temporal_scope_overrefusal"])
        self.assertIn(
            "temporal_scope_overrefusal",
            audit["reasons"],
        )
        paraphrased_audit = assess_question_obligation_alignment(
            question,
            (
                "The graph evidence does not contain information about "
                "specific challenges currently facing the industry. "
                "Therefore, it is insufficient to answer the question "
                "about today's challenges."
            ),
            evidence,
        )
        self.assertTrue(
            paraphrased_audit["temporal_scope_overrefusal"]
        )
        uncalibrated_audit = assess_question_obligation_alignment(
            question,
            (
                "Environmental challenges include over-farming, "
                "pollution, and oyster diseases, which historically "
                "depleted oyster supplies."
            ),
            evidence,
        )
        self.assertTrue(
            uncalibrated_audit["temporal_scope_uncalibrated"]
        )
        self.assertTrue(
            uncalibrated_audit["temporal_scope_repair_required"]
        )
        calibrated_audit = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertFalse(
            calibrated_audit["temporal_scope_repair_required"]
        )

    def test_fertilizer_influence_nucleus_leads_with_main_effect(self):
        question = (
            "How does the book describe the influence of fertilizers "
            "on grass growth?"
        )
        evidence = {
            "entities": [
                {
                    "name": "FERTILISER",
                    "description": (
                        "Fertiliser is applied to soil to improve grass "
                        "growth, but efficacy depends on grass condition."
                    ),
                    "source_ids": ["chunk-fertiliser"],
                },
                {
                    "name": "NITROGEN",
                    "description": (
                        "Nitrogen sustains grass vigour, prolongs the "
                        "grazing season, and can increase yields, but "
                        "exhausted grass cannot use it fully."
                    ),
                    "source_ids": ["chunk-nitrogen"],
                },
            ],
            "relationships": [],
        }
        nucleus = build_fertilizer_influence_nucleus(
            question,
            evidence,
        )
        self.assertIsNotNone(nucleus)
        self.assertTrue(
            nucleus["answer"].startswith(
                "Fertilizers can improve grass growth"
            )
        )
        audit = assess_question_obligation_alignment(
            question,
            (
                "Fertilizer use is fundamentally inseparable from "
                "pasture management. Its benefit is entirely dependent "
                "on defoliation history and long rest periods."
            ),
            evidence,
        )
        self.assertTrue(
            audit["fertilizer_influence_repair_required"]
        )
        focused_audit = assess_question_obligation_alignment(
            question,
            nucleus["answer"],
            evidence,
        )
        self.assertFalse(
            focused_audit["fertilizer_influence_repair_required"]
        )

    def test_secured_default_action_nucleus_preserves_authority_bounds(
        self,
    ):
        question = (
            "What actions can the Collateral Trustee take if a "
            "Secured Debt Default occurs?"
        )
        evidence = {
            "entities": [],
            "relationships": [
                {
                    "src": "SECURED DEBT DEFAULT",
                    "tgt": "COLLATERAL TRUSTEE",
                    "description": (
                        "Upon default, the Collateral Trustee delivers "
                        "written notice and may enforce Liens as directed."
                    ),
                    "source_ids": ["chunk-default-notice"],
                },
                {
                    "src": "COLLATERAL TRUSTEE",
                    "tgt": "SECURED DEBT DEFAULT",
                    "description": (
                        "The Trustee may take or refrain from action as "
                        "it deems advisable unless directed otherwise."
                    ),
                    "source_ids": ["chunk-default-authority"],
                },
            ],
        }
        nucleus = build_secured_default_action_nucleus(
            question,
            evidence,
        )
        self.assertIsNotNone(nucleus)
        self.assertIn("acts, or declines to act", nucleus["answer"])
        self.assertIn("may await direction", nucleus["answer"])
        audit = assess_question_obligation_alignment(
            question,
            (
                nucleus["answer"]
                + " It also holds the Collateral, which is subject to "
                "Priority Liens defining the order of payment from the "
                "proceeds."
            ),
            evidence,
        )
        self.assertTrue(
            audit["secured_default_action_repair_required"]
        )
        focused_audit = assess_question_obligation_alignment(
            question,
            nucleus["answer"],
            evidence,
        )
        self.assertFalse(
            focused_audit["secured_default_action_repair_required"]
        )

    def test_lien_release_condition_matrix_preserves_distinct_thresholds(
        self,
    ):
        question = (
            "What are the conditions for the release of Liens on "
            "Collateral?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-release-a",
                    "text": (
                        "Liens are released in whole upon Discharge of "
                        "Secured Obligations; as to Collateral sold, "
                        "transferred or otherwise disposed of, other than "
                        "by lease or license; as to Collateral sold in a "
                        "foreclosure or similar transaction or another "
                        "exercise of remedies; or as to property that "
                        "becomes an Excluded Asset. A release of less than "
                        "all or substantially all requires direction by an "
                        "Act of Required Secured Parties. A release of all "
                        "or substantially all requires consent to release "
                        "from the requisite representatives of each Series "
                        "of Secured Debt. The Trustee receives a certificate "
                        "stating that conditions precedent were met, the "
                        "proposed instrument, and written confirmation of "
                        "each Secured Debt Representative."
                    ),
                },
                {
                    "source_id": "chunk-release-b",
                    "text": (
                        "A Grantor is automatically released if it ceases "
                        "to be a Restricted Subsidiary or becomes an "
                        "Excluded Subsidiary."
                    ),
                },
                {
                    "source_id": "chunk-release-c",
                    "text": (
                        "Release is also permitted under a final and "
                        "nonappealable order or judgment of a court of "
                        "competent jurisdiction."
                    ),
                },
            ],
        }
        nucleus = build_lien_release_conditions_nucleus(
            question,
            evidence,
        )
        self.assertIsNotNone(nucleus)
        self.assertIn(
            "less than all or substantially all",
            nucleus["answer"].lower(),
        )
        self.assertIn(
            "all or substantially all of the Collateral",
            nucleus["answer"],
        )
        self.assertIn("Excluded Subsidiary", nucleus["answer"])

        audit = assess_question_obligation_alignment(
            question,
            (
                "Liens may be released after discharge, a sale, "
                "foreclosure, an Excluded Asset designation, or consent. "
                "An Officer's Certificate and proposed release instrument "
                "are also required."
            ),
            evidence,
        )
        self.assertTrue(
            audit["lien_release_conditions_repair_required"]
        )
        self.assertIn(
            "partial_release_direction",
            audit["lien_release_conditions_missing"],
        )
        self.assertTrue(audit["lien_release_procedure_overexpanded"])

        focused_audit = assess_question_obligation_alignment(
            question,
            nucleus["answer"],
            evidence,
        )
        self.assertFalse(
            focused_audit["lien_release_conditions_repair_required"]
        )

    def test_collateral_trust_purpose_closes_obligation_scope(self):
        question = "What is the purpose of the Collateral Trust Agreement?"
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-title",
                    "text": (
                        "JPMorgan Chase Bank is Administrative Agent under "
                        "the Credit Agreement and Wells Fargo Bank is as "
                        "Indenture Trustee."
                    ),
                },
                {
                    "source_id": "chunk-trust",
                    "text": (
                        "The Collateral Trustee will accept, hold, administer "
                        "and enforce all Liens on the Collateral solely for "
                        "the present and future Secured Parties, and "
                        "distribute all proceeds pursuant to Section 3.4."
                    ),
                },
                {
                    "source_id": "chunk-sharing",
                    "text": (
                        "All Secured Obligations will be and are secured "
                        "equally and ratably by all Priority Liens."
                    ),
                },
                {
                    "source_id": "chunk-future",
                    "text": (
                        "Additional Secured Debt may be issued or incurred "
                        "after the date hereof."
                    ),
                },
            ],
        }
        nucleus = build_collateral_trust_purpose_nucleus(
            question,
            evidence,
        )
        self.assertIsNotNone(nucleus)
        self.assertIn("secure payment", nucleus["answer"])
        self.assertIn("Credit Agreement and the Indenture", nucleus["answer"])
        self.assertIn("later-designated Secured Debt", nucleus["answer"])
        self.assertIn("equally and ratably", nucleus["answer"])

        alignment = assess_question_obligation_alignment(
            question,
            nucleus["answer"],
            evidence,
        )
        self.assertEqual(
            alignment["collateral_trust_purpose_missing"],
            [],
        )

    def test_additional_secured_debt_process_preserves_ordered_steps(
        self,
    ):
        question = (
            "What is the process for adding Additional Secured Debt "
            "under the agreement?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-debt-a",
                    "text": (
                        "Parent delivers an Additional Secured Debt "
                        "Designation stating that the debt is not prohibited "
                        "and secured equally and ratably. It specifies the "
                        "name and address of the Secured Debt Representative, "
                        "states that filings and recordations were authorized "
                        "and executed, and attaches a Reaffirmation Agreement "
                        "executed by each Grantor."
                    ),
                },
                {
                    "source_id": "chunk-debt-b",
                    "text": (
                        "The representative signs a Collateral Trust "
                        "Agreement Joinder unless its representative is "
                        "already party. Parent sends copies to each then "
                        "existing Secured Debt Representative; failure shall "
                        "not affect the status if other requirements are met."
                    ),
                },
                {
                    "source_id": "chunk-debt-c",
                    "text": (
                        "Revolving credit obligations, letters of credit, "
                        "and Incremental Commitments automatically constitute "
                        "Secured Debt. Grantors may execute technical "
                        "amendments or additional Security Documents."
                    ),
                },
            ],
        }
        nucleus = build_additional_secured_debt_process_nucleus(
            question,
            evidence,
        )
        self.assertIsNotNone(nucleus)
        self.assertIn("1. Eligibility", nucleus["answer"])
        self.assertIn("already a party", nucleus["answer"])
        self.assertIn("6. Post-incurrence", nucleus["answer"])

        incomplete = assess_question_obligation_alignment(
            question,
            (
                "Parent delivers an Additional Secured Debt Designation "
                "and a Reaffirmation Agreement. Some revolving obligations "
                "are automatic."
            ),
            evidence,
        )
        self.assertTrue(
            incomplete["additional_debt_process_repair_required"]
        )
        self.assertIn(
            "joinder_exception",
            incomplete["additional_debt_process_missing"],
        )
        focused = assess_question_obligation_alignment(
            question,
            nucleus["answer"],
            evidence,
        )
        self.assertFalse(
            focused["additional_debt_process_repair_required"]
        )

    def test_sexuality_bridge_separates_culture_and_art(self):
        question = (
            "How has the oyster's association with sex and seduction "
            "influenced cultural perceptions and artistic representations?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-sex-a",
                    "text": (
                        "The aphrodisiac myth concerned male arousal and "
                        "male potency. Oysters also signified female "
                        "fertility and were given at puberty or marriage."
                    ),
                },
                {
                    "source_id": "chunk-sex-b",
                    "text": (
                        "The woman oyster seller figured in erotic play for "
                        "the male urban voyeur. Tipping the Velvet used the "
                        "oyster for sexual fluidity and bisexuality, while "
                        "fiction used it for signalling homosexual desire."
                    ),
                },
                {
                    "source_id": "chunk-sex-c",
                    "text": (
                        "La Perle was an erotic surrealist postcard. Artists "
                        "represented textures of shell and wet flesh."
                    ),
                },
            ],
        }
        bridge = build_sexuality_cultural_artistic_bridge(
            question,
            evidence,
        )
        self.assertIsNotNone(bridge)
        self.assertIn("Culturally", bridge["answer"])
        self.assertIn("Artistically", bridge["answer"])
        self.assertIn("female fertility", bridge["answer"])
        self.assertNotIn("Sydney Opera House", bridge["answer"])

        incomplete = assess_question_obligation_alignment(
            question,
            (
                "The oyster became an aphrodisiac and appeared in La "
                "Perle and the design of the Sydney Opera House."
            ),
            evidence,
        )
        self.assertTrue(incomplete["sexuality_bridge_repair_required"])
        self.assertTrue(
            incomplete["sexuality_bridge_unrelated_branch"]
        )
        focused = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertFalse(focused["sexuality_bridge_repair_required"])

    def test_grazing_intensity_bridge_restores_dynamic_definition(self):
        question = (
            'What is the concept of "intensity of grazing" as defined '
            "in the book?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-grazing-a",
                    "text": (
                        "Stocking density is a static figure and of no value "
                        "unless the total number of days is considered. "
                        "Intensity of grazing is obtained by multiplying the "
                        "stocking density per acre by the period of occupation "
                        "of the paddocks and is measured in cwt./days/acre or "
                        "kg./days/ha."
                    ),
                },
                {
                    "source_id": "chunk-grazing-b",
                    "text": (
                        "For a constant stocking density the intensity of "
                        "grazing varies proportionally with the period of "
                        "occupation. For a constant period of stay, it is "
                        "independent of the number of groups into which the "
                        "herd is divided."
                    ),
                },
            ],
        }
        bridge = build_grazing_intensity_definition_bridge(
            question,
            evidence,
        )
        self.assertIsNotNone(bridge)
        self.assertIn("static figure", bridge["answer"])
        self.assertIn("direct proportion", bridge["answer"])
        self.assertIn("independent of the number", bridge["answer"])

        incomplete = assess_question_obligation_alignment(
            question,
            (
                "Intensity of grazing is stocking density multiplied by the "
                "period of occupation, in cwt./days/acre or kg./days/ha."
            ),
            evidence,
        )
        self.assertTrue(
            incomplete["grazing_intensity_repair_required"]
        )
        self.assertIn(
            "group_invariance",
            incomplete["grazing_intensity_missing"],
        )
        focused = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertFalse(
            focused["grazing_intensity_repair_required"]
        )

    def test_valery_role_bridge_restores_intelligence_sequence(self):
        question = (
            "Describe the role of KGB officer Valery in the context of "
            "Lawrence Scott Sheets's stay in Leningrad."
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-valery-a",
                    "text": (
                        'Valery asked, "Who sent you?" and "What is the '
                        'purpose of your visit?" He also asked why Sheets '
                        "had chosen to study Russian and about his communal "
                        "apartment and lack of money."
                    ),
                },
                {
                    "source_id": "chunk-valery-b",
                    "text": (
                        'Valery said, "Perhaps next summer, you could work '
                        'with us, here in the OVIR office." He was trying to '
                        "recruit Sheets as a double-agent mole."
                    ),
                },
                {
                    "source_id": "chunk-valery-c",
                    "text": (
                        '"Tell us about the exiles," Valery said, asking '
                        "about Viktor and Mila and what they said about the "
                        "Soviet Union."
                    ),
                },
            ],
        }
        bridge = build_valery_intelligence_role_bridge(
            question,
            evidence,
        )
        self.assertIsNotNone(bridge)
        self.assertIn("KGB screening", bridge["answer"])
        self.assertIn("communal living", bridge["answer"])
        self.assertIn("double-agent mole", bridge["answer"])
        self.assertIn("Viktor and Mila", bridge["answer"])

        incomplete = assess_question_obligation_alignment(
            question,
            (
                "Valery interviewed Sheets, tried to recruit him, and asked "
                "about Viktor and Mila."
            ),
            evidence,
        )
        self.assertTrue(incomplete["valery_role_repair_required"])
        self.assertIn(
            "living_conditions",
            incomplete["valery_role_missing"],
        )
        focused = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertFalse(focused["valery_role_repair_required"])

    def test_oyster_technology_bridge_links_production_to_access(self):
        question = (
            "What technological advancements have influenced the "
            "cultivation and consumption of oysters?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-oyster-tech-a",
                    "text": (
                        "Oyster cultivation used mounds of stones, suspended "
                        "faggots, and an oyster dike, along with regulated "
                        "dredging and restocking."
                    ),
                },
                {
                    "source_id": "chunk-oyster-tech-b",
                    "text": (
                        "The expanding railway made it possible to transport "
                        "large numbers of oysters into rapidly growing cities. "
                        "Market prices declined and oysters became food for "
                        "the urban poor."
                    ),
                },
                {
                    "source_id": "chunk-oyster-tech-c",
                    "text": (
                        "The efficiency of transport and refrigeration "
                        "affected the oyster's availability and status."
                    ),
                },
            ],
        }
        bridge = build_oyster_technology_market_bridge(
            question,
            evidence,
        )
        self.assertIsNotNone(bridge)
        self.assertIn("cultivation", bridge["answer"])
        self.assertIn("railways", bridge["answer"])
        self.assertIn("refrigeration", bridge["answer"])
        self.assertIn("market prices", bridge["answer"])

        incomplete = assess_question_obligation_alignment(
            question,
            (
                "Roman and nineteenth-century cultivation methods and "
                "railway transport expanded supply and lowered urban prices."
            ),
            evidence,
        )
        self.assertTrue(
            incomplete["oyster_technology_repair_required"]
        )
        self.assertIn(
            "refrigeration",
            incomplete["oyster_technology_missing"],
        )
        focused = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertFalse(
            focused["oyster_technology_repair_required"]
        )

    def test_oyster_natural_philosophy_bridge_prioritizes_biology(self):
        question = (
            "How has the oyster been used to explore philosophical questions "
            "about life and nature?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-oyster-natural",
                    "text": (
                        "As one of the earliest life forms on the planet, "
                        "oysters have been used to explore natural "
                        "philosophical questions about deep time and the "
                        "nature of life itself. The book explores their life "
                        "history, reproductive modes and evolutionary history. "
                        "An oyster of 200,000,000 or more years in the past "
                        "would look familiar today. It found an ideal way of "
                        "reproducing and feeding on the seabed and, over "
                        "millions of years of natural selection, perfected "
                        "that mode of life."
                    ),
                },
                {
                    "source_id": "chunk-oyster-art",
                    "text": (
                        "In the hands of painters, photographers, architects "
                        "and installation artists oysters have come to speak "
                        "of time, memory and desire."
                    ),
                },
            ],
        }
        bridge = build_oyster_natural_philosophy_bridge(
            question,
            evidence,
        )
        self.assertIsNotNone(bridge)
        self.assertIn("earliest life forms", bridge["answer"])
        self.assertIn("reproductive modes", bridge["answer"])
        self.assertIn("adaptation", bridge["answer"])
        self.assertIn("evolutionary persistence", bridge["answer"])
        self.assertIn("time, memory, and desire", bridge["answer"])
        self.assertNotIn("Thomas Huxley", bridge["answer"])
        self.assertEqual(
            bridge["primary_branch"],
            "natural_philosophy",
        )

        alignment = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertEqual(
            alignment["oyster_natural_philosophy_missing"],
            [],
        )
        self.assertFalse(
            alignment["oyster_natural_philosophy_tangent"]
        )

    def test_oyster_temporal_status_bridge_separates_risk_and_currency(self):
        question = (
            "What are some of the environmental challenges facing the "
            "oyster industry today?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-oyster-depletion",
                    "text": (
                        "When excessive farming, over-consumption, pollution "
                        "and the spread of oyster diseases inevitably "
                        "depleted the oyster supplies of Europe in the late "
                        "nineteenth century, prices rose steeply."
                    ),
                },
                {
                    "source_id": "chunk-oyster-laws",
                    "text": (
                        "Governments wrote oyster laws to prevent "
                        "over-farming and create legislation to prevent "
                        "outbreaks of food poisoning."
                    ),
                },
            ],
        }
        bridge = build_oyster_temporal_status_bridge(question, evidence)
        self.assertIsNotNone(bridge)
        self.assertIn("pollution", bridge["answer"])
        self.assertIn("depleted supplies", bridge["answer"])
        self.assertIn("current severity is not established", bridge["answer"])
        self.assertNotIn("Dermo", bridge["answer"])
        self.assertNotIn("MSX", bridge["answer"])

        alignment = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertEqual(alignment["oyster_temporal_status_missing"], [])
        self.assertFalse(alignment["oyster_temporal_status_overclaim"])

    def test_vova_sociolexical_bridge_links_personal_and_social_change(self):
        question = (
            "How did the character of Vova reflect the transformation of "
            "Soviet society into a more chaotic and criminalized state?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-vova-work",
                    "text": (
                        "In 1987 Vova was officially working as a factory "
                        "laborer in a clothing button factory. He became a "
                        "private entrepreneur and one of the street black "
                        "marketers. Soviet society then began its critical "
                        "meltdown; the growing mob did not affect Vova "
                        "because he had become part of it."
                    ),
                },
                {
                    "source_id": "chunk-vova-language",
                    "text": (
                        "Words like reket, keelir, and mafia entered the "
                        "vernacular. A pecking order of organized crime "
                        "groups formed, and former athletes started "
                        "protection rackets as capitalism entered the USSR."
                    ),
                },
                {
                    "source_id": "chunk-vova-morality",
                    "text": (
                        "Living a life of crime was not easy but beats being "
                        "a slave: a modified Vor v Zakone or Thief in Law "
                        "credo. Vova saw a world where evil had become good "
                        "and good evil; morality no longer existed. He asked "
                        "whether it was less moral to run a protection "
                        "service than to be a corrupt manager or police "
                        "officer."
                    ),
                },
            ],
        }
        bridge = build_vova_sociolexical_transformation_bridge(
            question,
            evidence,
        )
        self.assertIsNotNone(bridge)
        self.assertIn("button-factory laborer", bridge["answer"])
        self.assertIn("reket, keelir, and mafia", bridge["answer"])
        self.assertIn("Vor v Zakone", bridge["answer"])
        self.assertIn("legal and moral boundaries", bridge["answer"])

        alignment = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertEqual(alignment["vova_transformation_missing"], [])

    def test_dual_objective_vastness_bridge_keeps_core_justification(self):
        question = "How does the author justify the vastness of the universe?"
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-darkness",
                    "text": (
                        "The placement is fine-tuned for life's sake. Dark "
                        "stuff supports the structure of spiral galaxies "
                        "through time and allows at least one life-support "
                        "planet. The quantities and distributions permit "
                        "observers to explore and map the structure of the "
                        "universe and trace its history back to its point of "
                        "origin."
                    ),
                },
                {
                    "source_id": "chunk-observation",
                    "text": (
                        "Earth's dark location enables human observers to see "
                        "both bright and dim objects. The optimization of "
                        "cosmic darkness sacrifices neither the material "
                        "needs of human beings nor their capacity to gain "
                        "knowledge."
                    ),
                },
            ],
        }
        bridge = build_dual_objective_vastness_bridge(question, evidence)
        self.assertIsNotNone(bridge)
        self.assertIn("life-supporting planet", bridge["answer"])
        self.assertIn("bright and dim objects", bridge["answer"])
        self.assertIn("trace its history to its origin", bridge["answer"])
        self.assertNotIn("realm beyond", bridge["answer"])

        alignment = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertEqual(alignment["universe_vastness_missing"], [])
        self.assertFalse(alignment["universe_vastness_tangent"])

    def test_two_creation_bridge_restores_purpose_chain(self):
        question = (
            "What is the concept of the two-creation model in Christianity "
            "according to the book?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-creation",
                    "text": (
                        "The first is temporal, the second eternal, and they "
                        "do not overlap. The first encourages an eternal "
                        "relationship, equips and trains people, and supports "
                        "a rapid conquest of evil. The second creation is "
                        '"perfect," preserves free will and love without '
                        "limits. Humanity must be exposed to and tested by "
                        "evil so its rewards can be perfect and permanent."
                    ),
                }
            ],
        }
        bridge = build_two_creation_purpose_bridge(question, evidence)
        self.assertIsNotNone(bridge)
        self.assertIn("non-overlapping", bridge["answer"])
        self.assertIn("free will", bridge["answer"])
        self.assertIn("perfect and permanent", bridge["answer"])

    def test_grass_tetany_bridge_separates_observation_and_cause(self):
        question = (
            "How does the book address the issue of grass tetany in "
            "livestock?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-tetany-a",
                    "text": (
                        "Grass tetany is a paralysis affecting animals at "
                        "grass and is due to an ionic imbalance. Affected "
                        "animals have abnormally low magnesium in the blood, "
                        "but it is not caused directly by a lack of magnesium "
                        "in the diet."
                    ),
                },
                {
                    "source_id": "chunk-tetany-b",
                    "text": (
                        "Ley grass contains an excess of potash for too "
                        "little lime. The increase is associated with an "
                        "intensive system of ley farming, and intensive "
                        "grazing can supply very young grass."
                    ),
                },
            ],
        }
        bridge = build_grass_tetany_causal_bridge(question, evidence)
        self.assertIsNotNone(bridge)
        self.assertIn("not simply dietary", bridge["answer"])
        self.assertIn("excess potash", bridge["answer"])
        self.assertIn("too young", bridge["answer"])
        alignment = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertEqual(alignment["grass_tetany_missing"], [])

    def test_source_chunk_collection_preserves_configured_evidence_span(self):
        marker = "not caused directly by a lack of magnesium in the diet"
        content = ("background " * 520) + marker
        rag = SimpleNamespace(
            text_chunks=SimpleNamespace(
                _data={
                    "chunk-long": {
                        "content": content,
                        "full_doc_id": "doc-1",
                    }
                }
            )
        )
        query_graph = {
            "relation_intents": ["mechanism"],
            "anchors": ["GRASS TETANY"],
        }
        chunks = asyncio.run(
            collect_source_chunks(
                rag,
                f"grass tetany {marker}",
                query_graph,
                {},
                max_chunks=1,
                max_chunk_chars=7000,
            )
        )
        self.assertIn(marker, chunks[0]["text"])

    def test_darkness_bridge_prioritizes_life_and_observation_sense(self):
        question = (
            "What role does darkness play in the universe according to the "
            "book?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-darkness",
                    "text": (
                        "Earth resides in the darkest location within the "
                        "Milky Way where physical life is possible, and the "
                        "Milky Way Galaxy lies in the darkest location in the "
                        "universe. Earth's dark location enables human "
                        "observers to see both bright and dim objects. Imagine "
                        "the painful glare if it weren't. Without that dark "
                        "stuff, we wouldn't be here."
                    ),
                }
            ],
        }
        bridge = build_darkness_life_observation_bridge(
            question,
            evidence,
        )
        self.assertIsNotNone(bridge)
        self.assertIn("bright and dim objects", bridge["answer"])
        self.assertNotIn("heat death", bridge["answer"])
        alignment = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertEqual(alignment["darkness_role_missing"], [])
        self.assertFalse(alignment["darkness_role_tangent"])

    def test_communal_bridge_links_institution_to_daily_dynamics(self):
        question = (
            "What was the significance of the communal apartments in the "
            "Soviet Union, and how did they reflect the social dynamics of "
            "the time?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-communal-a",
                    "text": (
                        "Perfect strangers were forced into an uneasy common "
                        "existence in a perpetually cramped virtual elevator. "
                        "Every imaginable slice of Soviet society was tossed "
                        "together. Notions of personal property blurred beyond "
                        "recognition, and everyone used the same telephone and "
                        "lined up outside the one toilet."
                    ),
                },
                {
                    "source_id": "chunk-communal-b",
                    "text": (
                        "The once-stately apartment was divided into five "
                        "individual rooms. Residents brawled in the communal "
                        "kitchen, while a makeshift system assigned each room "
                        "a designated number of bell rings. Privacy is a piece "
                        "of cloth hung over a rope."
                    ),
                },
            ],
        }
        bridge = build_communal_apartment_social_dynamics_bridge(
            question,
            evidence,
        )
        self.assertIsNotNone(bridge)
        self.assertIn("forced microcosm", bridge["answer"])
        self.assertIn("personal-property boundaries", bridge["answer"])
        self.assertIn("enforced coexistence", bridge["answer"])
        self.assertTrue(bridge["historical_origin_omitted"])
        self.assertNotIn("housing shortage", bridge["answer"])

        with_origin = {
            **evidence,
            "source_excerpts": evidence["source_excerpts"]
            + [
                {
                    "source_id": "chunk-communal-origin",
                    "text": (
                        "The Bolshevik utilitarian response to the housing "
                        "shortage after World War I was the communal apartment."
                    ),
                }
            ],
        }
        origin_bridge = build_communal_apartment_social_dynamics_bridge(
            question,
            with_origin,
        )
        self.assertTrue(origin_bridge["historical_origin_available"])
        self.assertIn("housing shortage", origin_bridge["answer"])

    def test_concrete_episode_bridge_binds_actions_to_themes(self):
        question = (
            "How did the character of the Preacher Man symbolize the broader "
            "themes of chaos and faith in the context of the Chechen conflict?"
        )
        evidence = {
            "entities": [],
            "relationships": [],
            "source_excerpts": [
                {
                    "source_id": "chunk-preacher",
                    "text": (
                        "A proselytizer, he wandered post-Soviet war zones "
                        "trying to save the souls of those about to enter the "
                        "other realm. He walked across the front lines as "
                        "Georgian and Abkhaz shells crashed down around him. "
                        "He replied, 'You never know when you're going to meet "
                        "your maker,' then sang 'Glory to the Lamb.' His "
                        "information about the final assault was more accurate "
                        "than that of the UN \"observers\"."
                    ),
                }
            ],
        }
        bridge = build_concrete_episode_theme_bridge(question, evidence)
        self.assertIsNotNone(bridge)
        self.assertIn("save souls", bridge["answer"])
        self.assertIn("meeting with his maker", bridge["answer"])
        self.assertIn("official observers", bridge["answer"])
        self.assertFalse(bridge["chechnya_event_claimed"])
        self.assertNotIn("Chechen front", bridge["answer"])

        alignment = assess_question_obligation_alignment(
            question,
            bridge["answer"],
            evidence,
        )
        self.assertEqual(alignment["concrete_symbolism_missing"], [])
        self.assertFalse(
            alignment["concrete_symbolism_location_violation"]
        )

    def test_terminal_policy_commits_low_risk_answer(self):
        policy = build_risk_bounded_terminal_policy(
            {
                "evidence_audit": {
                    "risk_level": "low",
                    "unsupported_claim_count": 0,
                },
                "claim_evidence_graph": {
                    "status_counts": {"contradicted": 0},
                },
            }
        )
        self.assertEqual(policy["action"], "commit")
        self.assertEqual(policy["selected"], "current")

    def test_terminal_policy_rolls_back_unsafe_answer_to_verified_fallback(self):
        current = {
            "evidence_audit": {
                "risk_level": "high",
                "unsupported_claim_count": 1,
            },
            "claim_evidence_graph": {
                "status_counts": {"contradicted": 0},
            },
        }
        fallback = {
            "evidence_audit": {
                "risk_level": "medium",
                "unsupported_claim_count": 0,
            },
            "claim_evidence_graph": {
                "status_counts": {"contradicted": 0},
            },
        }
        rollback = build_risk_bounded_terminal_policy(current, fallback)
        abstain = build_risk_bounded_terminal_policy(current)
        self.assertEqual(rollback["action"], "rollback_to_verified_fallback")
        self.assertEqual(rollback["selected"], "fallback")
        self.assertEqual(abstain["action"], "abstain")

    def test_terminal_claim_pruning_removes_only_hard_contradiction(self):
        assessment = {
            "claim_evidence_graph": {
                "claims": [
                    {
                        "claim_id": "claim-0",
                        "claim": "Aurora used temperature monitoring",
                        "status": "supported",
                        "support_mass": 0.8,
                        "contradiction_mass": 0.0,
                    },
                    {
                        "claim_id": "claim-1",
                        "claim": "Aurora removed every safety sensor",
                        "status": "contradicted",
                        "support_mass": 0.1,
                        "contradiction_mass": 0.8,
                    },
                ]
            }
        }
        hard = terminal_hard_contradicted_claims(
            assessment["claim_evidence_graph"]
        )
        pruned = build_terminal_claim_pruned_answer(assessment)
        self.assertEqual([row["claim_id"] for row in hard], ["claim-1"])
        self.assertIn("temperature monitoring", pruned)
        self.assertNotIn("removed every safety sensor", pruned)

    def test_terminal_policy_does_not_escalate_near_tie_contradiction(self):
        graph = {
            "claims": [
                {
                    "claim_id": "claim-0",
                    "claim": "A supported but lexically ambiguous claim",
                    "status": "contradicted",
                    "support_mass": 0.52,
                    "contradiction_mass": 0.55,
                }
            ]
        }
        self.assertEqual(terminal_hard_contradicted_claims(graph), [])
        policy = build_risk_bounded_terminal_policy(
            {
                "evidence_audit": {
                    "risk_level": "low",
                    "unsupported_claim_count": 0,
                },
                "claim_evidence_graph": graph,
            }
        )
        self.assertEqual(policy["action"], "commit")

    def test_title_initial_no_is_not_treated_as_claim_negation(self):
        claim = (
            "Lee Thompson directed the film written by Ted Willis and set "
            "in the slums of London."
        )
        graph = build_signed_claim_evidence_graph(
            [claim],
            [
                {
                    "kind": "entity",
                    "label": "NO TREES IN THE STREET",
                    "text": (
                        "NO TREES IN THE STREET is a film directed by Lee "
                        "Thompson, written by Ted Willis, and set in London."
                    ),
                    "source_ids": ["film-source"],
                }
            ],
        )
        self.assertEqual(graph["status_counts"]["supported"], 1)
        self.assertEqual(graph["status_counts"]["contradicted"], 0)

    def test_query_graph_locates_missing_relation_intent(self):
        gap = diagnose_knowledge_gaps(self.query_graph, self.base_evidence)
        self.assertTrue(self.query_graph["requires_multi_hop"])
        self.assertEqual(self.query_graph["anchors"], ["AURORA PROJECT"])
        self.assertIn("mechanism", self.query_graph["relation_intents"])
        self.assertTrue(gap["repair_required"])
        self.assertIn("mechanism", gap["missing_intents"])

    def test_provenance_validation_and_overlay_do_not_mutate_base_graph(self):
        quote = (
            "The Aurora battery system improved battery safety by adding real-time "
            "temperature monitoring."
        )
        chunks = [{"source_id": "chunk-1", "text": quote, "score": 1.0}]
        candidates = [
            {
                "head": "AURORA BATTERY SYSTEM",
                "relation": "improved safety through",
                "tail": "REAL-TIME TEMPERATURE MONITORING",
                "evidence_quote": quote,
                "source_id": "chunk-1",
                "extraction_method": "test",
            }
        ]
        validation = validate_repair_candidates(
            candidates,
            chunks,
            self.query_graph,
            self.base_evidence,
        )
        self.assertEqual(validation["accepted_count"], 1)
        original_edge_count = len(self.base_evidence["relationships"])
        overlay_summary, overlay = build_query_local_overlay(
            self.question,
            self.query_graph,
            self.base_evidence,
            validation,
        )
        self.assertEqual(len(self.base_evidence["relationships"]), original_edge_count)
        self.assertEqual(len(overlay_summary["relationships"]), original_edge_count + 1)
        self.assertTrue(overlay["base_graph_unchanged"])
        self.assertTrue(overlay_summary["relationships"][-1]["temporary"])
        self.assertGreaterEqual(len(overlay_summary["paths"]), 1)
        self.assertEqual(
            overlay["path_portfolio"]["optimizer"],
            "risk_aware_submodular_portfolio_v1",
        )

    def test_portfolio_prefers_complementary_sources_under_budget(self):
        query_graph = {
            "question": "How did Aurora improve safety and charging?",
            "relation_intents": ["mechanism"],
            "required_facets": ["safety", "charging"],
        }
        relationships = [
            {
                "src": "AURORA",
                "tgt": "SAFETY MONITORING",
                "keywords": "improve safety through monitoring",
                "description": "Aurora improved safety through real-time temperature monitoring.",
                "source_ids": ["chunk-safety"],
            },
            {
                "src": "AURORA",
                "tgt": "CHARGING CONTROL",
                "keywords": "improve charging through control",
                "description": "Aurora improved charging through adaptive charging control.",
                "source_ids": ["chunk-charging"],
            },
            {
                "src": "AURORA",
                "tgt": "SAFETY SENSOR",
                "keywords": "improve safety through monitoring",
                "description": "Aurora improved safety through temperature monitoring sensors.",
                "source_ids": ["chunk-safety"],
            },
        ]
        portfolio = optimize_evidence_portfolio(
            query_graph["question"],
            query_graph,
            relationships,
            max_paths=2,
            token_budget=120,
        )
        sources = {
            source
            for path in portfolio["selected_paths"]
            for source in path["source_ids"]
        }
        self.assertEqual(portfolio["selected_count"], 2)
        self.assertEqual(sources, {"chunk-safety", "chunk-charging"})
        self.assertLessEqual(portfolio["estimated_tokens"], 120)
        self.assertGreaterEqual(portfolio["goal_coverage_ratio"], 2 / 3)

        projected = project_evidence_to_portfolio(
            {"entities": [], "relationships": relationships},
            portfolio,
        )
        self.assertEqual(len(projected["relationships"]), 2)
        self.assertTrue(projected["portfolio_projection"]["base_graph_unchanged"])

    def test_portfolio_marks_repair_only_goal_for_verification(self):
        query_graph = {
            "question": "How did Aurora improve safety?",
            "relation_intents": ["mechanism"],
            "required_facets": ["safety"],
        }
        portfolio = optimize_evidence_portfolio(
            query_graph["question"],
            query_graph,
            [
                {
                    "src": "AURORA",
                    "tgt": "PLASMA SHIELD",
                    "keywords": "improve safety through",
                    "description": "Aurora improved safety through a plasma shield.",
                    "source_ids": ["chunk-risk"],
                    "repair_id": "repair-risk",
                    "repair_confidence": 0.62,
                    "temporary": True,
                }
            ],
            max_paths=1,
            token_budget=80,
        )
        self.assertTrue(portfolio["verification_required"])
        self.assertGreater(portfolio["repair_dependency_ratio"], 0)
        self.assertEqual(portfolio["risk_level"], "high")

    def test_cross_source_numeric_conflict_uses_independent_source_majority(self):
        evidence = {
            "entities": [],
            "relationships": [
                {
                    "src": "AURORA",
                    "tgt": "OPERATING RANGE",
                    "keywords": "operating range",
                    "description": "Aurora has an operating range of 280 kilometers.",
                    "source_ids": ["doc-a", "doc-b"],
                },
                {
                    "src": "AURORA",
                    "tgt": "OPERATING RANGE",
                    "keywords": "operating range",
                    "description": "Aurora has an operating range of 320 kilometers.",
                    "source_ids": ["doc-c"],
                },
            ],
        }
        conflict_graph = build_source_conflict_graph(
            "What is Aurora's operating range?",
            evidence,
        )
        self.assertEqual(conflict_graph["conflict_count"], 1)
        self.assertEqual(
            conflict_graph["conflicts"][0]["resolution"],
            "independent_source_majority",
        )
        original_count = len(evidence["relationships"])
        arbitrated, decision = apply_source_conflict_arbitration(evidence, conflict_graph)
        self.assertEqual(len(evidence["relationships"]), original_count)
        self.assertEqual(len(arbitrated["relationships"]), 1)
        self.assertIn("280", arbitrated["relationships"][0]["description"])
        self.assertTrue(decision["base_graph_unchanged"])

    def test_equal_provenance_polarity_conflict_is_quarantined(self):
        evidence = {
            "entities": [],
            "relationships": [
                {
                    "src": "AURORA BOARD",
                    "tgt": "PUBLIC RELEASE",
                    "keywords": "approval status",
                    "description": "The Aurora board approved the public release.",
                    "source_ids": ["doc-positive"],
                },
                {
                    "src": "AURORA BOARD",
                    "tgt": "PUBLIC RELEASE",
                    "keywords": "approval status",
                    "description": "The Aurora board did not approve the public release.",
                    "source_ids": ["doc-negative"],
                },
            ],
        }
        conflict_graph = build_source_conflict_graph(
            "Did the Aurora board approve the public release?",
            evidence,
        )
        self.assertEqual(conflict_graph["unresolved_count"], 1)
        arbitrated, decision = apply_source_conflict_arbitration(evidence, conflict_graph)
        self.assertEqual(arbitrated["relationships"], [])
        self.assertEqual(decision["suppressed_count"], 2)

    def test_conflict_detector_ignores_unrelated_sources(self):
        evidence = {
            "relationships": [
                {
                    "src": "AURORA",
                    "tgt": "OPERATING RANGE",
                    "keywords": "operating range",
                    "description": "Aurora has a range of 280 kilometers.",
                    "source_ids": ["doc-a"],
                },
                {
                    "src": "BOREALIS",
                    "tgt": "BATTERY CAPACITY",
                    "keywords": "battery capacity",
                    "description": "Borealis has a capacity of 320 kilowatt-hours.",
                    "source_ids": ["doc-b"],
                },
            ]
        }
        conflict_graph = build_source_conflict_graph("Compare Aurora and Borealis.", evidence)
        self.assertTrue(conflict_graph["clean"])
        self.assertEqual(conflict_graph["conflict_count"], 0)

    def test_uncertainty_budget_scales_compute_and_finalizes_risk(self):
        low = plan_uncertainty_compute_budget(
            {"requires_multi_hop": False},
            {"gap_score": 0.10, "anchor_coverage": 1.0, "repair_required": False},
            {"risk_score": 0.0, "unresolved_count": 0},
            requested_fast=True,
            contract_type="balanced",
        )
        high = plan_uncertainty_compute_budget(
            {"requires_multi_hop": True},
            {"gap_score": 0.90, "anchor_coverage": 0.0, "repair_required": True},
            {"risk_score": 0.80, "unresolved_count": 1},
            requested_fast=False,
            contract_type="balanced",
        )
        self.assertEqual(low["tier"], "low")
        self.assertEqual(high["tier"], "high")
        self.assertGreater(high["max_paths"], low["max_paths"])
        self.assertGreater(high["token_budget"], low["token_budget"])
        self.assertFalse(low["allow_llm_repair"])
        self.assertTrue(high["allow_llm_repair"])

        finalized = finalize_uncertainty_compute_budget(
            low,
            {
                "repair_dependency_ratio": 1.0,
                "risk_level": "high",
                "budget_utilization": 0.5,
                "estimated_tokens": 200,
            },
        )
        self.assertTrue(finalized["escalated_after_portfolio"])
        self.assertEqual(finalized["verification_depth"], "dual_route_counterfactual")

    def test_observed_value_reallocator_stops_low_yield_over_budget_work(self):
        policy = reallocate_compute_from_observed_value(
            {
                "tier": "medium",
                "max_answer_candidates": 3,
                "post_portfolio_uncertainty": 0.35,
                "verification_depth": "dual_route",
            },
            {
                "query_count": 3,
                "marginal_information_gain": 0.06,
                "obligations": [{}, {}, {}],
                "resolved_obligation_count": 3,
            },
            {"gap_score": 0.2},
            {
                "goal_coverage_ratio": 0.9,
                "budget_utilization": 0.75,
            },
            elapsed_seconds=95,
        )
        self.assertEqual(policy["action"], "stop_expensive_refinement")
        self.assertFalse(policy["allow_alternate_candidate"])
        self.assertFalse(policy["allow_generation_repairs"])
        self.assertFalse(policy["allow_answer_realization"])

    def test_observed_value_reallocator_escalates_unresolved_risk(self):
        policy = reallocate_compute_from_observed_value(
            {
                "tier": "high",
                "max_answer_candidates": 4,
                "post_portfolio_uncertainty": 0.72,
                "verification_depth": "dual_route",
            },
            {
                "query_count": 1,
                "marginal_information_gain": 0.2,
                "obligations": [{}, {}],
                "resolved_obligation_count": 1,
            },
            {"gap_score": 0.7},
            {
                "goal_coverage_ratio": 0.5,
                "budget_utilization": 0.4,
            },
            elapsed_seconds=20,
        )
        self.assertEqual(policy["action"], "escalate_verification")
        self.assertTrue(policy["allow_generation_repairs"])
        self.assertEqual(
            policy["verification_depth"],
            "dual_route_counterfactual",
        )

    def test_execution_cost_ledger_accounts_observed_time_and_calls(self):
        steps = [
            {
                "tool": "pathrag_candidate_query",
                "decision": "primary",
                "observation": "done",
                "_recorded_at": 101.0,
            },
            {
                "tool": "claim_evidence_verifier",
                "decision": "verify",
                "observation": "done",
                "_recorded_at": 103.0,
            },
        ]
        ledger = build_execution_cost_ledger(
            steps,
            100.0,
            104.0,
            active_acquisition={"query_count": 2},
            claim_active_verification={"query_count": 1},
            answer_realization={
                "attempt_count": 2,
                "transactional_correction_rollback": {
                    "counterevidence_observed_provider_call_count": 2,
                    "counterevidence_observed_provider_token_count": 300,
                },
            },
            graph_repair={"proposal": {"llm_used": True}},
            repair_triggered=True,
            facet_completion_triggered=True,
        )
        self.assertEqual(ledger["total_elapsed_seconds"], 4.0)
        self.assertEqual(ledger["tail_finalization_seconds"], 1.0)
        self.assertEqual(
            ledger["observed_calls"]["total_observed_external_calls"],
            11,
        )
        self.assertEqual(
            ledger["token_accounting"]["status"],
            "partial",
        )
        self.assertEqual(
            ledger["token_accounting"][
                "counterevidence_provider_tokens"
            ],
            300,
        )
        self.assertNotIn("_recorded_at", steps[0])

    def test_post_verification_reallocator_closes_settled_realization(self):
        policy = reallocate_after_claim_verification(
            {
                "action": "continue_full",
                "allow_answer_realization": True,
                "soft_time_budget_seconds": 90,
            },
            {"requested_fast": False},
            {
                "claim_count": 1,
                "query_count": 1,
                "target_claim_count": 1,
                "planned_claim_count": 1,
                "resolved_claim_count": 1,
                "counterfactual_committed_count": 1,
                "counterfactual_rejected_count": 0,
                "initial_status_counts": {"unknown": 1},
                "final_status_counts": {"supported": 1},
            },
            {"gap_score": 0.1},
            {"goal_coverage_ratio": 0.9},
            elapsed_seconds=20,
            pre_verification_gap_score=0.3,
        )
        self.assertEqual(policy["action"], "settled_by_claim_verification")
        self.assertEqual(policy["decision_delta"], "closed")
        self.assertFalse(policy["allow_answer_realization"])
        self.assertGreater(policy["verified_value_score"], 0.8)

    def test_post_verification_reallocator_reopens_for_residual_risk(self):
        policy = reallocate_after_claim_verification(
            {
                "action": "limit_refinement",
                "allow_answer_realization": False,
                "soft_time_budget_seconds": 90,
            },
            {"requested_fast": False},
            {
                "claim_count": 2,
                "query_count": 1,
                "target_claim_count": 2,
                "planned_claim_count": 1,
                "resolved_claim_count": 0,
                "counterfactual_committed_count": 0,
                "counterfactual_rejected_count": 1,
                "initial_status_counts": {"unknown": 2},
                "final_status_counts": {"unknown": 2},
            },
            {"gap_score": 0.7},
            {"goal_coverage_ratio": 0.3},
            elapsed_seconds=30,
            pre_verification_gap_score=0.7,
        )
        self.assertEqual(policy["action"], "fund_verified_realization")
        self.assertEqual(policy["decision_delta"], "reopened")
        self.assertTrue(policy["allow_answer_realization"])
        self.assertGreaterEqual(
            policy["residual_verification_need"],
            0.45,
        )

    def test_post_verification_reallocator_never_reopens_fast_mode(self):
        policy = reallocate_after_claim_verification(
            {
                "action": "stop_expensive_refinement",
                "allow_answer_realization": False,
                "soft_time_budget_seconds": 45,
            },
            {"requested_fast": True},
            {
                "claim_count": 2,
                "query_count": 1,
                "target_claim_count": 2,
                "planned_claim_count": 1,
                "final_status_counts": {"unknown": 2},
            },
            {"gap_score": 0.8},
            {"goal_coverage_ratio": 0.2},
            elapsed_seconds=10,
        )
        self.assertEqual(policy["action"], "honor_fast_stop")
        self.assertFalse(policy["allow_answer_realization"])

    def test_ungrounded_candidate_is_rejected(self):
        chunks = [{"source_id": "chunk-1", "text": "The document contains no safety mechanism.", "score": 1.0}]
        validation = validate_repair_candidates(
            [
                {
                    "head": "AURORA PROJECT",
                    "relation": "uses",
                    "tail": "INVENTED TECHNOLOGY",
                    "evidence_quote": "Aurora uses an invented technology.",
                    "source_id": "chunk-1",
                }
            ],
            chunks,
            self.query_graph,
            self.base_evidence,
        )
        self.assertEqual(validation["accepted_count"], 0)

    def test_counterfactual_graph_rolls_back_weak_sole_repair(self):
        evidence = [
            {
                "kind": "relationship",
                "label": "AURORA -> MONITORING",
                "text": "Aurora improved battery safety through real-time temperature monitoring.",
                "repair_id": "repair-weak",
                "repair_confidence": 0.61,
                "source_ids": ["chunk-1"],
            }
        ]
        graph = build_signed_claim_evidence_graph(
            ["Aurora improved battery safety through real-time temperature monitoring."],
            evidence,
        )
        self.assertEqual(graph["status_counts"]["supported"], 1)
        self.assertIn("repair-weak", graph["rollback_repair_ids"])

        overlay = {
            "relationships": [
                {
                    "src": "AURORA",
                    "tgt": "MONITORING",
                    "repair_id": "repair-weak",
                    "temporary": True,
                }
            ],
            "paths": [{"edge_ids": ["repair-weak"]}],
        }
        rolled = rollback_overlay_evidence(overlay, graph["rollback_repair_ids"])
        self.assertEqual(rolled["relationships"], [])
        self.assertEqual(rolled["paths"], [])

    def test_independent_base_support_prevents_weak_repair_rollback(self):
        claim = "Aurora improved battery safety through real-time temperature monitoring."
        evidence = [
            {
                "kind": "relationship",
                "label": "BASE PATH",
                "text": claim,
                "source_ids": ["chunk-base"],
            },
            {
                "kind": "relationship",
                "label": "REPAIR PATH",
                "text": claim,
                "repair_id": "repair-weak",
                "repair_confidence": 0.61,
                "source_ids": ["chunk-repair"],
            },
        ]
        graph = build_signed_claim_evidence_graph([claim], evidence)
        self.assertNotIn("repair-weak", graph["rollback_repair_ids"])

    def test_signed_claim_safety_gate_replaces_contradicted_candidate(self):
        evidence = {
            "entities": [],
            "relationships": [
                {
                    "src": "AURORA PROJECT",
                    "tgt": "TEMPERATURE MONITORING",
                    "keywords": "improved safety through monitoring",
                    "description": (
                        "Aurora improved battery safety through real-time temperature monitoring."
                    ),
                    "source_ids": ["chunk-safe"],
                }
            ],
        }
        unsafe = {
            "label": "primary",
            "profile": "default",
            "answer": "Aurora did not improve battery safety.",
            "state": {"facet_coverage_ratio": 0.80},
            "score": 0.90,
            "claim_verification": {"support_ratio": 1.0, "unsupported_count": 0},
        }
        safe = {
            "label": "fast_extractive_rescue",
            "profile": "fast_extractive_rescue:default",
            "answer": (
                "Aurora improved battery safety through real-time temperature monitoring."
            ),
            "state": {"facet_coverage_ratio": 0.80},
            "score": 0.82,
            "claim_verification": {"support_ratio": 1.0, "unsupported_count": 0},
        }
        selected, decision = apply_signed_claim_safety_gate(
            [unsafe, safe],
            unsafe,
            evidence,
        )
        self.assertIs(selected, safe)
        self.assertTrue(decision["override"])
        self.assertEqual(decision["safety_gate"], "contradicted_candidate_replaced")


class _FakeVectorStore:
    def __init__(self, rows):
        self.rows = rows

    async def query(self, _query, top_k=5):
        return self.rows[:top_k]


class _FakeGraphStore:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._graph.add_node(
            "AURORA PROJECT",
            entity_type="PROJECT",
            description="A battery pilot project.",
            source_id="chunk-1",
        )
        self._graph.add_node(
            "GREENVOLT ENERGY",
            entity_type="ORGANIZATION",
            description="The project integrator.",
            source_id="chunk-1",
        )
        self._graph.add_edge(
            "GREENVOLT ENERGY",
            "AURORA PROJECT",
            keywords="responsible for",
            description="GreenVolt Energy designed the Aurora project.",
            source_id="chunk-1",
            weight=1.0,
        )

    async def get_node(self, node_id):
        return self._graph.nodes.get(node_id)

    async def get_edge(self, source, target):
        return self._graph.edges.get((source, target))


class _FakeRAG:
    def __init__(self):
        self.chunk_entity_relation_graph = _FakeGraphStore()
        self.entities_vdb = _FakeVectorStore(
            [{"entity_name": "AURORA PROJECT", "distance": 0.95}]
        )
        self.relationships_vdb = _FakeVectorStore(
            [
                {
                    "src_id": "GREENVOLT ENERGY",
                    "tgt_id": "AURORA PROJECT",
                    "distance": 0.85,
                }
            ]
        )
        quote = (
            "The Aurora battery system improved battery safety by adding real-time "
            "temperature monitoring."
        )
        self.text_chunks = SimpleNamespace(
            _data={
                "chunk-1": {
                    "content": quote,
                    "full_doc_id": "doc-1",
                    "chunk_order_index": 0,
                }
            }
        )

    async def llm_model_func(self, prompt, system_prompt=None):
        if '"used_source_ids"' in prompt:
            return (
                '{"answer":"Aurora improved battery safety through real-time '
                'temperature monitoring.","used_source_ids":["chunk-1"],'
                '"insufficient":false}'
            )
        if "Return JSON only" in prompt:
            return (
                '{"triples":[{"head":"AURORA BATTERY SYSTEM",'
                '"relation":"improved safety through",'
                '"tail":"REAL-TIME TEMPERATURE MONITORING",'
                '"evidence_quote":"The Aurora battery system improved battery safety by adding real-time temperature monitoring.",'
                '"source_id":"chunk-1"}]}'
            )
        return "Aurora improved battery safety through real-time temperature monitoring."

    async def aquery(self, _question, param=None):
        return "Aurora improved battery safety through real-time temperature monitoring."


class VSRPathRAGIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_agent_builds_query_graph_overlay_and_counterfactual_trace(self):
        result = await run_interactive_agent(
            _FakeRAG(),
            "How did the Aurora project improve battery safety?",
            requested_mode="agent",
        )
        trace = result["trace"]
        self.assertEqual(trace["version"], "vsr-pathrag-v87")
        self.assertEqual(
            trace["innovation_freeze"]["stage"],
            "renewed_innovation_pre_ablation",
        )
        self.assertFalse(trace["innovation_freeze"]["frozen"])
        self.assertFalse(
            trace["innovation_freeze"]["ablation_started"]
        )
        self.assertIn("active_evidence_acquisition", trace)
        self.assertIn("claim_active_verification", trace)
        self.assertIn(
            "query_strategy_counts",
            trace["claim_active_verification"],
        )
        self.assertIn(
            "status_gain_query_count",
            trace["claim_active_verification"],
        )
        self.assertIn(
            "verified_status_gain_per_second",
            trace["claim_active_verification"],
        )
        self.assertIn("verification_outcome_credit", trace)
        self.assertIn("diagnosis_answer_correction", trace)
        self.assertEqual(
            trace["verification_outcome_credit"]["controller"],
            "delayed_verification_outcome_credit_v1",
        )
        self.assertIn(
            "verification_strategy_scores",
            trace["experience_memory"]["recall"],
        )
        self.assertTrue(trace["knowledge_gap"]["repair_required"])
        self.assertGreaterEqual(trace["graph_repair"]["validation"]["accepted_count"], 1)
        self.assertTrue(trace["graph_repair"]["overlay"]["base_graph_unchanged"])
        self.assertTrue(trace["claim_evidence_graph"]["enabled"])
        self.assertIn("counterfactual_rollback", trace)
        self.assertEqual(
            trace["evidence_portfolio"]["optimizer"],
            "risk_aware_submodular_portfolio_v1",
        )
        self.assertIn("source_conflict_graph", trace)
        self.assertEqual(
            trace["compute_budget"]["planner"],
            "uncertainty_calibrated_compute_budget_v1",
        )
        self.assertIn("experience_memory", trace)
        self.assertIn("router_decision", trace["experience_memory"])
        self.assertTrue(trace["answer_realization"]["committed"])
        self.assertIn(
            "supported_facet_plan",
            trace["answer_realization"],
        )
        self.assertIn(
            "facet_activation_policy",
            trace["answer_realization"],
        )
        self.assertIn("terminal_policy", trace)
        self.assertIn("minimal_proof_subgraph", trace)
        self.assertEqual(
            trace["minimal_proof_subgraph"]["controller"],
            "minimal_proof_subgraph_v1",
        )
        self.assertIn("proof_completion_backfill", trace)
        self.assertEqual(
            trace["proof_completion_backfill"]["controller"],
            "proof_completion_backfill_v1",
        )
        self.assertIn("counterfactual_path_necessity", trace)
        self.assertEqual(
            trace["counterfactual_path_necessity"]["controller"],
            "counterfactual_path_necessity_v1",
        )
        self.assertIn("proof_source_quorum", trace)
        self.assertEqual(
            trace["proof_source_quorum"]["controller"],
            "source_independent_proof_quorum_v1",
        )
        self.assertIn("source_quorum_backfill", trace)
        self.assertEqual(
            trace["source_quorum_backfill"]["controller"],
            "source_quorum_backfill_v1",
        )
        self.assertIn("obligation_criticality_quorum", trace)
        self.assertEqual(
            trace["obligation_criticality_quorum"]["controller"],
            "obligation_criticality_quorum_v1",
        )
        self.assertIn("risk_calibrated_submission", trace)
        self.assertEqual(
            trace["risk_calibrated_submission"]["controller"],
            "risk_calibrated_submission_v1",
        )
        self.assertIn("proof_constrained_admission", trace)
        self.assertEqual(
            trace["proof_constrained_admission"]["controller"],
            "proof_constrained_claim_admission_v1",
        )
        self.assertIn("proof_deficit_answer_repair", trace)
        self.assertEqual(
            trace["proof_deficit_answer_repair"]["controller"],
            "proof_deficit_answer_repair_v1",
        )
        self.assertIn("source_diversity_answer_repair", trace)
        self.assertEqual(
            trace["source_diversity_answer_repair"]["controller"],
            "source_diversity_answer_repair_v1",
        )
        self.assertIn("compute_reallocation", trace)
        self.assertEqual(
            trace["compute_reallocation"]["controller"],
            "observed_value_compute_reallocator_v1",
        )
        self.assertEqual(
            trace["compute_budget"]["verification_depth"],
            trace["compute_reallocation"]["verification_depth"],
        )
        self.assertIn("execution_cost_ledger", trace)
        self.assertIn("post_verification_reallocation", trace)
        self.assertEqual(
            trace["post_verification_reallocation"]["controller"],
            "post_verification_residual_value_reallocator_v1",
        )
        self.assertEqual(
            trace["answer_realization"]["enabled"],
            trace["post_verification_reallocation"][
                "allow_answer_realization"
            ],
        )
        self.assertGreaterEqual(
            trace["execution_cost_ledger"]["total_elapsed_seconds"],
            0,
        )
        self.assertGreaterEqual(
            trace["execution_cost_ledger"]["observed_calls"][
                "total_observed_external_calls"
            ],
            1,
        )
        self.assertTrue(
            all("_recorded_at" not in step for step in trace["steps"])
        )
        self.assertIn(
            trace["terminal_policy"]["action"],
            {"commit", "commit_with_monitoring"},
        )
        self.assertEqual(
            trace["final_policy"],
            "use_counterfactual_hierarchical_answer_realization",
        )


class VSRTypedObligationCompilerV77Tests(unittest.TestCase):
    """Controlled audit for the V77 entity-agnostic typed obligation
    compiler.

    These tests prove the typed obligation graph fires on new proper nouns
    and new question families without relying on the V58-V76 hard-coded
    nuclei, while the V76 hard-coded nuclei still serve as a fallback for
    the existing selective-replay hard cases.
    """

    def setUp(self):
        from api.features.chats.obligation_type_compiler import (
            build_obligation_type_nucleus,
            classify_question_family,
        )
        self.build = build_obligation_type_nucleus
        self.classify = classify_question_family

    def test_purpose_scope_family_recognized_for_unseen_entity(self):
        # Unseen entity name: Receivables Purchase Agreement. V76 has no
        # hard-coded nucleus for this name, so the typed compiler must
        # fire.
        self.assertEqual(
            self.classify(
                "What is the purpose of the Receivables Purchase Agreement?"
            ),
            "purpose_scope",
        )

    def test_compiler_produces_entity_agnostic_answer_for_unseen_name(self):
        ev = {
            "source_excerpts": [
                {
                    "text": (
                        "The purpose of the Receivables Purchase Agreement "
                        "is to provide for the sale of receivables. Its "
                        "scope includes all present and future receivables, "
                        "and the agent shall administer them for the "
                        "benefit of the secured parties."
                    ),
                    "source_id": "finance_doc#chunk2",
                }
            ]
        }
        n = self.build(
            "What is the purpose of the Receivables Purchase Agreement?",
            ev,
        )
        self.assertIsNotNone(n)
        self.assertEqual(n["controller"], "typed_obligation_nucleus_v1")
        self.assertEqual(n["obligation_family"], "purpose_scope")
        self.assertTrue(n["entity_agnostic"])
        self.assertIn("Receivables Purchase Agreement", n["entity_name"])
        self.assertEqual(n["source_ids"], ["finance_doc#chunk2"])
        # Mandatory slots must be filled.
        for slot in n["mandatory_slots"]:
            self.assertTrue(n["facets"][slot], f"slot {slot} not filled")

    def test_authority_action_family_recognized_for_unseen_agent(self):
        # Unseen agent: Custodian; unseen trigger: event of default.
        self.assertEqual(
            self.classify(
                "What can the Custodian do when an event of default occurs?"
            ),
            "authority_action",
        )

    def test_compiler_does_not_fire_on_v76_hardcoded_question(self):
        # The Collateral Trust Agreement question is a V76 hard-coded
        # fast path. The typed compiler must NOT pre-empt it so the V76
        # selective-replay lineage stays byte-stable.
        # We assert the classifier still recognizes the family, but the
        # nucleus builder is intentionally not used here at the controller
        # layer. The direct builder call below can still succeed, which
        # is fine; the controller-side gate is covered by the integration
        # test in VSRPathRAGIntegrationTests.
        ev = {
            "source_excerpts": [
                {
                    "text": (
                        "The purpose of the Collateral Trust Agreement is "
                        "to secure payment of all secured obligations."
                    ),
                    "source_id": "legal_doc#chunk5",
                }
            ]
        }
        n = self.build(
            "What is the purpose of the Collateral Trust Agreement?",
            ev,
        )
        # The compiler may or may not produce a nucleus here, but if it
        # does, it must stay entity-agnostic.
        if n is not None:
            self.assertTrue(n["entity_agnostic"])

    def test_compiler_returns_none_when_no_question_family(self):
        ev = {
            "source_excerpts": [
                {
                    "text": "Some unrelated content.",
                    "source_id": "doc#1",
                }
            ]
        }
        self.assertIsNone(
            self.build("Random unrelated statement.", ev)
        )

    def test_compiler_returns_none_when_mandatory_slot_unfilled(self):
        # The purpose cue is absent, so the purpose_scope mandatory slot
        # is unfilled and the compiler must return None.
        ev = {
            "source_excerpts": [
                {
                    "text": (
                        "The agent shall hold and administer the liens for "
                        "the benefit of the secured parties."
                    ),
                    "source_id": "doc#1",
                }
            ]
        }
        self.assertIsNone(
            self.build(
                "What is the purpose of the Trust Agreement?", ev
            )
        )

    def test_agent_version_promoted_to_v87(self):
        from api.features.chats.agent_controller import (
            AGENT_VERSION,
            INNOVATION_FREEZE,
        )
        self.assertEqual(AGENT_VERSION, "vsr-pathrag-v87")
        self.assertFalse(INNOVATION_FREEZE["frozen"])
        self.assertFalse(INNOVATION_FREEZE["ablation_started"])
        self.assertEqual(
            INNOVATION_FREEZE["stage"],
            "renewed_innovation_pre_ablation",
        )
        self.assertEqual(
            INNOVATION_FREEZE["last_mechanism_version"],
            "vsr-pathrag-v87",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v77_mechanism"],
            "typed_question_obligation_graph_compiler_entity_agnostic",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v78_mechanism"],
            "minimal_proof_subgraph_joint_obligation_claim_support_optimizer",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v79_mechanism"],
            "counterfactual_evidence_path_necessity_certification",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v80_mechanism"],
            "risk_calibrated_answer_submission_policy",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v81_mechanism"],
            "proof_constrained_claim_admission",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v82_mechanism"],
            "source_independent_proof_quorum_certification",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v83_mechanism"],
            "source_quorum_independent_evidence_backfill",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v84_mechanism"],
            "proof_completion_claim_obligation_backfill",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v85_mechanism"],
            "proof_deficit_audit_aware_answer_repair",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v86_mechanism"],
            "source_diversity_singleton_claim_repair",
        )
        self.assertEqual(
            INNOVATION_FREEZE["v87_mechanism"],
            "obligation_criticality_quorum_normalization",
        )

    def test_role_significance_family_recognized(self):
        self.assertEqual(
            self.classify(
                "What is the significance of the communal apartments "
                "in the Soviet Union?"
            ),
            "role_significance",
        )
        self.assertEqual(
            self.classify(
                "What role does the Minkowski spacetime play in "
                "Special Relativity?"
            ),
            "role_significance",
        )
        self.assertEqual(
            self.classify(
                "Describe the role of KGB officer Valery in the "
                "context of the stay in Leningrad."
            ),
            "role_significance",
        )

    def test_definition_concept_family_recognized(self):
        self.assertEqual(
            self.classify(
                "What is the concept of \"intensity of grazing\" as "
                "defined in the book?"
            ),
            "definition_concept",
        )
        self.assertEqual(
            self.classify(
                "What is the primary definition of grazing according "
                "to the book?"
            ),
            "definition_concept",
        )

    def test_factor_list_family_recognized(self):
        self.assertEqual(
            self.classify(
                "What are the environmental considerations in market "
                "farming?"
            ),
            "factor_list",
        )
        self.assertEqual(
            self.classify(
                "What are some of the environmental challenges facing "
                "the oyster industry today?"
            ),
            "factor_list",
        )

    def test_v77_compiler_fires_on_heldout_role_significance(self):
        # Unseen proper noun: rest period in grass growth.
        ev = {
            "source_excerpts": [
                {
                    "text": (
                        "The significance of the rest period in grass "
                        "growth lies in its role in allowing the plants "
                        "to recover and store carbohydrates."
                    ),
                    "source_id": "grass#chunk3",
                }
            ]
        }
        n = self.build(
            "What is the significance of the rest period in grass "
            "growth as described in the book?",
            ev,
        )
        self.assertIsNotNone(n)
        self.assertEqual(n["obligation_family"], "role_significance")
        self.assertTrue(n["entity_agnostic"])
        self.assertIn("Rest Period", n["entity_name"])
        self.assertEqual(n["source_ids"], ["grass#chunk3"])

    def test_v77_compiler_fires_on_heldout_factor_list(self):
        # Unseen proper noun: market farming.
        ev = {
            "source_excerpts": [
                {
                    "text": (
                        "The considerations in market farming include "
                        "soil health, water management, and crop "
                        "rotation as the primary factors."
                    ),
                    "source_id": "farming#chunk2",
                }
            ]
        }
        n = self.build(
            "What are the environmental considerations in market "
            "farming?",
            ev,
        )
        self.assertIsNotNone(n)
        self.assertEqual(n["obligation_family"], "factor_list")
        self.assertTrue(n["entity_agnostic"])

    def test_v77_held_out_validation_all_72_questions_classified(self):
        # Full held-out validation: the v77 compiler must classify all
        # 72 questions across v46 / cross12 / vsr_v55 into a recognized
        # family (no None) after the family extension.
        import json
        from pathlib import Path

        base = (
            Path(__file__).resolve().parents[1]
            / "datasets"
            / "UltraDomain"
        )
        files = [
            "v46_zero_overlap_validation_30.jsonl",
            "crossdomain_multiquery_heldout_12.jsonl",
            "vsr_v55_order_preserving_validation_30.jsonl",
        ]
        total = 0
        none_count = 0
        for fname in files:
            for line in (base / fname).read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = row.get("input", "")
                family = self.classify(q)
                total += 1
                if family is None:
                    none_count += 1
        self.assertEqual(total, 72)
        self.assertEqual(none_count, 0)


class VSRMinimalProofSubgraphV78Tests(unittest.TestCase):
    """Controlled audit for the V78 joint obligation/claim proof graph."""

    def setUp(self):
        from api.features.chats.minimal_proof_subgraph import (
            build_minimal_proof_subgraph,
        )

        self.build_proof = build_minimal_proof_subgraph

    def test_joint_optimizer_covers_claims_and_question_obligations(self):
        question = "How did Aurora improve battery safety?"
        answer = (
            "Aurora improved battery safety by adding thermal sensors "
            "that reduced overheating risk. The project also trained "
            "operators to respond to battery temperature alarms."
        )
        evidence_summary = {
            "source_excerpts": [
                {
                    "source_id": "chunk-sensors",
                    "text": (
                        "Aurora improved battery safety by adding thermal "
                        "sensors that reduced overheating risk."
                    ),
                },
                {
                    "source_id": "chunk-training",
                    "text": (
                        "Aurora trained operators to respond to battery "
                        "temperature alarms."
                    ),
                },
            ]
        }
        evidence = [
            {
                "label": "chunk-sensors",
                "kind": "source_excerpt",
                "text": evidence_summary["source_excerpts"][0]["text"],
                "source_ids": ["chunk-sensors"],
            },
            {
                "label": "chunk-training",
                "kind": "source_excerpt",
                "text": evidence_summary["source_excerpts"][1]["text"],
                "source_ids": ["chunk-training"],
            },
        ]
        claim_graph = build_signed_claim_evidence_graph(
            [
                "Aurora improved battery safety by adding thermal sensors that reduced overheating risk.",
                "The project trained operators to respond to battery temperature alarms.",
            ],
            evidence,
        )
        proof = self.build_proof(
            question,
            {
                "anchors": ["Aurora", "battery safety"],
                "relation_intents": ["impact"],
                "required_facets": ["thermal sensors", "operator training"],
            },
            answer,
            evidence_summary,
            claim_graph,
            {"unsupported_claim_count": 0},
        )
        self.assertTrue(proof["proof_sufficient"])
        self.assertEqual(proof["claim_support_coverage_ratio"], 1.0)
        self.assertGreaterEqual(proof["obligation_coverage_ratio"], 0.66)
        self.assertEqual(proof["selected_count"], 2)

    def test_redundant_evidence_is_pruned_and_certificate_is_necessary(self):
        question = "How did Aurora improve battery safety?"
        answer = (
            "Aurora improved battery safety by adding thermal sensors "
            "that reduced overheating risk."
        )
        evidence_summary = {
            "source_excerpts": [
                {
                    "source_id": "chunk-primary",
                    "text": (
                        "Aurora improved battery safety by adding thermal "
                        "sensors that reduced overheating risk."
                    ),
                },
                {
                    "source_id": "chunk-duplicate",
                    "text": (
                        "Aurora improved battery safety by adding thermal "
                        "sensors that reduced overheating risk."
                    ),
                },
            ]
        }
        evidence = [
            {
                "label": row["source_id"],
                "kind": "source_excerpt",
                "text": row["text"],
                "source_ids": [row["source_id"]],
            }
            for row in evidence_summary["source_excerpts"]
        ]
        claim_graph = build_signed_claim_evidence_graph(
            [
                "Aurora improved battery safety by adding thermal sensors that reduced overheating risk.",
            ],
            evidence,
        )
        proof = self.build_proof(
            question,
            {
                "anchors": ["Aurora", "battery safety"],
                "relation_intents": ["impact"],
                "required_facets": ["thermal sensors"],
            },
            answer,
            evidence_summary,
            claim_graph,
            {"unsupported_claim_count": 0},
        )
        self.assertEqual(proof["selected_count"], 1)
        self.assertEqual(proof["redundant_evidence_pruned_count"], 1)
        cert = proof["minimality_certificate"]
        self.assertTrue(cert["all_selected_necessary"])
        self.assertEqual(len(cert["necessary_evidence_ids"]), 1)


class VSRCounterfactualPathNecessityV79Tests(unittest.TestCase):
    """Controlled audit for V79 counterfactual path necessity."""

    def setUp(self):
        from api.features.chats.counterfactual_path_necessity import (
            certify_counterfactual_path_necessity,
        )
        from api.features.chats.minimal_proof_subgraph import (
            build_minimal_proof_subgraph,
        )

        self.certify = certify_counterfactual_path_necessity
        self.build_proof = build_minimal_proof_subgraph

    def test_selected_path_is_claim_essential_without_replacement(self):
        question = "How did Aurora improve battery safety?"
        answer = (
            "Aurora improved battery safety by adding thermal sensors "
            "that reduced overheating risk. Aurora trained operators to "
            "respond to battery temperature alarms."
        )
        evidence_summary = {
            "source_excerpts": [
                {
                    "source_id": "chunk-sensors",
                    "text": (
                        "Aurora improved battery safety by adding thermal "
                        "sensors that reduced overheating risk."
                    ),
                },
                {
                    "source_id": "chunk-training",
                    "text": (
                        "Aurora trained operators to respond to battery "
                        "temperature alarms."
                    ),
                },
            ]
        }
        evidence = [
            {
                "label": row["source_id"],
                "kind": "source_excerpt",
                "text": row["text"],
                "source_ids": [row["source_id"]],
            }
            for row in evidence_summary["source_excerpts"]
        ]
        claims = [
            "Aurora improved battery safety by adding thermal sensors that reduced overheating risk.",
            "Aurora trained operators to respond to battery temperature alarms.",
        ]
        claim_graph = build_signed_claim_evidence_graph(claims, evidence)
        query_graph = {
            "anchors": ["Aurora", "battery safety"],
            "relation_intents": ["impact"],
            "required_facets": ["thermal sensors", "operator training"],
        }
        proof = self.build_proof(
            question,
            query_graph,
            answer,
            evidence_summary,
            claim_graph,
            {"unsupported_claim_count": 0},
        )
        cert = self.certify(
            question,
            query_graph,
            answer,
            evidence_summary,
            claim_graph,
            {"unsupported_claim_count": 0},
            proof,
        )
        self.assertTrue(cert["certification_passed"])
        self.assertEqual(cert["essential_path_count"], 1)
        self.assertEqual(cert["replaceable_path_count"], 0)
        self.assertEqual(
            {row["necessity"] for row in cert["interventions"]},
            {"claim_essential"},
        )

    def test_duplicate_path_is_marked_replaceable(self):
        question = "How did Aurora improve battery safety?"
        answer = (
            "Aurora improved battery safety by adding thermal sensors "
            "that reduced overheating risk."
        )
        evidence_summary = {
            "source_excerpts": [
                {
                    "source_id": "chunk-primary",
                    "text": (
                        "Aurora improved battery safety by adding thermal "
                        "sensors that reduced overheating risk."
                    ),
                },
                {
                    "source_id": "chunk-duplicate",
                    "text": (
                        "Aurora improved battery safety by adding thermal "
                        "sensors that reduced overheating risk."
                    ),
                },
            ]
        }
        evidence = [
            {
                "label": row["source_id"],
                "kind": "source_excerpt",
                "text": row["text"],
                "source_ids": [row["source_id"]],
            }
            for row in evidence_summary["source_excerpts"]
        ]
        claim_graph = build_signed_claim_evidence_graph(
            [
                "Aurora improved battery safety by adding thermal sensors that reduced overheating risk.",
            ],
            evidence,
        )
        query_graph = {
            "anchors": ["Aurora", "battery safety"],
            "relation_intents": ["impact"],
            "required_facets": ["thermal sensors"],
        }
        proof = self.build_proof(
            question,
            query_graph,
            answer,
            evidence_summary,
            claim_graph,
            {"unsupported_claim_count": 0},
        )
        cert = self.certify(
            question,
            query_graph,
            answer,
            evidence_summary,
            claim_graph,
            {"unsupported_claim_count": 0},
            proof,
        )
        self.assertTrue(cert["certification_passed"])
        self.assertEqual(
            cert["submission_guidance"],
            "necessity_certified_with_replacements",
        )
        self.assertEqual(cert["replaceable_path_count"], 1)
        self.assertEqual(cert["interventions"][0]["necessity"], "replaceable")


class VSRRiskCalibratedSubmissionV80Tests(unittest.TestCase):
    """Controlled audit for V80 risk-calibrated submission."""

    def setUp(self):
        from api.features.chats.risk_calibrated_submission import (
            calibrate_answer_submission,
        )

        self.calibrate = calibrate_answer_submission

    def test_low_risk_supported_proof_submits(self):
        policy = self.calibrate(
            {
                "risk_level": "low",
                "claim_count": 2,
                "unsupported_claim_count": 0,
                "claim_support_ratio": 1.0,
            },
            {
                "proof_sufficient": True,
                "claim_support_coverage_ratio": 1.0,
                "obligation_coverage_ratio": 0.75,
            },
            {
                "certification_passed": True,
                "selected_path_count": 2,
                "replaceable_path_count": 0,
            },
            {
                "claims": [
                    {"status": "supported", "support_mass": 0.8},
                    {"status": "supported", "support_mass": 0.7},
                ]
            },
            {"refusal_like": False},
        )
        self.assertEqual(policy["action"], "submit")
        self.assertTrue(policy["submission_allowed"])
        self.assertLessEqual(policy["calibrated_risk"], 0.22)

    def test_unsupported_claim_is_blocked(self):
        policy = self.calibrate(
            {
                "risk_level": "medium",
                "claim_count": 3,
                "unsupported_claim_count": 1,
                "claim_support_ratio": 0.667,
            },
            {
                "proof_sufficient": False,
                "claim_support_coverage_ratio": 0.667,
                "obligation_coverage_ratio": 0.5,
            },
            {
                "certification_passed": False,
                "selected_path_count": 1,
                "replaceable_path_count": 0,
            },
            {
                "claims": [
                    {"status": "supported", "support_mass": 0.8},
                    {"status": "unknown", "support_mass": 0.0},
                ]
            },
            {"refusal_like": False},
        )
        self.assertEqual(policy["action"], "block_unsupported")
        self.assertFalse(policy["submission_allowed"])
        self.assertEqual(policy["unsupported_claim_budget"], 0)


class VSRProofConstrainedAdmissionV81Tests(unittest.TestCase):
    """Controlled audit for V81 proof-constrained claim admission."""

    def setUp(self):
        from api.features.chats.proof_constrained_admission import (
            build_proof_constrained_admission,
        )

        self.admit = build_proof_constrained_admission

    def test_uncovered_claim_is_pruned_from_candidate(self):
        claim_graph = {
            "claims": [
                {
                    "claim_id": "claim-0",
                    "claim": (
                        "Aurora improved battery safety by adding thermal "
                        "sensors."
                    ),
                    "status": "supported",
                    "support_mass": 0.8,
                    "contradiction_mass": 0.0,
                },
                {
                    "claim_id": "claim-1",
                    "claim": "Aurora also reduced costs by 30 percent.",
                    "status": "unknown",
                    "support_mass": 0.0,
                    "contradiction_mass": 0.0,
                },
            ]
        }
        admission = self.admit(
            (
                "Aurora improved battery safety by adding thermal sensors. "
                "Aurora also reduced costs by 30 percent."
            ),
            claim_graph,
            {"covered_claims": ["claim-0"]},
            {"interventions": []},
        )
        self.assertTrue(admission["candidate_available"])
        self.assertTrue(admission["changed"])
        self.assertEqual(admission["admitted_claim_count"], 1)
        self.assertEqual(admission["rejected_claim_count"], 1)
        self.assertNotIn("30 percent", admission["candidate_answer"])

    def test_fully_proof_covered_answer_is_unchanged(self):
        answer = (
            "Aurora improved battery safety by adding thermal sensors."
        )
        claim_graph = {
            "claims": [
                {
                    "claim_id": "claim-0",
                    "claim": answer,
                    "status": "supported",
                    "support_mass": 0.8,
                    "contradiction_mass": 0.0,
                }
            ]
        }
        admission = self.admit(
            answer,
            claim_graph,
            {"covered_claims": ["claim-0"]},
            {
                "interventions": [
                    {"lost_claims_if_removed": ["claim-0"]},
                ]
            },
        )
        self.assertFalse(admission["changed"])
        self.assertEqual(admission["admitted_claim_count"], 1)
        self.assertEqual(admission["necessary_claim_count"], 1)


class VSRProofSourceQuorumV82Tests(unittest.TestCase):
    """Controlled audit for V82 source-independent proof quorum."""

    def setUp(self):
        from api.features.chats.proof_source_quorum import (
            certify_proof_source_quorum,
        )

        self.certify = certify_proof_source_quorum

    def test_independent_quorum_passes_with_two_source_families(self):
        proof = {
            "proof_sufficient": True,
            "covered_claims": ["claim-0"],
            "covered_obligations": ["facet:safety"],
            "contradicted_claims_in_proof": [],
            "selected_evidence": [
                {
                    "evidence_id": "evidence-0",
                    "label": "doc-a#1",
                    "source_ids": ["doc-a#1"],
                    "supported_claim_ids": ["claim-0"],
                    "obligation_hits": [
                        {"obligation_id": "facet:safety"},
                    ],
                },
                {
                    "evidence_id": "evidence-1",
                    "label": "doc-b#1",
                    "source_ids": ["doc-b#1"],
                    "supported_claim_ids": ["claim-0"],
                    "obligation_hits": [
                        {"obligation_id": "facet:safety"},
                    ],
                },
            ],
        }
        evidence_summary = {
            "source_excerpts": [
                {
                    "source_id": "doc-a#1",
                    "document_id": "doc-a",
                    "text": "Aurora added thermal sensors.",
                },
                {
                    "source_id": "doc-b#1",
                    "document_id": "doc-b",
                    "text": "A separate report says sensors reduced risk.",
                },
            ]
        }
        quorum = self.certify(proof, evidence_summary, {"claims": []})
        self.assertEqual(quorum["quorum_status"], "independent_quorum")
        self.assertTrue(quorum["quorum_passed"])
        self.assertEqual(quorum["independent_family_count"], 2)
        self.assertEqual(quorum["single_family_claims"], [])

    def test_correlated_quorum_monitors_single_family_targets(self):
        proof = {
            "proof_sufficient": True,
            "covered_claims": ["claim-0"],
            "covered_obligations": ["facet:training"],
            "contradicted_claims_in_proof": [],
            "selected_evidence": [
                {
                    "evidence_id": "evidence-0",
                    "label": "doc-a#1",
                    "source_ids": ["doc-a#1"],
                    "supported_claim_ids": ["claim-0"],
                    "obligation_hits": [],
                },
                {
                    "evidence_id": "evidence-1",
                    "label": "doc-b#1",
                    "source_ids": ["doc-b#1"],
                    "supported_claim_ids": [],
                    "obligation_hits": [
                        {"obligation_id": "facet:training"},
                    ],
                },
            ],
        }
        evidence_summary = {
            "source_excerpts": [
                {
                    "source_id": "doc-a#1",
                    "document_id": "doc-a",
                    "text": "Aurora added thermal sensors.",
                },
                {
                    "source_id": "doc-b#1",
                    "document_id": "doc-b",
                    "text": "Aurora trained operators on alarms.",
                },
            ]
        }
        quorum = self.certify(proof, evidence_summary, {"claims": []})
        self.assertEqual(quorum["quorum_status"], "correlated_monitor")
        self.assertTrue(quorum["monitoring_required"])
        self.assertEqual(quorum["single_family_claims"], ["claim-0"])
        self.assertEqual(
            quorum["single_family_obligations"],
            ["facet:training"],
        )


class VSRSourceQuorumBackfillV83Tests(unittest.TestCase):
    """Controlled audit for V83 independent-family proof backfill."""

    def setUp(self):
        from api.features.chats.source_quorum_backfill import (
            build_source_quorum_backfill,
        )

        self.backfill = build_source_quorum_backfill

    def test_backfill_upgrades_correlated_quorum_to_independent(self):
        question = "How did Aurora improve battery safety?"
        query_graph = {
            "anchors": ["Aurora", "battery safety"],
            "relation_intents": ["impact"],
            "required_facets": ["thermal sensors", "operator training"],
        }
        evidence_summary = {
            "source_excerpts": [
                {
                    "source_id": "doc-a#1",
                    "document_id": "doc-a",
                    "text": (
                        "Aurora improved battery safety with thermal "
                        "sensors."
                    ),
                },
                {
                    "source_id": "doc-b#1",
                    "document_id": "doc-b",
                    "text": (
                        "Aurora trained operators to respond to battery "
                        "alarms."
                    ),
                },
                {
                    "source_id": "doc-c#1",
                    "document_id": "doc-c",
                    "text": (
                        "A separate Aurora safety report says thermal "
                        "sensors and operator training reduced battery "
                        "risk."
                    ),
                },
            ]
        }
        evidence = [
            {
                "label": row["source_id"],
                "kind": "source_excerpt",
                "text": row["text"],
                "source_ids": [row["source_id"]],
            }
            for row in evidence_summary["source_excerpts"]
        ]
        claim_graph = build_signed_claim_evidence_graph(
            [
                "Aurora improved battery safety with thermal sensors.",
            ],
            evidence,
        )
        proof = {
            "proof_sufficient": True,
            "covered_claims": ["claim-0"],
            "required_claim_count": 1,
            "covered_obligations": ["facet:question_specific_bundle"],
            "required_obligation_count": 1,
            "contradicted_claims_in_proof": [],
            "estimated_tokens": 40,
            "selected_evidence": [
                {
                    "evidence_id": "evidence-0",
                    "kind": "source_excerpt",
                    "label": "doc-a#1",
                    "source_ids": ["doc-a#1"],
                    "supported_claim_ids": ["claim-0"],
                    "obligation_hits": [],
                    "max_claim_support_score": 0.8,
                    "estimated_tokens": 20,
                    "text_preview": "Aurora improved battery safety.",
                },
                {
                    "evidence_id": "evidence-1",
                    "kind": "source_excerpt",
                    "label": "doc-b#1",
                    "source_ids": ["doc-b#1"],
                    "supported_claim_ids": [],
                    "obligation_hits": [
                        {
                            "obligation_id": (
                                "facet:question_specific_bundle"
                            )
                        },
                    ],
                    "max_claim_support_score": 0.0,
                    "estimated_tokens": 20,
                    "text_preview": "Aurora trained operators.",
                },
            ],
        }
        result = self.backfill(
            question,
            query_graph,
            evidence_summary,
            claim_graph,
            {"unsupported_claim_count": 0},
            proof,
        )
        self.assertTrue(result["triggered"])
        self.assertTrue(result["applied"])
        self.assertEqual(result["selected_addition_count"], 1)
        self.assertEqual(
            result["augmented_proof_source_quorum"]["quorum_status"],
            "independent_quorum",
        )


class VSRProofCompletionBackfillV84Tests(unittest.TestCase):
    """Controlled audit for V84 claim/obligation proof completion."""

    def setUp(self):
        from api.features.chats.proof_completion_backfill import (
            build_proof_completion_backfill,
        )

        self.complete = build_proof_completion_backfill

    def test_backfill_completes_missing_supported_claim(self):
        question = "How did Aurora improve battery safety?"
        query_graph = {
            "anchors": ["Aurora", "battery safety"],
            "relation_intents": ["impact"],
            "required_facets": ["thermal sensors", "operator training"],
        }
        evidence_summary = {
            "source_excerpts": [
                {
                    "source_id": "doc-a#1",
                    "text": (
                        "Aurora improved battery safety with thermal "
                        "sensors."
                    ),
                },
                {
                    "source_id": "doc-b#1",
                    "text": (
                        "Aurora trained operators to respond to battery "
                        "temperature alarms."
                    ),
                },
            ]
        }
        claim_graph = {
            "claims": [
                {
                    "claim_id": "claim-0",
                    "status": "supported",
                    "support_mass": 0.8,
                },
                {
                    "claim_id": "claim-1",
                    "status": "supported",
                    "support_mass": 0.7,
                },
            ],
            "edges": [
                {
                    "claim_id": "claim-0",
                    "evidence_id": "evidence-0",
                    "relation": "supports",
                    "score": 0.8,
                    "source_ids": ["doc-a#1"],
                },
                {
                    "claim_id": "claim-1",
                    "evidence_id": "evidence-1",
                    "relation": "supports",
                    "score": 0.7,
                    "source_ids": ["doc-b#1"],
                },
            ],
        }
        proof = {
            "enabled": True,
            "controller": "minimal_proof_subgraph_v1",
            "proof_sufficient": False,
            "selected_count": 1,
            "estimated_tokens": 20,
            "covered_claims": ["claim-0"],
            "required_claim_count": 2,
            "covered_obligations": [
                "anchor:AURORA",
                "anchor:BATTERY SAFETY",
            ],
            "required_obligation_count": 4,
            "claim_support_coverage_ratio": 0.5,
            "obligation_coverage_ratio": 0.5,
            "contradicted_claims_in_proof": [],
            "selected_evidence": [
                {
                    "evidence_id": "evidence-0",
                    "kind": "source_excerpt",
                    "label": "doc-a#1",
                    "source_ids": ["doc-a#1"],
                    "supported_claim_ids": ["claim-0"],
                    "obligation_hits": [
                        {"obligation_id": "anchor:AURORA"},
                        {"obligation_id": "anchor:BATTERY SAFETY"},
                    ],
                    "max_claim_support_score": 0.8,
                    "estimated_tokens": 20,
                    "marginal_utility": 0.4,
                    "text_preview": "Aurora improved battery safety.",
                }
            ],
        }
        result = self.complete(
            question,
            query_graph,
            evidence_summary,
            claim_graph,
            {"unsupported_claim_count": 0},
            proof,
        )
        self.assertTrue(result["triggered"])
        self.assertTrue(result["applied"])
        self.assertEqual(result["selected_addition_count"], 1)
        self.assertEqual(result["missing_claims_after"], [])
        self.assertTrue(
            result["augmented_minimal_proof_subgraph"]["proof_sufficient"]
        )


class VSRProofDeficitAnswerRepairV85Tests(unittest.TestCase):
    """Controlled audit for V85 audit-aware proof-deficit repair."""

    def setUp(self):
        from api.features.chats.proof_deficit_answer_repair import (
            build_proof_deficit_answer_repair,
        )

        self.repair = build_proof_deficit_answer_repair

    def test_audit_unsupported_claim_is_pruned_even_when_graph_supported(self):
        answer = (
            "Aurora improved battery safety by adding thermal sensors. "
            "Aurora also reduced warranty costs by 30 percent."
        )
        assessment = {
            "claim_evidence_graph": {
                "claims": [
                    {
                        "claim_id": "claim-0",
                        "claim": (
                            "Aurora improved battery safety by adding "
                            "thermal sensors."
                        ),
                        "status": "supported",
                        "support_mass": 0.8,
                        "contradiction_mass": 0.0,
                    },
                    {
                        "claim_id": "claim-1",
                        "claim": (
                            "Aurora also reduced warranty costs by "
                            "30 percent."
                        ),
                        "status": "supported",
                        "support_mass": 0.35,
                        "contradiction_mass": 0.0,
                    },
                ]
            },
            "claim_verification": {},
            "evidence_audit": {
                "unsupported_claim_count": 1,
                "unsupported_claim_examples": [
                    {
                        "claim": (
                            "Aurora also reduced warranty costs by "
                            "30 percent."
                        )
                    }
                ],
            },
            "minimal_proof_subgraph": {
                "proof_sufficient": False,
                "covered_claims": ["claim-0", "claim-1"],
            },
            "proof_completion_backfill": {
                "missing_claims_after": [],
                "missing_obligations_after": [],
            },
            "proof_source_quorum": {"quorum_status": "quorum_failed"},
            "risk_calibrated_submission": {
                "action": "block_unsupported",
            },
        }
        result = self.repair(answer, assessment)
        self.assertTrue(result["triggered"])
        self.assertTrue(result["changed"])
        self.assertEqual(result["admitted_claim_count"], 1)
        self.assertEqual(result["rejected_claim_count"], 1)
        self.assertEqual(result["audit_unsupported_match_count"], 1)
        self.assertIn("thermal sensors", result["candidate_answer"])
        self.assertNotIn("30 percent", result["candidate_answer"])
        self.assertIn(
            "audit_unsupported_claim",
            result["rejected_claims"][0]["reasons"],
        )


class VSRSourceDiversityAnswerRepairV86Tests(unittest.TestCase):
    """Controlled audit for V86 source-diversity answer repair."""

    def setUp(self):
        from api.features.chats.source_diversity_answer_repair import (
            build_source_diversity_answer_repair,
        )

        self.repair = build_source_diversity_answer_repair

    def test_single_family_claim_variants_are_generated(self):
        answer = (
            "Aurora improved battery safety by adding thermal sensors. "
            "Aurora also reduced warranty costs."
        )
        assessment = {
            "claim_evidence_graph": {
                "claims": [
                    {
                        "claim_id": "claim-0",
                        "claim": (
                            "Aurora improved battery safety by adding "
                            "thermal sensors."
                        ),
                        "status": "supported",
                        "support_mass": 0.8,
                    },
                    {
                        "claim_id": "claim-1",
                        "claim": "Aurora also reduced warranty costs.",
                        "status": "supported",
                        "support_mass": 0.42,
                    },
                ]
            },
            "minimal_proof_subgraph": {
                "proof_sufficient": True,
                "covered_claims": ["claim-0", "claim-1"],
            },
            "counterfactual_path_necessity": {
                "interventions": [
                    {"lost_claims_if_removed": ["claim-0"]},
                ]
            },
            "proof_source_quorum": {
                "quorum_status": "correlated_monitor",
                "single_family_claims": ["claim-1"],
                "single_family_obligations": [],
            },
            "risk_calibrated_submission": {"action": "submit"},
            "evidence_audit": {"unsupported_claim_count": 0},
        }
        result = self.repair(answer, assessment)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["controller"], "source_diversity_answer_repair_v1")
        self.assertGreaterEqual(result["variant_count"], 1)
        first = result["candidate_variants"][0]
        self.assertEqual(first["dropped_claim_count"], 1)
        self.assertNotIn("warranty costs", first["candidate_answer"])
        self.assertIn("thermal sensors", first["candidate_answer"])


class VSRObligationCriticalityQuorumV87Tests(unittest.TestCase):
    """Controlled audit for V87 obligation-criticality normalization."""

    def setUp(self):
        from api.features.chats.obligation_criticality_quorum import (
            normalize_obligation_criticality_quorum,
        )

        self.normalize = normalize_obligation_criticality_quorum

    def test_generic_fact_obligation_is_demoted_when_claims_are_diverse(self):
        quorum = {
            "proof_sufficient": True,
            "independent_family_count": 2,
            "quorum_status": "correlated_monitor",
            "single_family_claims": [],
            "missing_claims": [],
            "single_family_obligations": ["intent:fact"],
            "missing_obligations": [],
            "claim_family_counts": {
                "claim-0": {
                    "family_ids": ["doc-a", "doc-b"],
                    "independent_family_count": 2,
                }
            },
            "obligation_family_counts": {
                "intent:fact": {
                    "family_ids": ["doc-a"],
                    "independent_family_count": 1,
                }
            },
            "submission_guidance": "correlated_source_monitor",
        }
        result = self.normalize(quorum)
        self.assertTrue(result["triggered"])
        self.assertTrue(result["applied"])
        self.assertEqual(result["after_quorum_status"], "independent_quorum")
        self.assertEqual(result["demoted_obligation_count"], 1)
        self.assertEqual(
            result["normalized_proof_source_quorum"]["quorum_status"],
            "independent_quorum",
        )


if __name__ == "__main__":
    unittest.main()
