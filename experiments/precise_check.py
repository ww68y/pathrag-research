#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precise check for the 13 modules flagged as not active."""
import re
from pathlib import Path

content = Path("api/features/chats/agent_controller.py").read_text(encoding="utf-8")
pipeline = content[5000:]

# These are the specific import aliases and call patterns used
checks = {
    "V107 collaborative_reasoning": [
        "CollaborativeReasoningEngine(",
        "v107_collaborative_result",
    ],
    "V108 uncertainty_quantification": [
        "quantify_answer_uncertainty(",
        "v108_uncertainty_result",
    ],
    "V110 self_reflective_retrieval": [
        "build_self_reflective_retrieval(",
        "v110_self_reflective_result",
    ],
    "V118 knowledge_distillation": [
        "build_knowledge_distillation_compression(",
        "v118_compression_result",
    ],
    "V119 reinforcement_learning": [
        "build_reinforcement_learning_optimizer(",
        "v119_rl_result",
    ],
    "V122 neural_symbolic": [
        "build_neural_symbolic_reasoning(",
        "v122_neural_symbolic_result",
    ],
    "V123 hierarchical_debate": [
        "build_hierarchical_multi_agent_debate(",
        "v123_hierarchical_debate_result",
    ],
    "V128 quantum_inspired": [
        "build_quantum_inspired_retrieval(",
        "v128_quantum_result",
    ],
    "V129 nerf_visualization": [
        "build_nerf_knowledge_visualization(",
        "v129_nerf_result",
    ],
    "V130 bio_evolution": [
        "build_bio_inspired_evolution(",
        "v130_bio_result",
    ],
    "V131 cross_modal": [
        "build_zero_shot_cross_modal_transfer(",
        "v131_cross_modal_result",
    ],
    "V132 continual_learning": [
        "build_continual_lifelong_learning(",
        "v132_continual_result",
    ],
    "V133 adversarial_robustness": [
        "build_adversarial_robustness_training(",
        "v133_adversarial_result",
    ],
    "V134 gnn_reasoning": [
        "build_gnn_reasoning_enhancement(",
        "v134_gnn_result",
    ],
    "V135 edge_computing": [
        "build_edge_computing_deployment(",
        "v135_edge_result",
    ],
    "V136 multimodal_fusion": [
        "build_v136_multimodal_fusion(",
    ],
    "V137 temporal_reasoning": [
        "build_v137_temporal_reasoning(",
    ],
    "V138 cross_lingual": [
        "build_v138_cross_lingual_alignment(",
    ],
    "V139 privacy_preserving": [
        "build_v139_privacy_preserving(",
    ],
    "V140 explainability": [
        "build_v140_explainability_dashboard(",
    ],
    "V141 rl_optimizer_v2": [
        "build_v141_rl_optimizer_v2(",
    ],
    "V142 incremental_graph_update": [
        "build_incremental_update_pipeline(",
        "v127_incremental_graph_update",
    ],
    "V143 edge_deployment": [
        "build_v143_edge_deployment(",
    ],
    # Check the remaining dormant ones
    "personalized_adaptation": [
        "build_personalized_adaptation(",
        "personalized_adaptation",
    ],
    "autonomous_tool_learning": [
        "build_autonomous_tool_learning(",
    ],
    "causal_reasoning": [
        "build_causal_reasoning",
    ],
    "meta_learning_optimizer": [
        "build_meta_learning",
    ],
    "federated_learning": [
        "build_federated_learning",
    ],
}

print("PRECISE INTEGRATION CHECK")
print("=" * 60)
for name, patterns in checks.items():
    found = any(p in pipeline for p in patterns)
    status = "INTEGRATED" if found else "NOT INTEGRATED"
    print(f"  {name}: {status}")

# Also check V115/V116/V117
extras = {
    "V115 multilingual": "build_multilingual_crosslingual_retrieval(",
    "V116 emotion_aware": "build_emotion_aware_reasoning(",
    "V117 dialogue_state": "build_dialogue_state_management(",
    "V101 adversarial_defense": "build_adversarial_robustness_defense(",
    "V103 multimodal_fusion_v1": "build_multimodal_evidence_fusion(",
    "V105 realtime_stream": "build_realtime_evidence_stream_processing(",
}
print("\nExtra modules:")
for name, pattern in extras.items():
    found = pattern in pipeline
    print(f"  {name}: {'INTEGRATED' if found else 'NOT INTEGRATED'}")
