#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full audit: check ALL modules are now integrated."""
import re
from pathlib import Path

content = Path("api/features/chats/agent_controller.py").read_text(encoding="utf-8")
pipeline = content[5000:]

checks = {
    "V107 collaborative_reasoning": "CollaborativeReasoningEngine(",
    "V108 uncertainty_quantification": "quantify_answer_uncertainty(",
    "V110 self_reflective_retrieval": "build_self_reflective_retrieval(",
    "V118 knowledge_distillation": "build_knowledge_distillation_compression(",
    "V119 reinforcement_learning": "build_reinforcement_learning_optimizer(",
    "V122 neural_symbolic_reasoning": "build_neural_symbolic_reasoning(",
    "V123 hierarchical_debate": "build_hierarchical_multi_agent_debate(",
    "V128 quantum_inspired": "build_quantum_inspired_retrieval(",
    "V129 nerf_visualization": "build_nerf_knowledge_visualization(",
    "V130 bio_evolution": "build_bio_inspired_evolution(",
    "V131 cross_modal_transfer": "build_zero_shot_cross_modal_transfer(",
    "V132 continual_learning": "build_continual_lifelong_learning(",
    "V133 adversarial_robustness": "build_adversarial_robustness_training(",
    "V134 gnn_reasoning": "build_gnn_reasoning_enhancement(",
    "V135 edge_computing": "build_edge_computing_deployment(",
    "V136 multimodal_fusion": "build_v136_multimodal_fusion(",
    "V137 temporal_reasoning": "build_v137_temporal_reasoning(",
    "V138 cross_lingual": "build_v138_cross_lingual_alignment(",
    "V139 privacy_preserving": "build_v139_privacy_preserving(",
    "V140 explainability": "build_v140_explainability_dashboard(",
    "V141 rl_optimizer_v2": "build_v141_rl_optimizer_v2(",
    "V142 incremental_graph_update": "build_incremental_update_pipeline(",
    "V143 edge_deployment": "build_v143_edge_deployment(",
}

print("FULL MODULE INTEGRATION AUDIT")
print("=" * 60)
integrated = 0
dormant = 0
for name, pattern in checks.items():
    found = pattern in pipeline
    status = "INTEGRATED" if found else "DORMANT"
    if found:
        integrated += 1
    else:
        dormant += 1
    print(f"  {name}: {status}")

print("=" * 60)
print(f"Integrated: {integrated}/{len(checks)}")
print(f"Dormant: {dormant}/{len(checks)}")

# Check for remaining None placeholders
none_count = len(re.findall(r': None,\s*#\s*V\d+', pipeline))
print(f"Remaining None placeholders: {none_count}")

# Check for remaining hardcoded dicts
fake_count = len(re.findall(r'#\s*V\d+.*\n\w+_result\s*=\s*\{', pipeline))
print(f"Remaining hardcoded dicts: {fake_count}")

if dormant == 0 and none_count == 0 and fake_count == 0:
    print("\nALL MODULES INTEGRATED! No placeholders, no fake data.")
else:
    print(f"\nISSUES REMAIN: {dormant} dormant, {none_count} None placeholders, {fake_count} fake dicts")
