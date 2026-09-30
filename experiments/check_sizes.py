#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick size check for dormant modules."""
import os

modules = [
    'collaborative_reasoning',
    'uncertainty_quantification',
    'personalized_adaptation',
    'self_reflective_retrieval',
    'autonomous_tool_learning',
    'neural_symbolic_reasoning',
    'hierarchical_multi_agent_debate',
    'quantum_inspired_retrieval',
    'nerf_knowledge_visualization',
    'bio_inspired_evolution',
    'zero_shot_cross_modal',
    'continual_lifelong_learning',
    'adversarial_robustness_training',
    'gnn_reasoning_enhancement',
    'edge_computing_deployment',
    'knowledge_distillation_compression',
    'reinforcement_learning_optimizer',
    'incremental_graph_update',
]

base = os.path.join("api", "features", "chats")
print(f"{'Module':<40} {'Lines':>6} {'Bytes':>8}")
print("-" * 56)
for m in modules:
    path = os.path.join(base, m + ".py")
    if os.path.exists(path):
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        print(f"{m:<40} {lines:>6} {size:>8}")
    else:
        print(f"{m:<40}   N/A      N/A")
