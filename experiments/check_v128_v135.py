#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')
import importlib

mods = {
    'V128': ('api.features.chats.quantum_inspired_retrieval', 'build_quantum_inspired_retrieval'),
    'V129': ('api.features.chats.nerf_knowledge_visualization', 'build_nerf_knowledge_visualization'),
    'V130': ('api.features.chats.bio_inspired_evolution', 'build_bio_inspired_evolution'),
    'V131': ('api.features.chats.zero_shot_cross_modal', 'build_zero_shot_cross_modal_transfer'),
    'V132': ('api.features.chats.continual_lifelong_learning', 'build_continual_lifelong_learning'),
    'V133': ('api.features.chats.adversarial_robustness_training', 'build_adversarial_robustness_training'),
    'V134': ('api.features.chats.gnn_reasoning_enhancement', 'build_gnn_reasoning_enhancement'),
    'V135': ('api.features.chats.edge_computing_deployment', 'build_edge_computing_deployment'),
}

for k, (mod_path, func_name) in mods.items():
    try:
        mod = importlib.import_module(mod_path)
        has_func = hasattr(mod, func_name)
        print(f'{k}: {"OK - " + func_name + "()" if has_func else "NO FUNC"}')
    except Exception as e:
        print(f'{k}: IMPORT ERROR: {e}')
