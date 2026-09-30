#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Find all function signature mismatches in agent_controller.py by doing a dry import."""
import sys
import importlib
import inspect
from pathlib import Path

sys.path.insert(0, ".")

# Try to import and check each function used in agent_controller
functions_to_check = [
    ("api.features.chats.answer_realization", "build_correction_persistence_constraints"),
    ("api.features.chats.answer_realization", "materialize_correction_source_chunks"),
    ("api.features.chats.answer_realization", "prepare_retrieved_excerpts"),
    ("api.features.chats.answer_realization", "realize_answer"),
    ("api.features.chats.answer_realization", "build_claim_evidence_traceability"),
    ("api.features.chats.experience_memory", "build_routing_context"),
    ("api.features.chats.experience_memory", "recall_experiences"),
    ("api.features.chats.experience_memory", "select_candidate_with_memory"),
    ("api.features.chats.experience_memory", "build_experience_episode"),
    ("api.features.chats.experience_memory", "append_experience_episode"),
    ("api.features.chats.claim_active_verification", "run_claim_level_active_verification"),
    ("api.features.chats.claim_active_verification", "apply_diagnosed_claim_correction"),
    ("api.features.chats.claim_active_verification", "build_delayed_verification_credit"),
    ("api.features.chats.counterevidence_cost_memory", "load_counterevidence_cost_memory"),
    ("api.features.chats.counterevidence_cost_memory", "append_counterevidence_cost_updates"),
    ("api.features.chats.counterevidence_cost_memory", "build_cross_query_evidence_memory"),
    ("api.features.chats.counterevidence_information_memory", "load_counterevidence_information_memory"),
    ("api.features.chats.counterevidence_information_memory", "append_counterevidence_information_updates"),
]

print("Function signatures:")
print("=" * 80)
for module_name, func_name in functions_to_check:
    try:
        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        print(f"{func_name}({', '.join(params)})")
    except Exception as e:
        print(f"{func_name}: IMPORT ERROR: {e}")
