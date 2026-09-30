#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick check: are V107/V108/V110/V122/V123 now called in the pipeline?"""

import re
from pathlib import Path

controller = Path("api/features/chats/agent_controller.py")
content = controller.read_text(encoding="utf-8")

checks = {
    "V108 uncertainty_quantification": "quantify_answer_uncertainty(",
    "V110 self_reflective_retrieval": "build_self_reflective_retrieval(",
    "V107 collaborative_reasoning": "CollaborativeReasoningEngine(",
    "V122 neural_symbolic_reasoning": "build_neural_symbolic_reasoning(",
    "V123 hierarchical_debate": "build_hierarchical_multi_agent_debate(",
}

print("Module Integration Check (after fix):")
print("=" * 50)
all_ok = True
for name, pattern in checks.items():
    found = pattern in content[5000:]
    status = "INTEGRATED" if found else "STILL DORMANT"
    if not found:
        all_ok = False
    print(f"  {name}: {status}")

print("=" * 50)
if all_ok:
    print("All 5 deep innovation modules are now INTEGRATED!")
else:
    print("Some modules are still not integrated.")

# Also verify trace fields
trace_checks = {
    "v107_collaborative_reasoning": "v107_collaborative_result",
    "v108_uncertainty_quantification": "v108_uncertainty_result",
    "v110_self_reflective_retrieval": "v110_self_reflective_result",
    "v122_neural_symbolic_reasoning": "v122_neural_symbolic_result",
    "v123_hierarchical_debate": "v123_hierarchical_debate_result",
}

print("\nTrace Field Check:")
print("=" * 50)
for name, var in trace_checks.items():
    found = var in content[5000:]
    status = "IN TRACE" if found else "NOT IN TRACE"
    print(f"  {name}: {status}")
