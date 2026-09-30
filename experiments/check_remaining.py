#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path

BASE = Path("api/features/chats")
modules = [
    ("personalized_adaptation", "personalized_adaptation.py"),
    ("autonomous_tool_learning", "autonomous_tool_learning.py"),
    ("causal_reasoning", "causal_reasoning_enhancement.py"),
    ("meta_learning_optimizer", "meta_learning_optimizer.py"),
    ("federated_learning", "federated_learning_privacy.py"),
    ("incremental_graph_update", "incremental_graph_update.py"),
]

print(f"{'Module':<35} {'Lines':>6} {'Bytes':>8} {'Entry Func':>20}")
print("-" * 75)
import re
for name, filename in modules:
    path = BASE / filename
    if path.exists():
        content = path.read_text(encoding="utf-8")
        lines = content.count("\n") + 1
        size = len(content)
        # Find entry function
        match = re.search(r'^def (build_\w+|run_\w+)', content, re.MULTILINE)
        entry = match.group(1) if match else "NONE"
        print(f"{name:<35} {lines:>6} {size:>8} {entry:>20}")
    else:
        print(f"{name:<35} {'N/A':>6} {'N/A':>8} {'N/A':>20}")
