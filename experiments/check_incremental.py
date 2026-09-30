#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final precise check for incremental_graph_update."""
from pathlib import Path

content = Path("api/features/chats/agent_controller.py").read_text(encoding="utf-8")

# Direct search
pattern = "build_incremental_update_pipeline("
found = pattern in content
print(f"build_incremental_update_pipeline( in controller: {found}")

# Also check the precise check patterns
patterns = [
    "build_incremental_graph",
    "build_incremental_update_pipeline(",
]
for p in patterns:
    print(f"  '{p}' found: {p in content}")

# Find the line
for i, line in enumerate(content.split('\n'), 1):
    if 'build_incremental_update_pipeline' in line:
        print(f"  Line {i}: {line.strip()}")
