#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit which innovation modules are actually called in the agent pipeline."""

import re
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
controller_path = ROOT / "api" / "features" / "chats" / "agent_controller.py"

with open(controller_path, "r", encoding="utf-8") as f:
    content = f.read()

# Find all imports from api.features.chats
import_pattern = re.compile(r"from api\.features\.chats\.(\w+) import\s+(?:\n\s*)?([\w,\s\(\)]+?)(?=\nfrom|\n\n|\Z)", re.MULTILINE)
imports = {}
for match in import_pattern.finditer(content):
    module_name = match.group(1)
    imported_names = re.findall(r"\b(\w+)\b", match.group(2))
    imports[module_name] = imported_names

# For each imported function name, check if it's actually called (appears with parentheses)
# Exclude the import section (first 5000 chars)
pipeline_code = content[5000:]

results = []
for module_name, func_names in imports.items():
    if module_name in ("agent_controller", "routes", "schemas"):
        continue
    
    called_funcs = []
    not_called_funcs = []
    
    for func_name in func_names:
        if func_name in ("Any", "List", "Dict", "Optional", "Tuple", "Set", "Union"):
            continue
        # Check if function is called with (
        call_pattern = rf"\b{re.escape(func_name)}\s*\("
        if re.search(call_pattern, pipeline_code):
            called_funcs.append(func_name)
        else:
            not_called_funcs.append(func_name)
    
    status = "ACTIVE" if called_funcs else "DORMANT"
    results.append({
        "module": module_name,
        "status": status,
        "called": called_funcs,
        "not_called": not_called_funcs,
        "total_funcs": len(func_names),
    })

# Sort: dormant first, then by name
results.sort(key=lambda x: (x["status"] != "DORMANT", x["module"]))

print("=" * 70)
print("VSR-PathRAG Module Integration Audit")
print("=" * 70)
print(f"Total modules imported: {len(results)}")
active_count = sum(1 for r in results if r["status"] == "ACTIVE")
dormant_count = sum(1 for r in results if r["status"] == "DORMANT")
print(f"Active (called in pipeline): {active_count}")
print(f"Dormant (imported but never called): {dormant_count}")
print("=" * 70)

print("\n--- DORMANT MODULES (imported but never called) ---")
for r in results:
    if r["status"] == "DORMANT":
        print(f'  [DORMANT] {r["module"]}')
        print(f"     Functions: {', '.join(r['not_called'])}")

print("\n--- ACTIVE MODULES (actually called in pipeline) ---")
for r in results:
    if r["status"] == "ACTIVE":
        print(f'  [ACTIVE] {r["module"]}')
        if r["not_called"]:
            print(f"     Called: {', '.join(r['called'])}")
            print(f"     Not called: {', '.join(r['not_called'])}")

# Also check for placeholder results (None in trace)
placeholders = re.findall(r'"(\w+)":\s*None.*?#\s*V(\d+)', pipeline_code)
if placeholders:
    print(f"\n--- PLACEHOLDER MODULES (set to None in trace) ---")
    for name, version in placeholders:
        print(f'  [PLACEHOLDER] V{version} {name} = None')

# Check for hardcoded dict results (fake modules)
fake_pattern = re.compile(r'#\s*V\d+.*\n\w+_result\s*=\s*\{', re.MULTILINE)
fake_matches = re.findall(r'#\s*(V\d+).*?\n(\w+_result)\s*=\s*\{', pipeline_code)
if fake_matches:
    print(f"\n--- HARDCODED DICT RESULTS (not real module calls) ---")
    for version, var_name in fake_matches:
        print(f'  [HARDCODED] {version} {var_name} = {{...}}')
