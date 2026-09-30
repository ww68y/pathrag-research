#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep audit: check implementation depth of ALL innovation modules."""
import os
import re
import ast
import importlib
import sys
from pathlib import Path

sys.path.insert(0, '.')

BASE = Path("api/features/chats")

# All innovation modules (excluding infrastructure files)
MODULES = {
    # Core VSR pipeline
    "V1_vsr_pathrag": "vsr_pathrag.py",
    "V2_agent_controller": "agent_controller.py",
    # Evidence & Verification
    "V77_obligation_type_compiler": "obligation_type_compiler.py",
    "V78_minimal_proof_subgraph": "minimal_proof_subgraph.py",
    "V79_counterfactual_path_necessity": "counterfactual_path_necessity.py",
    "V80_risk_calibrated_submission": "risk_calibrated_submission.py",
    "V81_proof_constrained_admission": "proof_constrained_admission.py",
    "V82_proof_completion_backfill": "proof_completion_backfill.py",
    "V83_proof_deficit_answer_repair": "proof_deficit_answer_repair.py",
    "V84_proof_source_quorum": "proof_source_quorum.py",
    "V85_source_quorum_backfill": "source_quorum_backfill.py",
    "V86_source_diversity_answer_repair": "source_diversity_answer_repair.py",
    "V87_obligation_source_rescue": "obligation_source_rescue.py",
    "V88_obligation_criticality_quorum": "obligation_criticality_quorum.py",
    # Evidence processing
    "active_evidence": "active_evidence.py",
    "adaptive_evidence_compression": "adaptive_evidence_compression.py",
    "adaptive_evidence_routing": "adaptive_evidence_routing.py",
    "answer_realization": "answer_realization.py",
    "claim_active_verification": "claim_active_verification.py",
    "counterevidence_cost_memory": "counterevidence_cost_memory.py",
    "counterevidence_information_memory": "counterevidence_information_memory.py",
    "experience_memory": "experience_memory.py",
    "fidelity_scoring": "fidelity_scoring.py",
    "incremental_graph_update": "incremental_graph_update.py",
    "intra_answer_consistency": "intra_answer_consistency.py",
    "iterative_refinement": "iterative_refinement.py",
    "salience_reordering": "salience_reordering.py",
    # Deep innovation modules
    "V107_collaborative_reasoning": "collaborative_reasoning.py",
    "V108_uncertainty_quantification": "uncertainty_quantification.py",
    "V110_self_reflective_retrieval": "self_reflective_retrieval.py",
    "V118_knowledge_distillation": "knowledge_distillation_compression.py",
    "V119_reinforcement_learning": "reinforcement_learning_optimizer.py",
    "V122_neural_symbolic_reasoning": "neural_symbolic_reasoning.py",
    "V123_hierarchical_debate": "hierarchical_multi_agent_debate.py",
    # V128-V135
    "V128_quantum_inspired": "quantum_inspired_retrieval.py",
    "V129_nerf_visualization": "nerf_knowledge_visualization.py",
    "V130_bio_evolution": "bio_inspired_evolution.py",
    "V131_cross_modal": "zero_shot_cross_modal.py",
    "V132_continual_learning": "continual_lifelong_learning.py",
    "V133_adversarial_robustness": "adversarial_robustness_training.py",
    "V134_gnn_reasoning": "gnn_reasoning_enhancement.py",
    "V135_edge_computing": "edge_computing_deployment.py",
    # V136-V143
    "V136_multimodal_fusion": "multimodal_knowledge_fusion.py",
    "V137_temporal_reasoning": "temporal_reasoning_enhancement.py",
    "V138_cross_lingual": "cross_lingual_alignment.py",
    "V139_privacy_preserving": "privacy_preserving_retrieval.py",
    "V140_explainability": "explainability_dashboard.py",
    "V141_rl_optimizer_v2": "reinforcement_learning_optimizer_v2.py",
    "V143_edge_deployment": "edge_deployment_support.py",
    # Other
    "V115_multilingual": "multilingual_crosslingual.py",
    "V116_emotion_aware": "emotion_aware_reasoning.py",
    "V117_dialogue_state": "dialogue_state_management.py",
    "V101_adversarial_defense": "adversarial_robustness_training.py",
    "V103_multimodal_fusion_v1": "multimodal_knowledge_fusion.py",
    "V105_realtime_stream": "adaptive_evidence_routing.py",
    "personalized_adaptation": "personalized_adaptation.py",
    "autonomous_tool_learning": "autonomous_tool_learning.py",
    "meta_learning_optimizer": "meta_learning_optimizer.py",
    "causal_reasoning": "causal_reasoning_enhancement.py",
    "federated_learning": "federated_learning_privacy.py",
    "self_reflective_retrieval": "self_reflective_retrieval.py",
}

# Check which are called in agent_controller.py
controller = Path("api/features/chats/agent_controller.py").read_text(encoding="utf-8")
pipeline_code = controller[5000:]

results = []
for label, filename in MODULES.items():
    filepath = BASE / filename
    if not filepath.exists():
        results.append({
            "label": label,
            "file": filename,
            "lines": 0,
            "classes": 0,
            "functions": 0,
            "has_entry_func": False,
            "called_in_pipeline": False,
            "status": "FILE_NOT_FOUND",
        })
        continue
    
    content = filepath.read_text(encoding="utf-8")
    lines = content.count("\n") + 1
    
    # Count classes and functions
    try:
        tree = ast.parse(content)
        classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        functions = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef))
    except:
        classes = 0
        functions = 0
    
    # Check if has build_ or quantify_ entry function
    has_entry = bool(re.search(r'^def (build_|quantify_|run_|certify_|normalize_|assess_|build_delayed_|apply_|enrich_|select_)', content, re.MULTILINE))
    
    # Check if called in pipeline
    # Extract function names and check if any appears in pipeline
    func_names = re.findall(r'^def (\w+)\(', content, re.MULTILINE)
    called = any(re.search(rf'\b{re.escape(fn)}\s*\(', pipeline_code) for fn in func_names if not fn.startswith('_'))
    
    if called:
        status = "ACTIVE"
    elif has_entry:
        status = "HAS_ENTRY_NOT_CALLED"
    else:
        status = "DORMANT"
    
    results.append({
        "label": label,
        "file": filename,
        "lines": lines,
        "classes": classes,
        "functions": functions,
        "has_entry_func": has_entry,
        "called_in_pipeline": called,
        "status": status,
    })

# Sort by status then lines
results.sort(key=lambda x: (x["status"] != "ACTIVE", -x["lines"]))

print("=" * 80)
print("VSR-PathRAG COMPLETE INNOVATION AUDIT")
print("=" * 80)
total = len(results)
active = sum(1 for r in results if r["status"] == "ACTIVE")
dormant = sum(1 for r in results if r["status"] in ("DORMANT", "HAS_ENTRY_NOT_CALLED"))
missing = sum(1 for r in results if r["status"] == "FILE_NOT_FOUND")
total_lines = sum(r["lines"] for r in results if r["status"] == "ACTIVE")

print(f"Total modules: {total}")
print(f"Active (called in pipeline): {active}")
print(f"Dormant (not called): {dormant}")
print(f"File not found: {missing}")
print(f"Total active code lines: {total_lines}")
print("=" * 80)

print(f"\n{'Label':<40} {'Lines':>6} {'Classes':>8} {'Funcs':>6} {'Status':>20}")
print("-" * 80)
for r in results:
    print(f"{r['label']:<40} {r['lines']:>6} {r['classes']:>8} {r['functions']:>6} {r['status']:>20}")

print("=" * 80)
