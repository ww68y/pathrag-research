#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final comprehensive audit of ALL innovation modules."""
import re
import ast
from pathlib import Path

BASE = Path("api/features/chats")
controller = Path("api/features/chats/agent_controller.py").read_text(encoding="utf-8")

# All innovation modules with their version numbers and categories
ALL_MODULES = [
    # === Core VSR Pipeline (V1-V5) ===
    ("V1", "VSR-PathRAG Core", "vsr_pathrag.py", "build_query_graph"),
    ("V2", "Agent Controller", "agent_controller.py", "agent_query"),
    ("V3", "Active Evidence", "active_evidence.py", "run_active_evidence_acquisition"),
    ("V4", "Adaptive Evidence Routing", "adaptive_evidence_routing.py", "build_adaptive_reasoning_path"),
    ("V5", "Adaptive Evidence Compression", "adaptive_evidence_compression.py", "build_adaptive_evidence_compression"),
    
    # === Evidence & Verification (V77-V88) ===
    ("V77", "Obligation Type Compiler", "obligation_type_compiler.py", "build_obligation_type_nucleus"),
    ("V78", "Minimal Proof Subgraph", "minimal_proof_subgraph.py", "build_minimal_proof_subgraph"),
    ("V79", "Counterfactual Path Necessity", "counterfactual_path_necessity.py", "certify_counterfactual_path_necessity"),
    ("V80", "Risk Calibrated Submission", "risk_calibrated_submission.py", "calibrate_answer_submission"),
    ("V81", "Proof Constrained Admission", "proof_constrained_admission.py", "build_proof_constrained_admission"),
    ("V82", "Proof Completion Backfill", "proof_completion_backfill.py", "build_proof_completion_backfill"),
    ("V83", "Proof Deficit Answer Repair", "proof_deficit_answer_repair.py", "build_proof_deficit_answer_repair"),
    ("V84", "Proof Source Quorum", "proof_source_quorum.py", "certify_proof_source_quorum"),
    ("V85", "Source Quorum Backfill", "source_quorum_backfill.py", "build_source_quorum_backfill"),
    ("V86", "Source Diversity Answer Repair", "source_diversity_answer_repair.py", "build_source_diversity_answer_repair"),
    ("V87", "Obligation Source Rescue", "obligation_source_rescue.py", "build_obligation_source_rescue"),
    ("V88", "Obligation Criticality Quorum", "obligation_criticality_quorum.py", "normalize_obligation_criticality_quorum"),
    
    # === Evidence Processing ===
    ("V89", "Claim Active Verification", "claim_active_verification.py", "run_claim_level_active_verification"),
    ("V90", "Answer Realization", "answer_realization.py", "realize_answer"),
    ("V91", "Experience Memory", "experience_memory.py", "recall_experiences"),
    ("V92", "Counterevidence Cost Memory", "counterevidence_cost_memory.py", "load_counterevidence_cost_memory"),
    ("V93", "Counterevidence Info Memory", "counterevidence_information_memory.py", "load_counterevidence_information_memory"),
    ("V94", "Fidelity Scoring", "fidelity_scoring.py", "build_fidelity_scoring"),
    ("V95", "Intra-Answer Consistency", "intra_answer_consistency.py", "build_intra_answer_consistency_check"),
    ("V96", "Iterative Refinement", "iterative_refinement.py", "build_iterative_refinement"),
    ("V97", "Salience Reordering", "salience_reordering.py", "build_salience_reordering"),
    
    # === V101-V106 (registered in pipeline) ===
    ("V101", "Adversarial Robustness Defense", "claim_active_verification.py", "build_adversarial_robustness_defense"),
    ("V103", "Multimodal Evidence Fusion", "active_evidence.py", "build_multimodal_evidence_fusion"),
    ("V105", "Realtime Evidence Stream", "experience_memory.py", "build_realtime_evidence_stream_processing"),
    
    # === V107-V119: Deep Innovation ===
    ("V107", "Collaborative Multi-Agent Reasoning", "collaborative_reasoning.py", "CollaborativeReasoningEngine"),
    ("V108", "Bayesian Uncertainty Quantification", "uncertainty_quantification.py", "quantify_answer_uncertainty"),
    ("V109", "Personalized Adaptation", "personalized_adaptation.py", "personalize_response"),
    ("V110", "Self-Reflective Retrieval", "self_reflective_retrieval.py", "build_self_reflective_retrieval"),
    ("V111", "Autonomous Tool Learning", "autonomous_tool_learning.py", "build_autonomous_tool_learning"),
    ("V115", "Multilingual Crosslingual", "multilingual_crosslingual.py", "build_multilingual_crosslingual_retrieval"),
    ("V116", "Emotion-Aware Reasoning", "emotion_aware_reasoning.py", "build_emotion_aware_reasoning"),
    ("V117", "Dialogue State Management", "dialogue_state_management.py", "build_dialogue_state_management"),
    ("V118", "Knowledge Distillation", "knowledge_distillation_compression.py", "build_knowledge_distillation_compression"),
    ("V119", "Reinforcement Learning Optimizer", "reinforcement_learning_optimizer.py", "build_reinforcement_learning_optimizer"),
    
    # === V122-V127: Advanced Reasoning ===
    ("V122", "Neural-Symbolic Reasoning Bridge", "neural_symbolic_reasoning.py", "build_neural_symbolic_reasoning"),
    ("V123", "Hierarchical Multi-Agent Debate", "hierarchical_multi_agent_debate.py", "build_hierarchical_multi_agent_debate"),
    ("V124", "Causal Reasoning Enhancement", "causal_reasoning_enhancement.py", "build_causal_reasoning_enhancement"),
    ("V125", "Meta-Learning Optimizer", "meta_learning_optimizer.py", "build_meta_learning_optimization"),
    ("V126", "Federated Learning Privacy", "federated_learning_privacy.py", "build_federated_learning_privacy"),
    ("V127", "Incremental Graph Update", "incremental_graph_update.py", "build_incremental_update_pipeline"),
    
    # === V128-V135: Ultimate Experience Innovation ===
    ("V128", "Quantum-Inspired Retrieval", "quantum_inspired_retrieval.py", "build_quantum_inspired_retrieval"),
    ("V129", "NeRF Knowledge Visualization", "nerf_knowledge_visualization.py", "build_nerf_knowledge_visualization"),
    ("V130", "Bio-Inspired Evolution", "bio_inspired_evolution.py", "build_bio_inspired_evolution"),
    ("V131", "Zero-Shot Cross-Modal Transfer", "zero_shot_cross_modal.py", "build_zero_shot_cross_modal_transfer"),
    ("V132", "Continual Lifelong Learning", "continual_lifelong_learning.py", "build_continual_lifelong_learning"),
    ("V133", "Adversarial Robustness Training", "adversarial_robustness_training.py", "build_adversarial_robustness_training"),
    ("V134", "GNN Reasoning Enhancement", "gnn_reasoning_enhancement.py", "build_gnn_reasoning_enhancement"),
    ("V135", "Edge Computing Deployment", "edge_computing_deployment.py", "build_edge_computing_deployment"),
    
    # === V136-V143: Eight Innovation Domains ===
    ("V136", "Multimodal Knowledge Fusion", "multimodal_knowledge_fusion.py", "build_multimodal_evidence_fusion"),
    ("V137", "Temporal Reasoning Enhancement", "temporal_reasoning_enhancement.py", "build_temporal_reasoning_enhancement"),
    ("V138", "Cross-Lingual Alignment", "cross_lingual_alignment.py", "build_cross_lingual_alignment"),
    ("V139", "Privacy-Preserving Retrieval", "privacy_preserving_retrieval.py", "build_privacy_preserving_retrieval"),
    ("V140", "Explainability Dashboard", "explainability_dashboard.py", "build_explainability_dashboard_data"),
    ("V141", "RL Optimizer V2", "reinforcement_learning_optimizer_v2.py", "build_rl_optimizer_v2"),
    ("V142", "Incremental Graph Update V2", "incremental_graph_update.py", "build_incremental_update_pipeline"),
    ("V143", "Edge Deployment Support", "edge_deployment_support.py", "build_edge_deployment_support"),
]

print("=" * 90)
print("VSR-PathRAG COMPLETE INNOVATION AUDIT - FINAL")
print("=" * 90)

# Check each module
results = []
for version, name, filename, entry_func in ALL_MODULES:
    filepath = BASE / filename
    if not filepath.exists():
        results.append((version, name, filename, 0, 0, 0, "FILE_NOT_FOUND", False))
        continue
    
    content = filepath.read_text(encoding="utf-8")
    lines = content.count("\n") + 1
    
    try:
        tree = ast.parse(content)
        classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        functions = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except:
        classes = 0
        functions = 0
    
    # Check if called in controller (by entry function name or alias). The
    # controller itself is the pipeline host, so it is active by definition.
    called = filename == "agent_controller.py" or entry_func in controller
    
    status = "ACTIVE" if called else "DORMANT"
    results.append((version, name, filename, lines, classes, functions, status, called))

# Summary
total = len(results)
active = sum(1 for r in results if r[6] == "ACTIVE")
dormant = sum(1 for r in results if r[6] == "DORMANT")
missing = sum(1 for r in results if r[6] == "FILE_NOT_FOUND")
total_lines = sum(r[3] for r in results if r[6] == "ACTIVE")
total_classes = sum(r[4] for r in results if r[6] == "ACTIVE")
total_functions = sum(r[5] for r in results if r[6] == "ACTIVE")

print(f"Total modules: {total}")
print(f"Active (integrated in pipeline): {active}")
print(f"Dormant (not integrated): {dormant}")
print(f"File not found: {missing}")
print(f"Total active code lines: {total_lines}")
print(f"Total classes: {total_classes}")
print(f"Total functions: {total_functions}")
print(f"Integration rate: {active}/{total} = {active/total*100:.1f}%")
print("=" * 90)

print(f"\n{'Ver':<6} {'Module Name':<40} {'Lines':>6} {'Cls':>4} {'Fn':>4} {'Status':>8}")
print("-" * 90)
for r in results:
    print(f"{r[0]:<6} {r[1]:<40} {r[3]:>6} {r[4]:>4} {r[5]:>4} {r[6]:>8}")

# Category summary
categories = {}
for r in results:
    v = r[0]
    if v.startswith("V1") and int(v[1:]) <= 5:
        cat = "Core VSR Pipeline"
    elif v.startswith("V7") or v.startswith("V8"):
        cat = "Evidence & Verification"
    elif v.startswith("V9"):
        cat = "Evidence Processing"
    elif v in ("V101", "V103", "V105"):
        cat = "Advanced Evidence"
    elif v.startswith("V1") and int(v[1:]) <= 119:
        cat = "Deep Innovation"
    elif v.startswith("V12"):
        cat = "Advanced Reasoning"
    elif v.startswith("V13"):
        cat = "Ultimate Experience"
    else:
        cat = "Eight Innovation Domains"
    
    if cat not in categories:
        categories[cat] = {"total": 0, "active": 0, "lines": 0}
    categories[cat]["total"] += 1
    if r[6] == "ACTIVE":
        categories[cat]["active"] += 1
        categories[cat]["lines"] += r[3]

print("\n" + "=" * 60)
print("CATEGORY SUMMARY")
print("=" * 60)
print(f"{'Category':<35} {'Total':>6} {'Active':>7} {'Lines':>8}")
print("-" * 60)
for cat, stats in categories.items():
    print(f"{cat:<35} {stats['total']:>6} {stats['active']:>7} {stats['lines']:>8}")
