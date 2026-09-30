#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check the 15 'dormant' modules more carefully."""
from pathlib import Path
import re

controller = Path("api/features/chats/agent_controller.py").read_text(encoding="utf-8")

# The 'dormant' modules from the audit - let's check what patterns are actually in the controller
dormant_checks = [
    ("V2 Agent Controller", "agent_controller.py", ["agent_controller"]),  # It IS the controller
    ("V77 Obligation Type Compiler", "obligation_type_compiler.py", [
        "build_obligation_type_compiler(",
        "compile_obligation_types(",
        "obligation_type_compiler",
        "ObligationTypeCompiler",
    ]),
    ("V79 Counterfactual Path Necessity", "counterfactual_path_necessity.py", [
        "build_counterfactual_path_necessity(",
        "counterfactual_path_necessity",
        "CounterfactualPath",
    ]),
    ("V80 Risk Calibrated Submission", "risk_calibrated_submission.py", [
        "build_risk_calibrated_submission(",
        "risk_calibrated_submission",
        "RiskCalibrated",
        "certify_risk_calibrated",
    ]),
    ("V84 Proof Source Quorum", "proof_source_quorum.py", [
        "build_proof_source_quorum(",
        "proof_source_quorum",
        "ProofSourceQuorum",
        "normalize_proof_source_quorum",
    ]),
    ("V88 Obligation Criticality Quorum", "obligation_criticality_quorum.py", [
        "build_obligation_criticality_quorum(",
        "obligation_criticality_quorum",
        "ObligationCriticality",
        "normalize_obligation_criticality_quorum",
    ]),
    ("V3 Active Evidence", "active_evidence.py", [
        "build_active_evidence(",
        "active_evidence",
        "ActiveEvidence",
        "build_active_evidence_collection",
    ]),
    ("V5 Adaptive Evidence Routing", "adaptive_evidence_routing.py", [
        "build_adaptive_evidence_routing(",
        "adaptive_evidence_routing",
        "AdaptiveEvidenceRouting",
    ]),
    ("V6 Answer Realization", "answer_realization.py", [
        "build_answer_realization(",
        "answer_realization",
        "AnswerRealization",
        "realize_answer",
    ]),
    ("V7 Claim Active Verification", "claim_active_verification.py", [
        "build_claim_active_verification(",
        "claim_active_verification",
        "ClaimActiveVerification",
        "verify_claims",
    ]),
    ("V8 Counterevidence Cost Memory", "counterevidence_cost_memory.py", [
        "build_counterevidence_cost_memory(",
        "counterevidence_cost_memory",
        "CounterevidenceCost",
    ]),
    ("V9 Counterevidence Info Memory", "counterevidence_information_memory.py", [
        "build_counterevidence_information_memory(",
        "counterevidence_information_memory",
        "CounterevidenceInformation",
    ]),
    ("V10 Experience Memory", "experience_memory.py", [
        "build_experience_memory(",
        "experience_memory",
        "ExperienceMemory",
    ]),
    ("V12 Intra-Answer Consistency", "intra_answer_consistency.py", [
        "build_intra_answer_consistency(",
        "intra_answer_consistency",
        "IntraAnswerConsistency",
        "check_intra_answer",
    ]),
    ("V105 Realtime Evidence Stream", "adaptive_evidence_routing.py", [
        "build_realtime_evidence_stream_processing(",
        "realtime_evidence_stream",
        "realtime_stream",
    ]),
]

print("DETAILED DORMANT CHECK")
print("=" * 80)
for name, filename, patterns in dormant_checks:
    found_patterns = [p for p in patterns if p in controller]
    status = "INTEGRATED" if found_patterns else "TRULY DORMANT"
    print(f"  {name}: {status}")
    if found_patterns:
        print(f"    Matched: {found_patterns[0]}")
    else:
        # Search for any reference to the module file
        module_name = filename.replace(".py", "")
        if module_name in controller:
            print(f"    Module referenced but function not called")
        else:
            print(f"    No reference found at all")
print("=" * 80)
