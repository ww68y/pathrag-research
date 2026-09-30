from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "reproducibility"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


SCRIPT_FILES = [
    "experiments/run_aligned_ablation_rerun.py",
    "experiments/run_retrieval_baseline.py",
    "experiments/summarize_retrieval_baselines.py",
    "experiments/run_graphrag_lite_baseline.py",
    "experiments/run_lightrag_baseline.py",
    "experiments/evaluate_pairwise_winrate.py",
    "experiments/build_retrieval_prefiltered_dataset.py",
    "experiments/summarize_retrieval_prefiltered_vsr.py",
    "experiments/summarize_cost_latency.py",
    "experiments/summarize_instrumented_cost_quality_cases.py",
    "experiments/build_i18n_abstention_language_guard.py",
    "experiments/summarize_legacy_baseline_overlap.py",
    "experiments/summarize_sage_candidate_scan.py",
    "experiments/summarize_sage_vs_baselines.py",
    "experiments/summarize_sage_external_probe.py",
    "experiments/summarize_sage_external_vs_baselines.py",
    "experiments/summarize_sage_locked_fullslice_ladder.py",
    "experiments/summarize_sage_unseen_validation.py",
    "experiments/summarize_sage_late_stage_candidates.py",
    "experiments/summarize_sage_promotion_mix130.py",
    "experiments/summarize_sage_promotion_external.py",
    "experiments/summarize_sage_final_candidate.py",
    "experiments/build_sage_v33_freeze_holdout_audit.py",
    "experiments/build_sage_locked_unseen_holdout.py",
    "experiments/summarize_sage_locked_unseen75_baselines.py",
    "experiments/summarize_sage_locked_unseen75_judge.py",
    "experiments/summarize_sage_locked_unseen75_pairwise.py",
    "experiments/summarize_sage_locked_unseen_repeat.py",
    "experiments/build_locked_holdout_meta_summary.py",
    "experiments/build_locked_holdout_overlap_audit.py",
    "experiments/build_paper_ready_tables.py",
    "experiments/build_paper_latex_tables.py",
    "experiments/validate_sage_reproduction.py",
    "experiments/evaluate_repro_results.py",
    "experiments/merge_jsonl_rows.py",
    "experiments/build_learned_contract_gate.py",
    "experiments/build_evidence_delta_guard.py",
    "experiments/build_evidence_delta_guard_v33.py",
    "experiments/build_v54_query_expanded_contract_gap_repair.py",
    "experiments/build_v55_supported_facet_guard.py",
    "experiments/build_v56_supported_facet_fusion.py",
    "experiments/summarize_sage_v33_external_probe.py",
    "experiments/train_contract_gap_router.py",
    "experiments/build_answer_contract_agent.py",
    "experiments/build_claim_evidence_agent.py",
    "experiments/summarize_cross_dataset_quality_flags.py",
    "experiments/audit_cross_dataset_quality_flags.py",
    "experiments/run_cross_dataset_context_rescue_diagnostic.py",
    "experiments/summarize_legal_stability_repeat.py",
    "experiments/analyze_evidence_contract_depth.py",
]

DATASET_FILES = [
    "datasets/UltraDomain/mix.jsonl",
    "datasets/UltraDomain/biology_fullcontext_50.jsonl",
    "datasets/UltraDomain/legal_fullcontext_20.jsonl",
    "datasets/UltraDomain/domain_expansion_windowed_50.jsonl",
    "datasets/UltraDomain/legal_retrieved_top5_20.jsonl",
    "datasets/UltraDomain/legal_retrieved_top5_prepended_full_20.jsonl",
    "datasets/UltraDomain/biology_retrieved_top5_50.jsonl",
    "datasets/UltraDomain/domain_expansion_retrieved_top5_50.jsonl",
    "datasets/UltraDomain/v46_zero_overlap_validation_30.jsonl",
    "datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl",
    "datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl",
    "datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl",
    "datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl",
    "datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl",
    "datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl",
    "datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl",
    "datasets/UltraDomain/sage_locked_legal40_repeat7_windowed.jsonl",
    "datasets/UltraDomain/sage_locked_legal54_repeat8_windowed.jsonl",
]

RESULT_FILES = [
    "paper/generated_experiment_tables.tex",
    "results/evidence_contract_depth/EVIDENCE_CONTRACT_DEPTH_MIX130.md",
    "results/evidence_contract_depth/evidence_contract_depth_mix130.json",
    "results/evidence_contract_depth/IMPACT_AWARE_ESCALATION_MIX130.md",
    "results/evidence_contract_depth/impact_aware_escalation_mix130.json",
    "results/evidence_contract_depth/IMPACT_AWARE_ESCALATION_DEPTH_MIX130.md",
    "results/evidence_contract_depth/IMPACT_AWARE_ESCALATION_DEPTH_MIX130.json",
    "results/evidence_contract_depth/impact_aware_escalation_policy_v1.json",
    "results/evidence_contract_depth/impact_aware_escalation_policy_v1.joblib",
    "results/enhanced_ablation_aligned/CROSS_DATASET_ALIGNED_PROGRESS_SUMMARY.md",
    "results/enhanced_ablation_aligned/cross_dataset_aligned_progress_summary.json",
    "results/enhanced_ablation_aligned/full_v2_rescue_formal_valid_n130.json",
    "results/enhanced_ablation_aligned/FULL_V2_RESCUE_FORMAL_VALID_N130_SUMMARY.md",
    "results/enhanced_ablation_aligned/aligned_ablation_study_n130.json",
    "results/enhanced_ablation_aligned/aligned_ablation_stat_tests_n130.json",
    "results/enhanced_ablation_aligned/aligned_ablation_report_n130.md",
    "results/enhanced_ablation_aligned/STRICT_ABLATION_RESUME_STATUS.md",
    "results/enhanced_ablation_aligned/w_o_all_aligned_n130.json",
    "results/enhanced_ablation_aligned/w_o_evidence_extraction_aligned_n130.json",
    "results/enhanced_ablation_aligned/w_o_verification_aligned_n130.json",
    "results/enhanced_ablation_aligned/w_o_repair_aligned_n130.json",
    "results/enhanced_ablation_aligned/full_v2_aligned_n130.json",
    "results/enhanced_ablation_aligned/full_v2_rescue_aligned_n130.json",
    "results/enhanced_ablation_aligned/biology_fullcontext_50/w_o_repair_aligned_n50.json",
    "results/enhanced_ablation_aligned/biology_fullcontext_50/full_v2_aligned_n50.json",
    "results/enhanced_ablation_aligned/biology_fullcontext_50/full_v2_rescue_aligned_n50.json",
    "results/enhanced_ablation_aligned/legal_fullcontext_20/w_o_repair_aligned_n20.json",
    "results/enhanced_ablation_aligned/legal_fullcontext_20/full_v2_aligned_n20.json",
    "results/enhanced_ablation_aligned/legal_fullcontext_20/full_v2_rescue_aligned_n20.json",
    "results/enhanced_ablation_aligned/domain_expansion_windowed_50/w_o_repair_aligned_n50.json",
    "results/enhanced_ablation_aligned/domain_expansion_windowed_50/full_v2_aligned_n50.json",
    "results/enhanced_ablation_aligned/domain_expansion_windowed_50/full_v2_rescue_aligned_n50.json",
    "results/enhanced_ablation_aligned/CROSS_DATASET_QUALITY_FLAGS_AND_GATES.md",
    "results/enhanced_ablation_aligned/cross_dataset_quality_flags_and_gates.json",
    "results/enhanced_ablation_aligned/CROSS_DATASET_QUALITY_FLAG_LLM_AUDIT.md",
    "results/enhanced_ablation_aligned/cross_dataset_quality_flag_llm_audit.json",
    "results/enhanced_ablation_aligned/CROSS_DATASET_CONTEXT_RESCUE_DIAGNOSTIC_n7_ctx80000.md",
    "results/enhanced_ablation_aligned/cross_dataset_context_rescue_diagnostic_n7_ctx80000.json",
    "results/enhanced_ablation_stability/LEGAL_FULLCONTEXT_20_STABILITY_SUMMARY.md",
    "results/enhanced_ablation_stability/legal_fullcontext_20_stability_summary.json",
    "results/historical_baselines/RETRIEVAL_BASELINE_ALIGNED_SUMMARY.md",
    "results/historical_baselines/retrieval_baseline_aligned_summary.json",
    "results/legal_fullcontext_20/naive_aligned_n20.jsonl",
    "results/legal_fullcontext_20/hyde_aligned_n20.jsonl",
    "results/biology_fullcontext_50/naive_aligned_n50.jsonl",
    "results/biology_fullcontext_50/hyde_aligned_n50.jsonl",
    "results/domain_expansion_windowed_50/naive_aligned_n50.jsonl",
    "results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl",
    "results/historical_baselines/RETRIEVAL_PREFILTERED_VSR_SUMMARY.md",
    "results/historical_baselines/retrieval_prefiltered_vsr_summary.json",
    "results/legal_retrieved_top5_20/full_v2_rescue_aligned_n20.json",
    "results/legal_retrieved_top5_20_ctx12k/full_v2_rescue_aligned_n20.json",
    "results/legal_retrieved_top5_prepended_full_20/w_o_repair_aligned_n20.json",
    "results/legal_retrieved_top5_prepended_full_20/full_v2_aligned_n20.json",
    "results/legal_retrieved_top5_prepended_full_20/full_v2_rescue_aligned_n20.json",
    "results/cost_instrumented/legal_fullcontext_20_n5/w_o_repair_aligned_n5.json",
    "results/cost_instrumented/legal_fullcontext_20_n5/full_v2_aligned_n5.json",
    "results/cost_instrumented/legal_fullcontext_20_n5/full_v2_rescue_aligned_n5.json",
    "results/cost_latency/COST_LATENCY_SUMMARY.md",
    "results/cost_latency/cost_latency_summary.json",
    "results/cost_latency/COST_QUALITY_CASE_AUDIT.md",
    "results/cost_latency/cost_quality_case_audit.json",
    "results/cost_latency/I18N_ABSTENTION_LANGUAGE_GUARD_AUDIT.md",
    "results/cost_latency/i18n_abstention_language_guard_audit.json",
    "results/paper_tables/PAPER_READY_EXPERIMENT_TABLES.md",
    "results/paper_tables/paper_ready_experiment_tables.json",
    "results/legacy_baseline_overlap/LEGACY_BASELINE_OVERLAP_SUMMARY.md",
    "results/legacy_baseline_overlap/legacy_baseline_overlap_summary.json",
    "results/historical_baselines/ultradomain_mix_graphrag_lite.jsonl",
    "results/historical_baselines/ultradomain_mix_graphrag_lite_eval.jsonl",
    "results/historical_baselines/ultradomain_mix_graphrag_lite_eval_summary.md",
    "results/historical_baselines/ultradomain_mix_graphrag_lite_report.md",
    "results/historical_pairwise/ultradomain_mix_pairwise_pathrag_vs_graphrag_lite.jsonl",
    "results/historical_pairwise/ultradomain_mix_pairwise_pathrag_vs_graphrag_lite_summary.md",
    "results/historical_baselines/legal_fullcontext_20_graphrag_lite.jsonl",
    "results/historical_baselines/legal_fullcontext_20_graphrag_lite_eval.jsonl",
    "results/historical_baselines/legal_fullcontext_20_graphrag_lite_eval_summary.md",
    "results/historical_pairwise/legal_fullcontext_20_pairwise_pathrag_vs_graphrag_lite.jsonl",
    "results/historical_pairwise/legal_fullcontext_20_pairwise_pathrag_vs_graphrag_lite_summary.md",
    "results/historical_baselines/legal_fullcontext_20_graphrag_lite_report.md",
    "results/historical_baselines/biology_fullcontext_50_graphrag_lite.jsonl",
    "results/historical_baselines/biology_fullcontext_50_graphrag_lite_eval.jsonl",
    "results/historical_baselines/biology_fullcontext_50_graphrag_lite_eval_summary.md",
    "results/historical_baselines/domain_expansion_windowed_50_graphrag_lite.jsonl",
    "results/historical_baselines/domain_expansion_windowed_50_graphrag_lite_eval.jsonl",
    "results/historical_baselines/domain_expansion_windowed_50_graphrag_lite_eval_summary.md",
    "results/historical_baselines/domain_expansion_windowed_50_graphrag_lite_report.md",
    "results/historical_baselines/lightrag_representative_subset.jsonl",
    "results/historical_baselines/lightrag_representative_subset_eval.jsonl",
    "results/historical_baselines/lightrag_representative_subset_eval_summary.md",
    "results/historical_baselines/lightrag_representative_subset_report.md",
    "results/historical_pairwise/pairwise_pathrag_vs_lightrag_subset.jsonl",
    "results/historical_pairwise/pairwise_pathrag_vs_lightrag_subset_summary.md",
    "results/historical_pairwise/pairwise_adaptive_hyde_fallback_vs_lightrag_subset.jsonl",
    "results/historical_pairwise/pairwise_adaptive_hyde_fallback_vs_lightrag_subset_summary.md",
    "results/historical_baselines/lightrag_current.jsonl",
    "results/historical_baselines/lightrag_current_eval.jsonl",
    "results/historical_baselines/lightrag_current_eval_summary.md",
    "results/historical_pairwise/pairwise_pathrag_vs_lightrag_current.jsonl",
    "results/historical_pairwise/pairwise_pathrag_vs_lightrag_current_summary.md",
    "results/historical_baselines/lightrag_current_extension_report.md",
    "results/historical_baselines/lightrag_current55_extension_report.md",
    "results/historical_baselines/lightrag_cached35_extension_report.md",
    "results/historical_sage/SAGE_CANDIDATE_SCAN_SUMMARY.md",
    "results/historical_sage/sage_candidate_scan_summary.json",
    "results/historical_sage/SAGE_VS_BASELINES_SUMMARY.md",
    "results/historical_sage/sage_vs_baselines_summary.json",
    "results/historical_datasets/legal_hyde_contract_completion_dry.jsonl",
    "results/historical_datasets/legal_hyde_contract_completion_dry_summary.md",
    "results/historical_datasets/biology_hyde_contract_completion_dry.jsonl",
    "results/historical_datasets/biology_hyde_contract_completion_dry_summary.md",
    "results/historical_datasets/domain_hyde_contract_completion_dry.jsonl",
    "results/historical_datasets/domain_hyde_contract_completion_dry_summary.md",
    "results/historical_datasets/domain_hyde_contract_completion_lines12_14_31_39_42.jsonl",
    "results/historical_datasets/domain_hyde_contract_completion_lines12_14_31_39_42_summary.md",
    "results/historical_datasets/domain_hyde_evidence_delta_lines12_14_31_39_42.jsonl",
    "results/historical_datasets/domain_hyde_evidence_delta_lines12_14_31_39_42_summary.md",
    "results/historical_datasets/domain_hyde_base_lines12_14_31_39_42_eval.jsonl",
    "results/historical_datasets/domain_hyde_base_lines12_14_31_39_42_eval_summary.md",
    "results/historical_datasets/domain_hyde_v32_lines12_14_31_39_42_eval.jsonl",
    "results/historical_datasets/domain_hyde_v32_lines12_14_31_39_42_eval_summary.md",
    "results/historical_sage/SAGE_EXTERNAL_PROBE_SUMMARY.md",
    "results/historical_sage/sage_external_probe_summary.json",
    "results/historical_vsr/v31_full_rf_report.md",
    "results/historical_vsr/v31_full_rf_oof.jsonl",
    "results/historical_sage/mse_pathrag_sage_v21_text_first_utility_guard_mix_130.jsonl",
    "results/historical_vsr/v31_mix130_next30_contract_completion.jsonl",
    "results/historical_vsr/v31_mix130_next30_contract_completion_summary.md",
    "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130.jsonl",
    "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_summary.md",
    "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_eval.jsonl",
    "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_eval_summary.md",
    "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130.jsonl",
    "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_summary.md",
    "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_eval.jsonl",
    "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_eval_summary.md",
    "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_guard_report.md",
    "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl",
    "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130_summary.md",
    "results/historical_vsr/v33_mix_unseen_context_manifest.jsonl",
    "results/historical_vsr/v33_mix_unseen_context_manifest_summary.md",
    "results/historical_vsr/v33_mix_unseen_context_contract_completion_merged.jsonl",
    "results/historical_vsr/v33_mix_unseen_context_contract_completion_merged_summary.md",
    "results/historical_vsr/v33_mix_unseen_context_changed_eval.jsonl",
    "results/historical_vsr/v33_mix_unseen_context_changed_eval_summary.md",
    "results/historical_baselines/v33_mix_unseen34_baseline_eval.jsonl",
    "results/historical_baselines/v33_mix_unseen34_baseline_eval_summary.md",
    "results/historical_vsr/v33_mix_unseen34_v33_eval.jsonl",
    "results/historical_vsr/v33_mix_unseen34_v33_eval_summary.md",
    "results/historical_sage/mse_pathrag_sage_v33_frozen_evidence_delta_mix_unseen34.jsonl",
    "results/historical_sage/mse_pathrag_sage_v33_frozen_evidence_delta_mix_unseen34_summary.md",
    "results/historical_sage/mse_pathrag_sage_v33_frozen_heldout_validation_report.md",
    "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix_unseen34.jsonl",
    "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix_unseen34_summary.md",
    "results/historical_baselines/v46_heldout30_hyde.jsonl",
    "results/historical_baselines/v46_heldout30_hyde_eval.jsonl",
    "results/historical_baselines/v46_heldout30_hyde_eval_summary.md",
    "results/historical_vsr/v46_heldout30_text_only_contract_completion_low5.jsonl",
    "results/historical_vsr/v46_heldout30_text_only_contract_completion_low5_summary.md",
    "results/historical_vsr/v46_heldout30_text_only_evidence_delta_low5.jsonl",
    "results/historical_vsr/v46_heldout30_text_only_evidence_delta_low5_summary.md",
    "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30.jsonl",
    "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30_summary.md",
    "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30_eval.jsonl",
    "results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30_eval_summary.md",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v46_30.jsonl",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v46_30_summary.md",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_report.md",
    "results/historical_sage/mse_pathrag_sage_v54_frozen_contract_gap_v40_12.jsonl",
    "results/historical_sage/mse_pathrag_sage_v54_frozen_contract_gap_v40_12_routed_eval.jsonl",
    "results/historical_sage/mse_pathrag_sage_v54_frozen_contract_gap_v40_12_routed_eval_summary.md",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v40_12.jsonl",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v40_12_routed_eval.jsonl",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v40_12_routed_eval_summary.md",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v42_12.jsonl",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v42_12_routed_eval.jsonl",
    "results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v42_12_routed_eval_summary.md",
    "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v40_12.jsonl",
    "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v40_12_routed_eval.jsonl",
    "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v40_12_routed_eval_summary.md",
    "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v42_12.jsonl",
    "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_v46_30.jsonl",
    "results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_report.md",
    "results/historical_pairwise/v54_frozen_v40_vs_hyde_pairwise.jsonl",
    "results/historical_pairwise/v55_supported_facet_v40_vs_hyde_pairwise.jsonl",
    "results/historical_pairwise/v55_supported_facet_v42_vs_hyde_pairwise.jsonl",
    "results/historical_pairwise/v56_supported_facet_fusion_v40_vs_hyde_pairwise.jsonl",
    "results/historical_pairwise/v46_v54_contract_gap_vs_hyde_pairwise_merged.jsonl",
    "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_repro.jsonl",
    "results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_repro_summary.md",
    "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_repro.jsonl",
    "results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_repro_summary.md",
    "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130_repro.jsonl",
    "results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130_repro_summary.md",
    "results/historical_sage/SAGE_REPRODUCTION_VALIDATION.md",
    "results/historical_sage/sage_reproduction_validation.json",
    "results/historical_sage/SAGE_V33_FREEZE_HOLDOUT_AUDIT.md",
    "results/historical_sage/sage_v33_freeze_holdout_audit.json",
    "paper/EXPERIMENT_READINESS_FOR_SCI.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int | None:
    if path.suffix.lower() not in {".py", ".md", ".jsonl", ".json", ".txt"}:
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def file_entry(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    entry: dict[str, Any] = {"path": relative, "exists": path.exists()}
    if path.exists() and path.is_file():
        entry.update(
            {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "lines": line_count(path),
            }
        )
    return entry


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def env_snapshot() -> dict[str, Any]:
    return {
        "DEEPSEEK_API_BASE": os.getenv("DEEPSEEK_API_BASE", ""),
        "DEEPSEEK_MODEL": os.getenv("DEEPSEEK_MODEL", ""),
        "DEEPSEEK_API_KEY_SET": bool(os.getenv("DEEPSEEK_API_KEY")),
        "EMBEDDING_API_BASE": os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1/embeddings"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        "EMBEDDING_DIM": os.getenv("EMBEDDING_DIM", "1024"),
        "EMBEDDING_MAX_TOKENS": os.getenv("EMBEDDING_MAX_TOKENS", "8192"),
    }


def experiment_commands() -> list[dict[str, str]]:
    return [
        {
            "name": "Aligned VSR UltraDomain mix",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/mix.jsonl --dataset-name ultradomain_mix --configs w/o_repair full_v2 full_v2_rescue --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Aligned VSR Biology full context",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/biology_fullcontext_50.jsonl --dataset-name biology_fullcontext_50 --output-dir results/enhanced_ablation_aligned/biology_fullcontext_50 --configs w/o_repair full_v2 full_v2_rescue --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Aligned VSR Legal full context",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/legal_fullcontext_20.jsonl --dataset-name legal_fullcontext_20 --output-dir results/enhanced_ablation_aligned/legal_fullcontext_20 --configs w/o_repair full_v2 full_v2_rescue --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Aligned VSR Domain expansion windowed",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/domain_expansion_windowed_50.jsonl --dataset-name domain_expansion_windowed_50 --output-dir results/enhanced_ablation_aligned/domain_expansion_windowed_50 --configs w/o_repair full_v2 full_v2_rescue --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Naive/HyDE retrieval baselines",
            "command": "python experiments/run_retrieval_baseline.py --dataset <dataset-jsonl> --output <baseline-jsonl> --mode naive|hyde --skip-existing-output --append --retry-errors",
        },
        {
            "name": "Retrieval baseline summary",
            "command": "python experiments/summarize_retrieval_baselines.py",
        },
        {
            "name": "UltraDomain Mix130 GraphRAG-lite community baseline",
            "command": "python experiments/run_graphrag_lite_baseline.py --dataset datasets/UltraDomain/mix.jsonl --pathrag-results results/historical_datasets/ultradomain_mix_repro_current.jsonl --output results/historical_baselines/ultradomain_mix_graphrag_lite.jsonl --append --skip-existing-output",
        },
        {
            "name": "UltraDomain Mix130 GraphRAG-lite judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_baselines/ultradomain_mix_graphrag_lite.jsonl --output results/historical_baselines/ultradomain_mix_graphrag_lite_eval.jsonl --summary results/historical_baselines/ultradomain_mix_graphrag_lite_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "UltraDomain Mix130 PathRAG vs GraphRAG-lite pairwise",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_datasets/ultradomain_mix_repro_current.jsonl --baseline results/historical_baselines/ultradomain_mix_graphrag_lite.jsonl --output results/historical_pairwise/ultradomain_mix_pairwise_pathrag_vs_graphrag_lite.jsonl --summary results/historical_pairwise/ultradomain_mix_pairwise_pathrag_vs_graphrag_lite_summary.md --resume",
        },
        {
            "name": "Legal GraphRAG-lite community baseline",
            "command": "python experiments/run_graphrag_lite_baseline.py --dataset datasets/UltraDomain/legal_fullcontext_20.jsonl --pathrag-results results/historical_datasets/legal_fullcontext_20_pathrag.jsonl --output results/historical_baselines/legal_fullcontext_20_graphrag_lite.jsonl --append --skip-existing-output",
        },
        {
            "name": "Legal GraphRAG-lite judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_baselines/legal_fullcontext_20_graphrag_lite.jsonl --output results/historical_baselines/legal_fullcontext_20_graphrag_lite_eval.jsonl --summary results/historical_baselines/legal_fullcontext_20_graphrag_lite_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Legal PathRAG vs GraphRAG-lite pairwise",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_datasets/legal_fullcontext_20_pathrag.jsonl --baseline results/historical_baselines/legal_fullcontext_20_graphrag_lite.jsonl --output results/historical_pairwise/legal_fullcontext_20_pairwise_pathrag_vs_graphrag_lite.jsonl --summary results/historical_pairwise/legal_fullcontext_20_pairwise_pathrag_vs_graphrag_lite_summary.md --resume",
        },
        {
            "name": "LightRAG current query extension",
            "command": "python experiments/run_lightrag_baseline.py --output results/historical_baselines/lightrag_current.jsonl --resume --retry-errors --line-nos 0-129 --query-mode hybrid --chunk-token-size 1200 --top-k 10 --llm-max-async 1 --embedding-max-async 1 --llm-timeout 300 --embedding-timeout 60",
        },
        {
            "name": "LightRAG current judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_baselines/lightrag_current.jsonl --output results/historical_baselines/lightrag_current_eval.jsonl --summary results/historical_baselines/lightrag_current_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "PathRAG vs LightRAG current pairwise",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_datasets/ultradomain_mix_repro_current.jsonl --baseline results/historical_baselines/lightrag_current.jsonl --output results/historical_pairwise/pairwise_pathrag_vs_lightrag_current.jsonl --summary results/historical_pairwise/pairwise_pathrag_vs_lightrag_current_summary.md --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Build retrieved-top5 dataset",
            "command": "python experiments/build_retrieval_prefiltered_dataset.py --dataset <dataset-jsonl> --retrieval-output <naive-baseline-jsonl> --output <retrieved-top5-jsonl>",
        },
        {
            "name": "Legal retrieved-top5 ctx12k completed run",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/legal_retrieved_top5_20.jsonl --dataset-name legal_retrieved_top5_20_ctx12k --output-dir results/legal_retrieved_top5_20_ctx12k --configs w/o_repair full_v2 full_v2_rescue --evidence-context-chars 12000 --answer-context-chars-with-evidence 12000 --verification-context-chars 5000 --repair-context-chars 5000 --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Legal retrieved-top5 prepended-full dataset",
            "command": "python experiments/build_retrieval_prefiltered_dataset.py --dataset datasets/UltraDomain/legal_fullcontext_20.jsonl --retrieval-output results/legal_fullcontext_20/naive_aligned_n20.jsonl --output datasets/UltraDomain/legal_retrieved_top5_prepended_full_20.jsonl --context-mode prepend_full",
        },
        {
            "name": "Legal retrieved-top5 prepended-full VSR",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/legal_retrieved_top5_prepended_full_20.jsonl --dataset-name legal_retrieved_top5_prepended_full_20 --output-dir results/legal_retrieved_top5_prepended_full_20 --configs w/o_repair full_v2 full_v2_rescue --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Legal full-context instrumented cost sample",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/legal_fullcontext_20.jsonl --dataset-name legal_fullcontext_20_cost_instrumented --output-dir results/cost_instrumented/legal_fullcontext_20_n5 --configs w/o_repair full_v2 full_v2_rescue --limit 5 --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Legal full-context instrumented cost sample lines 5-9",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/legal_fullcontext_20.jsonl --dataset-name legal_fullcontext_20_cost_instrumented_lines5_9 --output-dir results/cost_instrumented/legal_fullcontext_20_lines5_9 --configs w/o_repair full_v2 full_v2_rescue --line-nos 5-9 --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Biology full-context instrumented cost sample lines 0-4",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/biology_fullcontext_50.jsonl --dataset-name biology_fullcontext_50_cost_instrumented_lines0_4 --output-dir results/cost_instrumented/biology_fullcontext_50_lines0_4 --configs w/o_repair full_v2 full_v2_rescue --line-nos 0-4 --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Domain expansion windowed instrumented cost sample lines 0-4",
            "command": "python experiments/run_aligned_ablation_rerun.py --dataset-file datasets/UltraDomain/domain_expansion_windowed_50.jsonl --dataset-name domain_expansion_windowed_50_cost_instrumented_lines0_4 --output-dir results/cost_instrumented/domain_expansion_windowed_50_lines0_4 --configs w/o_repair full_v2 full_v2_rescue --line-nos 0-4 --question-timeout 420 --max-retries 3 --retry-errors",
        },
        {
            "name": "Cost/latency summary",
            "command": "python experiments/summarize_cost_latency.py",
        },
        {
            "name": "Cost-instrumented quality case audit",
            "command": "python experiments/summarize_instrumented_cost_quality_cases.py",
        },
        {
            "name": "I18n abstention/language guard audit",
            "command": "python experiments/build_i18n_abstention_language_guard.py",
        },
        {
            "name": "Legacy baseline lexical overlap summary",
            "command": "python experiments/summarize_legacy_baseline_overlap.py",
        },
        {
            "name": "SAGE candidate scan",
            "command": "python experiments/summarize_sage_candidate_scan.py",
        },
        {
            "name": "SAGE vs baselines summary",
            "command": "python experiments/summarize_sage_vs_baselines.py",
        },
        {
            "name": "SAGE external dry risk probes",
            "command": "python experiments/build_contract_completion_agent.py --input <hyde-aligned-jsonl> --dataset <matching-dataset-jsonl> --full-context-root datasets/UltraDomain --output results/<slice>_hyde_contract_completion_dry.jsonl --summary results/<slice>_hyde_contract_completion_dry_summary.md --evidence-mode text-only",
        },
        {
            "name": "SAGE Domain-windowed external API probe",
            "command": "python experiments/build_contract_completion_agent.py --input results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl --dataset datasets/UltraDomain/domain_expansion_windowed_50.jsonl --full-context-root datasets/UltraDomain --output results/historical_datasets/domain_hyde_contract_completion_lines12_14_31_39_42.jsonl --summary results/historical_datasets/domain_hyde_contract_completion_lines12_14_31_39_42_summary.md --evidence-mode text-only --execute --concurrency 2 --verification-line-nos 12,14,31,39,42",
        },
        {
            "name": "SAGE Domain-windowed external API probe batch 2",
            "command": "python experiments/build_contract_completion_agent.py --input results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl --dataset datasets/UltraDomain/domain_expansion_windowed_50.jsonl --full-context-root datasets/UltraDomain --output results/historical_datasets/domain_hyde_contract_completion_lines13_16_18_19_32_33_36_38_41_45.jsonl --summary results/historical_datasets/domain_hyde_contract_completion_lines13_16_18_19_32_33_36_38_41_45_summary.md --evidence-mode text-only --execute --concurrency 2 --verification-line-nos 13,16,18,19,32,33,36,38,41,45",
        },
        {
            "name": "SAGE Domain-windowed external API probe batch 3",
            "command": "python experiments/build_contract_completion_agent.py --input results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl --dataset datasets/UltraDomain/domain_expansion_windowed_50.jsonl --full-context-root datasets/UltraDomain --output results/historical_datasets/domain_hyde_contract_completion_lines11_17_30_35_40_46_47_49.jsonl --summary results/historical_datasets/domain_hyde_contract_completion_lines11_17_30_35_40_46_47_49_summary.md --evidence-mode text-only --execute --concurrency 2 --verification-line-nos 11,17,30,35,40,46,47,49",
        },
        {
            "name": "SAGE Domain-windowed external API probe batch 4",
            "command": "python experiments/build_contract_completion_agent.py --input results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl --dataset datasets/UltraDomain/domain_expansion_windowed_50.jsonl --full-context-root datasets/UltraDomain --output results/historical_datasets/domain_hyde_contract_completion_lines0_1_4_5_6_7_8_9_20_21_22_24_25_26_27_28_29.jsonl --summary results/historical_datasets/domain_hyde_contract_completion_lines0_1_4_5_6_7_8_9_20_21_22_24_25_26_27_28_29_summary.md --evidence-mode text-only --execute --concurrency 2 --verification-line-nos 0,1,4,5,6,7,8,9,20,21,22,24,25,26,27,28,29",
        },
        {
            "name": "SAGE Legal full-context external API probe",
            "command": "python experiments/build_contract_completion_agent.py --input results/legal_fullcontext_20/hyde_aligned_n20.jsonl --dataset datasets/UltraDomain/legal_fullcontext_20.jsonl --full-context-root datasets/UltraDomain --output results/historical_datasets/legal_hyde_contract_completion_lines0_1_4_5_11_13_14_15_18.jsonl --summary results/historical_datasets/legal_hyde_contract_completion_lines0_1_4_5_11_13_14_15_18_summary.md --evidence-mode text-only --execute --concurrency 2 --verification-line-nos 0,1,4,5,11,13,14,15,18",
        },
        {
            "name": "SAGE Biology full-context external API probe",
            "command": "python experiments/build_contract_completion_agent.py --input results/biology_fullcontext_50/hyde_aligned_n50.jsonl --dataset datasets/UltraDomain/biology_fullcontext_50.jsonl --full-context-root datasets/UltraDomain --output results/historical_datasets/biology_hyde_contract_completion_lines0_49_risk32.jsonl --summary results/historical_datasets/biology_hyde_contract_completion_lines0_49_risk32_summary.md --evidence-mode text-only --execute --concurrency 2 --verification-line-nos 0,1,3,6,7,8,9,10,11,12,14,15,16,19,21,22,23,26,27,29,31,32,35,37,38,39,40,41,42,46,47,49",
        },
        {
            "name": "SAGE external v32 guard",
            "command": "python experiments/build_evidence_delta_guard.py --source results/<slice>_hyde_contract_completion_<batch>.jsonl --output results/<slice>_hyde_evidence_delta_<batch>.jsonl --summary results/<slice>_hyde_evidence_delta_<batch>_summary.md",
        },
        {
            "name": "SAGE external v33 preservation guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/<slice>_hyde_contract_completion_cumulative.jsonl --output results/<slice>_hyde_evidence_delta_v33_cumulative.jsonl --summary results/<slice>_hyde_evidence_delta_v33_cumulative_summary.md",
        },
        {
            "name": "SAGE external judge evals",
            "command": "python experiments/evaluate_repro_results.py --input <base-or-v32-jsonl> --output results/<eval-output>.jsonl --summary results/<eval-summary>.md --line-nos <line-nos> --concurrency 1",
        },
        {
            "name": "SAGE external probe summary",
            "command": "python experiments/summarize_sage_external_probe.py",
        },
        {
            "name": "SAGE v33 external probe summary",
            "command": "python experiments/summarize_sage_v33_external_probe.py",
        },
        {
            "name": "SAGE external vs baselines summary",
            "command": "python experiments/summarize_sage_external_vs_baselines.py",
        },
        {
            "name": "SAGE locked full-slice ladder summary",
            "command": "python experiments/summarize_sage_locked_fullslice_ladder.py",
        },
        {
            "name": "SAGE v33 unseen34 preservation guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_vsr/v33_mix_unseen_context_contract_completion_merged.jsonl --output results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix_unseen34.jsonl --summary results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix_unseen34_summary.md",
        },
        {
            "name": "SAGE v46 zero-overlap v33 preservation guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_vsr/v46_heldout30_text_only_contract_completion_low5.jsonl --output results/historical_vsr/v46_zero_overlap_v33_preservation_delta_30.jsonl --summary results/historical_vsr/v46_zero_overlap_v33_preservation_delta_30_summary.md",
        },
        {
            "name": "SAGE v46 zero-overlap v33 judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_vsr/v46_zero_overlap_v33_preservation_delta_30.jsonl --output results/historical_vsr/v46_zero_overlap_v33_preservation_eval.jsonl --summary results/historical_vsr/v46_zero_overlap_v33_preservation_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE v46 zero-overlap v55 supported-facet guard",
            "command": "python experiments/build_v55_supported_facet_guard.py --input results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_v46_30.jsonl --output results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v46_30.jsonl --summary results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_v46_30_summary.md",
        },
        {
            "name": "SAGE unseen validation summary",
            "command": "python experiments/summarize_sage_unseen_validation.py",
        },
        {
            "name": "SAGE late-stage candidate ladder summary",
            "command": "python experiments/summarize_sage_late_stage_candidates.py",
        },
        {
            "name": "SAGE Mix130 v54 promotion replay high-risk long22",
            "command": "python experiments/build_v54_query_expanded_contract_gap_repair.py --input results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl --dataset datasets/UltraDomain/mix.jsonl --full-context-root datasets/UltraDomain --output results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_long22.jsonl --summary results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_long22_summary.md --line-nos 3,9,16,22,25,27,28,42,46,47,51,56,60,66,68,71,72,89,98,109,111,112",
        },
        {
            "name": "SAGE Mix130 v54 promotion replay lowjudge30",
            "command": "python experiments/build_v54_query_expanded_contract_gap_repair.py --input results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl --dataset datasets/UltraDomain/mix.jsonl --full-context-root datasets/UltraDomain --output results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_lowjudge30.jsonl --summary results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_lowjudge30_summary.md --line-nos 30,70,128,77,11,41,17,95,13,7,117,2,90,33,26,5,78,40,55,15,127,119,94,19,96,54,32,43,129,125",
        },
        {
            "name": "SAGE Mix130 v54 promotion replay highjudge30",
            "command": "python experiments/build_v54_query_expanded_contract_gap_repair.py --input results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl --dataset datasets/UltraDomain/mix.jsonl --full-context-root datasets/UltraDomain --output results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_highjudge30.jsonl --summary results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_highjudge30_summary.md --line-nos 116,6,91,12,80,110,86,123,97,84,118,18,75,4,121,87,69,65,99,100,74,29,106,37,14,10,24,20,103,44",
        },
        {
            "name": "SAGE Mix130 v54 line71 retry merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_long22.jsonl --updates results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_line71_retry.jsonl --output results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_long22_clean.jsonl --line-nos 71",
        },
        {
            "name": "SAGE Mix130 v54 candidate52 merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_long22_clean.jsonl --updates results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_lowjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_candidate52.jsonl --line-nos 30,70,128,77,11,41,17,95,13,7,117,2,90,33,26,5,78,40,55,15,127,119,94,19,96,54,32,43,129,125",
        },
        {
            "name": "SAGE Mix130 v54 candidate82 merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_candidate52.jsonl --updates results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_highjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_candidate82.jsonl --line-nos 116,6,91,12,80,110,86,123,97,84,118,18,75,4,121,87,69,65,99,100,74,29,106,37,14,10,24,20,103,44",
        },
        {
            "name": "SAGE Mix130 v55 supported-facet guard high-risk long22",
            "command": "python experiments/build_v55_supported_facet_guard.py --input results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_long22_clean.jsonl --output results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_long22_clean.jsonl --summary results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_long22_clean_summary.md",
        },
        {
            "name": "SAGE Mix130 v55 supported-facet guard lowjudge30",
            "command": "python experiments/build_v55_supported_facet_guard.py --input results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_lowjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_lowjudge30.jsonl --summary results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_lowjudge30_summary.md",
        },
        {
            "name": "SAGE Mix130 v55 supported-facet guard highjudge30",
            "command": "python experiments/build_v55_supported_facet_guard.py --input results/historical_sage/mse_pathrag_sage_v54_contract_gap_repair_mix130_highjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_highjudge30.jsonl --summary results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_highjudge30_summary.md",
        },
        {
            "name": "SAGE Mix130 v55 candidate52 merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_long22_clean.jsonl --updates results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_lowjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_candidate52.jsonl --line-nos 30,70,128,77,11,41,17,95,13,7,117,2,90,33,26,5,78,40,55,15,127,119,94,19,96,54,32,43,129,125",
        },
        {
            "name": "SAGE Mix130 v55 candidate82 merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_candidate52.jsonl --updates results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_highjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_candidate82.jsonl --line-nos 116,6,91,12,80,110,86,123,97,84,118,18,75,4,121,87,69,65,99,100,74,29,106,37,14,10,24,20,103,44",
        },
        {
            "name": "SAGE Mix130 v56 supported-facet fusion high-risk long22",
            "command": "python experiments/build_v56_supported_facet_fusion.py --input results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_long22_clean.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_clean.jsonl --summary results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_clean_summary.md",
        },
        {
            "name": "SAGE Mix130 v56 supported-facet fusion lowjudge30",
            "command": "python experiments/build_v56_supported_facet_fusion.py --input results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_lowjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_lowjudge30.jsonl --summary results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_lowjudge30_summary.md",
        },
        {
            "name": "SAGE Mix130 v56 supported-facet fusion highjudge30",
            "command": "python experiments/build_v56_supported_facet_fusion.py --input results/historical_sage/mse_pathrag_sage_v55_supported_facet_guard_mix130_highjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_highjudge30.jsonl --summary results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_highjudge30_summary.md",
        },
        {
            "name": "SAGE Mix130 v56 candidate52 merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_clean.jsonl --updates results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_lowjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate52.jsonl --line-nos 30,70,128,77,11,41,17,95,13,7,117,2,90,33,26,5,78,40,55,15,127,119,94,19,96,54,32,43,129,125",
        },
        {
            "name": "SAGE Mix130 v56 candidate82 merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate52.jsonl --updates results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_highjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82.jsonl --line-nos 116,6,91,12,80,110,86,123,97,84,118,18,75,4,121,87,69,65,99,100,74,29,106,37,14,10,24,20,103,44",
        },
        {
            "name": "SAGE Mix130 v56 changed-row judge high-risk long22",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_clean.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_changed_eval.jsonl --summary results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_changed_eval_summary.md --line-nos 9,16,22,42,47,66,98,112 --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE Mix130 v56 changed-row judge lowjudge30",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_lowjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_lowjudge30_changed_eval.jsonl --summary results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_lowjudge30_changed_eval_summary.md --line-nos 7,19,30,55,78,94,96,129 --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE Mix130 v56 changed-row judge highjudge30",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_highjudge30.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_highjudge30_changed_eval.jsonl --summary results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_highjudge30_changed_eval_summary.md --line-nos 4,29,118 --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE Mix130 v56 candidate52 changed-row judge merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_changed_eval.jsonl --updates results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_lowjudge30_changed_eval.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate52_changed_eval.jsonl --append-missing",
        },
        {
            "name": "SAGE Mix130 v56 candidate82 changed-row judge merge",
            "command": "python experiments/merge_jsonl_rows.py --base results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate52_changed_eval.jsonl --updates results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_highjudge30_changed_eval.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82_changed_eval.jsonl --append-missing",
        },
        {
            "name": "SAGE Mix130 v56 routed judge high-risk long22",
            "command": "python experiments/build_routed_eval_from_sources.py --routed results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_clean.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_clean_routed_eval.jsonl --summary results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_clean_routed_eval_summary.md --source v33:results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl:results/historical_sage/mse_pathrag_sage_v33_frozen_evidence_delta_mix130_eval.jsonl --source v56_changed:results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_clean.jsonl:results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_long22_changed_eval.jsonl",
        },
        {
            "name": "SAGE Mix130 v56 routed judge candidate82",
            "command": "python experiments/build_routed_eval_from_sources.py --routed results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82.jsonl --output results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82_routed_eval.jsonl --summary results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82_routed_eval_summary.md --source v33:results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130.jsonl:results/historical_sage/mse_pathrag_sage_v33_frozen_evidence_delta_mix130_eval.jsonl --source v56_changed:results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82.jsonl:results/historical_sage/mse_pathrag_sage_v56_supported_facet_fusion_mix130_candidate82_changed_eval.jsonl",
        },
        {
            "name": "SAGE Mix130 promotion replay summary",
            "command": "python experiments/summarize_sage_promotion_mix130.py",
        },
        {
            "name": "SAGE external promotion replay summary",
            "command": "python experiments/summarize_sage_promotion_external.py",
        },
        {
            "name": "SAGE v33 Legal full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_datasets/legal_hyde_evidence_delta_v33_cumulative.jsonl --output results/historical_datasets/legal_sage_v33_fullslice_eval.jsonl --summary results/historical_datasets/legal_sage_v33_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE v33 Biology full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_datasets/biology_hyde_evidence_delta_v33_cumulative.jsonl --output results/historical_datasets/biology_sage_v33_fullslice_eval.jsonl --summary results/historical_datasets/biology_sage_v33_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE v33 Domain full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_datasets/domain_hyde_evidence_delta_v33_cumulative.jsonl --output results/historical_datasets/domain_sage_v33_fullslice_eval.jsonl --summary results/historical_datasets/domain_sage_v33_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Legal Naive full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/legal_fullcontext_20/naive_aligned_n20.jsonl --output results/historical_datasets/legal_naive_fullslice_eval.jsonl --summary results/historical_datasets/legal_naive_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Legal HyDE full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/legal_fullcontext_20/hyde_aligned_n20.jsonl --output results/historical_datasets/legal_hyde_fullslice_eval.jsonl --summary results/historical_datasets/legal_hyde_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Legal VSR rescue full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/enhanced_ablation_aligned/legal_fullcontext_20/full_v2_rescue_aligned_n20.json --output results/historical_datasets/legal_vsr_rescue_fullslice_eval.jsonl --summary results/historical_datasets/legal_vsr_rescue_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Biology Naive full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/biology_fullcontext_50/naive_aligned_n50.jsonl --output results/historical_datasets/biology_naive_fullslice_eval.jsonl --summary results/historical_datasets/biology_naive_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Biology HyDE full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/biology_fullcontext_50/hyde_aligned_n50.jsonl --output results/historical_datasets/biology_hyde_fullslice_eval.jsonl --summary results/historical_datasets/biology_hyde_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Biology VSR rescue full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/enhanced_ablation_aligned/biology_fullcontext_50/full_v2_rescue_aligned_n50.json --output results/historical_datasets/biology_vsr_rescue_fullslice_eval.jsonl --summary results/historical_datasets/biology_vsr_rescue_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Domain Naive full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/domain_expansion_windowed_50/naive_aligned_n50.jsonl --output results/historical_datasets/domain_naive_fullslice_eval.jsonl --summary results/historical_datasets/domain_naive_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Domain HyDE full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl --output results/historical_datasets/domain_hyde_fullslice_eval.jsonl --summary results/historical_datasets/domain_hyde_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "Domain VSR rescue full-slice judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/enhanced_ablation_aligned/domain_expansion_windowed_50/full_v2_rescue_aligned_n50.json --output results/historical_datasets/domain_vsr_rescue_fullslice_eval.jsonl --summary results/historical_datasets/domain_vsr_rescue_fullslice_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE final candidate summary",
            "command": "python experiments/summarize_sage_final_candidate.py",
        },
        {
            "name": "SAGE v33 freeze and holdout audit",
            "command": "python experiments/build_sage_v33_freeze_holdout_audit.py",
        },
        {
            "name": "SAGE locked unseen75 holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py",
        },
        {
            "name": "SAGE locked unseen75 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --output results/historical_baselines/naive_aligned_n75.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen75 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --output results/historical_baselines/hyde_aligned_n75.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen75 method summary",
            "command": "python experiments/summarize_sage_locked_unseen75_baselines.py",
        },
        {
            "name": "SAGE locked unseen75 contract completion dry pass",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n75.jsonl --dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only_dry.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_dry_summary.md --evidence-mode text-only",
        },
        {
            "name": "SAGE locked unseen75 contract completion execution",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n75.jsonl --dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2",
        },
        {
            "name": "SAGE locked unseen75 v33 preservation guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked unseen75 v33 judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only_eval.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen75 HyDE judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_baselines/hyde_aligned_n75.jsonl --output results/historical_baselines/hyde_aligned_n75_eval.jsonl --summary results/historical_baselines/hyde_aligned_n75_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen75 Naive judge eval",
            "command": "python experiments/evaluate_repro_results.py --input results/historical_baselines/naive_aligned_n75.jsonl --output results/historical_baselines/naive_aligned_n75_eval.jsonl --summary results/historical_baselines/naive_aligned_n75_eval_summary.md --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen75 judge summary",
            "command": "python experiments/summarize_sage_locked_unseen75_judge.py",
        },
        {
            "name": "SAGE locked unseen75 pairwise vs HyDE",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline results/historical_baselines/hyde_aligned_n75.jsonl --output results/historical_pairwise/pairwise_sage_v33_vs_hyde.jsonl --summary results/historical_pairwise/pairwise_sage_v33_vs_hyde_summary.md --resume",
        },
        {
            "name": "SAGE locked unseen75 pairwise vs Naive",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline results/historical_baselines/naive_aligned_n75.jsonl --output results/historical_pairwise/pairwise_sage_v33_vs_naive.jsonl --summary results/historical_pairwise/pairwise_sage_v33_vs_naive_summary.md --resume",
        },
        {
            "name": "SAGE locked unseen75 pairwise summary",
            "command": "python experiments/summarize_sage_locked_unseen75_pairwise.py",
        },
        {
            "name": "SAGE locked unseen25 repeat1 holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py --output datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --manifest-md results/historical_sage/SAGE_LOCKED_UNSEEN25_REPEAT1_MANIFEST.md --manifest-json results/historical_sage/sage_locked_unseen25_repeat1_manifest.json --samples-per-domain 5 --max-samples-per-context 2 --exclude-dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl",
        },
        {
            "name": "SAGE locked unseen25 repeat1 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --output results/historical_baselines/naive_aligned_n25.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen25 repeat1 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --output results/historical_baselines/hyde_aligned_n25.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen25 repeat1 contract completion",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n25.jsonl --dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2",
        },
        {
            "name": "SAGE locked unseen25 repeat1 v33 guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked unseen25 repeat1 judge eval",
            "command": "python experiments/evaluate_repro_results.py --input <repeat1-method-jsonl> --output <repeat1-eval-jsonl> --summary <repeat1-eval-summary.md> --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen25 repeat1 pairwise eval",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline <repeat1-baseline-jsonl> --output <repeat1-pairwise-jsonl> --summary <repeat1-pairwise-summary.md> --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen25 repeat1 summary",
            "command": "python experiments/summarize_sage_locked_unseen_repeat.py",
        },
        {
            "name": "SAGE locked unseen40 repeat2 holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py --output datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --manifest-md results/historical_sage/SAGE_LOCKED_UNSEEN40_REPEAT2_MANIFEST.md --manifest-json results/historical_sage/sage_locked_unseen40_repeat2_manifest.json --samples-per-domain 10 --max-samples-per-context 2 --domains legal,biology,history,physics --exclude-dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl",
        },
        {
            "name": "SAGE locked unseen40 repeat2 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --output results/historical_baselines/naive_aligned_n40.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen40 repeat2 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --output results/historical_baselines/hyde_aligned_n40.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen40 repeat2 contract completion dry pass",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n40.jsonl --dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only_dry.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_dry_summary.md --evidence-mode text-only",
        },
        {
            "name": "SAGE locked unseen40 repeat2 contract completion",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n40.jsonl --dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2",
        },
        {
            "name": "SAGE locked unseen40 repeat2 v33 guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked unseen40 repeat2 judge eval",
            "command": "python experiments/evaluate_repro_results.py --input <repeat2-method-jsonl> --output <repeat2-eval-jsonl> --summary <repeat2-eval-summary.md> --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen40 repeat2 pairwise eval",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline <repeat2-baseline-jsonl> --output <repeat2-pairwise-jsonl> --summary <repeat2-pairwise-summary.md> --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen40 repeat2 summary",
            "command": "python experiments/summarize_sage_locked_unseen_repeat.py --dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --output-dir results/historical_sage/sage_locked_unseen40_repeat2 --expected-n 40 --file-suffix n40 --title \"SAGE Locked Unseen40 Repeat2 Summary\" --summary-json results/historical_sage/sage_locked_unseen40_repeat2_summary.json --summary-md results/historical_sage/SAGE_LOCKED_UNSEEN40_REPEAT2_SUMMARY.md",
        },
        {
            "name": "SAGE locked unseen24 repeat3 holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py --output datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --manifest-md results/historical_sage/SAGE_LOCKED_UNSEEN24_REPEAT3_MANIFEST.md --manifest-json results/historical_sage/sage_locked_unseen24_repeat3_manifest.json --samples-per-domain 6 --max-samples-per-context 2 --domains legal,biology,history,physics --exclude-dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl",
        },
        {
            "name": "SAGE locked unseen24 repeat3 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --output results/historical_baselines/naive_aligned_n24.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen24 repeat3 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --output results/historical_baselines/hyde_aligned_n24.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen24 repeat3 contract completion dry pass",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n24.jsonl --dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only_dry.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_dry_summary.md --evidence-mode text-only",
        },
        {
            "name": "SAGE locked unseen24 repeat3 contract completion",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n24.jsonl --dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2",
        },
        {
            "name": "SAGE locked unseen24 repeat3 v33 guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked unseen24 repeat3 judge eval",
            "command": "python experiments/evaluate_repro_results.py --input <repeat3-method-jsonl> --output <repeat3-eval-jsonl> --summary <repeat3-eval-summary.md> --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen24 repeat3 pairwise eval",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline <repeat3-baseline-jsonl> --output <repeat3-pairwise-jsonl> --summary <repeat3-pairwise-summary.md> --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen24 repeat3 summary",
            "command": "python experiments/summarize_sage_locked_unseen_repeat.py --dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --output-dir results/historical_sage/sage_locked_unseen24_repeat3 --expected-n 24 --file-suffix n24 --title \"SAGE Locked Unseen24 Repeat3 Summary\" --summary-json results/historical_sage/sage_locked_unseen24_repeat3_summary.json --summary-md results/historical_sage/SAGE_LOCKED_UNSEEN24_REPEAT3_SUMMARY.md",
        },
        {
            "name": "SAGE locked unseen24 repeat4 holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py --output datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --manifest-md results/historical_sage/SAGE_LOCKED_UNSEEN24_REPEAT4_MANIFEST.md --manifest-json results/historical_sage/sage_locked_unseen24_repeat4_manifest.json --samples-per-domain 8 --max-samples-per-context 2 --domains legal,biology,history --exclude-dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl",
        },
        {
            "name": "SAGE locked unseen24 repeat4 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --output results/historical_baselines/naive_aligned_n24.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen24 repeat4 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --output results/historical_baselines/hyde_aligned_n24.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked unseen24 repeat4 contract completion dry pass",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n24.jsonl --dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only_dry.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_dry_summary.md --evidence-mode text-only",
        },
        {
            "name": "SAGE locked unseen24 repeat4 contract completion",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n24.jsonl --dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2",
        },
        {
            "name": "SAGE locked unseen24 repeat4 v33 guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked unseen24 repeat4 judge eval",
            "command": "python experiments/evaluate_repro_results.py --input <repeat4-method-jsonl> --output <repeat4-eval-jsonl> --summary <repeat4-eval-summary.md> --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen24 repeat4 pairwise eval",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline <repeat4-baseline-jsonl> --output <repeat4-pairwise-jsonl> --summary <repeat4-pairwise-summary.md> --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked unseen24 repeat4 summary",
            "command": "python experiments/summarize_sage_locked_unseen_repeat.py --dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --output-dir results/historical_sage/sage_locked_unseen24_repeat4 --expected-n 24 --file-suffix n24 --title \"SAGE Locked Unseen24 Repeat4 Summary\" --summary-json results/historical_sage/sage_locked_unseen24_repeat4_summary.json --summary-md results/historical_sage/SAGE_LOCKED_UNSEEN24_REPEAT4_SUMMARY.md",
        },
        {
            "name": "SAGE locked legal40 repeat5 stress holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py --output datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl --manifest-md results/historical_sage/SAGE_LOCKED_LEGAL40_REPEAT5_MANIFEST.md --manifest-json results/historical_sage/sage_locked_legal40_repeat5_manifest.json --samples-per-domain 40 --max-samples-per-context 2 --domains legal --exclude-dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl",
        },
        {
            "name": "SAGE locked legal40 repeat5 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl --output results/historical_baselines/naive_aligned_n40.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked legal40 repeat5 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl --output results/historical_baselines/hyde_aligned_n40.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked legal40 repeat5 contract completion dry pass",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n40.jsonl --dataset datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only_dry.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_dry_summary.md --evidence-mode text-only",
        },
        {
            "name": "SAGE locked legal40 repeat5 contract completion",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n40.jsonl --dataset datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2",
        },
        {
            "name": "SAGE locked legal40 repeat5 v33 guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked legal40 repeat5 judge eval",
            "command": "python experiments/evaluate_repro_results.py --input <repeat5-method-jsonl> --output <repeat5-eval-jsonl> --summary <repeat5-eval-summary.md> --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked legal40 repeat5 pairwise eval",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline <repeat5-baseline-jsonl> --output <repeat5-pairwise-jsonl> --summary <repeat5-pairwise-summary.md> --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked legal40 repeat5 summary",
            "command": "python experiments/summarize_sage_locked_unseen_repeat.py --dataset datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl --output-dir results/historical_sage/sage_locked_legal40_repeat5 --expected-n 40 --file-suffix n40 --title \"SAGE Locked Legal40 Repeat5 Summary\" --summary-json results/historical_sage/sage_locked_legal40_repeat5_summary.json --summary-md results/historical_sage/SAGE_LOCKED_LEGAL40_REPEAT5_SUMMARY.md",
        },
        {
            "name": "SAGE locked history4 repeat6 tail holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py --output datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl --manifest-md results/historical_sage/SAGE_LOCKED_HISTORY4_REPEAT6_MANIFEST.md --manifest-json results/historical_sage/sage_locked_history4_repeat6_manifest.json --samples-per-domain 4 --max-samples-per-context 2 --domains history --exclude-dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl",
        },
        {
            "name": "SAGE locked history4 repeat6 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl --output results/historical_baselines/naive_aligned_n4.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked history4 repeat6 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl --output results/historical_baselines/hyde_aligned_n4.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked history4 repeat6 SAGE chain",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n4.jsonl --dataset datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2; python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked history4 repeat6 judge and pairwise eval",
            "command": "python experiments/evaluate_repro_results.py --input <repeat6-method-jsonl> --output <repeat6-eval-jsonl> --summary <repeat6-eval-summary.md> --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5; python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline <repeat6-baseline-jsonl> --output <repeat6-pairwise-jsonl> --summary <repeat6-pairwise-summary.md> --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked history4 repeat6 summary",
            "command": "python experiments/summarize_sage_locked_unseen_repeat.py --dataset datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl --output-dir results/historical_sage/sage_locked_history4_repeat6 --expected-n 4 --file-suffix n4 --title \"SAGE Locked History4 Repeat6 Tail Summary\" --summary-json results/historical_sage/sage_locked_history4_repeat6_summary.json --summary-md results/historical_sage/SAGE_LOCKED_HISTORY4_REPEAT6_SUMMARY.md",
        },
        {
            "name": "SAGE locked legal40 repeat7 stress holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py --output datasets/UltraDomain/sage_locked_legal40_repeat7_windowed.jsonl --manifest-md results/historical_sage/SAGE_LOCKED_LEGAL40_REPEAT7_MANIFEST.md --manifest-json results/historical_sage/sage_locked_legal40_repeat7_manifest.json --samples-per-domain 40 --max-samples-per-context 2 --domains legal --exclude-dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl",
        },
        {
            "name": "SAGE locked legal40 repeat7 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_legal40_repeat7_windowed.jsonl --output results/historical_baselines/naive_aligned_n40.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked legal40 repeat7 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_legal40_repeat7_windowed.jsonl --output results/historical_baselines/hyde_aligned_n40.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked legal40 repeat7 SAGE chain",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n40.jsonl --dataset datasets/UltraDomain/sage_locked_legal40_repeat7_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2; python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked legal40 repeat7 judge and pairwise eval",
            "command": "python experiments/evaluate_repro_results.py --input <repeat7-method-jsonl> --output <repeat7-eval-jsonl> --summary <repeat7-eval-summary.md> --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5; python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline <repeat7-baseline-jsonl> --output <repeat7-pairwise-jsonl> --summary <repeat7-pairwise-summary.md> --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked legal40 repeat7 summary",
            "command": "python experiments/summarize_sage_locked_unseen_repeat.py --dataset datasets/UltraDomain/sage_locked_legal40_repeat7_windowed.jsonl --output-dir results/historical_sage/sage_locked_legal40_repeat7 --expected-n 40 --file-suffix n40 --title \"SAGE Locked Legal40 Repeat7 Stress Summary\" --summary-json results/historical_sage/sage_locked_legal40_repeat7_summary.json --summary-md results/historical_sage/SAGE_LOCKED_LEGAL40_REPEAT7_SUMMARY.md",
        },
        {
            "name": "SAGE locked legal54 repeat8 exhaustive stress holdout build",
            "command": "python experiments/build_sage_locked_unseen_holdout.py --output datasets/UltraDomain/sage_locked_legal54_repeat8_windowed.jsonl --manifest-md results/historical_sage/SAGE_LOCKED_LEGAL54_REPEAT8_MANIFEST.md --manifest-json results/historical_sage/sage_locked_legal54_repeat8_manifest.json --samples-per-domain 54 --max-samples-per-context 2 --domains legal --exclude-dataset datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl --exclude-dataset datasets/UltraDomain/sage_locked_legal40_repeat7_windowed.jsonl",
        },
        {
            "name": "SAGE locked legal54 repeat8 Naive baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_legal54_repeat8_windowed.jsonl --output results/historical_baselines/naive_aligned_n54.jsonl --mode naive --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked legal54 repeat8 HyDE baseline",
            "command": "python experiments/run_retrieval_baseline.py --dataset datasets/UltraDomain/sage_locked_legal54_repeat8_windowed.jsonl --output results/historical_baselines/hyde_aligned_n54.jsonl --mode hyde --chunk-chars 1800 --overlap-chars 200 --top-k 5 --embedding-batch-size 16 --request-timeout 180 --max-retries 3 --append --skip-existing-output --retry-errors --shared-context-cache",
        },
        {
            "name": "SAGE locked legal54 repeat8 contract completion dry pass",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n54.jsonl --dataset datasets/UltraDomain/sage_locked_legal54_repeat8_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only_dry.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_dry_summary.md --evidence-mode text-only",
        },
        {
            "name": "SAGE locked legal54 repeat8 contract completion",
            "command": "python experiments/build_contract_completion_agent.py --input results/historical_baselines/hyde_aligned_n54.jsonl --dataset datasets/UltraDomain/sage_locked_legal54_repeat8_windowed.jsonl --full-context-root datasets/UltraDomain --output results/historical_baselines/hyde_contract_completion_text_only.jsonl --summary results/historical_baselines/hyde_contract_completion_text_only_summary.md --evidence-mode text-only --execute --concurrency 2",
        },
        {
            "name": "SAGE locked legal54 repeat8 v33 guard",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_baselines/hyde_contract_completion_text_only.jsonl --output results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --summary results/historical_baselines/hyde_evidence_delta_v33_text_only_summary.md",
        },
        {
            "name": "SAGE locked legal54 repeat8 judge eval",
            "command": "python experiments/evaluate_repro_results.py --input <repeat8-method-jsonl> --output <repeat8-eval-jsonl> --summary <repeat8-eval-summary.md> --resume --concurrency 1 --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked legal54 repeat8 pairwise eval",
            "command": "python experiments/evaluate_pairwise_winrate.py --pathrag results/historical_baselines/hyde_evidence_delta_v33_text_only.jsonl --baseline <repeat8-baseline-jsonl> --output <repeat8-pairwise-jsonl> --summary <repeat8-pairwise-summary.md> --resume --max-retries 5 --retry-delay-seconds 5",
        },
        {
            "name": "SAGE locked legal54 repeat8 summary",
            "command": "python experiments/summarize_sage_locked_unseen_repeat.py --dataset datasets/UltraDomain/sage_locked_legal54_repeat8_windowed.jsonl --output-dir results/historical_sage/sage_locked_legal54_repeat8 --expected-n 54 --file-suffix n54 --title \"SAGE Locked Legal54 Repeat8 Exhaustive Stress Summary\" --summary-json results/historical_sage/sage_locked_legal54_repeat8_summary.json --summary-md results/historical_sage/SAGE_LOCKED_LEGAL54_REPEAT8_SUMMARY.md",
        },
        {
            "name": "SAGE locked unseen holdout meta summary",
            "command": "python experiments/build_locked_holdout_meta_summary.py",
        },
        {
            "name": "SAGE locked holdout overlap audit",
            "command": "python experiments/build_locked_holdout_overlap_audit.py",
        },
        {
            "name": "SAGE v31 strict deterministic replay",
            "command": "python experiments/build_learned_contract_gate.py --dataset-name Mix --baseline results/historical_sage/mse_pathrag_sage_v21_text_first_utility_guard_mix_130.jsonl --completion results/historical_vsr/v31_mix130_next30_contract_completion.jsonl --oof results/historical_vsr/v31_full_rf_oof.jsonl --output results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_repro.jsonl --summary results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_repro_summary.md --version v31",
        },
        {
            "name": "SAGE v32 strict deterministic replay",
            "command": "python experiments/build_evidence_delta_guard.py --source results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_repro.jsonl --output results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_repro.jsonl --summary results/historical_sage/mse_pathrag_sage_v32_evidence_delta_mix130_repro_summary.md",
        },
        {
            "name": "SAGE v33 strict deterministic replay",
            "command": "python experiments/build_evidence_delta_guard_v33.py --source results/historical_sage/mse_pathrag_sage_v31_learned_contract_gap_mix130_repro.jsonl --output results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130_repro.jsonl --summary results/historical_sage/mse_pathrag_sage_v33_preservation_delta_mix130_repro_summary.md",
        },
        {
            "name": "SAGE reproduction validation",
            "command": "python experiments/validate_sage_reproduction.py",
        },
        {
            "name": "Paper LaTeX table fragments",
            "command": "python experiments/build_paper_latex_tables.py",
        },
        {
            "name": "Evidence-contract depth audit",
            "command": "python experiments/analyze_evidence_contract_depth.py",
        },
        {
            "name": "Impact-aware escalation offline proxy",
            "command": "python experiments/analyze_impact_aware_escalation.py",
        },
        {
            "name": "Impact-aware escalation depth audit",
            "command": "python experiments/deepen_impact_aware_escalation.py",
        },
        {
            "name": "Freeze impact-aware escalation v1",
            "command": "python experiments/freeze_impact_aware_policy.py",
        },
        {
            "name": "Reproducibility manifest",
            "command": "python experiments/build_reproducibility_manifest.py",
        },
    ]


def expanded_result_files() -> list[str]:
    dynamic_dirs = [
        ROOT / "results" / "sage_external_probe",
        ROOT / "results" / "sage_repro",
        ROOT / "results" / "sage_vs_baselines",
        ROOT / "results" / "sage_candidate_scan",
        ROOT / "results" / "sage_final_candidate",
        ROOT / "results" / "sage_locked_fullslice",
        ROOT / "results" / "sage_locked_unseen75",
        ROOT / "results" / "sage_locked_unseen25_repeat1",
        ROOT / "results" / "sage_locked_unseen40_repeat2",
        ROOT / "results" / "sage_locked_unseen24_repeat3",
        ROOT / "results" / "sage_locked_unseen24_repeat4",
        ROOT / "results" / "sage_locked_unseen_meta",
        ROOT / "results" / "sage_locked_legal40_repeat5",
        ROOT / "results" / "sage_locked_history4_repeat6",
        ROOT / "results" / "sage_locked_legal40_repeat7",
        ROOT / "results" / "sage_locked_legal54_repeat8",
        ROOT / "results" / "sage_locked_holdout_overlap",
        ROOT / "results" / "sage_unseen_validation",
        ROOT / "results" / "sage_late_stage_candidates",
        ROOT / "results" / "sage_promotion_mix130",
        ROOT / "results" / "sage_promotion_external",
        ROOT / "results" / "cost_instrumented",
        ROOT / "results" / "cost_latency",
    ]
    paths = list(RESULT_FILES)
    for directory in dynamic_dirs:
        if not directory.exists():
            continue
        paths.extend(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        )
    return sorted(dict.fromkeys(paths))


def build_manifest() -> dict[str, Any]:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "workspace": str(ROOT),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "git": {
            "commit": run_git(["rev-parse", "HEAD"]),
            "branch": run_git(["branch", "--show-current"]),
            "status_short": run_git(["status", "--short"]),
            "note": "Working tree may contain unrelated dirty/untracked files; use file hashes below for experiment reproducibility.",
        },
        "environment": env_snapshot(),
        "scripts": [file_entry(path) for path in SCRIPT_FILES],
        "datasets": [file_entry(path) for path in DATASET_FILES],
        "results": [file_entry(path) for path in expanded_result_files()],
        "commands": experiment_commands(),
    }


def md_table(entries: list[dict[str, Any]]) -> list[str]:
    lines = ["| Path | Exists | Bytes | SHA256 |", "|---|---:|---:|---|"]
    for entry in entries:
        sha = entry.get("sha256", "")
        lines.append(
            f"| `{entry['path']}` | {str(entry.get('exists')).lower()} | {entry.get('bytes', '')} | `{sha[:16]}` |"
        )
    return lines


def write_markdown(manifest: dict[str, Any]) -> str:
    env = manifest["environment"]
    lines = [
        "# Reproducibility Manifest",
        "",
        f"> Generated: {manifest['generated_at']}",
        f"> Workspace: `{manifest['workspace']}`",
        "",
        "## Environment",
        "",
        f"- Python: `{manifest['python']['version'].split()[0]}`",
        f"- Platform: `{manifest['python']['platform']}`",
        f"- Git branch: `{manifest['git'].get('branch') or 'unknown'}`",
        f"- Git commit: `{manifest['git'].get('commit') or 'unknown'}`",
        f"- LLM model: `{env.get('DEEPSEEK_MODEL') or 'unset'}`",
        f"- LLM base URL: `{env.get('DEEPSEEK_API_BASE') or 'unset'}`",
        f"- Embedding model: `{env.get('EMBEDDING_MODEL') or 'unset'}`",
        f"- Embedding dim: `{env.get('EMBEDDING_DIM') or 'unset'}`",
        f"- API key set: `{env.get('DEEPSEEK_API_KEY_SET')}`",
        "",
        "## Scripts",
        "",
        *md_table(manifest["scripts"]),
        "",
        "## Datasets",
        "",
        *md_table(manifest["datasets"]),
        "",
        "## Result Artifacts",
        "",
        *md_table(manifest["results"]),
        "",
        "## Key Commands",
        "",
    ]
    for command in manifest["commands"]:
        lines.extend([f"### {command['name']}", "", "```powershell", command["command"], "```", ""])
    lines.extend(
        [
            "## Notes",
            "",
            "- The manifest intentionally records no API secrets; it only records whether an API key was present in the environment.",
            "- File hashes are the reproducibility anchor because the repository currently contains many dirty or untracked experiment files.",
            "- The Legal retrieved-top5 ctx12k and top5+full-prepend runs are complete but should be treated as negative diagnostics because they remain below Naive/HyDE and default/full-context VSR.",
            "- SAGE v31/v32/v33 strict replay uses `results/historical_vsr/v31_full_rf_oof.jsonl`; using `results/historical_sage/mse_pathrag_sage_v31_contract_gap_router_oof.jsonl` changes route probabilities/thresholds and is not the original mix130 protocol.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    json_path = OUT_DIR / "reproducibility_manifest.json"
    md_path = OUT_DIR / "REPRODUCIBILITY_MANIFEST.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(write_markdown(manifest), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
