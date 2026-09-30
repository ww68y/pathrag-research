from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "sage_final_candidate"
OUTPUT_JSON = OUT_DIR / "sage_final_candidate_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_FINAL_CANDIDATE_SUMMARY.md"


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_optional_json(path: str) -> dict[str, Any] | None:
    candidate = ROOT / path
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def find_mix_method(summary: dict[str, Any], method_name: str) -> dict[str, Any]:
    for row in summary.get("mix130_methods", []):
        if row.get("method") == method_name:
            return row
    raise ValueError(f"Missing method: {method_name}")


def final_candidate_summary() -> dict[str, Any]:
    sage_vs = load_json("results/historical_sage/sage_vs_baselines_summary.json")
    external = load_json("results/historical_sage/sage_external_vs_baselines_summary.json")
    v33 = load_json("results/historical_sage/sage_v33_external_probe_summary.json")
    repro = load_json("results/historical_sage/sage_reproduction_validation.json")
    locked = load_optional_json("results/historical_sage/sage_locked_fullslice_ladder_summary.json")
    unseen = load_optional_json("results/historical_sage/sage_unseen_validation_summary.json")
    late = load_optional_json("results/historical_sage/sage_late_stage_candidates_summary.json")
    promotion = load_optional_json("results/historical_sage/sage_promotion_mix130_summary.json")
    external_promotion = load_optional_json("results/historical_sage/sage_promotion_external_summary.json")

    mix_v33 = find_mix_method(sage_vs, "SAGE v33 preservation-delta")
    locked_summary = None
    if locked:
        v33_weighted = next(
            (row for row in locked.get("external_weighted", []) if row.get("key") == "sage_v33"),
            None,
        )
        weighted_judges = locked.get("external_weighted_judges", [])
        locked_summary = {
            "all_slices_complete": all(row.get("complete") for row in locked.get("slices", [])),
            "external_weighted_top": locked.get("external_weighted", [None])[0],
            "external_weighted_v33": v33_weighted,
            "external_weighted_judge": locked.get("external_weighted_judge"),
            "external_weighted_judges": weighted_judges,
            "judge_pairwise": {
                row["dataset"]: {
                    key: {
                        "delta": value.get("delta"),
                        "wins": value.get("wins"),
                        "losses": value.get("losses"),
                        "ties": value.get("ties"),
                        "t_p": value.get("t_p"),
                        "cohens_d": value.get("cohens_d"),
                    }
                    for key, value in row.get("judge_pairwise_tests", {}).items()
                    if key in {"sage_v33_vs_best_retrieval", "sage_v33_vs_vsr_rescue"}
                }
                for row in locked.get("slices", [])
                if row.get("judge_pairwise_tests")
            },
            "v33_v32_prediction_diffs": {
                row["dataset"]: row.get("prediction_diffs", {}).get("sage_v33_vs_sage_v32", 0)
                for row in locked.get("slices", [])
            },
        }
    unseen_summary = None
    if unseen:
        baseline = unseen["scores"]["baseline"]
        current = unseen["scores"]["current_v33"]
        baseline_judge = unseen["judges"]["baseline"]
        current_judge = unseen["judges"]["current_v33"]
        f1_pairwise = unseen["pairwise"]["v33_vs_baseline_f1"]
        judge_pairwise = unseen["judge_pairwise"]["v33_vs_baseline_overall"]
        unseen_summary = {
            "n": unseen["protocol"]["n"],
            "scope": unseen["protocol"]["scope"],
            "f1_baseline": baseline["mean_f1"],
            "f1_v33": current["mean_f1"],
            "f1_delta": f1_pairwise["delta"],
            "f1_t_p": f1_pairwise["t_p"],
            "judge_overall_baseline": baseline_judge["overall"],
            "judge_overall_v33": current_judge["overall"],
            "judge_overall_delta": judge_pairwise["delta"],
            "judge_t_p": judge_pairwise["t_p"],
            "changed_rows_vs_baseline": unseen["prediction_diffs"]["current_v33_vs_baseline_heldout"],
            "candidate_changed_rows_vs_baseline": unseen["prediction_diffs"]["contract_candidate_vs_baseline_heldout"],
            "old_frozen_prediction_diffs_all_mix": len(unseen["prediction_diffs"]["current_v33_vs_old_frozen_all_mix"]),
            "guard_decisions": unseen["guard_decisions"],
            "reused_judge_scores": unseen["protocol"].get("reused_judge_scores", False),
            "additional_slices": [
                {
                    "name": row["name"],
                    "n": row["n"],
                    "f1_baseline": row["scores"]["baseline"]["mean_f1"],
                    "f1_v33": row["scores"]["current_v33"]["mean_f1"],
                    "f1_delta": row["pairwise"]["v33_vs_baseline_f1"]["delta"],
                    "judge_overall_baseline": row["judges"]["baseline"]["overall"],
                    "judge_overall_v33": row["judges"]["current_v33"]["overall"],
                    "judge_overall_delta": row["judge_pairwise"]["v33_vs_baseline_overall"]["delta"],
                    "changed_rows_vs_baseline": row["prediction_diffs"]["current_v33_vs_baseline"],
                    "guard_decisions": row.get("guard_decisions", []),
                    "exploratory_v55": (
                        {
                            "f1": row["scores"]["sage_v55_supported_facet"]["mean_f1"],
                            "rouge_l": row["scores"]["sage_v55_supported_facet"]["mean_rouge_l"],
                            "judge_overall": row["judges"]["sage_v55_supported_facet"]["overall"],
                            "judge_correctness": row["judges"]["sage_v55_supported_facet"]["correctness"],
                            "delta_f1_vs_baseline": row["pairwise"]["v55_vs_baseline_f1"]["delta"],
                            "delta_f1_vs_v33": row["pairwise"]["v55_vs_v33_f1"]["delta"],
                            "delta_judge_vs_baseline": row["judge_pairwise"]["v55_vs_baseline_overall"]["delta"],
                            "delta_judge_vs_v33": row["judge_pairwise"]["v55_vs_v33_overall"]["delta"],
                            "changed_rows_vs_baseline": row["prediction_diffs"]["v55_vs_baseline"],
                            "prediction_diffs_vs_v54": row["prediction_diffs"].get("v55_vs_v54", []),
                        }
                        if "sage_v55_supported_facet" in row.get("scores", {})
                        else None
                    ),
                }
                for row in unseen.get("additional_slices", [])
            ],
        }
    late_summary = None
    if late:
        late_rows = []
        best_rows = []
        for slice_row in late.get("slices", []):
            non_baseline = []
            for key, method in slice_row.get("methods", {}).items():
                judge = method.get("eval") or {}
                vs_hyde = method.get("vs_hyde") or {}
                pairwise_win = method.get("pairwise_win") or {}
                row = {
                    "slice": slice_row["name"],
                    "complete": slice_row.get("complete"),
                    "key": key,
                    "label": method["label"],
                    "status": method["status"],
                    "n": method["n"],
                    "f1": method["mean_f1"],
                    "rouge_l": method["mean_rouge_l"],
                    "judge_overall": judge.get("overall"),
                    "judge_correctness": judge.get("correctness"),
                    "judge_coverage": judge.get("coverage"),
                    "delta_f1_vs_hyde": (vs_hyde.get("f1") or {}).get("delta"),
                    "delta_judge_vs_hyde": (vs_hyde.get("judge_overall") or {}).get("delta"),
                    "pairwise_win_rate_vs_hyde": pairwise_win.get("average_win_rate"),
                    "prediction_diffs_vs_hyde": vs_hyde.get("prediction_diffs") if vs_hyde else [],
                }
                late_rows.append(row)
                if key != "hyde":
                    non_baseline.append(row)
            if non_baseline:
                best_rows.append(max(non_baseline, key=lambda item: (item.get("judge_overall") or 0.0, item.get("f1") or 0.0)))
        late_summary = {
            "all_slices_complete": all(row.get("complete") for row in late.get("slices", [])),
            "rows": late_rows,
            "best_by_slice": best_rows,
            "protocol": late.get("protocol", {}),
        }
    promotion_summary = None
    if promotion:
        methods = promotion.get("method_scores", {})
        compact_scores = {}
        for key in ["v33", "v54", "v55", "v56"]:
            row = methods.get(key) or {}
            judge = row.get("judge") or {}
            compact_scores[key] = {
                "label": row.get("label"),
                "n": row.get("n"),
                "target_n": row.get("target_n"),
                "f1": row.get("mean_f1"),
                "rouge_l": row.get("mean_rouge_l"),
                "target_f1": row.get("target_mean_f1"),
                "target_rouge_l": row.get("target_mean_rouge_l"),
                "judge_overall": judge.get("overall"),
                "judge_coverage": judge.get("coverage"),
                "status": row.get("status"),
                "path": row.get("path"),
            }
        promotion_summary = {
            "target_lines": promotion.get("target_lines", []),
            "changed_lines_vs_v33": promotion.get("changed_lines_vs_v33", []),
            "scores": compact_scores,
            "actions": promotion.get("actions", {}),
            "v56_vs_v33_full_f1": (promotion.get("lexical_pairwise", {}) or {}).get("v56_vs_v33_full_f1"),
            "v56_vs_v33_target_f1": (promotion.get("lexical_pairwise", {}) or {}).get("v56_vs_v33_target_f1"),
            "v56_vs_v33_full_judge_overall": (promotion.get("judge_pairwise", {}) or {}).get("v56_vs_v33_full_overall"),
            "changed_judge_rows": promotion.get("changed_judge_rows", []),
            "risk_notes": promotion.get("risk_notes", []),
        }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": "SAGE v33-preservation",
        "positioning": "lead candidate method, not yet final locked paper claim",
        "mix130": {
            "n": mix_v33["n"],
            "f1": mix_v33["mean_f1"],
            "rouge_l": mix_v33["mean_rouge_l"],
            "runtime_seconds": mix_v33["mean_runtime_seconds"],
            "sage_minus_vsr_f1": sage_vs["sage_vsr_pairwise"]["mean_delta"],
            "sage_vs_vsr_wins": sage_vs["sage_vsr_pairwise"]["wins"],
            "sage_vs_vsr_losses": sage_vs["sage_vsr_pairwise"]["losses"],
        },
        "external_slices": external["slices"],
        "v33_safety": v33["probes"],
        "reproduction": {
            "all_object_exact": repro.get("all_object_exact"),
            "all_predictions_exact": repro.get("all_predictions_exact"),
            "pairs": [
                {
                    "name": row["name"],
                    "rows": row["reproduction_rows"],
                    "f1": row["mean_f1"],
                    "rouge_l": row["mean_rouge_l"],
                    "object_diffs": row["exact_object_diffs"],
                    "prediction_diffs": row["prediction_diffs"],
                    "route_diffs": row["route_metadata_diffs"],
                }
                for row in repro.get("pairs", [])
            ],
        },
        "locked_fullslice": locked_summary,
        "unseen_validation": unseen_summary,
        "late_stage_candidates": late_summary,
        "mix130_promotion_replay": promotion_summary,
        "external_promotion_replay": external_promotion,
    }


def write_report(payload: dict[str, Any]) -> None:
    mix = payload["mix130"]
    lines = [
        "# SAGE Final Candidate Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Candidate Position",
        "",
        f"- Candidate method: `{payload['candidate']}`.",
        "- Current status: lead experimental candidate, but not yet a final locked SCI claim.",
        "- v33 adds deterministic high-content preservation on top of v31/v32 adaptive routing and evidence-delta safety.",
        "",
        "## Mix130 Result",
        "",
        "| N | F1 | ROUGE-L | Runtime s/q | Delta vs VSR rescue | Wins | Losses |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {mix['n']} | {mix['f1']:.4f} | {mix['rouge_l']:.4f} | {mix['runtime_seconds']:.1f} | {mix['sage_minus_vsr_f1']:+.4f} | {mix['sage_vs_vsr_wins']} | {mix['sage_vs_vsr_losses']} |",
        "",
        "## External Slice Snapshot",
        "",
        "| Dataset | N | Best retrieval | Best retrieval F1 | SAGE v33 F1 | SAGE - best retrieval | SAGE - VSR rescue | Significance note |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["external_slices"]:
        methods = row["methods"]
        best = row["best_retrieval"]
        best_delta = row["sage_vs_best_retrieval"]
        vsr_delta = row["sage_vs_vsr_rescue"]
        sig = "significant vs HyDE/VSR; not significant vs Naive" if row["dataset"] == "Domain expansion windowed" else "not significant vs best retrieval"
        if row["dataset"] == "Biology full context":
            sig = "safety-neutral tie vs HyDE"
        lines.append(
            f"| {row['dataset']} | {row['n']} | {best} | {methods[best]['f1']:.4f} | {methods['sage_v33']['f1']:.4f} | "
            f"{best_delta['delta']:+.4f} | {vsr_delta['delta']:+.4f} | {sig} |"
        )

    lines.extend(
        [
            "",
            "## v33 Safety Check",
            "",
            "| Probe | Verified | v33 diffs vs v32 | All-row F1 base -> v32 -> v33 | Judge overall base -> v32 -> v33 |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in payload["v33_safety"]:
        all_f1 = row["all_mean_f1"]
        judge = row["selected_mean_judge"]
        lines.append(
            f"| {row['title']} | {row['verified']} | {len(row['v33_diffs_vs_v32'])} | "
            f"{all_f1['base']:.4f} -> {all_f1['v32']:.4f} -> {all_f1['v33']:.4f} | "
            f"{judge['base_overall']:.2f} -> {judge['v32_overall']:.2f} -> {judge['v33_overall']:.2f} |"
        )

    locked = payload.get("locked_fullslice")
    if locked:
        top = locked.get("external_weighted_top") or {}
        v33_weighted = locked.get("external_weighted_v33") or {}
        weighted_judges = locked.get("external_weighted_judges") or []
        judge = next((row for row in weighted_judges if row.get("key") == "sage_v33"), {})
        best_judge = next((row for row in weighted_judges if row.get("key") == "best_retrieval"), {})
        vsr_judge = next((row for row in weighted_judges if row.get("key") == "vsr_rescue"), {})
        pairwise = locked.get("judge_pairwise") or {}
        pairwise_text = "; ".join(
            f"{dataset}: best {tests.get('sage_v33_vs_best_retrieval', {}).get('delta', 0.0):+.2f}, "
            f"VSR {tests.get('sage_v33_vs_vsr_rescue', {}).get('delta', 0.0):+.2f}"
            for dataset, tests in pairwise.items()
        )
        diffs = locked.get("v33_v32_prediction_diffs") or {}
        diff_text = ", ".join(f"{name}:{count}" for name, count in diffs.items())
        lines.extend(
            [
                "",
                "## Locked Full-Slice Ladder",
                "",
                f"- All scored slices complete: `{locked['all_slices_complete']}`.",
                f"- Weighted external top: `{top.get('label')}` F1={top.get('weighted_f1', 0.0):.4f}; v33 weighted F1={v33_weighted.get('weighted_f1', 0.0):.4f}.",
                f"- v33 weighted external judge: N={judge.get('n', 0)}, overall={judge.get('overall', 0.0):.2f}, correctness={judge.get('correctness', 0.0):.2f}, coverage={judge.get('coverage', 0.0):.2f}.",
                f"- Judge comparison: v33 overall {judge.get('overall', 0.0):.2f} vs best retrieval {best_judge.get('overall', 0.0):.2f} and VSR rescue {vsr_judge.get('overall', 0.0):.2f} on the same 120 external rows.",
                f"- Paired judge overall deltas: {pairwise_text}.",
                "- v33 is kept as the freeze candidate because it preserves Mix130 parity and removes the Biology v32 regression, not because it maximizes every external lexical aggregate.",
                f"- v33-v32 prediction diffs: {diff_text}.",
            ]
        )

    unseen = payload.get("unseen_validation")
    if unseen:
        decision_text = "; ".join(
            f"line {row['line_no']} {row['action']} (base {row['baseline_overall']} -> candidate {row['candidate_overall']} -> final {row['final_overall']})"
            for row in unseen.get("guard_decisions", [])
        )
        lines.extend(
            [
                "",
                "## Unseen Validation",
                "",
                f"- Context-heldout Mix risk candidates: N={unseen['n']}; judge scores reused=`{unseen['reused_judge_scores']}` because current v33 matches old frozen v33 on all Mix130 predictions.",
                f"- Lexical F1: baseline {unseen['f1_baseline']:.4f} -> v33 {unseen['f1_v33']:.4f} ({unseen['f1_delta']:+.4f}, t-p={unseen['f1_t_p']:.4g}).",
                f"- Judge overall: baseline {unseen['judge_overall_baseline']:.3f} -> v33 {unseen['judge_overall_v33']:.3f} ({unseen['judge_overall_delta']:+.3f}, t-p={unseen['judge_t_p']:.4g}).",
                f"- Guard decisions: {decision_text}.",
                f"- Current v33 final changes vs baseline: {unseen['changed_rows_vs_baseline']}; candidate repair attempts changed rows: {unseen['candidate_changed_rows_vs_baseline']}.",
                "- Framing: use as context-heldout safety validation, not as a large unseen benchmark claim.",
            ]
        )
        additional = unseen.get("additional_slices") or []
        if additional:
            lines.extend(
                [
                    "",
                    "| Additional slice | N | Baseline F1 | v33 F1 | Delta F1 | Baseline judge | v33 judge | Delta judge | Changed rows |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in additional:
                lines.append(
                    f"| {row['name']} | {row['n']} | {row['f1_baseline']:.4f} | {row['f1_v33']:.4f} | "
                    f"{row['f1_delta']:+.4f} | {row['judge_overall_baseline']:.3f} | "
                    f"{row['judge_overall_v33']:.3f} | {row['judge_overall_delta']:+.3f} | {row['changed_rows_vs_baseline']} |"
                )
            exploratory = [row for row in additional if row.get("exploratory_v55")]
            if exploratory:
                lines.extend(
                    [
                        "",
                        "| Exploratory branch | Slice | F1 | ROUGE-L | Judge overall | Delta F1 vs baseline | Delta overall vs baseline | Delta overall vs v33 | Changed rows |",
                        "|---|---|---:|---:|---:|---:|---:|---:|---|",
                    ]
                )
                for row in exploratory:
                    v55 = row["exploratory_v55"]
                    lines.append(
                        f"| SAGE v55 supported-facet guard | {row['name']} | {v55['f1']:.4f} | {v55['rouge_l']:.4f} | "
                        f"{v55['judge_overall']:.3f} | {v55['delta_f1_vs_baseline']:+.4f} | "
                        f"{v55['delta_judge_vs_baseline']:+.3f} | {v55['delta_judge_vs_v33']:+.3f} | {v55['changed_rows_vs_baseline']} |"
                    )
                lines.append("- v55 is a stronger v46 exploratory branch, but it is not yet reconciled with the v33 frozen full-slice/reproducibility protocol.")

    late = payload.get("late_stage_candidates")
    if late:
        lines.extend(
            [
                "",
                "## Late-Stage Candidates",
                "",
                f"- All late-stage slices complete: `{late['all_slices_complete']}`.",
                "- These v54/v55/v56 rows are promotion candidates only; v33 remains the freeze candidate until the late-stage branch is replayed on Mix130/full-slice protocol.",
                "",
                "| Slice | Best late-stage method | N | F1 | Judge overall | Delta F1 vs HyDE | Delta judge vs HyDE | Pairwise win vs HyDE |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in late.get("best_by_slice", []):
            win = row.get("pairwise_win_rate_vs_hyde")
            win_text = "-" if win is None else f"{win:.2%}"
            lines.append(
                f"| {row['slice']} | {row['label']} | {row['n']} | {row['f1']:.4f} | "
                f"{row['judge_overall']:.3f} | {(row.get('delta_f1_vs_hyde') or 0.0):+.4f} | "
                f"{(row.get('delta_judge_vs_hyde') or 0.0):+.3f} | {win_text} |"
            )
        lines.append("- Next promotion decision point: inspect Mix130 candidate82 regressions before any external full-slice replay or main-method upgrade.")

    promotion = payload.get("mix130_promotion_replay")
    if promotion:
        scores = promotion.get("scores") or {}
        v33_score = scores.get("v33") or {}
        v56_score = scores.get("v56") or {}
        full_f1 = promotion.get("v56_vs_v33_full_f1") or {}
        target_f1 = promotion.get("v56_vs_v33_target_f1") or {}
        judge_overall = promotion.get("v56_vs_v33_full_judge_overall") or {}
        changed = promotion.get("changed_lines_vs_v33") or []
        actions = promotion.get("actions") or {}
        v54_actions = actions.get("v54") or {}
        v55_actions = actions.get("v55") or {}
        v56_actions = actions.get("v56") or {}
        regressions = [row for row in promotion.get("changed_judge_rows", []) if row.get("delta_overall", 0) < 0]
        remaining_candidates = max(0, 82 - len(promotion.get("target_lines") or []))
        replay_scope = (
            "Mix130 v54->v55->v56 promotion replay"
            if remaining_candidates == 0
            else "First high-risk v54->v55->v56 promotion batch"
        )
        lines.extend(
            [
                "",
                "## Mix130 Promotion Replay",
                "",
                f"- {replay_scope} complete: {len(promotion.get('target_lines') or [])} targeted Mix130 rows out of 82 v54 candidates.",
                f"- v54 accepted {v54_actions.get('use_contract_gap_repair', 0)} repairs; v55 reverted {v55_actions.get('revert_to_pre_repair_answer', 0)} over-compressed repair; v56 accepted {v56_actions.get('use_supported_facet_fusion', 0)} fusion.",
                f"- v56 full-Mix130 lexical F1: {v33_score.get('f1', 0.0):.4f} -> {v56_score.get('f1', 0.0):.4f} ({full_f1.get('delta', 0.0):+.4f}); target-{len(promotion.get('target_lines') or [])} F1 delta {target_f1.get('delta', 0.0):+.4f}.",
                f"- Routed judge overall: {v33_score.get('judge_overall', 0.0):.4f} -> {v56_score.get('judge_overall', 0.0):.4f} ({judge_overall.get('delta', 0.0):+.4f}) using new judge calls only for changed predictions.",
                f"- Final changed rows vs v33: {changed}.",
                f"- Promotion status: positive but not promotable yet; {len(regressions)} changed rows regress by judge score and line 47 remains poor.",
                "",
                "| Line | Dataset | v33 overall | v56 overall | Delta | v56 verdict |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        for row in promotion.get("changed_judge_rows", []):
            lines.append(
                f"| {row['line_no']} | {row.get('dataset')} | {row['v33_overall']:.0f} | "
                f"{row['v56_overall']:.0f} | {row['delta_overall']:+.0f} | {row.get('v56_verdict')} |"
            )

    external_promotion = payload.get("external_promotion_replay")
    if external_promotion:
        weighted = external_promotion.get("external_weighted") or {}
        lines.extend(
            [
                "",
                "## External Full-Slice Promotion Replay",
                "",
                "- Legal20, Biology50, and Domain50 have completed the v54 -> v55 -> v56 replay and routed evaluation.",
                f"- Weighted external F1: {weighted.get('v33_f1', 0.0):.4f} -> {weighted.get('v56_f1', 0.0):.4f} ({weighted.get('f1_delta', 0.0):+.4f}); "
                f"routed judge overall: {weighted.get('v33_judge_overall', 0.0):.4f} -> {weighted.get('v56_judge_overall', 0.0):.4f} ({weighted.get('judge_overall_delta', 0.0):+.4f}).",
                "- The result is mixed: Legal judge overall falls by 0.05, while Biology and Domain improve; all three slices include changed-row judge regressions.",
                "- v55 reverted no external v54 repair and v56 triggered no external fusion, so this replay does not demonstrate that the guard/fusion stages bound repair risk.",
                "- Decision: retain v33-preservation as the freeze candidate and keep v56 as an exploratory branch.",
            ]
        )

    repro = payload["reproduction"]
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Object-exact reproduction: `{repro['all_object_exact']}`.",
            f"- Prediction-exact reproduction: `{repro['all_predictions_exact']}`.",
            "- v31/v32/v33 mix130 artifacts all replay with 0 object, prediction, and route-metadata diffs under the frozen `v31_full_rf_oof` router.",
            "",
            "## Paper Implication",
            "",
            "- Best current paper framing: adaptive SAGE as the headline candidate; VSR rescue as the verification/repair ablation layer.",
            "- Strongest positive evidence: mix130 paired gain vs VSR and Domain external gain vs HyDE/VSR; Domain vs Naive is only a nonsignificant numerical lead.",
            "- Remaining risk: Legal/Domain best-retrieval deltas are small and not significant; v56's external replay is mixed despite its positive aggregate shift.",
            "- Next experiment: use a reference-free guard that actually rejects risky repairs, then validate it on a new locked holdout rather than retuning on these replay slices.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = final_candidate_summary()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
