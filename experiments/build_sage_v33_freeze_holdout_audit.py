from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "sage_final_candidate"
OUT_JSON = OUT_DIR / "sage_v33_freeze_holdout_audit.json"
OUT_MD = OUT_DIR / "SAGE_V33_FREEZE_HOLDOUT_AUDIT.md"


def read_json(relative: str) -> Any:
    path = ROOT / relative
    if not path.exists():
        return {} if path.suffix.lower() == ".json" else []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def fmt_delta(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    prefix = "+" if value > 0 else ""
    return f"{prefix}{value:.{digits}f}"


def fmt_p(value: Any) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if value == 0:
        return "0"
    if value < 0.001:
        return f"{value:.3g}"
    return f"{value:.4f}"


def find_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for row in rows:
        if row.get("key") == key:
            return row
    return {}


def v77_summary(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    return {
        "slices": [
            {
                "tag": row.get("tag"),
                "short": row.get("short"),
                "total_questions": row.get("total_questions", 0),
                "fired_count": row.get("fired_count", 0),
                "no_family_count": row.get("no_family_count", 0),
                "mandatory_filled_count": row.get("mandatory_filled_count", 0),
                "entity_agnostic_count": row.get("entity_agnostic_count", 0),
            }
            for row in rows
        ],
        "total_questions": sum(row.get("total_questions", 0) for row in rows),
        "fired_count": sum(row.get("fired_count", 0) for row in rows),
        "no_family_count": sum(row.get("no_family_count", 0) for row in rows),
        "mandatory_filled_count": sum(row.get("mandatory_filled_count", 0) for row in rows),
        "entity_agnostic_count": sum(row.get("entity_agnostic_count", 0) for row in rows),
    }


def build_audit() -> dict[str, Any]:
    final = read_json("results/historical_sage/sage_final_candidate_summary.json")
    external_promotion = read_json("results/historical_sage/sage_promotion_external_summary.json")
    v77 = v77_summary(read_json("results/historical_vsr/v77_cross_question_cross_domain_heldout_validation.json"))

    locked = final.get("locked_fullslice", {})
    external_judges = locked.get("external_weighted_judges", [])
    mix_promotion = final.get("mix130_promotion_replay", {})
    unseen = final.get("unseen_validation", {})

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decision": {
            "status": "freeze_v33_for_paper_planning",
            "method": "SAGE v33-preservation",
            "components": [
                "v31 learned contract-gap routing",
                "v32 evidence-delta guard",
                "v33 high-content preservation guard",
            ],
            "runtime_reference_free": True,
            "runtime_inputs": [
                "question",
                "current prediction",
                "source evidence / retrieved context",
                "frozen router and deterministic guard metadata",
            ],
            "excluded_runtime_inputs": [
                "reference answers",
                "gold labels",
                "judge scores",
                "post-hoc metric deltas",
            ],
            "rationale": [
                "v33 is deterministic and object-exact reproducible under the frozen v31 OOF replay protocol.",
                "v33 preserves the strong Mix130 v31/v32 score while reverting the observed Biology v32 regression.",
                "v56 has positive aggregate signals but still exposes slice-level and changed-row regressions.",
                "No further tuning should be done on Mix130 or the completed external full-slice replay surfaces.",
            ],
        },
        "evidence": {
            "mix130": final.get("mix130", {}),
            "external_slices": final.get("external_slices", []),
            "locked_fullslice": locked,
            "reproduction": final.get("reproduction", {}),
            "unseen_validation": unseen,
            "mix130_promotion_replay": mix_promotion,
            "external_promotion_replay": final.get("external_promotion_replay") or external_promotion,
            "v77_mechanism_audit": v77,
            "weighted_judges": {
                "sage_v33": find_by_key(external_judges, "sage_v33"),
                "best_retrieval": find_by_key(external_judges, "best_retrieval"),
                "vsr_rescue": find_by_key(external_judges, "vsr_rescue"),
            },
        },
        "holdout_usability": [
            {
                "surface": "UltraDomain Mix130",
                "n": 130,
                "status": "development / main reported slice",
                "use": "Valid for frozen reporting and paired tables; not a fresh validation pool after repeated candidate selection.",
            },
            {
                "surface": "Legal20, Biology50, Domain50 external full slices",
                "n": 120,
                "status": "completed full-slice scoring and v56 replay",
                "use": "Valid as external evidence; do not tune further on these rows without declaring them development slices.",
            },
            {
                "surface": "Mix unseen34 context-heldout risk pool",
                "n": 34,
                "status": "guard-specific heldout validation",
                "use": "Supportive safety evidence only; small number of guard-active rows limits benchmark strength.",
            },
            {
                "surface": "V46 zero-overlap validation30",
                "n": 30,
                "status": "supplementary unseen-style validation already used by v55/v56 exploration",
                "use": "Report as supplementary; no longer clean for another promotion decision.",
            },
            {
                "surface": "V40/V42 heldout12 ladders",
                "n": 24,
                "status": "late-stage exploratory promotion probes",
                "use": "Useful for mechanism discussion, too small for headline claims.",
            },
            {
                "surface": "v77 cross-question/cross-domain heldout audit",
                "n": v77.get("total_questions"),
                "status": "mechanism coverage audit",
                "use": "Use as renewed-innovation/mechanism validation, not as SAGE answer-quality holdout evidence.",
            },
            {
                "surface": "New locked unseen holdout",
                "n": "TBD",
                "status": "not yet run",
                "use": "Required only if v56 or a new safety guard is promoted, or if the paper needs a cleaner final unseen claim.",
            },
        ],
        "next_actions": [
            "Write the paper method section around frozen SAGE v33-preservation.",
            "Keep v56 in ablation/discussion as an exploratory branch until a better reference-free safety guard exists.",
            "Spend API next only on a new locked unseen holdout or deliberate official GraphRAG/LightRAG reproduction.",
            "Do not use references, judge scores, or gold labels in runtime routing, guard, repair, or fusion decisions.",
        ],
    }


def write_markdown(audit: dict[str, Any]) -> str:
    decision = audit["decision"]
    evidence = audit["evidence"]
    mix = evidence.get("mix130", {})
    locked = evidence.get("locked_fullslice", {})
    unseen = evidence.get("unseen_validation", {})
    mix_promo = evidence.get("mix130_promotion_replay", {})
    external_promo = evidence.get("external_promotion_replay", {})
    weighted_judges = evidence.get("weighted_judges", {})
    sage_judge = weighted_judges.get("sage_v33", {})
    best_judge = weighted_judges.get("best_retrieval", {})
    vsr_judge = weighted_judges.get("vsr_rescue", {})
    v77 = evidence.get("v77_mechanism_audit", {})

    v56_scores = (mix_promo.get("scores") or {}).get("v56", {})
    v33_scores = (mix_promo.get("scores") or {}).get("v33", {})
    v56_vs_v33 = mix_promo.get("v56_vs_v33_full_f1", {})
    external_weighted = external_promo.get("external_weighted", {})
    external_slices = external_promo.get("slices", [])

    lines = [
        "# SAGE v33 Freeze and Holdout Audit",
        "",
        f"> Generated: {audit['generated_at']}",
        "> Scope: offline consolidation from completed result artifacts; no API calls.",
        "",
        "## Freeze Decision",
        "",
        f"- Decision: `{decision['status']}`.",
        f"- Frozen paper-planning method: `{decision['method']}`.",
        "- Method stack: " + " -> ".join(f"`{item}`" for item in decision["components"]) + ".",
        "- Runtime policy is reference-free: uses question, current prediction, source evidence, and frozen routing/guard metadata only.",
        "- Runtime policy excludes reference answers, gold labels, judge scores, and post-hoc metric deltas.",
        "",
        "## Evidence Snapshot",
        "",
        "| Evidence | N | Primary result | Interpretation |",
        "|---|---:|---|---|",
        (
            "| Mix130 paired candidate | "
            f"{mix.get('n', 'N/A')} | F1={fmt_num(mix.get('f1'))}, ROUGE-L={fmt_num(mix.get('rouge_l'))}, "
            f"vs VSR rescue F1 {fmt_delta(mix.get('sage_minus_vsr_f1'))}, W/L={mix.get('sage_vs_vsr_wins', 'N/A')}/{mix.get('sage_vs_vsr_losses', 'N/A')} | "
            "Strong large-N candidate evidence. |"
        ),
        (
            "| External weighted lexical | "
            f"{(locked.get('external_weighted_v33') or {}).get('n', 'N/A')} | "
            f"v33 F1={fmt_num((locked.get('external_weighted_v33') or {}).get('weighted_f1'))}; "
            f"top v31 F1={fmt_num((locked.get('external_weighted_top') or {}).get('weighted_f1'))} | "
            "v33 is chosen for safety, not because it maximizes every lexical aggregate. |"
        ),
        (
            "| External weighted judge | "
            f"{sage_judge.get('n', 'N/A')} | "
            f"v33 overall={fmt_num(sage_judge.get('overall'), 3)} vs best retrieval={fmt_num(best_judge.get('overall'), 3)} "
            f"and VSR rescue={fmt_num(vsr_judge.get('overall'), 3)} | "
            "Strong semantic-quality evidence under same rubric. |"
        ),
        (
            "| Unseen34 guard validation | "
            f"{unseen.get('n', 'N/A')} | "
            f"F1 {fmt_num(unseen.get('f1_baseline'))}->{fmt_num(unseen.get('f1_v33'))} "
            f"({fmt_delta(unseen.get('f1_delta'))}); judge {fmt_num(unseen.get('judge_overall_baseline'), 3)}->{fmt_num(unseen.get('judge_overall_v33'), 3)} "
            f"({fmt_delta(unseen.get('judge_overall_delta'), 3)}) | "
            "Supportive guard-safety evidence, not a large unseen benchmark. |"
        ),
    ]
    for item in unseen.get("additional_slices", []):
        lines.append(
            "| "
            f"{item.get('name', 'Additional unseen slice')} | {item.get('n', 'N/A')} | "
            f"F1 {fmt_num(item.get('f1_baseline'))}->{fmt_num(item.get('f1_v33'))} ({fmt_delta(item.get('f1_delta'))}); "
            f"judge {fmt_num(item.get('judge_overall_baseline'), 3)}->{fmt_num(item.get('judge_overall_v33'), 3)} ({fmt_delta(item.get('judge_overall_delta'), 3)}) | "
            "Supplementary unseen-style evidence. |"
        )
    lines.extend(
        [
            (
                "| v56 Mix130 promotion replay | "
                f"{v56_scores.get('n', 'N/A')} | "
                f"F1 {fmt_num(v33_scores.get('f1'))}->{fmt_num(v56_scores.get('f1'))} "
                f"({fmt_delta(v56_vs_v33.get('delta'))}); routed judge {fmt_num(v33_scores.get('judge_overall'), 3)}->{fmt_num(v56_scores.get('judge_overall'), 3)} | "
                "Positive but not promotable: changed-row regressions remain. |"
            ),
            (
                "| v56 external promotion replay | "
                f"{external_weighted.get('n', 'N/A')} | "
                f"F1 {fmt_num(external_weighted.get('v33_f1'))}->{fmt_num(external_weighted.get('v56_f1'))} "
                f"({fmt_delta(external_weighted.get('f1_delta'))}); judge {fmt_num(external_weighted.get('v33_judge_overall'), 3)}->{fmt_num(external_weighted.get('v56_judge_overall'), 3)} "
                f"({fmt_delta(external_weighted.get('judge_overall_delta'), 3)}) | "
                "Aggregate gain is mixed by slice; keep exploratory. |"
            ),
            "",
            "## Why v56 Is Not Promoted",
            "",
            "| Slice | Changed rows | Judge movement | Regressed changed rows | Decision impact |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in external_slices:
        judge = row.get("judge", {})
        lines.append(
            f"| {row.get('label')} | {len(row.get('changed_lines', []))} | "
            f"overall {fmt_num(judge.get('v33_overall'), 3)}->{fmt_num(judge.get('v56_overall'), 3)} "
            f"({fmt_delta(judge.get('overall_delta'), 3)}) | "
            f"{row.get('judge', {}).get('regressed_changed_lines', [])} | "
            "Shows remaining risk. |"
        )
    lines.extend(
        [
            "",
            "- v56 improves some aggregate lexical/judge metrics, but Legal judge overall declines and every external slice has changed-row judge regressions.",
            "- v55 reverts no external repair and v56 triggers no external fusion on these full slices, so the new stages do not yet bound repair risk.",
            "- v33 remains the cleaner paper-planning freeze because it is deterministic, reproducible, and has a simpler safety story.",
            "",
            "## Holdout Usability Audit",
            "",
            "| Surface | N | Status | Paper use |",
            "|---|---:|---|---|",
        ]
    )
    for row in audit["holdout_usability"]:
        lines.append(f"| {row['surface']} | {row.get('n', 'N/A')} | {row['status']} | {row['use']} |")
    lines.extend(
        [
            "",
            "## Mechanism Audit Note",
            "",
            (
                "- v77 cross-question/cross-domain audit totals: "
                f"{v77.get('total_questions', 'N/A')} questions, compiler fired {v77.get('fired_count', 'N/A')}, "
                f"no-family count {v77.get('no_family_count', 'N/A')}, mandatory slots filled {v77.get('mandatory_filled_count', 'N/A')}, "
                f"entity-agnostic traces {v77.get('entity_agnostic_count', 'N/A')}."
            ),
            "- This is useful renewed-innovation/mechanism validation, but it is not a new SAGE answer-quality holdout result.",
            "",
            "## Locked Claim Boundary",
            "",
            "- Claim as frozen: SAGE v33-preservation is the lead paper-planning candidate and the safest current adaptive-routing method.",
            "- Claim cautiously: external lexical gains over best retrieval are small or neutral, while same-rubric judge quality is stronger.",
            "- Do not claim: official GraphRAG/LightRAG full reproduction or broad F1 dominance over all strong retrieval baselines.",
            "- Do not tune further on: Mix130, Legal20, Biology50, Domain50, unseen34, v46, v40, or v42 unless they are relabeled as development probes.",
            "",
            "## Next Actions",
            "",
        ]
    )
    for idx, item in enumerate(audit["next_actions"], start=1):
        lines.append(f"{idx}. {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(write_markdown(audit), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
