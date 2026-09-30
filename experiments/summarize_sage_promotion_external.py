from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_enhanced_experiments import compute_f1, compute_rouge_l


OUT_DIR = ROOT / "results" / "sage_promotion_external"
OUTPUT_JSON = OUT_DIR / "sage_promotion_external_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_EXTERNAL_PROMOTION_REPLAY_SUMMARY.md"

SLICES = [
    {
        "key": "legal",
        "label": "Legal full context",
        "n": 20,
        "baseline": "results/historical_datasets/legal_hyde_evidence_delta_v33_cumulative.jsonl",
        "candidate": "results/historical_datasets/legal_sage_v56_supported_facet_fusion_full20.jsonl",
        "baseline_eval": "results/historical_datasets/legal_sage_v33_fullslice_eval.jsonl",
        "candidate_eval": "results/historical_datasets/legal_sage_v56_routed_eval.jsonl",
    },
    {
        "key": "biology",
        "label": "Biology full context",
        "n": 50,
        "baseline": "results/historical_datasets/biology_hyde_evidence_delta_v33_cumulative.jsonl",
        "candidate": "results/historical_datasets/biology_sage_v56_supported_facet_fusion_full50.jsonl",
        "baseline_eval": "results/historical_datasets/biology_sage_v33_fullslice_eval.jsonl",
        "candidate_eval": "results/historical_datasets/biology_sage_v56_routed_eval.jsonl",
    },
    {
        "key": "domain",
        "label": "Domain expansion windowed",
        "n": 50,
        "baseline": "results/historical_datasets/domain_hyde_evidence_delta_v33_cumulative.jsonl",
        "candidate": "results/historical_datasets/domain_sage_v56_supported_facet_fusion_full50.jsonl",
        "baseline_eval": "results/historical_datasets/domain_sage_v33_fullslice_eval.jsonl",
        "candidate_eval": "results/historical_datasets/domain_sage_v56_routed_eval.jsonl",
    },
]


def load_jsonl(relative_path: str) -> list[dict[str, Any]]:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def references(row: dict[str, Any]) -> list[str]:
    values = row.get("reference_answers") or row.get("answers") or row.get("gold_answer") or [""]
    return [str(value) for value in values] if isinstance(values, list) else [str(values)]


def lexical_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    return {
        int(row["line_no"]): {
            "f1": max(compute_f1(prediction(row), reference) for reference in references(row)),
            "rouge_l": max(compute_rouge_l(prediction(row), reference) for reference in references(row)),
        }
        for row in rows
    }


def judge_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["line_no"]): row.get("scores") or {} for row in rows}


def action_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str((row.get(field) or {}).get("action", "not_applicable")) for row in rows)
    return dict(sorted(counts.items()))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def score_slice(spec: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = load_jsonl(spec["baseline"])
    candidate_rows = load_jsonl(spec["candidate"])
    baseline_by_line = {int(row["line_no"]): row for row in baseline_rows}
    candidate_by_line = {int(row["line_no"]): row for row in candidate_rows}
    if set(baseline_by_line) != set(candidate_by_line):
        raise ValueError(f"Row mismatch for {spec['key']}")

    baseline_lexical = lexical_rows(baseline_rows)
    candidate_lexical = lexical_rows(candidate_rows)
    baseline_judge = judge_rows(load_jsonl(spec["baseline_eval"]))
    candidate_judge = judge_rows(load_jsonl(spec["candidate_eval"]))
    if set(baseline_by_line) != set(baseline_judge) or set(candidate_by_line) != set(candidate_judge):
        raise ValueError(f"Judge row mismatch for {spec['key']}")

    lines = sorted(baseline_by_line)
    changed = [
        line
        for line in lines
        if " ".join(prediction(baseline_by_line[line]).split())
        != " ".join(prediction(candidate_by_line[line]).split())
    ]
    f1_deltas = [candidate_lexical[line]["f1"] - baseline_lexical[line]["f1"] for line in lines]
    rouge_deltas = [candidate_lexical[line]["rouge_l"] - baseline_lexical[line]["rouge_l"] for line in lines]
    judge_deltas = [
        float(candidate_judge[line].get("overall", 0)) - float(baseline_judge[line].get("overall", 0))
        for line in lines
    ]
    changed_judge_deltas = [
        float(candidate_judge[line].get("overall", 0)) - float(baseline_judge[line].get("overall", 0))
        for line in changed
    ]

    def wlt(values: list[float]) -> dict[str, int]:
        return {
            "wins": sum(value > 1e-12 for value in values),
            "losses": sum(value < -1e-12 for value in values),
            "ties": sum(abs(value) <= 1e-12 for value in values),
        }

    return {
        "key": spec["key"],
        "label": spec["label"],
        "n": len(lines),
        "changed_lines": changed,
        "actions": {
            "v54": action_counts(candidate_rows, "query_expanded_contract_gap_repair"),
            "v55": action_counts(candidate_rows, "supported_facet_guard"),
            "v56": action_counts(candidate_rows, "supported_facet_fusion"),
        },
        "lexical": {
            "v33_f1": mean([baseline_lexical[line]["f1"] for line in lines]),
            "v56_f1": mean([candidate_lexical[line]["f1"] for line in lines]),
            "f1_delta": mean(f1_deltas),
            "f1_wlt": wlt(f1_deltas),
            "v33_rouge_l": mean([baseline_lexical[line]["rouge_l"] for line in lines]),
            "v56_rouge_l": mean([candidate_lexical[line]["rouge_l"] for line in lines]),
            "rouge_l_delta": mean(rouge_deltas),
            "rouge_l_wlt": wlt(rouge_deltas),
        },
        "judge": {
            "v33_overall": mean([float(baseline_judge[line].get("overall", 0)) for line in lines]),
            "v56_overall": mean([float(candidate_judge[line].get("overall", 0)) for line in lines]),
            "overall_delta": mean(judge_deltas),
            "overall_wlt": wlt(judge_deltas),
            "changed_v33_overall": mean([float(baseline_judge[line].get("overall", 0)) for line in changed]),
            "changed_v56_overall": mean([float(candidate_judge[line].get("overall", 0)) for line in changed]),
            "changed_overall_delta": mean(changed_judge_deltas),
            "changed_overall_wlt": wlt(changed_judge_deltas),
            "regressed_changed_lines": [
                line
                for line in changed
                if float(candidate_judge[line].get("overall", 0)) < float(baseline_judge[line].get("overall", 0))
            ],
        },
        "paths": {key: spec[key] for key in ("baseline", "candidate", "baseline_eval", "candidate_eval")},
    }


def build_summary() -> dict[str, Any]:
    slices = [score_slice(spec) for spec in SLICES]
    total_n = sum(row["n"] for row in slices)

    def weighted(section: str, key: str) -> float:
        return sum(row["n"] * row[section][key] for row in slices) / total_n

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "baseline": "SAGE v33-preservation",
            "candidate": "SAGE v54->v55->v56 external full-slice replay",
            "runtime_is_reference_free": True,
            "judge_policy": "Only changed candidate predictions were newly judged; unchanged rows reuse the v33 judgment after exact prediction matching.",
            "promotion_decision": "retain_v33_as_freeze_candidate",
        },
        "slices": slices,
        "external_weighted": {
            "n": total_n,
            "v33_f1": weighted("lexical", "v33_f1"),
            "v56_f1": weighted("lexical", "v56_f1"),
            "f1_delta": weighted("lexical", "f1_delta"),
            "v33_rouge_l": weighted("lexical", "v33_rouge_l"),
            "v56_rouge_l": weighted("lexical", "v56_rouge_l"),
            "rouge_l_delta": weighted("lexical", "rouge_l_delta"),
            "v33_judge_overall": weighted("judge", "v33_overall"),
            "v56_judge_overall": weighted("judge", "v56_overall"),
            "judge_overall_delta": weighted("judge", "overall_delta"),
        },
        "decision_rationale": [
            "v56 improves the weighted external lexical and routed-judge aggregates, but the improvement is not uniform.",
            "Legal judge overall regresses despite a lexical gain, and every slice has changed rows with judge regressions.",
            "v55 did not revert any v54 repair and v56 did not trigger a fusion on these full slices, so the guard/fusion stages did not bound the observed risk.",
            "Keep v33-preservation as the freeze candidate; report v56 as an exploratory external replay rather than a replacement main method.",
        ],
    }


def write_markdown(summary: dict[str, Any]) -> None:
    weighted = summary["external_weighted"]
    lines = [
        "# SAGE External Promotion Replay Summary",
        "",
        f"> Generated: {summary['generated_at']}",
        "",
        "## Protocol",
        "",
        "- Baseline: SAGE v33-preservation; candidate: external full-slice v54 -> v55 -> v56 replay.",
        "- Runtime uses the question, current prediction, and source evidence only. Reference answers and judge scores are excluded from routing and editing.",
        "- Only changed predictions received new judge calls. Unchanged rows reuse the v33 score only after exact prediction matching.",
        "",
        "## Per-Slice Results",
        "",
        "| Slice | N | v54 accepted | v55 reverted | v56 fused | Changed | F1 v33 -> v56 | ROUGE-L v33 -> v56 | Judge overall v33 -> v56 | Changed judge W/L/T |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in summary["slices"]:
        actions = row["actions"]
        lexical = row["lexical"]
        judge = row["judge"]
        wlt = judge["changed_overall_wlt"]
        lines.append(
            f"| {row['label']} | {row['n']} | {actions['v54'].get('use_contract_gap_repair', 0)} | "
            f"{actions['v55'].get('revert_to_pre_repair_answer', 0)} | {actions['v56'].get('use_supported_facet_fusion', 0)} | "
            f"{len(row['changed_lines'])} | {lexical['v33_f1']:.4f} -> {lexical['v56_f1']:.4f} ({lexical['f1_delta']:+.4f}) | "
            f"{lexical['v33_rouge_l']:.4f} -> {lexical['v56_rouge_l']:.4f} ({lexical['rouge_l_delta']:+.4f}) | "
            f"{judge['v33_overall']:.3f} -> {judge['v56_overall']:.3f} ({judge['overall_delta']:+.3f}) | "
            f"{wlt['wins']}/{wlt['losses']}/{wlt['ties']} |"
        )

    lines.extend(
        [
            "",
            "## Weighted External Aggregate",
            "",
            "| N | F1 v33 -> v56 | ROUGE-L v33 -> v56 | Judge overall v33 -> v56 |",
            "|---:|---|---|---|",
            f"| {weighted['n']} | {weighted['v33_f1']:.4f} -> {weighted['v56_f1']:.4f} ({weighted['f1_delta']:+.4f}) | "
            f"{weighted['v33_rouge_l']:.4f} -> {weighted['v56_rouge_l']:.4f} ({weighted['rouge_l_delta']:+.4f}) | "
            f"{weighted['v33_judge_overall']:.4f} -> {weighted['v56_judge_overall']:.4f} ({weighted['judge_overall_delta']:+.4f}) |",
            "",
            "## Risk and Decision",
            "",
        ]
    )
    for rationale in summary["decision_rationale"]:
        lines.append(f"- {rationale}")
    lines.append("")
    for row in summary["slices"]:
        judge = row["judge"]
        lines.append(
            f"- {row['label']} changed-row judge regressions: {judge['regressed_changed_lines']}; "
            f"all final changed rows: {row['changed_lines']}."
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
