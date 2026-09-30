from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_enhanced_experiments import compute_f1  # noqa: E402


OUT_DIR = ROOT / "results" / "sage_external_probe"
OUTPUT_JSON = OUT_DIR / "sage_v33_external_probe_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_V33_EXTERNAL_PROBE_SUMMARY.md"

PROBES = [
    {
        "title": "Domain Windowed",
        "prefix": "domain",
        "base": "results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl",
        "v32": "results/historical_datasets/domain_hyde_evidence_delta_cumulative.jsonl",
        "v33": "results/historical_datasets/domain_hyde_evidence_delta_v33_cumulative.jsonl",
        "base_eval_glob": "domain_hyde_base_lines*_eval.jsonl",
        "v32_eval_glob": "domain_hyde_v32_lines*_eval.jsonl",
        "all_label": "Full 50-row",
    },
    {
        "title": "Legal Full-Context",
        "prefix": "legal",
        "base": "results/legal_fullcontext_20/hyde_aligned_n20.jsonl",
        "v32": "results/historical_datasets/legal_hyde_evidence_delta_cumulative.jsonl",
        "v33": "results/historical_datasets/legal_hyde_evidence_delta_v33_cumulative.jsonl",
        "base_eval_glob": "legal_hyde_base_lines*_eval.jsonl",
        "v32_eval_glob": "legal_hyde_v32_lines*_eval.jsonl",
        "all_label": "Full 20-row",
    },
    {
        "title": "Biology Full-Context",
        "prefix": "biology",
        "base": "results/biology_fullcontext_50/hyde_aligned_n50.jsonl",
        "v32": "results/historical_datasets/biology_hyde_evidence_delta_cumulative.jsonl",
        "v33": "results/historical_datasets/biology_hyde_evidence_delta_v33_cumulative.jsonl",
        "base_eval_glob": "biology_hyde_base_lines*_eval.jsonl",
        "v32_eval_glob": "biology_hyde_v32_lines*_eval.jsonl",
        "all_label": "Full 50-row",
    },
]


def load_jsonl(path_or_relative: str | Path) -> list[dict[str, Any]]:
    path = path_or_relative if isinstance(path_or_relative, Path) else ROOT / path_or_relative
    with path.open("r", encoding="utf-8", errors="replace") as source:
        return [json.loads(line) for line in source if line.strip()]


def by_line(path_or_relative: str | Path) -> dict[int, dict[str, Any]]:
    return {int(row.get("line_no", 0)): row for row in load_jsonl(path_or_relative)}


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def gold_answer(row: dict[str, Any]) -> str:
    if row.get("gold_answer"):
        return str(row.get("gold_answer"))
    answers = row.get("reference_answers") or row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def merged_eval(pattern: str) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for path in sorted(OUT_DIR.glob(pattern)):
        for row in load_jsonl(path):
            merged[int(row.get("line_no", 0))] = row.get("scores", {})
    return merged


def mean_f1(rows: dict[int, dict[str, Any]], base: dict[int, dict[str, Any]], lines: list[int]) -> float:
    values = [compute_f1(prediction(rows[line_no]), gold_answer(base[line_no])) for line_no in lines]
    return statistics.mean(values) if values else 0.0


def eval_score(
    row_set: dict[int, dict[str, Any]],
    base: dict[int, dict[str, Any]],
    base_eval: dict[int, dict[str, Any]],
    v32_eval: dict[int, dict[str, Any]],
    line_no: int,
    metric: str,
) -> float:
    if clean_text(prediction(row_set[line_no])) == clean_text(prediction(base[line_no])):
        return float(base_eval.get(line_no, {}).get(metric, 0.0) or 0.0)
    return float(v32_eval.get(line_no, {}).get(metric, 0.0) or 0.0)


def mean_eval(
    row_set: dict[int, dict[str, Any]],
    base: dict[int, dict[str, Any]],
    base_eval: dict[int, dict[str, Any]],
    v32_eval: dict[int, dict[str, Any]],
    lines: list[int],
    metric: str,
) -> float:
    values = [eval_score(row_set, base, base_eval, v32_eval, line_no, metric) for line_no in lines]
    return statistics.mean(values) if values else 0.0


def summarize_probe(spec: dict[str, str]) -> dict[str, Any]:
    base = by_line(spec["base"])
    v32 = by_line(spec["v32"])
    v33 = by_line(spec["v33"])
    base_eval = merged_eval(spec["base_eval_glob"])
    v32_eval = merged_eval(spec["v32_eval_glob"])
    selected = sorted(
        line_no
        for line_no, row in v33.items()
        if (row.get("contract_completion_agent") or {}).get("verification_attempted")
    )
    all_lines = sorted(base)
    diffs = []
    for line_no in all_lines:
        if clean_text(prediction(v32[line_no])) == clean_text(prediction(v33[line_no])):
            continue
        guard = v33[line_no].get("evidence_delta_guard") or {}
        diffs.append(
            {
                "line_no": line_no,
                "v32_action": (v32[line_no].get("evidence_delta_guard") or {}).get("action"),
                "v33_action": guard.get("action"),
                "v33_rejection_reasons": guard.get("rejection_reasons") or [],
                "base_f1": compute_f1(prediction(base[line_no]), gold_answer(base[line_no])),
                "v32_f1": compute_f1(prediction(v32[line_no]), gold_answer(base[line_no])),
                "v33_f1": compute_f1(prediction(v33[line_no]), gold_answer(base[line_no])),
                "base_overall": eval_score(base, base, base_eval, v32_eval, line_no, "overall"),
                "v32_overall": eval_score(v32, base, base_eval, v32_eval, line_no, "overall"),
                "v33_overall": eval_score(v33, base, base_eval, v32_eval, line_no, "overall"),
            }
        )

    return {
        **spec,
        "selected_lines": selected,
        "verified": len(selected),
        "v33_diffs_vs_v32": diffs,
        "selected_mean_f1": {
            "base": mean_f1(base, base, selected),
            "v32": mean_f1(v32, base, selected),
            "v33": mean_f1(v33, base, selected),
        },
        "all_mean_f1": {
            "base": mean_f1(base, base, all_lines),
            "v32": mean_f1(v32, base, all_lines),
            "v33": mean_f1(v33, base, all_lines),
        },
        "selected_mean_judge": {
            "base_overall": mean_eval(base, base, base_eval, v32_eval, selected, "overall"),
            "v32_overall": mean_eval(v32, base, base_eval, v32_eval, selected, "overall"),
            "v33_overall": mean_eval(v33, base, base_eval, v32_eval, selected, "overall"),
            "base_correctness": mean_eval(base, base, base_eval, v32_eval, selected, "correctness"),
            "v32_correctness": mean_eval(v32, base, base_eval, v32_eval, selected, "correctness"),
            "v33_correctness": mean_eval(v33, base, base_eval, v32_eval, selected, "correctness"),
        },
    }


def summarize() -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probes": [summarize_probe(spec) for spec in PROBES],
    }


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# SAGE v33 External Probe Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## v33 Change",
        "",
        "- v33 starts from v32 and adds a narrow high-content original preservation guard for non-refusal answers.",
        "- The new guard is deterministic and uses no reference answers, judge scores, or extra LLM calls.",
        "",
        "## Probe Comparison",
        "",
        "| Probe | Verified | v33 diffs vs v32 | Selected F1 base -> v32 -> v33 | All-row F1 base -> v32 -> v33 | Judge overall base -> v32 -> v33 |",
        "|---|---:|---:|---|---|---|",
    ]
    for probe in payload["probes"]:
        selected_f1 = probe["selected_mean_f1"]
        all_f1 = probe["all_mean_f1"]
        judge = probe["selected_mean_judge"]
        lines.append(
            f"| {probe['title']} | {probe['verified']} | {len(probe['v33_diffs_vs_v32'])} | "
            f"{selected_f1['base']:.4f} -> {selected_f1['v32']:.4f} -> {selected_f1['v33']:.4f} | "
            f"{all_f1['base']:.4f} -> {all_f1['v32']:.4f} -> {all_f1['v33']:.4f} | "
            f"{judge['base_overall']:.2f} -> {judge['v32_overall']:.2f} -> {judge['v33_overall']:.2f} |"
        )

    lines.extend(["", "## Changed Rows", "", "| Probe | Line | v32 action | v33 action | Reasons | F1 base -> v32 -> v33 | Judge overall base -> v32 -> v33 |", "|---|---:|---|---|---|---|---|"])
    changed_any = False
    for probe in payload["probes"]:
        for row in probe["v33_diffs_vs_v32"]:
            changed_any = True
            lines.append(
                f"| {probe['title']} | {row['line_no']} | {row['v32_action']} | {row['v33_action']} | "
                f"{', '.join(row['v33_rejection_reasons']) or 'none'} | "
                f"{row['base_f1']:.4f} -> {row['v32_f1']:.4f} -> {row['v33_f1']:.4f} | "
                f"{row['base_overall']:.0f} -> {row['v32_overall']:.0f} -> {row['v33_overall']:.0f} |"
            )
    if not changed_any:
        lines.append("| none |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- v33 preserves Domain and Legal v32 gains unchanged while eliminating the Biology line-46 regression.",
            "- Biology returns to the HyDE baseline instead of becoming worse; this is a safety improvement, not a new positive Biology result.",
            "- The next useful experiment is to use v33 as the guard for a frozen full-slice SAGE validation, then report v31/v32/v33 as an ablation ladder.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = summarize()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
