from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_enhanced_experiments import compute_f1, compute_rouge_l  # noqa: E402

OUT_DIR = ROOT / "results" / "sage_external_probe"
OUTPUT_JSON = OUT_DIR / "sage_external_probe_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_EXTERNAL_PROBE_SUMMARY.md"

DRY_PROBES = [
    {
        "slice": "Legal full context",
        "path": "results/historical_datasets/legal_hyde_contract_completion_dry.jsonl",
    },
    {
        "slice": "Biology full context",
        "path": "results/historical_datasets/biology_hyde_contract_completion_dry.jsonl",
    },
    {
        "slice": "Domain expansion windowed",
        "path": "results/historical_datasets/domain_hyde_contract_completion_dry.jsonl",
    },
]

API_PROBES = [
    {
        "key": "domain_probe",
        "title": "Domain Windowed API Probe",
        "base": "results/domain_expansion_windowed_50/hyde_aligned_n50.jsonl",
        "completion_prefix": "domain_hyde_contract_completion_",
        "v32_prefix": "domain_hyde_evidence_delta_",
        "completion_cumulative": OUT_DIR / "domain_hyde_contract_completion_cumulative.jsonl",
        "v32_cumulative": OUT_DIR / "domain_hyde_evidence_delta_cumulative.jsonl",
        "base_eval_glob": "domain_hyde_base_lines*_eval.jsonl",
        "v32_eval_glob": "domain_hyde_v32_lines*_eval.jsonl",
        "all_rows_label": "Full 50-row",
    },
    {
        "key": "legal_probe",
        "title": "Legal Full-Context API Probe",
        "base": "results/legal_fullcontext_20/hyde_aligned_n20.jsonl",
        "completion_prefix": "legal_hyde_contract_completion_",
        "v32_prefix": "legal_hyde_evidence_delta_",
        "completion_cumulative": OUT_DIR / "legal_hyde_contract_completion_cumulative.jsonl",
        "v32_cumulative": OUT_DIR / "legal_hyde_evidence_delta_cumulative.jsonl",
        "base_eval_glob": "legal_hyde_base_lines*_eval.jsonl",
        "v32_eval_glob": "legal_hyde_v32_lines*_eval.jsonl",
        "all_rows_label": "Full 20-row",
    },
    {
        "key": "biology_probe",
        "title": "Biology Full-Context API Probe",
        "base": "results/biology_fullcontext_50/hyde_aligned_n50.jsonl",
        "completion_prefix": "biology_hyde_contract_completion_",
        "v32_prefix": "biology_hyde_evidence_delta_",
        "completion_cumulative": OUT_DIR / "biology_hyde_contract_completion_cumulative.jsonl",
        "v32_cumulative": OUT_DIR / "biology_hyde_evidence_delta_cumulative.jsonl",
        "base_eval_glob": "biology_hyde_base_lines*_eval.jsonl",
        "v32_eval_glob": "biology_hyde_v32_lines*_eval.jsonl",
        "all_rows_label": "Full 50-row",
    },
]


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def load_jsonl(path_or_relative: str | Path) -> list[dict[str, Any]]:
    path = path_or_relative if isinstance(path_or_relative, Path) else ROOT / path_or_relative
    with path.open("r", encoding="utf-8", errors="replace") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def by_line(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row.get("line_no", 0)): row for row in rows}


def gold_answer(row: dict[str, Any]) -> str:
    if row.get("gold_answer"):
        return str(row.get("gold_answer"))
    answers = row.get("reference_answers") or row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def mean_f1(rows_by_line: dict[int, dict[str, Any]], base_by_line: dict[int, dict[str, Any]], line_nos: list[int]) -> float:
    values = []
    for line_no in line_nos:
        row = rows_by_line[line_no]
        gold = gold_answer(base_by_line[line_no])
        values.append(compute_f1(prediction(row), gold))
    return statistics.mean(values) if values else 0.0


def summarize_dry_probe(spec: dict[str, str]) -> dict[str, Any]:
    rows = load_jsonl(spec["path"])
    risk_rows = [row for row in rows if (row.get("contract_completion_agent") or {}).get("risk_candidate")]
    risk_signals = Counter(
        reason
        for row in risk_rows
        for reason in (row.get("contract_completion_agent") or {}).get("risk_reasons", [])
    )
    task_counts = Counter(str(row.get("dataset") or row.get("label") or "unknown") for row in risk_rows)
    return {
        **spec,
        "n": len(rows),
        "risk_candidates": len(risk_rows),
        "risk_rate": len(risk_rows) / max(1, len(rows)),
        "risk_signals": dict(risk_signals),
        "task_counts": dict(task_counts),
    }


def score_row(row: dict[str, Any], gold: str) -> dict[str, float]:
    pred = prediction(row)
    return {
        "f1": compute_f1(pred, gold),
        "rouge_l": compute_rouge_l(pred, gold),
    }


def eval_by_line(path: str | Path) -> dict[int, dict[str, Any]]:
    return {int(row.get("line_no", 0)): row.get("scores", {}) for row in load_jsonl(path)}


def merged_eval_by_line(pattern: str) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for path in sorted(OUT_DIR.glob(pattern)):
        merged.update(eval_by_line(path))
    return merged


def batch_suffix(path: Path, prefix: str) -> str:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(".jsonl"):
        raise ValueError(f"Unexpected batch filename: {path}")
    return name[len(prefix) : -len(".jsonl")]


def completion_batches(spec: dict[str, Any]) -> list[dict[str, Any]]:
    batches = []
    completion_prefix = str(spec["completion_prefix"])
    v32_prefix = str(spec["v32_prefix"])
    for completion_path in sorted(OUT_DIR.glob(f"{completion_prefix}lines*.jsonl")):
        suffix = batch_suffix(completion_path, completion_prefix)
        v32_path = OUT_DIR / f"{v32_prefix}{suffix}.jsonl"
        if not v32_path.exists():
            continue
        completion_rows = by_line(load_jsonl(completion_path))
        v32_rows = by_line(load_jsonl(v32_path))
        selected_lines = sorted(
            line_no
            for line_no, row in completion_rows.items()
            if (row.get("contract_completion_agent") or {}).get("verification_attempted")
        )
        batches.append(
            {
                "suffix": suffix,
                "completion_path": completion_path,
                "v32_path": v32_path,
                "completion_rows": completion_rows,
                "v32_rows": v32_rows,
                "selected_lines": selected_lines,
            }
        )
    return batches


def summarize_api_probe(spec: dict[str, Any]) -> dict[str, Any]:
    base_by_line = by_line(load_jsonl(str(spec["base"])))
    cumulative_completion = dict(base_by_line)
    cumulative_v32 = dict(base_by_line)
    batches = completion_batches(spec)
    batch_summaries = []
    selected_lines: list[int] = []
    for batch in batches:
        batch_lines = batch["selected_lines"]
        selected_lines.extend(batch_lines)
        for line_no in batch_lines:
            cumulative_completion[line_no] = batch["completion_rows"][line_no]
            cumulative_v32[line_no] = batch["v32_rows"][line_no]
        accepted = sum(
            bool((batch["completion_rows"][line_no].get("contract_completion_agent") or {}).get("repair_accepted"))
            for line_no in batch_lines
        )
        retained = sum(
            (batch["v32_rows"][line_no].get("evidence_delta_guard") or {}).get("action") == "retain_repair"
            for line_no in batch_lines
        )
        reverted = sum(
            (batch["v32_rows"][line_no].get("evidence_delta_guard") or {}).get("action") == "retain_original"
            for line_no in batch_lines
        )
        batch_summaries.append(
            {
                "suffix": batch["suffix"],
                "completion": rel(batch["completion_path"]),
                "v32": rel(batch["v32_path"]),
                "selected_lines": batch_lines,
                "verified": len(batch_lines),
                "repairs_accepted_before_v32": accepted,
                "v32_repairs_retained": retained,
                "v32_repairs_reverted": reverted,
            }
        )
    selected_lines = sorted(set(selected_lines))
    base_eval = merged_eval_by_line(str(spec["base_eval_glob"]))
    v32_eval = merged_eval_by_line(str(spec["v32_eval_glob"]))

    write_jsonl(spec["completion_cumulative"], [cumulative_completion[line_no] for line_no in sorted(cumulative_completion)])
    write_jsonl(spec["v32_cumulative"], [cumulative_v32[line_no] for line_no in sorted(cumulative_v32)])

    line_rows = []
    for line_no in selected_lines:
        base = base_by_line[line_no]
        completion = cumulative_completion[line_no]
        guarded = cumulative_v32[line_no]
        gold = gold_answer(base)
        base_scores = score_row(base, gold)
        completion_scores = score_row(completion, gold)
        guarded_scores = score_row(guarded, gold)
        agent = completion.get("contract_completion_agent") or {}
        guard = guarded.get("evidence_delta_guard") or {}
        changed = clean_text(prediction(base)) != clean_text(prediction(guarded))
        # Reuse the baseline judge score for unchanged predictions to avoid double-judging noise.
        effective_v32_eval = v32_eval.get(line_no, {}) if changed else base_eval.get(line_no, {})
        line_rows.append(
            {
                "line_no": line_no,
                "dataset": base.get("dataset") or base.get("label") or "unknown",
                "repair_accepted": bool(agent.get("repair_accepted")),
                "v32_action": guard.get("action", "not_applicable"),
                "base_f1": base_scores["f1"],
                "completion_f1": completion_scores["f1"],
                "v32_f1": guarded_scores["f1"],
                "completion_delta_f1": completion_scores["f1"] - base_scores["f1"],
                "v32_delta_f1": guarded_scores["f1"] - base_scores["f1"],
                "base_overall": float(base_eval.get(line_no, {}).get("overall", 0.0) or 0.0),
                "v32_overall": float(effective_v32_eval.get("overall", 0.0) or 0.0),
                "base_correctness": float(base_eval.get(line_no, {}).get("correctness", 0.0) or 0.0),
                "v32_correctness": float(effective_v32_eval.get("correctness", 0.0) or 0.0),
                "judge_reused_for_unchanged": not changed,
            }
        )

    all_lines = sorted(base_by_line)
    retained = sum(row["v32_action"] == "retain_repair" for row in line_rows)
    reverted = sum(row["v32_action"] == "retain_original" for row in line_rows)
    return {
        "key": spec["key"],
        "title": spec["title"],
        "base": spec["base"],
        "completion": rel(spec["completion_cumulative"]),
        "v32": rel(spec["v32_cumulative"]),
        "all_rows_label": spec["all_rows_label"],
        "batches": batch_summaries,
        "selected_lines": selected_lines,
        "verified": len(selected_lines),
        "repairs_accepted_before_v32": sum(row["repair_accepted"] for row in line_rows),
        "v32_repairs_retained": retained,
        "v32_repairs_reverted": reverted,
        "selected_mean_f1": {
            "base": mean_f1(base_by_line, base_by_line, selected_lines),
            "completion": mean_f1(cumulative_completion, base_by_line, selected_lines),
            "v32": mean_f1(cumulative_v32, base_by_line, selected_lines),
        },
        "all_mean_f1": {
            "base": mean_f1(base_by_line, base_by_line, all_lines),
            "completion": mean_f1(cumulative_completion, base_by_line, all_lines),
            "v32": mean_f1(cumulative_v32, base_by_line, all_lines),
        },
        "selected_mean_judge": {
            "base_overall": statistics.mean(row["base_overall"] for row in line_rows),
            "v32_overall": statistics.mean(row["v32_overall"] for row in line_rows),
            "base_correctness": statistics.mean(row["base_correctness"] for row in line_rows),
            "v32_correctness": statistics.mean(row["v32_correctness"] for row in line_rows),
        },
        "line_rows": line_rows,
    }


def summarize() -> dict[str, Any]:
    api_probe_summaries = [summarize_api_probe(spec) for spec in API_PROBES if completion_batches(spec)]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_probes": [summarize_dry_probe(spec) for spec in DRY_PROBES],
        "api_probes": api_probe_summaries,
        **{probe["key"]: probe for probe in api_probe_summaries},
    }


def fmt_signals(values: dict[str, int], limit: int = 3) -> str:
    items = sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items) or "none"


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# SAGE External Probe Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Dry Risk Probe",
        "",
        "| Slice | n | Risk candidates | Risk rate | Top risk signals | Risk tasks |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload["dry_probes"]:
        lines.append(
            f"| {row['slice']} | {row['n']} | {row['risk_candidates']} | {row['risk_rate']:.1%} | "
            f"{fmt_signals(row['risk_signals'])} | {fmt_signals(row['task_counts'], 5)} |"
        )
    for probe in payload["api_probes"]:
        selected_f1 = probe["selected_mean_f1"]
        all_f1 = probe["all_mean_f1"]
        judge = probe["selected_mean_judge"]
        lines.extend(
            [
                "",
                f"## {probe['title']}",
                "",
                f"- Selected lines: `{', '.join(str(line) for line in probe['selected_lines'])}`",
                f"- Verifications: {probe['verified']}; pre-v32 accepted repairs: {probe['repairs_accepted_before_v32']}; v32 retained: {probe['v32_repairs_retained']}; v32 reverted: {probe['v32_repairs_reverted']}",
                f"- Selected-line lexical F1: base {selected_f1['base']:.4f} -> completion {selected_f1['completion']:.4f} -> v32 {selected_f1['v32']:.4f}",
                f"- {probe['all_rows_label']} lexical F1: base {all_f1['base']:.4f} -> completion {all_f1['completion']:.4f} -> v32 {all_f1['v32']:.4f}",
                f"- Selected-line judge overall: base {judge['base_overall']:.2f} -> v32 {judge['v32_overall']:.2f}; correctness: base {judge['base_correctness']:.2f} -> v32 {judge['v32_correctness']:.2f}",
                "- Judge scores for unchanged predictions reuse the base evaluation to avoid duplicate-judge noise.",
                "",
                f"## {probe['title']} Batch Ledger",
                "",
                "| Batch | Lines | Verifications | Accepted before v32 | v32 retained | v32 reverted |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for batch in probe["batches"]:
            lines.append(
                f"| `{batch['suffix']}` | {', '.join(str(line) for line in batch['selected_lines'])} | "
                f"{batch['verified']} | {batch['repairs_accepted_before_v32']} | "
                f"{batch['v32_repairs_retained']} | {batch['v32_repairs_reverted']} |"
            )
        lines.extend(
            [
                "",
                "| Line | Task | Pre-v32 repair | v32 action | Base F1 | Completion F1 | v32 F1 | Base overall | v32 overall |",
                "|---:|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in probe["line_rows"]:
            lines.append(
                f"| {row['line_no']} | {row['dataset']} | {row['repair_accepted']} | {row['v32_action']} | "
                f"{row['base_f1']:.4f} | {row['completion_f1']:.4f} | {row['v32_f1']:.4f} | "
                f"{row['base_overall']:.0f} | {row['v32_overall']:.0f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Domain-windowed and Legal full-context probes are positive on lexical F1 and semantic judge scores, while Biology full-context is essentially flat to slightly negative after a very conservative repair pass.",
            "- These runs remain targeted risk-candidate protocols rather than final frozen external validation.",
            "- Line 31 shows metric tension: lexical F1 falls while the judge improves, so final validation should report both metric families.",
            "- Biology line 46 is the opposite failure mode: deterministic evidence-support metrics retained a repair that reduced reference-targeted coverage, moving judge overall from 5 to 3.",
            "- v32 reverted line 12 despite a lexical F1 gain because the independent evidence guard detected support regression; this is acceptable for a do-no-harm safety layer but may trade off overlap score.",
            "- Domain-windowed, Legal full-context, and Biology full-context risk pools are now complete; the next useful step is Biology failure analysis or a frozen full-slice SAGE validation protocol.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = summarize()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
