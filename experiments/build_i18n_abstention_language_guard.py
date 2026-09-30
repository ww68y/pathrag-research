#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit an exploratory i18n abstention/language guard on cost samples.

The guard decision is intentionally reference-free: it only inspects the
question language, answer language, and abstention-like text. Gold answers and
metric deltas are used only after the decision for offline simulation/reporting.
No API calls are made, and original prediction artifacts are not modified.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "cost_latency"
OUTPUT_JSON = OUT_DIR / "i18n_abstention_language_guard_audit.json"
OUTPUT_MD = OUT_DIR / "I18N_ABSTENTION_LANGUAGE_GUARD_AUDIT.md"

SAMPLES = [
    ("Legal full-context cost sample", "legal_fullcontext_20"),
    ("Biology full-context cost sample", "biology_fullcontext_50"),
    ("Domain expansion windowed cost sample", "domain_expansion_windowed_50"),
]
METHOD_FILES = {
    "w/o_repair": "w_o_repair_aligned_n*.json",
    "full_v2": "full_v2_aligned_n*.json",
    "full_v2_rescue": "full_v2_rescue_aligned_n*.json",
}
ALIGNED_SURFACES = [
    ("UltraDomain mix130 full_v2_rescue", "results/enhanced_ablation_aligned/full_v2_rescue_aligned_n130.json"),
    (
        "Legal full-context20 full_v2_rescue",
        "results/enhanced_ablation_aligned/legal_fullcontext_20/full_v2_rescue_aligned_n20.json",
    ),
    (
        "Biology full-context50 full_v2_rescue",
        "results/enhanced_ablation_aligned/biology_fullcontext_50/full_v2_rescue_aligned_n50.json",
    ),
    (
        "Domain windowed50 full_v2_rescue",
        "results/enhanced_ablation_aligned/domain_expansion_windowed_50/full_v2_rescue_aligned_n50.json",
    ),
]

EN_ABSTENTION_PATTERNS = [
    r"\b(?:cannot|can't|can not|could not) be (?:determined|found|answered|identified|ascertained)\b",
    r"\bunable to (?:determine|find|answer|identify|ascertain)\b",
    r"\binsufficient (?:evidence|information|context|data)\b",
    r"\bnot enough (?:evidence|information|context|data)\b",
    r"\bno (?:relevant )?(?:evidence|information|context|data|details?|clauses?)\b",
    r"\b(?:does not|doesn't) (?:provide|specify|state|mention|contain|include)\b",
    r"\bnot (?:provided|specified|stated|mentioned|included|available|present)\b",
]

ZH_ABSTENTION_PATTERNS = [
    "\u672a\u627e\u5230",  # not found
    "\u6ca1\u6709\u627e\u5230",
    "\u65e0\u6cd5\u786e\u5b9a",
    "\u4e0d\u80fd\u786e\u5b9a",
    "\u65e0\u6cd5\u5224\u65ad",
    "\u4e0d\u80fd\u5224\u65ad",
    "\u65e0\u6cd5\u5f97\u51fa",
    "\u4e0d\u8db3\u4ee5(?:\u786e\u5b9a|\u5224\u65ad|\u56de\u7b54|\u652f\u6301)",
    "(?:\u6ca1\u6709|\u672a).{0,16}(?:\u8bc1\u636e|\u4fe1\u606f|\u4e0a\u4e0b\u6587|\u6761\u6b3e|\u6750\u6599)",
    "\u7f3a\u4e4f.{0,16}(?:\u8bc1\u636e|\u4fe1\u606f|\u4e0a\u4e0b\u6587|\u6761\u6b3e|\u6750\u6599)",
]

COMPILED_ABSTENTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in [*EN_ABSTENTION_PATTERNS, *ZH_ABSTENTION_PATTERNS]
]
CJK_RE = re.compile(r"[\u3400-\u9fff]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def answer(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = (row.get("metrics") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def bool_metric(row: dict[str, Any] | None, key: str) -> bool | None:
    if not row:
        return None
    metrics = row.get("metrics") or {}
    if key not in metrics:
        return None
    return bool(metrics[key])


def compact(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().replace("|", "/")
    return cleaned[:limit]


def detect_i18n_abstention(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    return any(pattern.search(normalized) for pattern in COMPILED_ABSTENTION_PATTERNS)


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text or ""))


def latin_word_count(text: str) -> int:
    return len(LATIN_WORD_RE.findall(text or ""))


def detect_language(text: str) -> str:
    text = str(text or "")
    cjk = cjk_count(text)
    latin = latin_word_count(text)
    alphaish = cjk + sum(len(word) for word in LATIN_WORD_RE.findall(text))
    cjk_ratio = cjk / alphaish if alphaish else 0.0
    if cjk >= 4 and (cjk_ratio >= 0.15 or latin <= 2):
        return "zh"
    if latin >= 3:
        return "en"
    if cjk > 0:
        return "zh"
    return "unknown"


def language_mismatch(question: str, candidate_answer: str) -> dict[str, Any]:
    expected = detect_language(question)
    observed = detect_language(candidate_answer)
    mismatch = False
    reason = ""
    if expected == "en" and observed == "zh":
        mismatch = True
        reason = "english_question_chinese_answer"
    elif expected == "zh" and observed == "en":
        mismatch = True
        reason = "chinese_question_english_answer"
    return {"expected": expected, "observed": observed, "mismatch": mismatch, "reason": reason}


def sample_dirs(prefix: str) -> list[Path]:
    base = ROOT / "results" / "cost_instrumented"
    if not base.exists():
        return []
    return sorted(
        directory
        for directory in base.iterdir()
        if directory.is_dir()
        and directory.name.startswith(prefix)
        and any(directory.glob(METHOD_FILES["full_v2_rescue"]))
    )


def row_key(row: dict[str, Any]) -> str:
    line_no = row.get("dataset_line_no")
    if isinstance(line_no, int):
        return f"line:{line_no}"
    source_line_no = row.get("source_line_no")
    if isinstance(source_line_no, int):
        return f"line:{source_line_no}"
    return str(row.get("question_id") or row.get("question") or "unknown")


def load_sample_rows(prefix: str) -> dict[str, dict[str, dict[str, Any]]]:
    aligned: dict[str, dict[str, dict[str, Any]]] = {}
    for directory in sample_dirs(prefix):
        for method, pattern in METHOD_FILES.items():
            for path in sorted(directory.glob(pattern)):
                data = read_json(path)
                for row in data.get("results") or []:
                    aligned.setdefault(row_key(row), {})[method] = row
    return aligned


def choose_fallback(rows: dict[str, dict[str, Any]], question: str) -> tuple[str | None, dict[str, Any] | None, str]:
    for method in ("w/o_repair", "full_v2"):
        candidate = rows.get(method)
        if not candidate or not answer(candidate).strip():
            continue
        lang = language_mismatch(question, answer(candidate))
        if not lang["mismatch"]:
            return method, candidate, "first_prior_answer_matching_question_language"
    return None, None, "no_prior_answer_matching_question_language"


def build_case(sample: str, key: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rescue = rows["full_v2_rescue"]
    question = str(rescue.get("question") or "")
    rescue_answer = answer(rescue)
    recorded_abstention = bool_metric(rescue, "abstention_language")
    i18n_abstention = detect_i18n_abstention(rescue_answer)
    missed_abstention = bool(i18n_abstention and not recorded_abstention)
    lang = language_mismatch(question, rescue_answer)
    flags = []
    if lang["mismatch"]:
        flags.append(lang["reason"])
    if missed_abstention:
        flags.append("missed_i18n_abstention")

    guard_triggered = bool(flags)
    auto_fallback_triggered = bool(lang["mismatch"])
    fallback_method, fallback_row, fallback_reason = (None, None, "not_triggered")
    if auto_fallback_triggered:
        fallback_method, fallback_row, fallback_reason = choose_fallback(rows, question)

    simulated_row = fallback_row or rescue
    if fallback_row:
        action = f"fallback_to_{fallback_method}"
    elif guard_triggered:
        action = "flag_only"
    else:
        action = "keep_full_v2_rescue"
    original_f1 = metric(rescue, "f1")
    simulated_f1 = metric(simulated_row, "f1")
    original_rouge = metric(rescue, "rouge_l")
    simulated_rouge = metric(simulated_row, "rouge_l")
    original_faith = metric(rescue, "faithfulness")
    simulated_faith = metric(simulated_row, "faithfulness")
    return {
        "sample": sample,
        "key": key,
        "dataset_line_no": rescue.get("dataset_line_no"),
        "question_id": rescue.get("question_id"),
        "question": question,
        "expected_language": lang["expected"],
        "answer_language": lang["observed"],
        "language_mismatch": lang["mismatch"],
        "language_mismatch_reason": lang["reason"],
        "recorded_abstention_language": recorded_abstention,
        "i18n_abstention_language": i18n_abstention,
        "missed_i18n_abstention": missed_abstention,
        "guard_flags": flags,
        "guard_triggered": guard_triggered,
        "auto_fallback_triggered": auto_fallback_triggered,
        "action": action,
        "fallback_method": fallback_method,
        "fallback_reason": fallback_reason,
        "original_f1": original_f1,
        "simulated_f1": simulated_f1,
        "delta_f1": simulated_f1 - original_f1
        if original_f1 is not None and simulated_f1 is not None
        else None,
        "original_rouge_l": original_rouge,
        "simulated_rouge_l": simulated_rouge,
        "delta_rouge_l": simulated_rouge - original_rouge
        if original_rouge is not None and simulated_rouge is not None
        else None,
        "original_faithfulness": original_faith,
        "simulated_faithfulness": simulated_faith,
        "delta_faithfulness": simulated_faith - original_faith
        if original_faith is not None and simulated_faith is not None
        else None,
        "original_prediction_preview": compact(rescue_answer),
        "fallback_prediction_preview": compact(answer(fallback_row)) if fallback_row else "",
    }


def mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if isinstance(value, (int, float))]
    return statistics.mean(valid) if valid else None


def count_flag(cases: list[dict[str, Any]], flag: str) -> int:
    return sum(1 for case in cases if flag in case.get("guard_flags", []))


def aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(cases),
        "guard_triggered": sum(1 for case in cases if case["guard_triggered"]),
        "auto_fallback_triggered": sum(1 for case in cases if case["auto_fallback_triggered"]),
        "fallback_applied": sum(1 for case in cases if case["fallback_method"]),
        "language_mismatch": count_flag(cases, "english_question_chinese_answer")
        + count_flag(cases, "chinese_question_english_answer"),
        "missed_i18n_abstention": count_flag(cases, "missed_i18n_abstention"),
        "original_mean_f1": mean([case.get("original_f1") for case in cases]),
        "simulated_mean_f1": mean([case.get("simulated_f1") for case in cases]),
        "mean_delta_f1": mean([case.get("delta_f1") for case in cases]),
        "original_mean_rouge_l": mean([case.get("original_rouge_l") for case in cases]),
        "simulated_mean_rouge_l": mean([case.get("simulated_rouge_l") for case in cases]),
        "mean_delta_rouge_l": mean([case.get("delta_rouge_l") for case in cases]),
        "original_mean_faithfulness": mean([case.get("original_faithfulness") for case in cases]),
        "simulated_mean_faithfulness": mean([case.get("simulated_faithfulness") for case in cases]),
        "mean_delta_faithfulness": mean([case.get("delta_faithfulness") for case in cases]),
    }


def scan_aligned_surface(name: str, relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.exists():
        return {"surface": name, "path": relative_path, "exists": False, "n": 0, "examples": []}
    rows = read_json(path).get("results") or []
    scanned = []
    for row in rows:
        question = str(row.get("question") or "")
        prediction = answer(row)
        lang = language_mismatch(question, prediction)
        recorded_abstention = bool_metric(row, "abstention_language")
        i18n_abstention = detect_i18n_abstention(prediction)
        flags = []
        if lang["mismatch"]:
            flags.append(lang["reason"])
        if i18n_abstention and not recorded_abstention:
            flags.append("missed_i18n_abstention")
        scanned.append(
            {
                "question_id": row.get("question_id"),
                "dataset_line_no": row.get("dataset_line_no"),
                "question": question,
                "expected_language": lang["expected"],
                "answer_language": lang["observed"],
                "language_mismatch": lang["mismatch"],
                "recorded_abstention_language": recorded_abstention,
                "i18n_abstention_language": i18n_abstention,
                "missed_i18n_abstention": bool(i18n_abstention and not recorded_abstention),
                "flags": flags,
                "prediction_preview": compact(prediction),
            }
        )
    flagged = [row for row in scanned if row["flags"]]
    return {
        "surface": name,
        "path": relative_path,
        "exists": True,
        "n": len(scanned),
        "flagged": len(flagged),
        "language_mismatch": sum(1 for row in scanned if row["language_mismatch"]),
        "recorded_abstention_language": sum(1 for row in scanned if row["recorded_abstention_language"]),
        "i18n_abstention_language": sum(1 for row in scanned if row["i18n_abstention_language"]),
        "missed_i18n_abstention": sum(1 for row in scanned if row["missed_i18n_abstention"]),
        "examples": flagged[:8],
    }


def aggregate_surface_scans(scans: list[dict[str, Any]]) -> dict[str, Any]:
    existing = [scan for scan in scans if scan.get("exists")]
    return {
        "surfaces": len(existing),
        "n": sum(scan.get("n", 0) for scan in existing),
        "flagged": sum(scan.get("flagged", 0) for scan in existing),
        "language_mismatch": sum(scan.get("language_mismatch", 0) for scan in existing),
        "recorded_abstention_language": sum(scan.get("recorded_abstention_language", 0) for scan in existing),
        "i18n_abstention_language": sum(scan.get("i18n_abstention_language", 0) for scan in existing),
        "missed_i18n_abstention": sum(scan.get("missed_i18n_abstention", 0) for scan in existing),
    }


def sample_summaries(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for sample in sorted({case["sample"] for case in cases}):
        sample_cases = [case for case in cases if case["sample"] == sample]
        item = aggregate_cases(sample_cases)
        item["sample"] = sample
        summaries.append(item)
    return summaries


def build_payload() -> dict[str, Any]:
    cases = []
    for label, prefix in SAMPLES:
        for key, rows in load_sample_rows(prefix).items():
            if {"w/o_repair", "full_v2", "full_v2_rescue"}.issubset(rows):
                cases.append(build_case(label, key, rows))
    cases = sorted(cases, key=lambda case: (case["sample"], case.get("dataset_line_no") or -1, case["key"]))
    flagged = [case for case in cases if case["guard_triggered"]]
    surface_scans = [scan_aligned_surface(name, path) for name, path in ALIGNED_SURFACES]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "offline exploratory guard simulation over cost-instrumented rows; no API calls",
        "guard_version": "i18n_abstention_language_guard_v0_exploratory",
        "decision_policy": {
            "uses_reference_answers_for_decision": False,
            "uses_metric_deltas_for_decision": False,
            "flag_trigger": "language mismatch between question and final answer, or abstention missed by recorded detector",
            "auto_fallback_trigger": "language mismatch between question and final answer",
            "missed_abstention_action": "flag-only diagnostic; not an automatic fallback",
            "fallback_order": ["w/o_repair", "full_v2"],
            "fallback_constraint": "prior answer must not mismatch the question language",
        },
        "aggregate": aggregate_cases(cases),
        "sample_summaries": sample_summaries(cases),
        "flagged_cases": flagged,
        "cases": cases,
        "aligned_surface_scan": {
            "aggregate": aggregate_surface_scans(surface_scans),
            "surfaces": surface_scans,
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_delta(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def md_table(headers: list[str], rows: list[list[Any]], numeric_cols: set[int] | None = None) -> list[str]:
    numeric_cols = numeric_cols or set()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---:" if index in numeric_cols else "---" for index in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def write_report(payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# I18n Abstention / Language Guard Audit",
        "",
        f"> Generated: {payload['generated_at']}",
        "> Scope: offline exploratory guard simulation over cost-instrumented rows; no LLM/API calls.",
        "> Important: this is a future-validation candidate, not a retroactive change to frozen reported outputs.",
        "",
        "## Decision Policy",
        "",
        "- Flags use only question language, answer language, and abstention-like text.",
        "- Automatic fallback is limited to language mismatch; missed-abstention-only rows are flag-only diagnostics.",
        "- The decision does not use reference answers, lexical metrics, judge scores, or post-hoc deltas.",
        "- Simulated fallback uses the first prior answer (`w/o_repair`, then `full_v2`) that matches the question language.",
        "",
        "## Aggregate Simulation",
        "",
    ]
    lines.extend(
        md_table(
            ["N", "Flagged", "Auto FB", "Fallbacks", "Lang Mismatch", "Missed Abstain", "F1 Orig", "F1 Sim", "Delta F1", "Faith Orig", "Faith Sim", "Delta Faith"],
            [[
                agg["n"],
                agg["guard_triggered"],
                agg["auto_fallback_triggered"],
                agg["fallback_applied"],
                agg["language_mismatch"],
                agg["missed_i18n_abstention"],
                fmt(agg["original_mean_f1"]),
                fmt(agg["simulated_mean_f1"]),
                fmt_delta(agg["mean_delta_f1"]),
                fmt(agg["original_mean_faithfulness"], 3),
                fmt(agg["simulated_mean_faithfulness"], 3),
                fmt_delta(agg["mean_delta_faithfulness"], 3),
            ]],
            numeric_cols=set(range(12)),
        )
    )

    lines.extend(["", "## Sample Summary", ""])
    lines.extend(
        md_table(
            ["Sample", "N", "Flagged", "Auto FB", "Fallbacks", "Lang Mismatch", "Missed Abstain", "Delta F1", "Delta Rouge", "Delta Faith"],
            [[
                row["sample"],
                row["n"],
                row["guard_triggered"],
                row["auto_fallback_triggered"],
                row["fallback_applied"],
                row["language_mismatch"],
                row["missed_i18n_abstention"],
                fmt_delta(row["mean_delta_f1"]),
                fmt_delta(row["mean_delta_rouge_l"]),
                fmt_delta(row["mean_delta_faithfulness"], 3),
            ] for row in payload["sample_summaries"]],
            numeric_cols={1, 2, 3, 4, 5, 6, 7, 8, 9},
        )
    )

    lines.extend(["", "## Flagged Cases", ""])
    flagged_rows = []
    for case in payload["flagged_cases"]:
        flagged_rows.append(
            [
                case["sample"],
                case.get("dataset_line_no"),
                ", ".join(case["guard_flags"]),
                case["action"],
                fmt(case.get("original_f1")),
                fmt(case.get("simulated_f1")),
                fmt_delta(case.get("delta_f1")),
                fmt_delta(case.get("delta_faithfulness"), 3),
                compact(case.get("question"), 80),
                case.get("original_prediction_preview"),
            ]
        )
    lines.extend(
        md_table(
            ["Sample", "Line", "Flags", "Action", "F1 Orig", "F1 Sim", "Delta F1", "Delta Faith", "Question", "Prediction Preview"],
            flagged_rows,
            numeric_cols={1, 4, 5, 6, 7},
        )
    )

    surface_scan = payload["aligned_surface_scan"]
    scan_agg = surface_scan["aggregate"]
    lines.extend(["", "## Full Aligned Surface Scan", ""])
    lines.extend(
        md_table(
            ["Surfaces", "N", "Flagged", "Lang Mismatch", "Recorded Abstain", "I18n Abstain", "Missed Abstain"],
            [[
                scan_agg["surfaces"],
                scan_agg["n"],
                scan_agg["flagged"],
                scan_agg["language_mismatch"],
                scan_agg["recorded_abstention_language"],
                scan_agg["i18n_abstention_language"],
                scan_agg["missed_i18n_abstention"],
            ]],
            numeric_cols=set(range(7)),
        )
    )
    lines.extend(["", "### Surface Breakdown", ""])
    lines.extend(
        md_table(
            ["Surface", "N", "Flagged", "Lang Mismatch", "Recorded Abstain", "I18n Abstain", "Missed Abstain"],
            [[
                scan["surface"],
                scan.get("n", 0),
                scan.get("flagged", 0),
                scan.get("language_mismatch", 0),
                scan.get("recorded_abstention_language", 0),
                scan.get("i18n_abstention_language", 0),
                scan.get("missed_i18n_abstention", 0),
            ] for scan in surface_scan["surfaces"]],
            numeric_cols={1, 2, 3, 4, 5, 6},
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The exploratory guard flags the known Domain line-1 regression because an English question received a Chinese abstention.",
            "- Missed-abstention-only rows are not automatically reverted because the broad fallback policy hurts several Legal rows in this sample.",
            "- The full aligned VSR scan finds no language-mismatch cases across Mix130/Legal/Biology/Domain full_v2_rescue outputs, but it does expose missed abstention-like answers for detector calibration.",
            "- Aggregate movement is small because this audit covers only 20 cost-instrumented rows; the result is best used to define a future locked validation run.",
            "- Do not fold this simulation into frozen v33 or current main result tables without a new pre-registered holdout validation.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
