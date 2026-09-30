#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit cost-instrumented rows for quality regressions and missed abstentions."""
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
OUTPUT_JSON = OUT_DIR / "cost_quality_case_audit.json"
OUTPUT_MD = OUT_DIR / "COST_QUALITY_CASE_AUDIT.md"

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

MULTILINGUAL_ABSTENTION_PATTERNS = [
    r"cannot be determined",
    r"unable to determine",
    r"insufficient (?:evidence|information|context)",
    r"not enough (?:evidence|information|context)",
    r"no (?:relevant )?(?:evidence|information)",
    r"未找到",
    r"无法确定",
    r"不能确定",
    r"无法判断",
    r"不能判断",
    r"没有.*(?:证据|信息|上下文|条款)",
    r"缺乏.*(?:证据|信息|上下文)",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


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
    return bool(metrics.get(key))


def answer(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("prediction") or row.get("answer") or "")


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def detect_multilingual_abstention(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return any(re.search(pattern, normalized) for pattern in MULTILINGUAL_ABSTENTION_PATTERNS)


def mean(values: list[float | None]) -> float | None:
    vals = [value for value in values if isinstance(value, (int, float))]
    return statistics.mean(vals) if vals else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def md_table(headers: list[str], rows: list[list[Any]], numeric_cols: set[int] | None = None) -> list[str]:
    numeric_cols = numeric_cols or set()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---:" if index in numeric_cols else "---" for index in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def sample_dirs(prefix: str) -> list[Path]:
    base = ROOT / "results" / "cost_instrumented"
    if not base.exists():
        return []
    return sorted(
        directory
        for directory in base.iterdir()
        if directory.is_dir()
        and directory.name.startswith(prefix)
        and any(directory.glob("full_v2_rescue_aligned_n*.json"))
    )


def row_key(row: dict[str, Any]) -> str:
    line_no = row.get("dataset_line_no")
    if isinstance(line_no, int):
        return f"line:{line_no}"
    return str(row.get("question_id") or row.get("question") or "unknown")


def load_sample_rows(prefix: str) -> dict[str, dict[str, dict[str, Any]]]:
    aligned: dict[str, dict[str, dict[str, Any]]] = {}
    for directory in sample_dirs(prefix):
        for method, pattern in METHOD_FILES.items():
            for path in sorted(directory.glob(pattern)):
                data = read_json(path)
                for row in data.get("results") or []:
                    key = row_key(row)
                    aligned.setdefault(key, {})[method] = row
    return aligned


def build_case(label: str, key: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    wo = rows.get("w/o_repair")
    full = rows.get("full_v2")
    rescue = rows.get("full_v2_rescue")
    anchor = rescue or full or wo or {}
    wo_f1 = metric(wo, "f1")
    full_f1 = metric(full, "f1")
    rescue_f1 = metric(rescue, "f1")
    full_faith = metric(full, "faithfulness")
    rescue_faith = metric(rescue, "faithfulness")
    rescue_answer = answer(rescue)
    recorded_abstention = bool_metric(rescue, "abstention_language")
    multilingual_abstention = detect_multilingual_abstention(rescue_answer)
    return {
        "sample": label,
        "key": key,
        "dataset_line_no": anchor.get("dataset_line_no"),
        "question_id": anchor.get("question_id"),
        "question": anchor.get("question"),
        "gold_answer": anchor.get("gold_answer"),
        "w_o_repair_f1": wo_f1,
        "full_v2_f1": full_f1,
        "full_v2_rescue_f1": rescue_f1,
        "rescue_delta_vs_w_o_repair_f1": rescue_f1 - wo_f1
        if rescue_f1 is not None and wo_f1 is not None
        else None,
        "rescue_delta_vs_full_v2_f1": rescue_f1 - full_f1
        if rescue_f1 is not None and full_f1 is not None
        else None,
        "full_v2_delta_vs_w_o_repair_f1": full_f1 - wo_f1
        if full_f1 is not None and wo_f1 is not None
        else None,
        "full_v2_faithfulness": full_faith,
        "full_v2_rescue_faithfulness": rescue_faith,
        "rescue_delta_vs_full_v2_faithfulness": rescue_faith - full_faith
        if rescue_faith is not None and full_faith is not None
        else None,
        "repair_performed": bool_metric(rescue, "repair_performed") or bool_metric(rescue, "repair_triggered"),
        "rescue_applied": bool_metric(rescue, "abstention_rescue_applied"),
        "recorded_abstention_language": recorded_abstention,
        "multilingual_abstention_language": multilingual_abstention,
        "missed_multilingual_abstention": bool(multilingual_abstention and not recorded_abstention),
        "cjk_output": has_cjk(rescue_answer),
        "prediction_preview": re.sub(r"\s+", " ", rescue_answer).strip()[:240],
    }


def summarize_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for sample in sorted({case["sample"] for case in cases}):
        sample_cases = [case for case in cases if case["sample"] == sample]
        deltas_vs_wo = [case.get("rescue_delta_vs_w_o_repair_f1") for case in sample_cases]
        deltas_vs_full = [case.get("rescue_delta_vs_full_v2_f1") for case in sample_cases]
        faith_deltas = [case.get("rescue_delta_vs_full_v2_faithfulness") for case in sample_cases]
        summaries.append(
            {
                "sample": sample,
                "n": len(sample_cases),
                "mean_rescue_delta_vs_w_o_repair_f1": mean(deltas_vs_wo),
                "mean_rescue_delta_vs_full_v2_f1": mean(deltas_vs_full),
                "mean_rescue_delta_vs_full_v2_faithfulness": mean(faith_deltas),
                "f1_regressions_vs_w_o_repair": sum(
                    1 for value in deltas_vs_wo if isinstance(value, (int, float)) and value < -1e-9
                ),
                "f1_gains_vs_w_o_repair": sum(
                    1 for value in deltas_vs_wo if isinstance(value, (int, float)) and value > 1e-9
                ),
                "faithfulness_drops_vs_full_v2": sum(
                    1 for value in faith_deltas if isinstance(value, (int, float)) and value < -1e-9
                ),
                "missed_multilingual_abstentions": sum(
                    1 for case in sample_cases if case["missed_multilingual_abstention"]
                ),
                "cjk_outputs": sum(1 for case in sample_cases if case["cjk_output"]),
            }
        )
    return summaries


def build_payload() -> dict[str, Any]:
    cases = []
    for label, prefix in SAMPLES:
        for key, rows in load_sample_rows(prefix).items():
            if {"w/o_repair", "full_v2", "full_v2_rescue"}.issubset(rows):
                cases.append(build_case(label, key, rows))
    cases = sorted(cases, key=lambda case: (case["sample"], case.get("dataset_line_no") or -1, case["key"]))
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "offline audit of cost-instrumented rows; no API calls",
        "sample_summaries": summarize_cases(cases),
        "cases": cases,
        "top_f1_regressions": sorted(
            [case for case in cases if (case.get("rescue_delta_vs_w_o_repair_f1") or 0) < -1e-9],
            key=lambda case: case.get("rescue_delta_vs_w_o_repair_f1") or 0,
        )[:10],
        "faithfulness_drops": sorted(
            [case for case in cases if (case.get("rescue_delta_vs_full_v2_faithfulness") or 0) < -1e-9],
            key=lambda case: case.get("rescue_delta_vs_full_v2_faithfulness") or 0,
        )[:10],
        "missed_multilingual_abstentions": [case for case in cases if case["missed_multilingual_abstention"]],
    }


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Cost-Instrumented Quality Case Audit",
        "",
        f"> Generated: {payload['generated_at']}",
        "> Scope: offline audit of existing cost-instrumented rows; no LLM/API calls.",
        "",
        "## Sample Summary",
        "",
    ]
    lines.extend(
        md_table(
            [
                "Sample",
                "n",
                "Mean F1 vs w/o",
                "Mean F1 vs full_v2",
                "Mean Faith vs full_v2",
                "F1 Gains",
                "F1 Regressions",
                "Faith Drops",
                "Missed Abstain",
                "CJK Outputs",
            ],
            [
                [
                    row["sample"],
                    row["n"],
                    fmt(row["mean_rescue_delta_vs_w_o_repair_f1"]),
                    fmt(row["mean_rescue_delta_vs_full_v2_f1"]),
                    fmt(row["mean_rescue_delta_vs_full_v2_faithfulness"], 3),
                    row["f1_gains_vs_w_o_repair"],
                    row["f1_regressions_vs_w_o_repair"],
                    row["faithfulness_drops_vs_full_v2"],
                    row["missed_multilingual_abstentions"],
                    row["cjk_outputs"],
                ]
                for row in payload["sample_summaries"]
            ],
            numeric_cols={1, 2, 3, 4, 5, 6, 7, 8, 9},
        )
    )

    lines.extend(["", "## Top F1 Regressions", ""])
    regression_rows = []
    for case in payload["top_f1_regressions"]:
        regression_rows.append(
            [
                case["sample"],
                case.get("dataset_line_no"),
                fmt(case.get("rescue_delta_vs_w_o_repair_f1")),
                fmt(case.get("w_o_repair_f1")),
                fmt(case.get("full_v2_rescue_f1")),
                case.get("repair_performed"),
                case.get("rescue_applied"),
                case.get("missed_multilingual_abstention"),
                str(case.get("question") or "").replace("|", "/")[:90],
                str(case.get("prediction_preview") or "").replace("|", "/"),
            ]
        )
    lines.extend(
        md_table(
            [
                "Sample",
                "Line",
                "Delta F1",
                "w/o F1",
                "Rescue F1",
                "Repair",
                "Rescue",
                "Missed Abstain",
                "Question",
                "Prediction Preview",
            ],
            regression_rows,
            numeric_cols={1, 2, 3, 4},
        )
    )

    lines.extend(["", "## Faithfulness Drops", ""])
    faith_rows = []
    for case in payload["faithfulness_drops"]:
        faith_rows.append(
            [
                case["sample"],
                case.get("dataset_line_no"),
                fmt(case.get("rescue_delta_vs_full_v2_faithfulness"), 3),
                fmt(case.get("full_v2_faithfulness"), 3),
                fmt(case.get("full_v2_rescue_faithfulness"), 3),
                fmt(case.get("rescue_delta_vs_full_v2_f1")),
                case.get("repair_performed"),
                case.get("rescue_applied"),
                str(case.get("question") or "").replace("|", "/")[:90],
            ]
        )
    lines.extend(
        md_table(
            ["Sample", "Line", "Faith Delta", "Full Faith", "Rescue Faith", "F1 Delta", "Repair", "Rescue", "Question"],
            faith_rows,
            numeric_cols={1, 2, 3, 4, 5},
        )
    )

    lines.extend(["", "## Missed Abstentions", ""])
    missed_rows = []
    for case in payload["missed_multilingual_abstentions"]:
        missed_rows.append(
            [
                case["sample"],
                case.get("dataset_line_no"),
                bool(case.get("cjk_output")),
                bool(case.get("recorded_abstention_language")),
                fmt(case.get("rescue_delta_vs_w_o_repair_f1")),
                str(case.get("prediction_preview") or "").replace("|", "/"),
            ]
        )
    lines.extend(
        md_table(
            ["Sample", "Line", "CJK", "Recorded Abstain", "Delta F1", "Prediction Preview"],
            missed_rows,
            numeric_cols={1, 4},
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Domain line 1 is the clearest F1 regression: repair converts a partially correct English abstention into a Chinese abstention, causing lexical F1 to fall to zero.",
            "- The current recorded abstention flag misses both an English and a Chinese abstention in the Domain sample, so the rescue branch is not triggered even when the answer is effectively an abstention.",
            "- Legal has the main faithfulness-drop signal after rescue; Biology is repair-heavy rather than rescue-heavy in this small sample.",
            "- This audit should guide a future pre-registered multilingual abstention detector or language-control guard, not post-hoc modification of already reported outputs.",
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
