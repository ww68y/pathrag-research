#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize pairwise win-rate results for the locked unseen75 SAGE holdout."""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "sage_locked_unseen75"
OUTPUT_JSON = OUT_DIR / "sage_locked_unseen75_pairwise_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_LOCKED_UNSEEN75_PAIRWISE_SUMMARY.md"
EXPECTED_N = 75

DIMENSIONS = [
    "comprehensiveness",
    "diversity",
    "logicality",
    "relevance",
    "coherence",
]

PAIRWISE_RUNS = {
    "sage_v33_preservation_delta_minus_hyde": {
        "label": "SAGE v33 - HyDE",
        "target": "sage_v33_preservation_delta",
        "baseline": "hyde",
        "path": OUT_DIR / "pairwise_sage_v33_vs_hyde.jsonl",
        "summary": OUT_DIR / "pairwise_sage_v33_vs_hyde_summary.md",
    },
    "sage_v33_preservation_delta_minus_naive": {
        "label": "SAGE v33 - Naive",
        "target": "sage_v33_preservation_delta",
        "baseline": "naive",
        "path": OUT_DIR / "pairwise_sage_v33_vs_naive.jsonl",
        "summary": OUT_DIR / "pairwise_sage_v33_vs_naive_summary.md",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_file_line_no", line_no)
            rows.append(row)
    return rows


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.mean(vals) if vals else None


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def md_table(headers: list[str], rows: Iterable[Iterable[Any]], numeric_cols: set[int] | None = None) -> list[str]:
    numeric_cols = numeric_cols or set()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---:" if idx in numeric_cols else "---" for idx in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def summarize_dimensions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dimensions: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        winners = Counter(str((row.get("winners") or {}).get(dimension, "missing")) for row in rows)
        target_wins = winners.get("pathrag", 0)
        baseline_wins = winners.get("baseline", 0)
        ties = winners.get("tie", 0)
        win_rate = (target_wins + 0.5 * ties) / len(rows) if rows else None
        dimensions[dimension] = {
            "target_wins": target_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "missing": winners.get("missing", 0),
            "target_win_rate": win_rate,
        }
    return dimensions


def average_dimension_win_rate(dimensions: dict[str, dict[str, Any]]) -> float | None:
    return mean(value["target_win_rate"] for value in dimensions.values() if value["target_win_rate"] is not None)


def summarize_pairwise(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    dimensions = summarize_dimensions(rows)
    by_domain = {}
    for domain in sorted({str(row.get("dataset") or "unknown") for row in rows}):
        domain_rows = [row for row in rows if str(row.get("dataset") or "unknown") == domain]
        domain_dimensions = summarize_dimensions(domain_rows)
        by_domain[domain] = {
            "sample_size": len(domain_rows),
            "dimensions": domain_dimensions,
            "average_target_win_rate": average_dimension_win_rate(domain_dimensions),
        }
    return {
        "sample_size": len(rows),
        "complete": len(rows) == EXPECTED_N,
        "line_no_min": min((int(row.get("line_no", 0)) for row in rows), default=None),
        "line_no_max": max((int(row.get("line_no", 0)) for row in rows), default=None),
        "unique_line_nos": len({int(row.get("line_no", -1)) for row in rows}),
        "dimensions": dimensions,
        "by_domain": by_domain,
        "average_target_win_rate": average_dimension_win_rate(dimensions),
    }


def build_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_n": EXPECTED_N,
        "winner_convention": "tie counts as 0.5 win; target is the SAGE v33 input passed via --pathrag",
        "comparisons": {},
    }
    for key, spec in PAIRWISE_RUNS.items():
        stats = summarize_pairwise(spec["path"])
        payload["comparisons"][key] = {
            "label": spec["label"],
            "target": spec["target"],
            "baseline": spec["baseline"],
            "path": spec["path"].relative_to(ROOT).as_posix(),
            "summary": spec["summary"].relative_to(ROOT).as_posix(),
            **stats,
        }
    return payload


def write_report(payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# SAGE Locked Unseen75 Pairwise Win-Rate Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Scope",
        "",
        "- Dataset: `datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl`",
        "- Target: frozen `sage_v33_preservation_delta`; baselines are HyDE and Naive on the same 75 questions.",
        "- Judge protocol: pairwise A/B comparison with position randomized by line number; ties count as 0.5 win.",
        "",
        "## Average Win Rates",
        "",
    ]
    lines.extend(
        md_table(
            ["Comparison", "n", "Complete", "Avg Target Win Rate", "Line Range", "Unique Lines"],
            [
                [
                    comparison["label"],
                    comparison["sample_size"],
                    "yes" if comparison["complete"] else "no",
                    fmt_pct(comparison["average_target_win_rate"]),
                    f"{comparison['line_no_min']}-{comparison['line_no_max']}",
                    comparison["unique_line_nos"],
                ]
                for comparison in payload["comparisons"].values()
            ],
            numeric_cols={1, 3, 5},
        )
    )

    lines.extend(["", "## Dimension Win Rates", ""])
    dimension_rows = []
    for comparison in payload["comparisons"].values():
        for dimension, stats in comparison["dimensions"].items():
            dimension_rows.append(
                [
                    comparison["label"],
                    dimension,
                    stats["target_wins"],
                    stats["baseline_wins"],
                    stats["ties"],
                    fmt_pct(stats["target_win_rate"]),
                ]
            )
    lines.extend(
        md_table(
            ["Comparison", "Dimension", "Target Wins", "Baseline Wins", "Ties", "Target Win Rate"],
            dimension_rows,
            numeric_cols={2, 3, 4, 5},
        )
    )

    lines.extend(["", "## Domain Win Rates", ""])
    domain_rows = []
    for comparison in payload["comparisons"].values():
        for domain, stats in comparison["by_domain"].items():
            domain_rows.append(
                [
                    comparison["label"],
                    domain,
                    stats["sample_size"],
                    fmt_pct(stats["average_target_win_rate"]),
                ]
            )
    lines.extend(
        md_table(
            ["Comparison", "Domain", "n", "Avg Target Win Rate"],
            domain_rows,
            numeric_cols={2, 3},
        )
    )

    lines.extend(["", "## Source Files", ""])
    for comparison in payload["comparisons"].values():
        lines.append(f"- `{comparison['label']}`: `{comparison['path']}`")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
