#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze repair_prompt_v2 losses and simple fallback gates."""
import json
import statistics
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
V2_DIR = ROOT_DIR / "results" / "repair_v2_postverify"
TRIGGERED_V2_FILE = V2_DIR / "repair_v2_postverify_triggered_n45.json"
REPORT_FILE = V2_DIR / "repair_v2_failure_analysis_n45.md"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def short(text: str, width: int = 180) -> str:
    clean = " ".join(str(text).split()).replace("|", "/")
    return textwrap.shorten(clean, width=width, placeholder="...")


def classify_loss(row: Dict[str, Any]) -> str:
    m = row["metrics"]
    v2_words = len(row.get("v2_prediction", "").split())
    gold_words = max(1, len(row.get("gold_answer", "").split()))
    if m.get("abstention_language"):
        return "abstention / under-answer"
    if m.get("post_repair_faithfulness", 1.0) < 0.9:
        return "postverify faithfulness failure"
    if m.get("post_repair_coverage", 1.0) < 0.5:
        return "evidence coverage collapse"
    if v2_words / gold_words < 0.55:
        return "over-compression"
    return "semantic / lexical coverage loss"


def gate_simulation(rows: List[Dict[str, Any]], name: str, predicate: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
    current = [r["metrics"]["v2_f1"] for r in rows]
    values = []
    gate_n = saved = hurt = 0
    for row in rows:
        m = row["metrics"]
        if predicate(m):
            gate_n += 1
            values.append(m["old_full_f1"])
            if m["old_full_f1"] > m["v2_f1"]:
                saved += 1
            elif m["old_full_f1"] < m["v2_f1"]:
                hurt += 1
        else:
            values.append(m["v2_f1"])
    triggered_delta = statistics.mean(values) - statistics.mean(current)
    return {
        "name": name,
        "gate_n": gate_n,
        "saved": saved,
        "hurt": hurt,
        "triggered_mean_f1": statistics.mean(values),
        "estimated_130_delta": triggered_delta * len(rows) / 130,
    }


def main() -> None:
    payload = load_json(TRIGGERED_V2_FILE)
    rows = [row for row in payload["results"] if "error" not in row]
    wins = [r for r in rows if r["metrics"]["delta_v2_vs_v1_full_f1"] > 1e-9]
    losses = [r for r in rows if r["metrics"]["delta_v2_vs_v1_full_f1"] < -1e-9]
    current_mean = statistics.mean(r["metrics"]["v2_f1"] for r in rows)
    old_full_mean = statistics.mean(r["metrics"]["old_full_f1"] for r in rows)
    no_repair_mean = statistics.mean(r["metrics"]["old_no_repair_f1"] for r in rows)
    oracle_values = [max(r["metrics"]["v2_f1"], r["metrics"]["old_full_f1"]) for r in rows]
    oracle_mean = statistics.mean(oracle_values)
    oracle_130_delta = (oracle_mean - current_mean) * len(rows) / 130

    gate_rows = [
        gate_simulation(rows, "fallback if postFaith < 0.9", lambda m: m["post_repair_faithfulness"] < 0.9),
        gate_simulation(rows, "fallback if coverage < 0.7", lambda m: m["post_repair_coverage"] < 0.7),
        gate_simulation(rows, "fallback if coverage < 0.8", lambda m: m["post_repair_coverage"] < 0.8),
        gate_simulation(rows, "fallback if abstention language", lambda m: m["abstention_language"]),
        gate_simulation(
            rows,
            "fallback if postFaith < 0.9 or coverage < 0.7",
            lambda m: m["post_repair_faithfulness"] < 0.9 or m["post_repair_coverage"] < 0.7,
        ),
        gate_simulation(
            rows,
            "fallback if postFaith < 0.9 or abstention",
            lambda m: m["post_repair_faithfulness"] < 0.9 or m["abstention_language"],
        ),
    ]

    lines = [
        "# Repair Prompt V2 Failure Analysis",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Triggered sample size | {len(rows)} |",
        f"| Wins vs v1 full | {len(wins)} |",
        f"| Losses vs v1 full | {len(losses)} |",
        f"| Old no-repair mean F1 | {no_repair_mean:.4f} |",
        f"| Old v1 full mean F1 | {old_full_mean:.4f} |",
        f"| V2 mean F1 | {current_mean:.4f} |",
        f"| Oracle max(v2, v1 full) mean F1 | {oracle_mean:.4f} |",
        f"| Oracle estimated 130-row uplift left | {oracle_130_delta:+.4f} |",
        "",
        "## Fallback Gate Simulation",
        "",
        "| Gate | Rows gated | Losses saved | Wins hurt | Triggered mean F1 | Estimated 130-row delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in gate_rows:
        lines.append(
            f"| {row['name']} | {row['gate_n']} | {row['saved']} | {row['hurt']} | "
            f"{row['triggered_mean_f1']:.4f} | {row['estimated_130_delta']:+.4f} |"
        )

    lines.extend([
        "",
        "## V2 Loss Cases vs V1 Full",
        "",
        "| ID | Domain | Delta | No-repair F1 | V1 F1 | V2 F1 | PostFaith | Coverage | Abstain | Diagnosis |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in sorted(losses, key=lambda r: r["metrics"]["delta_v2_vs_v1_full_f1"]):
        m = row["metrics"]
        lines.append(
            f"| {row['question_id']} | {row['domain']} | {m['delta_v2_vs_v1_full_f1']:+.4f} | "
            f"{m['old_no_repair_f1']:.4f} | {m['old_full_f1']:.4f} | {m['v2_f1']:.4f} | "
            f"{m['post_repair_faithfulness']:.2f} | {m['post_repair_coverage']:.2f} | "
            f"{m['abstention_language']} | {classify_loss(row)} |"
        )

    lines.extend([
        "",
        "## Largest Loss Snapshots",
        "",
    ])
    for row in sorted(losses, key=lambda r: r["metrics"]["delta_v2_vs_v1_full_f1"])[:5]:
        m = row["metrics"]
        lines.extend(
            [
                f"### {row['question_id']} ({row['domain']}, delta {m['delta_v2_vs_v1_full_f1']:+.4f})",
                "",
                f"- Question: {short(row['question'])}",
                f"- Gold: {short(row['gold_answer'])}",
                f"- V1 full: {short(row.get('v1_full_prediction', ''))}",
                f"- V2: {short(row.get('v2_prediction', ''))}",
                "",
            ]
        )

    lines.extend(
        [
            "## Takeaways",
            "",
            "- The remaining downside is small: an oracle fallback to v1 full adds only about +0.0044 F1 on the 130-row selective score.",
            "- Simple fallback gates are not attractive yet; low coverage and low faithfulness also catch several genuine v2 wins.",
            "- The next useful work is evidence coverage and answer coverage diagnostics on the 11 losses, not another broad repair wording change.",
            "- A conservative production fallback could still flag abstention-language rows for review, but it should not automatically revert all of them.",
        ]
    )
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_FILE}")
    print(f"wins={len(wins)} losses={len(losses)} oracle_130_delta={oracle_130_delta:+.4f}")


if __name__ == "__main__":
    main()
