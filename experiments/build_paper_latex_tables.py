#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build LaTeX table fragments from the completed paper-ready result payload.

The script is intentionally offline. It reads the consolidated JSON produced by
``build_paper_ready_tables.py`` and writes an include-ready table fragment under
``paper/`` without mutating the main paper draft.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSON = ROOT / "results" / "paper_tables" / "paper_ready_experiment_tables.json"
OUTPUT_TEX = ROOT / "paper" / "generated_experiment_tables.tex"


def read_payload() -> dict[str, Any]:
    with SOURCE_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tex_escape(value: Any) -> str:
    text = str(value if value is not None else "N/A")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_num(value: Any, digits: int = 4) -> str:
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


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_p(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0:
        return "<1e-15"
    if number < 0.001:
        return f"{number:.2e}"
    return f"{number:.4f}"


def table_block(
    *,
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    align: str,
    size: str = r"\small",
    note: str | None = None,
) -> list[str]:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        size,
        f"\\caption{{{tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(tex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if note:
        lines.append(r"\vspace{0.35em}")
        lines.append(r"\footnotesize{" + tex_escape(note) + "}")
    lines.append(r"\end{table}")
    lines.append("")
    return lines


def strict_ablation_table(payload: dict[str, Any]) -> list[str]:
    rows = []
    for row in payload["strict_ablation"]["aggregate"]:
        rows.append(
            [
                row["config"],
                row["rows"],
                fmt_num(row["f1"]),
                fmt_num(row["rouge_l"]),
                fmt_num(row["faithfulness"]),
                fmt_pct(row["repair_rate"]),
                fmt_pct(row["abstention_rate"]),
                fmt_num(row["runtime_seconds"], 1),
            ]
        )
    return table_block(
        caption="Strict VSR ablation on UltraDomain Mix130",
        label="tab:strict-vsr-ablation-mix130",
        headers=["Config", "Rows", "F1", "ROUGE-L", "Faith.", "Repair", "Abstain", "s/q"],
        rows=rows,
        align="lrrrrrrr",
        size=r"\scriptsize",
    )


def ablation_tests_table(payload: dict[str, Any]) -> list[str]:
    rows = []
    for row in payload["strict_ablation"]["comparisons"]:
        rows.append(
            [
                row["comparison"],
                row["n"],
                fmt_delta(row["f1_delta"]),
                fmt_p(row["f1_p"]),
                fmt_delta(row["faith_delta"]),
                fmt_p(row["faith_p"]),
            ]
        )
    return table_block(
        caption="Paired tests for strict VSR ablations",
        label="tab:strict-vsr-ablation-tests",
        headers=["Comparison", "N", "F1 Delta", "F1 p", "Faith. Delta", "Faith. p"],
        rows=rows,
        align="lrrrrr",
        size=r"\scriptsize",
    )


def retrieval_sage_table(payload: dict[str, Any]) -> list[str]:
    rows = []
    for row in payload["retrieval_and_sage"]:
        rows.append(
            [
                row["dataset"],
                row["n"],
                fmt_num(row["naive_f1"]),
                fmt_num(row["hyde_f1"]),
                fmt_num(row["vsr_rescue_f1"]),
                fmt_num(row["sage_v33_f1"]),
                row["best_retrieval"],
                fmt_delta(row["sage_minus_best"]),
            ]
        )
    return table_block(
        caption="External retrieval baselines and SAGE v33",
        label="tab:retrieval-sage-v33",
        headers=["Dataset", "N", "Naive", "HyDE", "VSR Rescue", "SAGE v33", "Best", "SAGE-Best"],
        rows=rows,
        align="lrrrrrlr",
        size=r"\scriptsize",
    )


def locked_meta_tables(payload: dict[str, Any]) -> list[str]:
    meta = payload["locked_unseen_meta"]
    lines: list[str] = []
    method_rows = []
    for method, row in meta["weighted_methods_all_repeats"].items():
        method_rows.append(
            [
                method,
                row["sample_size"],
                fmt_num(row["mean_f1"]),
                fmt_num(row["mean_rouge_l"]),
                fmt_num(row["mean_runtime_seconds"], 1),
                fmt_num(row["mean_prediction_chars"], 1),
            ]
        )
    lines.extend(
        table_block(
            caption="Locked holdout weighted lexical results, Unseen75 plus Repeat1-8",
            label="tab:locked-holdout-weighted-lexical",
            headers=["Method", "N", "F1", "ROUGE-L", "s/q", "Chars"],
            rows=method_rows,
            align="lrrrrr",
            size=r"\scriptsize",
        )
    )

    judge_rows = []
    for method, row in meta["weighted_judge_complete_repeats"].items():
        judge_rows.append(
            [
                method,
                row["sample_size"],
                fmt_num(row["correctness"], 3),
                fmt_num(row["coverage"], 3),
                fmt_num(row["overall"], 3),
            ]
        )
    lines.extend(
        table_block(
            caption="Locked holdout weighted judge results, complete 326-row subset",
            label="tab:locked-holdout-weighted-judge",
            headers=["Method", "N", "Correct", "Coverage", "Overall"],
            rows=judge_rows,
            align="lrrrr",
            size=r"\scriptsize",
        )
    )

    pairwise_rows = []
    for comparison, row in meta["weighted_pairwise_complete_repeats"].items():
        pairwise_rows.append([comparison, row["sample_size"], fmt_pct(row["avg_target_win_rate"], 2)])
    lines.extend(
        table_block(
            caption="Locked holdout weighted pairwise win rates, complete 326-row subset",
            label="tab:locked-holdout-weighted-pairwise",
            headers=["Comparison", "N", "SAGE Target Win Rate"],
            rows=pairwise_rows,
            align="lrr",
            size=r"\scriptsize",
        )
    )
    return lines


def repeat8_table(payload: dict[str, Any]) -> list[str]:
    repeat = payload["locked_legal54_repeat8"]
    rows = []
    for method in ["naive", "hyde", "sage_v33_preservation_delta"]:
        agg = repeat["method_aggregates"][method]
        judge = repeat["judge_summaries"][method]
        rows.append(
            [
                method,
                agg["sample_size"],
                fmt_num(agg["mean_f1"]),
                fmt_num(agg["mean_rouge_l"]),
                fmt_num(judge["means"].get("overall"), 3),
                fmt_num(agg["mean_runtime_seconds"], 1),
            ]
        )
    return table_block(
        caption="Repeat8 legal-only exhaustive stress holdout",
        label="tab:repeat8-legal54",
        headers=["Method", "N", "F1", "ROUGE-L", "Judge", "s/q"],
        rows=rows,
        align="lrrrrr",
        size=r"\scriptsize",
        note="Repeat8 exhausts the remaining strict legal holdout capacity under the max-two-samples-per-context policy.",
    )


def cost_table(payload: dict[str, Any]) -> list[str]:
    rows = []
    for row in payload["cost_latency"]["retrieval_runtime"]:
        rows.append(
            [
                row["label"],
                row["n"],
                fmt_num(row["naive_runtime"], 1),
                fmt_num(row["hyde_runtime"], 1),
                fmt_num(row["full_v2_rescue_runtime"], 1),
                fmt_num(row["vsr_vs_naive_runtime_ratio"], 2),
                fmt_num(row["vsr_vs_hyde_runtime_ratio"], 2),
                fmt_delta(row["vsr_minus_best_retrieval_f1"]),
            ]
        )
    return table_block(
        caption="Cost and latency summary",
        label="tab:cost-latency-summary",
        headers=["Dataset", "N", "Naive s/q", "HyDE s/q", "VSR s/q", "VSR/Naive", "VSR/HyDE", "VSR-Best F1"],
        rows=rows,
        align="lrrrrrrr",
        size=r"\scriptsize",
    )


def main() -> None:
    payload = read_payload()
    lines = [
        "% Auto-generated by experiments/build_paper_latex_tables.py",
        f"% Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "% Source: results/paper_tables/paper_ready_experiment_tables.json",
        "% Requires: \\usepackage{booktabs}",
        "",
    ]
    for builder in [
        strict_ablation_table,
        ablation_tests_table,
        retrieval_sage_table,
        locked_meta_tables,
        repeat8_table,
        cost_table,
    ]:
        lines.extend(builder(payload))
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
