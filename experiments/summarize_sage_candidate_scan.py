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

from experiments.run_enhanced_experiments import compute_f1, compute_rouge_l  # noqa: E402

OUT_DIR = ROOT / "results" / "sage_candidate_scan"
OUTPUT_JSON = OUT_DIR / "sage_candidate_scan_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_CANDIDATE_SCAN_SUMMARY.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def gold_answer(row: dict[str, Any]) -> str:
    if row.get("gold_answer"):
        return str(row.get("gold_answer"))
    answers = row.get("reference_answers") or row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def runtime(row: dict[str, Any]) -> float:
    value = float(row.get("query_seconds", 0.0) or 0.0) + float(row.get("insert_seconds", 0.0) or 0.0)
    if value:
        return value
    metrics = row.get("metrics") or {}
    return float(metrics.get("runtime_seconds", 0.0) or 0.0)


def source_label(row: dict[str, Any]) -> str:
    for key in ["source", "selected_source", "route", "selected_route", "method"]:
        value = row.get(key)
        if value:
            return str(value)
    return "unknown"


def infer_family(path: Path) -> str:
    name = path.stem
    match = re.search(r"sage_v(\d+)", name)
    if match:
        return f"sage_v{match.group(1)}"
    return "sage"


def include_file(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".jsonl"):
        return False
    if "_eval" in name or "_analysis" in name or "_select" in name or "_line" in name:
        return False
    return name.startswith("mse_pathrag_sage_")


def aggregate_file(path: Path) -> dict[str, Any] | None:
    rows = [row for row in load_jsonl(path) if not row.get("error") and gold_answer(row) and prediction(row)]
    if not rows:
        return None
    f1s = [compute_f1(prediction(row), gold_answer(row)) for row in rows]
    rouges = [compute_rouge_l(prediction(row), gold_answer(row)) for row in rows]
    runtimes = [runtime(row) for row in rows]
    datasets: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in rows:
        datasets[str(row.get("dataset") or row.get("label") or "unknown")] = datasets.get(
            str(row.get("dataset") or row.get("label") or "unknown"), 0
        ) + 1
        source = source_label(row)
        sources[source] = sources.get(source, 0) + 1
    return {
        "file": str(path.relative_to(ROOT)),
        "family": infer_family(path),
        "sample_size": len(rows),
        "mean_f1": statistics.mean(f1s),
        "mean_rouge_l": statistics.mean(rouges),
        "mean_runtime_seconds": statistics.mean(runtimes),
        "datasets": datasets,
        "sources": sources,
    }


def summarize() -> dict[str, Any]:
    candidates = []
    for path in sorted((ROOT / "results").rglob("mse_pathrag_sage_*.jsonl")):
        if include_file(path):
            row = aggregate_file(path)
            if row:
                candidates.append(row)
    candidates.sort(key=lambda row: (row["sample_size"] >= 20, row["mean_f1"]), reverse=True)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "top_n_ge_20": [row for row in candidates if row["sample_size"] >= 20][:25],
    }


def short_dict(values: dict[str, int], limit: int = 4) -> str:
    items = sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items)


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# SAGE Candidate Scan Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Scope",
        "",
        "- Offline lexical scan of existing `mse_pathrag_sage_*.jsonl` outputs; no API calls are made.",
        "- Files with `_eval`, `_analysis`, `_select`, or single-line debug outputs are skipped.",
        "- Use this as candidate discovery; rerun promising methods under the final aligned protocol before paper claims.",
        "",
        "## Top Candidates With n >= 20",
        "",
        "| Rank | File | n | F1 | ROUGE-L | Runtime s/q | Datasets | Sources/Routes |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(payload["top_n_ge_20"], start=1):
        lines.append(
            f"| {rank} | `{row['file']}` | {row['sample_size']} | {row['mean_f1']:.4f} | {row['mean_rouge_l']:.4f} | "
            f"{row['mean_runtime_seconds']:.1f} | {short_dict(row['datasets'])} | {short_dict(row['sources'])} |"
        )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "- The strongest existing candidates should be treated as adaptive-routing leads, not final evidence, until regenerated under one frozen protocol.",
            "- High lexical F1 from routed methods suggests the next productive branch is method selection/routing rather than more context concatenation.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = summarize()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
