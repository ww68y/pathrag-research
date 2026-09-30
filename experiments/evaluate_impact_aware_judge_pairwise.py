#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-hoc judge and pairwise evaluation for the new four-policy holdout.

Unchanged predictions reuse the never-escalate judge score and count as a
deterministic pairwise tie. Only changed predictions trigger new API calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from evaluate_pairwise_winrate import DIMENSIONS, judge_pair
from evaluate_repro_results import judge_item


ROOT = Path(__file__).resolve().parents[1]
QUALITY_KEYS = ("correctness", "coverage", "relevance", "conciseness", "overall")
POLICY_FILES = {
    "never_escalate": "new48_never_escalate.jsonl",
    "risk_only": "new48_risk_only.jsonl",
    "impact_aware_v1": "new48_impact_aware.jsonl",
    "always_escalate": "new48_always_escalate.jsonl",
}


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["line_no"])] = row
    return rows


def answer_text(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or "")


def same_answer(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return answer_text(left) == answer_text(right)


def read_lines(path: Path) -> set[int]:
    return {int(value) for value in path.read_text(encoding="utf-8").split(",") if value.strip()}


def load_partial(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    partial: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return partial
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("kind")), str(row.get("policy")), int(row.get("line_no")))
        partial[key] = row
    return partial


async def gather_quality(
    client: AsyncOpenAI,
    model: str,
    jobs: list[tuple[str, int, dict[str, Any]]],
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(policy: str, line_no: int, row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            result = await judge_item(client, model, row, 2, 3.0)
        return {"kind": "quality", "policy": policy, **result}

    return await asyncio.gather(*(run_one(*job) for job in jobs))


async def gather_pairwise(
    client: AsyncOpenAI,
    model: str,
    jobs: list[tuple[str, int, dict[str, Any], dict[str, Any]]],
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(
        policy: str,
        line_no: int,
        policy_row: dict[str, Any],
        base_row: dict[str, Any],
    ) -> dict[str, Any]:
        async with semaphore:
            result = await judge_pair(client, model, policy_row, base_row)
        return {"kind": "pairwise", "policy": policy, **result}

    return await asyncio.gather(*(run_one(*job) for job in jobs))


def quality_summary(
    policy: str,
    base_rows: dict[int, dict[str, Any]],
    policy_rows: dict[int, dict[str, Any]],
    changed_scores: dict[int, dict[str, Any]],
    base_scores: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    for line_no in sorted(base_rows):
        score = changed_scores.get(line_no) or base_scores[line_no]
        scores.append(score)
    means = {
        key: sum(float(item.get("scores", {}).get(key, 0)) for item in scores) / max(1, len(scores))
        for key in QUALITY_KEYS
    }
    verdicts = Counter(str(item.get("scores", {}).get("verdict", "unknown")) for item in scores)
    changed = sum(not same_answer(base_rows[line_no], policy_rows[line_no]) for line_no in base_rows)
    return {
        "policy": policy,
        "n": len(scores),
        "changed_rows": changed,
        "means": {key: round(value, 4) for key, value in means.items()},
        "verdict_counts": dict(sorted(verdicts.items())),
        "rows": scores,
    }


def pairwise_summary(
    policy: str,
    base_rows: dict[int, dict[str, Any]],
    policy_rows: dict[int, dict[str, Any]],
    changed_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for line_no in sorted(base_rows):
        if same_answer(base_rows[line_no], policy_rows[line_no]):
            results.append({
                "line_no": line_no,
                "dataset": base_rows[line_no].get("dataset"),
                "winners": {dimension: "tie" for dimension in DIMENSIONS},
                "short_reason": "Identical prediction; deterministic tie.",
            })
        else:
            results.append(changed_results[line_no])
    dimensions: dict[str, dict[str, Any]] = {}
    rates = []
    for dimension in DIMENSIONS:
        pathrag_wins = sum(row.get("winners", {}).get(dimension) == "pathrag" for row in results)
        baseline_wins = sum(row.get("winners", {}).get(dimension) == "baseline" for row in results)
        ties = sum(row.get("winners", {}).get(dimension) == "tie" for row in results)
        rate = (pathrag_wins + 0.5 * ties) / max(1, len(results))
        rates.append(rate)
        dimensions[dimension] = {
            "pathrag_wins": pathrag_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "win_rate": round(rate, 4),
        }
    return {
        "policy": policy,
        "n": len(results),
        "changed_rows": len(changed_results),
        "average_win_rate": round(sum(rates) / max(1, len(rates)), 4),
        "dimensions": dimensions,
        "rows": results,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/impact_aware_validation")
    parser.add_argument("--base-file", default=POLICY_FILES["never_escalate"])
    parser.add_argument("--risk-file", default=POLICY_FILES["risk_only"])
    parser.add_argument("--impact-file", default=POLICY_FILES["impact_aware_v1"])
    parser.add_argument("--always-file", default=POLICY_FILES["always_escalate"])
    parser.add_argument("--label", default="New48")
    parser.add_argument("--output-json", default="results/impact_aware_validation/IMPACT_AWARE_NEW48_JUDGE_PAIRWISE.json")
    parser.add_argument("--output-md", default="results/impact_aware_validation/IMPACT_AWARE_NEW48_JUDGE_PAIRWISE.md")
    parser.add_argument("--partial-jsonl", default="results/impact_aware_validation/IMPACT_AWARE_NEW48_JUDGE_PAIRWISE_PARTIAL.jsonl")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    policy_files = dict(POLICY_FILES)
    policy_files["never_escalate"] = args.base_file
    policy_files["risk_only"] = args.risk_file
    policy_files["impact_aware_v1"] = args.impact_file
    policy_files["always_escalate"] = args.always_file
    policy_rows = {
        policy: load_jsonl(input_dir / filename)
        for policy, filename in policy_files.items()
    }
    base_rows = policy_rows["never_escalate"]
    partial_path = ROOT / args.partial_jsonl
    partial = load_partial(partial_path) if args.resume else {}

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = os.getenv("EVAL_MODEL") or os.getenv("DEEPSEEK_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("Judge API configuration is incomplete in .env")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=1)

    quality_jobs: list[tuple[str, int, dict[str, Any]]] = []
    pairwise_jobs: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
    for policy, rows in policy_rows.items():
        if policy == "never_escalate":
            for line_no, row in base_rows.items():
                key = ("quality", policy, line_no)
                if key not in partial:
                    quality_jobs.append((policy, line_no, row))
            continue
        for line_no, row in rows.items():
            if same_answer(row, base_rows[line_no]):
                continue
            qkey = ("quality", policy, line_no)
            pkey = ("pairwise", policy, line_no)
            if qkey not in partial:
                quality_jobs.append((policy, line_no, row))
            if pkey not in partial:
                pairwise_jobs.append((policy, line_no, row, base_rows[line_no]))

    new_quality = await gather_quality(client, model, quality_jobs, args.concurrency) if quality_jobs else []
    new_pairwise = await gather_pairwise(client, model, pairwise_jobs, args.concurrency) if pairwise_jobs else []
    for result in [*new_quality, *new_pairwise]:
        partial[(result["kind"], result["policy"], int(result["line_no"]))] = result

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("w", encoding="utf-8") as handle:
        for key in sorted(partial, key=lambda item: (item[0], item[1], item[2])):
            handle.write(json.dumps(partial[key], ensure_ascii=False) + "\n")

    base_scores = {
        line_no: partial[("quality", "never_escalate", line_no)]
        for line_no in base_rows
    }
    quality_results = {}
    pairwise_results = {}
    for policy, rows in policy_rows.items():
        changed_scores = {
            line_no: partial[("quality", policy, line_no)]
            for line_no in rows
            if ("quality", policy, line_no) in partial
        }
        quality_results[policy] = quality_summary(
            policy, base_rows, rows, changed_scores, base_scores
        )
        if policy == "never_escalate":
            continue
        changed_pairs = {
            line_no: partial[("pairwise", policy, line_no)]
            for line_no in rows
            if ("pairwise", policy, line_no) in partial
        }
        pairwise_results[policy] = pairwise_summary(policy, base_rows, rows, changed_pairs)

    payload = {
        "protocol": "impact_aware_new48_posthoc_judge_pairwise",
        "holdout_label": args.label,
        "quality": quality_results,
        "pairwise": pairwise_results,
        "api_calls_new_this_run": len(new_quality) + len(new_pairwise),
        "reuse_rule": "Unchanged policy predictions reuse baseline judge scores and deterministic pairwise ties.",
    }
    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Impact-Aware {args.label} Judge and Pairwise Validation",
        "",
        "- Unchanged predictions reuse never-escalate judge scores and count as deterministic pairwise ties.",
        "- New judge calls cover one baseline answer plus only changed policy answers.",
        "",
        "## Judge Quality",
        "",
        "| Policy | N | Changed | Correctness | Coverage | Relevance | Conciseness | Overall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, row in quality_results.items():
        means = row["means"]
        lines.append(
            f"| {policy} | {row['n']} | {row['changed_rows']} | {means['correctness']:.3f} | "
            f"{means['coverage']:.3f} | {means['relevance']:.3f} | {means['conciseness']:.3f} | {means['overall']:.3f} |"
        )
    lines.extend([
        "",
        "## Pairwise vs Never",
        "",
        "| Policy | N | Changed | Comprehensiveness | Diversity | Logicality | Relevance | Coherence | Average |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for policy, row in pairwise_results.items():
        values = [row["dimensions"][dimension]["win_rate"] for dimension in DIMENSIONS]
        lines.append(
            f"| {policy} | {row['n']} | {row['changed_rows']} | "
            + " | ".join(f"{row['dimensions'][dimension]['win_rate']:.3f}" for dimension in DIMENSIONS)
            + f" | {row['average_win_rate']:.3f} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "new_api_calls": len(new_quality) + len(new_pairwise),
        "judge_overall": {policy: row["means"]["overall"] for policy, row in quality_results.items()},
        "pairwise_average": {policy: row["average_win_rate"] for policy, row in pairwise_results.items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
