#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM-audit the highest-priority quality flags across aligned dataset slices."""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from experiments.run_aligned_ablation_rerun import load_ultradomain_jsonl  # noqa: E402
from experiments.run_enhanced_experiments import get_api_config  # noqa: E402
from experiments.run_repair_v2_postverify import parse_json_object  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

BASE_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
QUALITY_FLAGS_FILE = BASE_DIR / "cross_dataset_quality_flags_and_gates.json"
OUTPUT_JSON = BASE_DIR / "cross_dataset_quality_flag_llm_audit.json"
OUTPUT_MD = BASE_DIR / "CROSS_DATASET_QUALITY_FLAG_LLM_AUDIT.md"
LOGFILE = ROOT_DIR / "logs" / "cross_dataset_quality_flag_llm_audit.log"

DATASETS = {
    "ultradomain_mix": {
        "name": "UltraDomain mix",
        "file": ROOT_DIR / "datasets" / "UltraDomain" / "mix.jsonl",
        "dir": BASE_DIR,
        "n": 130,
    },
    "biology_fullcontext_50": {
        "name": "Biology full context",
        "file": ROOT_DIR / "datasets" / "UltraDomain" / "biology_fullcontext_50.jsonl",
        "dir": BASE_DIR / "biology_fullcontext_50",
        "n": 50,
    },
    "legal_fullcontext_20": {
        "name": "Legal full context",
        "file": ROOT_DIR / "datasets" / "UltraDomain" / "legal_fullcontext_20.jsonl",
        "dir": BASE_DIR / "legal_fullcontext_20",
        "n": 20,
    },
    "domain_expansion_windowed_50": {
        "name": "Domain expansion windowed",
        "file": ROOT_DIR / "datasets" / "UltraDomain" / "domain_expansion_windowed_50.jsonl",
        "dir": BASE_DIR / "domain_expansion_windowed_50",
        "n": 50,
    },
}

ISSUE_TYPES = {
    "evidence_gap",
    "answer_undercoverage",
    "overconservative_abstention",
    "reasonable_abstention",
    "answer_hallucination",
    "metric_mismatch",
    "rescue_gate_overreach",
    "repair_overcorrection",
    "verifier_false_positive",
    "verifier_false_negative",
    "mixed",
}

ACTION_TYPES = {
    "keep_full_v2_rescue",
    "tighten_abstention_gate",
    "add_rescue_acceptance_gate",
    "rerun_with_more_context",
    "improve_evidence_extraction",
    "improve_answer_coverage",
    "fix_postverifier",
    "manual_review",
}


def log(message: str) -> None:
    print(message, flush=True)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with LOGFILE.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(message + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def compact(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, number))


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def normalize_audit(raw_text: str) -> Dict[str, Any]:
    parsed = parse_json_object(raw_text)
    issue_type = str(parsed.get("issue_type", "mixed")).strip()
    if issue_type not in ISSUE_TYPES:
        issue_type = "mixed"
    action = str(parsed.get("recommended_action", "manual_review")).strip()
    if action not in ACTION_TYPES:
        action = "manual_review"
    return {
        "issue_type": issue_type,
        "recommended_action": action,
        "gold_in_normal_context_score": bounded_float(parsed.get("gold_in_normal_context_score")),
        "answer_support_score": bounded_float(parsed.get("answer_support_score")),
        "answer_completeness_score": bounded_float(parsed.get("answer_completeness_score")),
        "rescue_gate_was_appropriate": normalize_bool(parsed.get("rescue_gate_was_appropriate", True)),
        "prefer_full_v2_over_rescue": normalize_bool(parsed.get("prefer_full_v2_over_rescue", False)),
        "metric_under_rewards_rescue": normalize_bool(parsed.get("metric_under_rewards_rescue", False)),
        "short_reason": compact(parsed.get("short_reason", ""), 500),
        "raw_audit": raw_text,
    }


def safe_name(config_name: str) -> str:
    return config_name.replace("/", "_")


def result_file(dataset_slug: str, config_name: str) -> Path:
    dataset = DATASETS[dataset_slug]
    return dataset["dir"] / f"{safe_name(config_name)}_aligned_n{dataset['n']}.json"


def result_rows(dataset_slug: str, config_name: str) -> Dict[str, Dict[str, Any]]:
    return {
        row["question_id"]: row
        for row in load_json(result_file(dataset_slug, config_name)).get("results", [])
        if "error" not in row
    }


def question_rows(dataset_slug: str) -> Dict[str, Dict[str, Any]]:
    dataset = DATASETS[dataset_slug]
    return {
        row["question_id"]: row
        for row in load_ultradomain_jsonl(dataset["file"], dataset_slug)
    }


def priority_score(flag: Dict[str, Any]) -> tuple:
    flags = set(flag.get("flags", []))
    return (
        0 if "rescue_worse_than_full_v2" in flags else 1,
        0 if "low_faithfulness" in flags else 1,
        0 if "final_abstention" in flags else 1,
        float(flag.get("delta_vs_full_v2", 0.0)),
    )


def select_flags(limit: Optional[int], include_ultradomain: bool, ids: List[str]) -> List[Dict[str, Any]]:
    payload = load_json(QUALITY_FLAGS_FILE)
    flags = []
    for slug, data in payload.get("datasets", {}).items():
        if not include_ultradomain and slug == "ultradomain_mix":
            continue
        for item in data.get("flagged_rows", []):
            if ids and item.get("question_id") not in ids:
                continue
            flags.append(item)
    flags = sorted(flags, key=priority_score)
    return flags[:limit] if limit else flags


async def chat_completion(client: AsyncOpenAI, model: str, prompt: str):
    return await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=650,
    )


async def audit_one(flag: Dict[str, Any], config: Dict[str, str]) -> Dict[str, Any]:
    dataset_slug = flag["dataset_slug"]
    qid = flag["question_id"]
    questions = question_rows(dataset_slug)
    no_repair = result_rows(dataset_slug, "w/o_repair").get(qid, {})
    full_v2 = result_rows(dataset_slug, "full_v2").get(qid, {})
    rescue = result_rows(dataset_slug, "full_v2_rescue")[qid]
    question = questions[qid]
    trace = rescue.get("trace", {})

    prompt = f"""You are auditing a retrieval-QA experiment row from VSR-PathRAG full_v2_rescue.

Classify the dominant failure mode or confirm that the row is acceptable. Be strict about unsupported claims, but recognize when lexical F1 under-rewards a semantically correct answer.

Allowed issue_type values:
- evidence_gap: normal/rescue evidence likely lacks the gold-answer facts.
- answer_undercoverage: evidence has the needed facts, but the final answer omits important gold facets.
- overconservative_abstention: enough answer evidence exists, but the final answer refuses or says unavailable.
- reasonable_abstention: evidence genuinely lacks key answer facts and abstention/hedging is appropriate.
- answer_hallucination: final answer adds unsupported or contradicted claims.
- metric_mismatch: final answer is substantially correct but lexical F1/ROUGE under-rewarded it.
- rescue_gate_overreach: rescue was triggered, but it worsened a useful full_v2 answer and should have been rejected.
- repair_overcorrection: repair removed useful supported details or made the answer too narrow.
- verifier_false_positive: verifier scored support too high despite unsupported final answer.
- verifier_false_negative: verifier scored support too low despite supported final answer.
- mixed: more than one issue is material.

Allowed recommended_action values:
- keep_full_v2_rescue
- tighten_abstention_gate
- add_rescue_acceptance_gate
- rerun_with_more_context
- improve_evidence_extraction
- improve_answer_coverage
- fix_postverifier
- manual_review

Dataset: {flag['dataset']}
Question ID: {qid}
Domain: {flag.get('domain', '')}
Flags: {', '.join(flag.get('flags', []))}

Question:
{question['question']}

Gold answer:
{compact(question.get('gold_answer'), 1200)}

Normal aligned context excerpt:
{compact(question.get('context'), 6000)}

w/o_repair answer, F1={no_repair.get('metrics', {}).get('f1', 0.0):.4f}, faith={no_repair.get('metrics', {}).get('faithfulness', 0.0):.4f}:
{compact(no_repair.get('prediction'), 900)}

full_v2 answer, F1={full_v2.get('metrics', {}).get('f1', 0.0):.4f}, faith={full_v2.get('metrics', {}).get('faithfulness', 0.0):.4f}:
{compact(full_v2.get('prediction'), 900)}

full_v2_rescue final answer, F1={rescue.get('metrics', {}).get('f1', 0.0):.4f}, faith={rescue.get('metrics', {}).get('faithfulness', 0.0):.4f}, rescue_applied={rescue.get('metrics', {}).get('abstention_rescue_applied', False)}:
{compact(rescue.get('prediction'), 1200)}

Same-run full_v2 baseline before rescue:
{compact(trace.get('baseline_full_v2_answer'), 900)}

Verifier output:
{compact(trace.get('raw_postverification') or trace.get('raw_verification'), 900)}

Rescue verifier output:
{compact(trace.get('raw_rescue_postverification'), 900)}

Return JSON only:
{{
  "issue_type": "one allowed issue_type",
  "recommended_action": "one allowed recommended_action",
  "gold_in_normal_context_score": 0.0,
  "answer_support_score": 0.0,
  "answer_completeness_score": 0.0,
  "rescue_gate_was_appropriate": true,
  "prefer_full_v2_over_rescue": false,
  "metric_under_rewards_rescue": false,
  "short_reason": "brief reason in one English sentence"
}}"""
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    response = await chat_completion(client, config["model"], prompt)
    audit = normalize_audit(response.choices[0].message.content)
    return {
        "dataset": flag["dataset"],
        "dataset_slug": dataset_slug,
        "question_id": qid,
        "domain": flag.get("domain", ""),
        "question": rescue.get("question", ""),
        "flags": flag.get("flags", []),
        "metrics": rescue.get("metrics", {}),
        "full_v2_metrics": full_v2.get("metrics", {}),
        "no_repair_metrics": no_repair.get("metrics", {}),
        "delta_vs_full_v2": flag.get("delta_vs_full_v2", 0.0),
        "delta_vs_no_repair": flag.get("delta_vs_no_repair", 0.0),
        "prediction": rescue.get("prediction", ""),
        "gold_answer": rescue.get("gold_answer", ""),
        "audit": audit,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    return {
        "sample_size": len(rows),
        "valid_sample_size": len(valid),
        "error_count": len(rows) - len(valid),
        "issue_counts": dict(Counter(row["audit"]["issue_type"] for row in valid)),
        "action_counts": dict(Counter(row["audit"]["recommended_action"] for row in valid)),
        "mean_gold_in_normal_context_score": statistics.mean([row["audit"]["gold_in_normal_context_score"] for row in valid]) if valid else 0.0,
        "mean_answer_support_score": statistics.mean([row["audit"]["answer_support_score"] for row in valid]) if valid else 0.0,
        "mean_answer_completeness_score": statistics.mean([row["audit"]["answer_completeness_score"] for row in valid]) if valid else 0.0,
        "prefer_full_v2_count": sum(1 for row in valid if row["audit"].get("prefer_full_v2_over_rescue")),
        "rescue_gate_appropriate_count": sum(1 for row in valid if row["audit"].get("rescue_gate_was_appropriate")),
    }


def write_report(payload: Dict[str, Any]) -> None:
    lines = [
        "# Cross-Dataset Quality Flag LLM Audit",
        "",
        f"> Generated: {payload['created_at']}",
        f"> Rows audited: {payload['aggregated']['valid_sample_size']}/{payload['target_sample_size']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Error count | {payload['aggregated']['error_count']} |",
        f"| Mean gold-in-context score | {payload['aggregated']['mean_gold_in_normal_context_score']:.3f} |",
        f"| Mean answer support score | {payload['aggregated']['mean_answer_support_score']:.3f} |",
        f"| Mean answer completeness score | {payload['aggregated']['mean_answer_completeness_score']:.3f} |",
        f"| Prefer full_v2 over rescue | {payload['aggregated']['prefer_full_v2_count']} |",
        f"| Rescue gate appropriate | {payload['aggregated']['rescue_gate_appropriate_count']} |",
        "",
        "## Issue Counts",
        "",
        "| Issue | Count |",
        "|---|---:|",
    ]
    for issue, count in sorted(payload["aggregated"]["issue_counts"].items()):
        lines.append(f"| {issue} | {count} |")
    lines.extend(["", "## Recommended Actions", "", "| Action | Count |", "|---|---:|"])
    for action, count in sorted(payload["aggregated"]["action_counts"].items()):
        lines.append(f"| {action} | {count} |")
    lines.extend([
        "",
        "## Rows",
        "",
        "| Dataset | ID | Flags | Delta vs full_v2 | Faith | Issue | Action | Prefer full_v2 | Reason |",
        "|---|---|---|---:|---:|---|---|---:|---|",
    ])
    for row in payload["results"]:
        if "error" in row:
            lines.append(f"| {row.get('dataset', '')} | {row.get('question_id', '')} | error | 0.0000 | 0.00 | error | manual_review | False | {row.get('error', '')} |")
            continue
        audit = row["audit"]
        reason = audit.get("short_reason", "").replace("|", "/")
        lines.append(
            f"| {row['dataset']} | {row['question_id']} | {', '.join(row['flags'])} | "
            f"{row.get('delta_vs_full_v2', 0.0):+.4f} | {row.get('metrics', {}).get('faithfulness', 0.0):.2f} | "
            f"{audit['issue_type']} | {audit['recommended_action']} | {audit['prefer_full_v2_over_rescue']} | {reason} |"
        )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    flags = select_flags(args.limit, args.include_ultradomain, args.ids or [])
    existing_by_key: Dict[str, Dict[str, Any]] = {}
    if OUTPUT_JSON.exists() and not args.force:
        existing = load_json(OUTPUT_JSON)
        existing_by_key = {
            f"{row.get('dataset_slug')}::{row.get('question_id')}": row
            for row in existing.get("results", [])
        }
        log(f"Loaded existing audit rows: {len(existing_by_key)}/{len(flags)}")

    config = get_api_config()
    if not config.get("api_key"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    log("=" * 70)
    log(f"CROSS-DATASET QUALITY FLAG LLM AUDIT n={len(flags)}")
    log("=" * 70)
    log(f"Model: {config.get('model')}")

    rows_by_key = dict(existing_by_key)
    for index, flag in enumerate(flags, 1):
        key = f"{flag['dataset_slug']}::{flag['question_id']}"
        existing = rows_by_key.get(key)
        if existing and ("error" not in existing or not args.retry_errors):
            log(f"[{index}/{len(flags)}] SKIP {flag['question_id']}: already complete")
            continue
        last_error: Optional[Exception] = None
        for attempt in range(1, args.max_retries + 1):
            try:
                log(f"[{index}/{len(flags)}] {flag['dataset']} {flag['question_id']}: {flag.get('question', '')[:70]}...")
                row = await asyncio.wait_for(audit_one(flag, config), timeout=args.question_timeout)
                rows_by_key[key] = row
                log(f"  OK issue={row['audit']['issue_type']} action={row['audit']['recommended_action']}")
                break
            except Exception as exc:
                last_error = exc
                log(f"  ERROR attempt {attempt}/{args.max_retries}: {exc}")
                if attempt < args.max_retries:
                    await asyncio.sleep(5 * attempt)
        else:
            rows_by_key[key] = {
                "dataset": flag.get("dataset", ""),
                "dataset_slug": flag.get("dataset_slug", ""),
                "question_id": flag.get("question_id", ""),
                "flags": flag.get("flags", []),
                "error": str(last_error),
            }

        ordered_rows = [rows_by_key[f"{flag['dataset_slug']}::{flag['question_id']}"] for flag in flags if f"{flag['dataset_slug']}::{flag['question_id']}" in rows_by_key]
        payload = {
            "experiment": "cross_dataset_quality_flag_llm_audit",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_sample_size": len(flags),
            "include_ultradomain": args.include_ultradomain,
            "target_ids": [flag["question_id"] for flag in flags],
            "aggregated": aggregate(ordered_rows),
            "results": ordered_rows,
        }
        write_json(OUTPUT_JSON, payload)

    payload = load_json(OUTPUT_JSON)
    write_report(payload)
    log(f"Saved {OUTPUT_JSON}")
    log(f"Report {OUTPUT_MD}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--include-ultradomain", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--question-timeout", type=int, default=180)
    args = parser.parse_args()
    LOGFILE.write_text("", encoding="utf-8")
    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
