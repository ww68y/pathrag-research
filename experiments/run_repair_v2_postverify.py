#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate repair_prompt_v2 with post-repair verification.

Default target set: questions where the aligned v1 repair branch triggered.
The script reuses the aligned no-repair answer as the original answer, extracts
fresh evidence from the same context profile, applies a stricter final-answer-only
repair prompt, then verifies the repaired answer against the extracted evidence.
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

from openai import AsyncOpenAI

from experiments.run_enhanced_experiments import (
    bootstrap_ci,
    cohens_d,
    compute_contains,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    get_api_config,
    load_ultradomain_full,
    paired_t_test,
    wilcoxon_signed_rank_test,
)


ALIGNED_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
OUTPUT_DIR = ROOT_DIR / "results" / "repair_v2_postverify"
LOGFILE = ROOT_DIR / "logs" / "repair_v2_postverify.log"

RUN_PROFILE = {
    "name": "repair_prompt_v2_postverify",
    "target_source": "aligned full repair-triggered cases by default",
    "evidence_context_chars": 3500,
    "repair_evidence_chars": 1800,
    "postverify_evidence_chars": 1800,
    "original_answer_source": "w_o_repair_aligned_n130.json prediction",
    "temperature_evidence": 0.0,
    "temperature_repair": 0.0,
    "temperature_verification": 0.0,
    "evidence_max_tokens": 550,
    "repair_max_tokens": 360,
    "postverify_max_tokens": 260,
    "postverify_schema_fixed": True,
}

META_AUDIT_LEAK_PATTERNS = [
    r"unsupported claims?",
    r"contradicted claims?",
    r"breakdown of (?:supported|unsupported|contradicted) claims?",
    r"corrected answer",
    r"verification",
    r"correction steps?",
]

ABSTENTION_PATTERNS = [
    r"the evidence does not",
    r"evidence does not (?:provide|specify|state|mention|contain|include|support)",
    r"the extracted evidence does not",
    r"the provided context does not",
    r"the provided evidence does not",
    r"the context does not",
    r"context does not (?:provide|specify|state|mention|contain|include|support)",
    r"the passage does not",
    r"passage does not (?:provide|specify|state|mention|contain|include|support)",
    r"cannot be described from this evidence",
    r"cannot be determined from this evidence",
    r"cannot be determined from (?:the )?(?:context|evidence|passage|excerpt)",
    r"unable to determine from (?:the )?(?:context|evidence|passage|excerpt)",
    r"insufficient (?:evidence|information|context)",
    r"not enough (?:evidence|information|context)",
    r"no (?:relevant )?(?:evidence|information) (?:is )?(?:provided|available)",
]


def log(message: str) -> None:
    print(message, flush=True)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with LOGFILE.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(message + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_path, path)


def safe_target_name(target_mode: str, target_ids: List[str]) -> str:
    if target_mode == "ids":
        return f"ids_n{len(target_ids)}"
    return f"{target_mode}_n{len(target_ids)}"


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
    return {}


async def chat_completion(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
):
    return await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def detect_meta_audit_leak(answer: str) -> bool:
    lower = answer.lower()
    return any(re.search(pattern, lower) for pattern in META_AUDIT_LEAK_PATTERNS)


def detect_abstention_language(answer: str) -> bool:
    lower = answer.lower()
    if any(re.search(pattern, lower) for pattern in ABSTENTION_PATTERNS):
        return True
    has_evidence_anchor = re.search(r"\b(context|evidence|passage|excerpt|provided|extracted)\b", lower)
    if has_evidence_anchor and re.search(r"\bnot (?:specified|provided|stated|mentioned|included)\b", lower):
        return True
    return False


def normalize_final_answer(answer: str) -> str:
    cleaned = answer.strip()
    cleaned = re.sub(r"^final answer\s*:\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^answer\s*:\s*", "", cleaned, flags=re.I)
    return cleaned.strip()


async def repair_v2_one(
    question: Dict[str, Any],
    aligned_no_repair: Dict[str, Any],
    aligned_full: Dict[str, Any],
    config: Dict[str, str],
) -> Dict[str, Any]:
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    model = config["model"]
    question_text = question["question"]
    context = question["context"]
    original_answer = aligned_no_repair.get("prediction", "")
    start = time.time()

    evidence_prompt = f"""Extract the strongest evidence snippets needed to answer the question.

Rules:
- Use only text from the context.
- Prefer snippets that directly answer the question.
- Include enough evidence to preserve coverage, not only to reject unsupported claims.
- Return 4-6 snippets, each prefixed [E1], [E2], etc.

Context: {context[:RUN_PROFILE['evidence_context_chars']]}

Question: {question_text}

Evidence snippets:"""
    evidence_response = await chat_completion(
        client,
        model,
        evidence_prompt,
        RUN_PROFILE["temperature_evidence"],
        RUN_PROFILE["evidence_max_tokens"],
    )
    evidence = evidence_response.choices[0].message.content.strip()

    previous_trace = aligned_full.get("trace", {})
    unsupported = previous_trace.get("unsupported_claims", aligned_full.get("metrics", {}).get("unsupported_claims", 0))
    contradicted = previous_trace.get("contradicted_claims", aligned_full.get("metrics", {}).get("contradicted_claims", 0))
    raw_verification = previous_trace.get("raw_verification", "")[:900]

    repair_prompt = f"""You are editing an answer for a retrieval QA benchmark.

Task: produce ONLY the final answer to the question.

Hard rules:
- Do not write a verification report.
- Do not mention unsupported claims, contradicted claims, correction steps, evidence IDs, or these instructions.
- Keep all answer content that is directly supported by the evidence.
- Remove or soften claims that are not supported by the evidence.
- If the evidence is incomplete, give the best concise answer supported by the evidence; do not add a long refusal.
- Prefer 1 short paragraph. Use at most 3 sentences.

Question:
{question_text}

Original answer:
{original_answer[:1000]}

Previous verification summary:
unsupported_claims={unsupported}; contradicted_claims={contradicted}
{raw_verification}

Evidence:
{evidence[:RUN_PROFILE['repair_evidence_chars']]}

Final answer only:"""
    repair_response = await chat_completion(
        client,
        model,
        repair_prompt,
        RUN_PROFILE["temperature_repair"],
        RUN_PROFILE["repair_max_tokens"],
    )
    repaired_answer = normalize_final_answer(repair_response.choices[0].message.content)

    postverify_prompt = f"""Verify the final answer against the evidence. Output JSON only.

Question: {question_text}

Final answer: {repaired_answer[:1000]}

Evidence:
{evidence[:RUN_PROFILE['postverify_evidence_chars']]}

Return one JSON object with this schema. Replace placeholders with your assessed values:
{{
  "supported_claims": <integer count>,
  "unsupported_claims": <integer count>,
  "contradicted_claims": <integer count>,
  "faithfulness_score": <number from 0.0 to 1.0>,
  "coverage_score": <number from 0.0 to 1.0>
}}"""
    postverify_response = await chat_completion(
        client,
        model,
        postverify_prompt,
        RUN_PROFILE["temperature_verification"],
        RUN_PROFILE["postverify_max_tokens"],
    )
    postverify_raw = postverify_response.choices[0].message.content
    postverify = parse_json_object(postverify_raw)

    elapsed = time.time() - start
    gold = question["gold_answer"]
    f1 = compute_f1(repaired_answer, gold)
    rouge_l = compute_rouge_l(repaired_answer, gold)
    em = compute_exact_match(repaired_answer, gold)
    contains = compute_contains(repaired_answer, gold)

    return {
        "question_id": question["question_id"],
        "question": question_text,
        "gold_answer": gold,
        "domain": question.get("domain", ""),
        "dataset": question.get("dataset", "ultradomain"),
        "original_no_repair_prediction": original_answer,
        "v1_full_prediction": aligned_full.get("prediction", ""),
        "v2_prediction": repaired_answer,
        "evidence": evidence,
        "metrics": {
            "v2_f1": f1,
            "v2_rouge_l": rouge_l,
            "v2_exact_match": em,
            "v2_contains": contains,
            "runtime_seconds": elapsed,
            "post_repair_faithfulness": float(postverify.get("faithfulness_score", 0.0) or 0.0),
            "post_repair_coverage": float(postverify.get("coverage_score", 0.0) or 0.0),
            "post_supported_claims": int(postverify.get("supported_claims", 0) or 0),
            "post_unsupported_claims": int(postverify.get("unsupported_claims", 0) or 0),
            "post_contradicted_claims": int(postverify.get("contradicted_claims", 0) or 0),
            "meta_audit_leak": detect_meta_audit_leak(repaired_answer),
            "abstention_language": detect_abstention_language(repaired_answer),
            "old_no_repair_f1": aligned_no_repair["metrics"]["f1"],
            "old_full_f1": aligned_full["metrics"]["f1"],
            "old_full_faithfulness": aligned_full["metrics"].get("faithfulness", 0),
            "delta_v2_vs_no_repair_f1": f1 - aligned_no_repair["metrics"]["f1"],
            "delta_v2_vs_v1_full_f1": f1 - aligned_full["metrics"]["f1"],
            "delta_v1_full_vs_no_repair_f1": aligned_full["metrics"]["f1"] - aligned_no_repair["metrics"]["f1"],
        },
        "postverify_raw": postverify_raw,
        "run_profile": RUN_PROFILE,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if "error" not in row]
    if not valid_rows:
        return {}
    metrics = [r["metrics"] for r in valid_rows]
    legacy_postverify_schema_rows = sum(
        1
        for row in valid_rows
        if not row.get("run_profile", {}).get("postverify_schema_fixed", False)
    )
    meta_audit_count = sum(
        1
        for row in valid_rows
        if row["metrics"].get(
            "meta_audit_leak",
            row["metrics"].get("meta_leak", detect_meta_audit_leak(row.get("v2_prediction", ""))),
        )
    )
    abstention_count = sum(
        1
        for row in valid_rows
        if row["metrics"].get("abstention_language", detect_abstention_language(row.get("v2_prediction", "")))
    )
    return {
        "sample_size": len(rows),
        "valid_sample_size": len(valid_rows),
        "error_count": len(rows) - len(valid_rows),
        "legacy_postverify_schema_rows": legacy_postverify_schema_rows,
        "mean_v2_f1": statistics.mean(m["v2_f1"] for m in metrics),
        "mean_v2_rouge_l": statistics.mean(m["v2_rouge_l"] for m in metrics),
        "mean_post_repair_faithfulness": statistics.mean(m["post_repair_faithfulness"] for m in metrics),
        "mean_post_repair_coverage": statistics.mean(m["post_repair_coverage"] for m in metrics),
        "mean_old_no_repair_f1": statistics.mean(m["old_no_repair_f1"] for m in metrics),
        "mean_old_full_f1": statistics.mean(m["old_full_f1"] for m in metrics),
        "mean_delta_v2_vs_no_repair_f1": statistics.mean(m["delta_v2_vs_no_repair_f1"] for m in metrics),
        "mean_delta_v2_vs_v1_full_f1": statistics.mean(m["delta_v2_vs_v1_full_f1"] for m in metrics),
        "mean_delta_v1_full_vs_no_repair_f1": statistics.mean(m["delta_v1_full_vs_no_repair_f1"] for m in metrics),
        "meta_audit_leak_rate": meta_audit_count / len(metrics),
        "abstention_language_rate": abstention_count / len(metrics),
        "v2_improved_vs_no_repair": sum(1 for m in metrics if m["delta_v2_vs_no_repair_f1"] > 1e-9),
        "v2_worse_vs_no_repair": sum(1 for m in metrics if m["delta_v2_vs_no_repair_f1"] < -1e-9),
        "v2_improved_vs_v1_full": sum(1 for m in metrics if m["delta_v2_vs_v1_full_f1"] > 1e-9),
        "v2_worse_vs_v1_full": sum(1 for m in metrics if m["delta_v2_vs_v1_full_f1"] < -1e-9),
    }


def paired_stats(rows: List[Dict[str, Any]], left_key: str, right_key: str) -> Dict[str, Any]:
    rows = [row for row in rows if "error" not in row]
    if len(rows) < 2:
        return {}
    left = [r["metrics"][left_key] for r in rows]
    right = [r["metrics"][right_key] for r in rows]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "left": left_key,
        "right": right_key,
        "n": len(rows),
        "mean_diff": statistics.mean(diffs) if diffs else 0.0,
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs) if len(diffs) >= 2 else (0, 0),
    }


def select_target_ids(target_mode: str, explicit_ids: List[str]) -> List[str]:
    full_rows = load_json(ALIGNED_DIR / "full_aligned_n130.json")["results"]
    no_rows = load_json(ALIGNED_DIR / "w_o_repair_aligned_n130.json")["results"]
    by_no = {r["question_id"]: r for r in no_rows}
    if target_mode == "ids":
        return explicit_ids
    if target_mode == "all":
        return [r["question_id"] for r in full_rows]
    if target_mode == "triggered":
        return [r["question_id"] for r in full_rows if r["metrics"].get("repair_triggered")]
    if target_mode == "losses":
        return [
            r["question_id"]
            for r in full_rows
            if r["metrics"].get("repair_triggered") and r["metrics"]["f1"] < by_no[r["question_id"]]["metrics"]["f1"]
        ]
    raise ValueError(f"Unsupported target mode: {target_mode}")


def save_progress(output_file: Path, rows_by_id: Dict[str, Dict[str, Any]], target_ids: List[str]) -> None:
    rows = [rows_by_id[qid] for qid in target_ids if qid in rows_by_id]
    error_count = sum(1 for row in rows if "error" in row)
    payload = {
        "experiment": "repair_prompt_v2_postverify",
        "sample_size": len(rows),
        "valid_sample_size": len(rows) - error_count,
        "error_count": error_count,
        "target_sample_size": len(target_ids),
        "complete": len(rows) >= len(target_ids) and error_count == 0,
        "run_profile": RUN_PROFILE,
        "aggregated": aggregate(rows),
        "stat_tests": {
            "v2_vs_no_repair_f1": paired_stats(rows, "v2_f1", "old_no_repair_f1"),
            "v2_vs_v1_full_f1": paired_stats(rows, "v2_f1", "old_full_f1"),
            "v1_full_vs_no_repair_f1": paired_stats(rows, "old_full_f1", "old_no_repair_f1"),
        },
        "results": rows,
    }
    write_json(output_file, payload)


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload["results"]
    agg = payload["aggregated"]
    stats = payload["stat_tests"]
    top_gains = sorted(rows, key=lambda r: r["metrics"]["delta_v2_vs_v1_full_f1"], reverse=True)[:10]
    top_losses = sorted(rows, key=lambda r: r["metrics"]["delta_v2_vs_v1_full_f1"])[:10]
    lines = [
        "# Repair Prompt V2 + Post-Verification",
        "",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Completed valid rows: {payload.get('valid_sample_size', payload['sample_size'])}/{payload['target_sample_size']}",
        f"> Error rows: {payload.get('error_count', 0)}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Old no-repair F1 | {agg.get('mean_old_no_repair_f1', 0):.4f} |",
        f"| Old v1 full F1 | {agg.get('mean_old_full_f1', 0):.4f} |",
        f"| V2 repair F1 | {agg.get('mean_v2_f1', 0):.4f} |",
        f"| V2 vs no-repair F1 delta | {agg.get('mean_delta_v2_vs_no_repair_f1', 0):+.4f} |",
        f"| V2 vs v1 full F1 delta | {agg.get('mean_delta_v2_vs_v1_full_f1', 0):+.4f} |",
        f"| V2 ROUGE-L | {agg.get('mean_v2_rouge_l', 0):.4f} |",
        f"| Post-repair faithfulness | {agg.get('mean_post_repair_faithfulness', 0):.4f} |",
        f"| Post-repair coverage | {agg.get('mean_post_repair_coverage', 0):.4f} |",
        f"| Legacy postverify-schema rows | {agg.get('legacy_postverify_schema_rows', 0)} |",
        f"| Meta-audit leak rate | {agg.get('meta_audit_leak_rate', agg.get('meta_leak_rate', 0)):.1%} |",
        f"| Abstention-language rate | {agg.get('abstention_language_rate', 0):.1%} |",
        "",
        "## Paired Tests",
        "",
        "| Comparison | n | Mean diff | t-p | Wilcoxon-p | Cohen's d |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in stats.items():
        if row:
            lines.append(
                f"| {label} | {row['n']} | {row['mean_diff']:+.4f} | {row['t_p_value']:.4f} | "
                f"{row['wilcoxon_p_value']:.4f} | {row['cohens_d']:+.2f} |"
            )
    lines.extend([
        "",
        "## Top V2 Gains vs V1 Full",
        "",
        "| Question | Domain | Delta | V1 F1 | V2 F1 | Post Faith | Meta Audit | Abstain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in [row for row in top_gains if "error" not in row]:
        m = row["metrics"]
        q = row["question"].replace("|", "/")[:90]
        lines.append(
            f"| {row['question_id']}: {q} | {row['domain']} | {m['delta_v2_vs_v1_full_f1']:+.4f} | "
            f"{m['old_full_f1']:.4f} | {m['v2_f1']:.4f} | {m['post_repair_faithfulness']:.2f} | "
            f"{m.get('meta_audit_leak', m.get('meta_leak', detect_meta_audit_leak(row.get('v2_prediction', ''))))} | "
            f"{m.get('abstention_language', detect_abstention_language(row.get('v2_prediction', '')))} |"
        )
    lines.extend([
        "",
        "## Top V2 Losses vs V1 Full",
        "",
        "| Question | Domain | Delta | V1 F1 | V2 F1 | Post Faith | Meta Audit | Abstain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in [row for row in top_losses if "error" not in row]:
        m = row["metrics"]
        q = row["question"].replace("|", "/")[:90]
        lines.append(
            f"| {row['question_id']}: {q} | {row['domain']} | {m['delta_v2_vs_v1_full_f1']:+.4f} | "
            f"{m['old_full_f1']:.4f} | {m['v2_f1']:.4f} | {m['post_repair_faithfulness']:.2f} | "
            f"{m.get('meta_audit_leak', m.get('meta_leak', detect_meta_audit_leak(row.get('v2_prediction', ''))))} | "
            f"{m.get('abstention_language', detect_abstention_language(row.get('v2_prediction', '')))} |"
        )
    lines.extend(["", "## Notes", ""])
    if agg.get("legacy_postverify_schema_rows", 0):
        lines.append(
            "- Post-repair faithfulness/coverage scores in legacy rows are invalid because the earlier postverify prompt used zero-valued schema examples; rerun those rows after API balance is restored."
        )
    lines.append("- F1 and ROUGE-L comparisons remain usable because they are computed locally from the generated answers.")
    path.write_text("\n".join(lines), encoding="utf-8")


async def run_targets(args: argparse.Namespace) -> Path:
    config = get_api_config()
    if not config.get("api_key"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    questions = {q["question_id"]: q for q in load_ultradomain_full()}
    target_ids = select_target_ids(args.target_mode, args.ids or [])
    if args.limit:
        target_ids = target_ids[: args.limit]

    aligned_no = {r["question_id"]: r for r in load_json(ALIGNED_DIR / "w_o_repair_aligned_n130.json")["results"]}
    aligned_full = {r["question_id"]: r for r in load_json(ALIGNED_DIR / "full_aligned_n130.json")["results"]}
    target_name = safe_target_name(args.target_mode, target_ids)
    output_file = OUTPUT_DIR / f"repair_v2_postverify_{target_name}.json"
    rows_by_id: Dict[str, Dict[str, Any]] = {}

    if output_file.exists() and not args.force:
        existing = load_json(output_file)
        rows_by_id = {r["question_id"]: r for r in existing.get("results", [])}
        log(f"Loaded existing rows: {len(rows_by_id)}/{len(target_ids)}")

    log("=" * 70)
    log(f"REPAIR PROMPT V2 POSTVERIFY: {target_name}")
    log("=" * 70)
    log(f"Model: {config.get('model')}")
    log(f"API Base: {config.get('base_url')}")

    for index, qid in enumerate(target_ids, 1):
        if qid in rows_by_id and not (args.retry_errors and "error" in rows_by_id[qid]):
            log(f"[{index}/{len(target_ids)}] SKIP {qid}: already complete")
            continue
        question = questions[qid]
        log(f"[{index}/{len(target_ids)}] {qid} {question.get('domain', '')}: {question['question'][:72]}...")
        start = time.time()
        last_error: Optional[Exception] = None
        for attempt in range(1, args.max_retries + 1):
            try:
                coro = repair_v2_one(question, aligned_no[qid], aligned_full[qid], config)
                row = await asyncio.wait_for(coro, timeout=args.question_timeout) if args.question_timeout else await coro
                rows_by_id[qid] = row
                metrics = row["metrics"]
                log(
                    f"  OK old_full={metrics['old_full_f1']:.3f} v2={metrics['v2_f1']:.3f} "
                    f"postFaith={metrics['post_repair_faithfulness']:.2f} "
                    f"metaAudit={'Y' if metrics['meta_audit_leak'] else 'N'} "
                    f"abstain={'Y' if metrics['abstention_language'] else 'N'} "
                    f"Time={metrics['runtime_seconds']:.1f}s"
                )
                break
            except Exception as exc:
                last_error = exc
                elapsed = time.time() - start
                log(f"  ERROR attempt {attempt}/{args.max_retries} after {elapsed:.1f}s: {exc}")
                if attempt < args.max_retries:
                    await asyncio.sleep(5 * attempt)
        else:
            assert last_error is not None
            rows_by_id[qid] = {
                "question_id": qid,
                "question": question["question"],
                "gold_answer": question["gold_answer"],
                "domain": question.get("domain", ""),
                "error": str(last_error),
                "metrics": {
                    "v2_f1": 0,
                    "v2_rouge_l": 0,
                    "post_repair_faithfulness": 0,
                    "post_repair_coverage": 0,
                    "meta_audit_leak": False,
                    "abstention_language": False,
                    "old_no_repair_f1": aligned_no[qid]["metrics"]["f1"],
                    "old_full_f1": aligned_full[qid]["metrics"]["f1"],
                    "delta_v2_vs_no_repair_f1": -aligned_no[qid]["metrics"]["f1"],
                    "delta_v2_vs_v1_full_f1": -aligned_full[qid]["metrics"]["f1"],
                    "delta_v1_full_vs_no_repair_f1": aligned_full[qid]["metrics"]["f1"] - aligned_no[qid]["metrics"]["f1"],
                },
                "run_profile": RUN_PROFILE,
            }
        save_progress(output_file, rows_by_id, target_ids)

    save_progress(output_file, rows_by_id, target_ids)
    payload = load_json(output_file)
    report_file = output_file.with_suffix(".md")
    write_report(report_file, payload)
    agg = payload.get("aggregated", {})
    log(
        f"Saved {output_file}: n={payload.get('sample_size')} v2F1={agg.get('mean_v2_f1', 0):.4f} "
        f"postFaith={agg.get('mean_post_repair_faithfulness', 0):.4f}"
    )
    log(f"Report: {report_file}")
    return output_file


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-mode", choices=["triggered", "losses", "all", "ids"], default="triggered")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--question-timeout", type=int, default=240)
    args = parser.parse_args()

    LOGFILE.write_text("", encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    await run_targets(args)


if __name__ == "__main__":
    asyncio.run(main())
