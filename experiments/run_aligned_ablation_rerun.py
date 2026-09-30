#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rerun ablation configs with the original/main experiment context profile.

This script exists because the previous `full` and `w/o_repair` ablation files
were regenerated in fast mode with a shorter context window. It writes aligned
rerun outputs to a separate directory and can resume after interruption.
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

from experiments.run_enhanced_ablation import run_vsr_ablation
from experiments.run_enhanced_experiments import (
    aggregate_results,
    bootstrap_ci,
    cohens_d,
    compute_contains,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    get_api_config,
    paired_t_test,
    wilcoxon_signed_rank_test,
)
from experiments.run_repair_v2_postverify import (
    detect_abstention_language,
    detect_meta_audit_leak,
    normalize_final_answer,
    parse_json_object,
)


ALIGNED_OUTPUT_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
SOURCE_ABLATION_DIR = ROOT_DIR / "results" / "enhanced_ablation"
LOGFILE = ROOT_DIR / "logs" / "aligned_ablation_rerun.log"
DEFAULT_DATASET_FILE = ROOT_DIR / "datasets" / "UltraDomain" / "mix.jsonl"
ACTIVE_DATASET_NAME = "ultradomain_mix"
ACTIVE_DATASET_SLUG = "ultradomain_mix"
ACTIVE_DATASET_FILE = DEFAULT_DATASET_FILE

ALL_CONFIGS = [
    "w/o_all",
    "w/o_evidence_extraction",
    "w/o_verification",
    "w/o_repair",
    "full",
    "full_v2",
    "full_v2_rescue",
]
DEFAULT_RERUN_CONFIGS = ["w/o_repair", "full"]
PAIRABLE_REPAIR_CONFIGS = {"w/o_repair", "full", "full_v2", "full_v2_rescue"}

RUN_PROFILE = {
    "name": "aligned_original_ablation_profile",
    "evidence_context_chars": 3500,
    "answer_context_chars_with_evidence": 3000,
    "answer_context_chars_without_evidence": 4000,
    "verification_context_chars": 1500,
    "repair_context_chars": 1500,
    "evidence_max_tokens": 500,
    "answer_max_tokens": 400,
    "verification_max_tokens": 300,
    "repair_max_tokens": 400,
    "notes": "Matches experiments/run_enhanced_ablation.py; avoids fast-mode CTX_LIMIT=2000.",
}

PAIRED_REPAIR_PROFILE = {
    **RUN_PROFILE,
    "paired_w_o_repair_full": True,
    "notes": (
        "Shares evidence extraction, answer generation, and verification for w/o_repair/full/full_v2; "
        "only the repair step differs. Context lengths match experiments/run_enhanced_ablation.py."
    ),
}

FULL_V2_REPAIR_PROFILE = {
    **PAIRED_REPAIR_PROFILE,
    "repair_prompt_version": "v2_final_answer_only",
    "temperature_repair_v2": 0.0,
    "repair_v2_max_tokens": 360,
    "post_repair_verification": True,
    "post_repair_verification_max_tokens": 260,
    "postverify_schema_fixed": True,
}

FULL_V2_RESCUE_PROFILE = {
    **FULL_V2_REPAIR_PROFILE,
    "name": "full_v2_abstention_rescue_profile",
    "abstention_rescue_enabled": True,
    "rescue_trigger": "abstention_language",
    "rescue_evidence_context_chars": 40000,
    "rescue_answer_context_chars": 6000,
    "rescue_verification_evidence_chars": 3500,
    "rescue_repair_evidence_chars": 3500,
    "rescue_postverify_evidence_chars": 3500,
    "rescue_evidence_max_tokens": 900,
    "rescue_answer_max_tokens": 450,
    "rescue_verification_max_tokens": 320,
    "rescue_repair_max_tokens": 380,
    "rescue_postverify_max_tokens": 280,
    "temperature_rescue": 0.0,
    "notes": (
        "Runs normal full_v2 first. If the final answer contains abstention language, "
        "reruns a 40k-context evidence extraction and v2-style final-answer repair path."
    ),
}

CONTEXT_LIMIT_KEYS = [
    "evidence_context_chars",
    "answer_context_chars_with_evidence",
    "answer_context_chars_without_evidence",
    "verification_context_chars",
    "repair_context_chars",
]
ACTIVE_CONTEXT_LIMITS = {key: RUN_PROFILE[key] for key in CONTEXT_LIMIT_KEYS}


def profile_with_active_context_limits(profile: Dict[str, Any]) -> Dict[str, Any]:
    active = dict(profile)
    for key, value in ACTIVE_CONTEXT_LIMITS.items():
        if key in active:
            active[key] = value
    return active


def active_context_limit(key: str) -> int:
    return int(ACTIVE_CONTEXT_LIMITS.get(key, RUN_PROFILE[key]))


def run_profile_for_config(config_name: str) -> Dict[str, Any]:
    if config_name == "full_v2_rescue":
        return profile_with_active_context_limits(FULL_V2_RESCUE_PROFILE)
    if config_name == "full_v2":
        return profile_with_active_context_limits(FULL_V2_REPAIR_PROFILE)
    if config_name in {"w/o_repair", "full"}:
        return profile_with_active_context_limits(PAIRED_REPAIR_PROFILE)
    return profile_with_active_context_limits(RUN_PROFILE)


def safe_name(config_name: str) -> str:
    return config_name.replace("/", "_")


def parse_line_nos(value: Optional[str], max_count: int) -> Optional[List[int]]:
    if not value:
        return None
    selected: List[int] = []
    seen = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"Invalid descending line range: {token}")
            values = range(start, end + 1)
        else:
            values = [int(token)]
        for index in values:
            if index < 0 or index >= max_count:
                raise ValueError(f"line_no {index} outside dataset range 0-{max_count - 1}")
            if index not in seen:
                selected.append(index)
                seen.add(index)
    if not selected:
        raise ValueError("--line-nos did not select any rows")
    return selected


def dataset_line_no(question: Dict[str, Any]) -> Optional[int]:
    value = question.get("dataset_line_no")
    if isinstance(value, int):
        return value
    question_id = str(question.get("question_id") or "")
    match = re.search(r"(\d+)$", question_id)
    return int(match.group(1)) if match else None


def target_line_nos(target_questions: List[Dict[str, Any]]) -> List[int]:
    line_nos = []
    for question in target_questions:
        line_no = dataset_line_no(question)
        if line_no is not None:
            line_nos.append(line_no)
    return line_nos


def select_target_questions(
    questions: List[Dict[str, Any]],
    limit: Optional[int],
    line_nos: Optional[str],
) -> List[Dict[str, Any]]:
    selected_indices = parse_line_nos(line_nos, len(questions))
    if selected_indices is None:
        return questions[:limit] if limit else questions
    selected = [questions[index] for index in selected_indices]
    return selected[:limit] if limit else selected


def log(message: str) -> None:
    print(message, flush=True)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with LOGFILE.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(message + "\n")


def is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return "402" in text or "balance is insufficient" in text or "insufficient" in text and "balance" in text


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_path, path)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower()).strip("_")
    return slug or "dataset"


def infer_dataset_name(dataset_file: Path) -> str:
    if dataset_file.resolve() == DEFAULT_DATASET_FILE.resolve():
        return "ultradomain_mix"
    return dataset_file.stem


def normalize_context(raw_context: Any) -> str:
    if isinstance(raw_context, str):
        return raw_context
    if isinstance(raw_context, list):
        parts = []
        for item in raw_context:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(
                    str(
                        item.get("text")
                        or item.get("content")
                        or item.get("passage")
                        or item.get("context")
                        or ""
                    )
                )
            else:
                parts.append(str(item))
        return "\n\n".join(part for part in parts if part)
    return "" if raw_context is None else str(raw_context)


def load_ultradomain_jsonl(dataset_file: Path, dataset_name: str) -> List[Dict[str, Any]]:
    dataset_slug = slugify(dataset_name)
    questions: List[Dict[str, Any]] = []
    with dataset_file.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            answers = item.get("answers") or []
            if isinstance(answers, list) and answers:
                gold_answer = str(answers[0])
            else:
                gold_answer = str(item.get("answer") or item.get("gold_answer") or "")
            question_text = str(item.get("input") or item.get("question") or item.get("query") or "")
            context = normalize_context(item.get("context") or item.get("contexts") or item.get("passages"))
            domain = str(item.get("dataset") or item.get("label") or item.get("domain") or dataset_name)
            questions.append(
                {
                    "question_id": f"{dataset_slug}_{index}",
                    "source_question_id": item.get("question_id") or item.get("_id") or item.get("id"),
                    "question": question_text,
                    "gold_answer": gold_answer,
                    "answers": answers,
                    "context": context,
                    "domain": domain,
                    "dataset": dataset_name,
                    "source_dataset": item.get("source_dataset") or item.get("dataset"),
                    "dataset_line_no": index,
                    "source_line_no": item.get("source_line_no"),
                    "source_context_chars": item.get("source_context_chars") or len(context),
                }
            )
    if not questions:
        raise RuntimeError(f"No questions loaded from {dataset_file}")
    return questions


def ordered_results(result_map: Dict[str, Dict[str, Any]], questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [result_map[q["question_id"]] for q in questions if q["question_id"] in result_map]


def result_record(
    question: Dict[str, Any],
    answer: str,
    trace: Dict[str, Any],
    ablation_config: str,
    elapsed: float,
    run_profile: Dict[str, Any] = RUN_PROFILE,
) -> Dict[str, Any]:
    f1 = compute_f1(answer, question["gold_answer"])
    rouge_l = compute_rouge_l(answer, question["gold_answer"])
    em = compute_exact_match(answer, question["gold_answer"])
    contains = compute_contains(answer, question["gold_answer"])
    metrics = {
        "f1": f1,
        "rouge_l": rouge_l,
        "exact_match": em,
        "contains": contains,
        "runtime_seconds": elapsed,
        "faithfulness": trace.get("faithfulness", 0),
        "repair_triggered": trace.get("repair_triggered", False),
        "supported_claims": trace.get("supported_claims", 0),
        "unsupported_claims": trace.get("unsupported_claims", 0),
        "contradicted_claims": trace.get("contradicted_claims", 0),
    }
    cost_accounting = trace.get("cost_accounting") or {}
    if cost_accounting:
        totals = cost_accounting.get("totals") or {}
        metrics["llm_call_count"] = cost_accounting.get("call_count")
        metrics["llm_usage_available"] = cost_accounting.get("usage_available", False)
        metrics["llm_elapsed_seconds"] = totals.get("elapsed_seconds")
        for usage_key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            if totals.get(usage_key):
                metrics[f"llm_{usage_key}"] = totals.get(usage_key)
    for key in [
        "coverage",
        "pre_repair_faithfulness",
        "pre_repair_coverage",
        "post_repair_faithfulness",
        "post_repair_coverage",
        "post_supported_claims",
        "post_unsupported_claims",
        "post_contradicted_claims",
        "repair_performed",
        "repair_prompt_version",
        "meta_audit_leak",
        "abstention_language",
        "abstention_rescue_triggered",
        "abstention_rescue_applied",
        "baseline_full_v2_abstention_language",
        "baseline_full_v2_faithfulness",
        "baseline_full_v2_repair_triggered",
        "rescue_evidence_count",
        "rescue_repair_triggered",
        "rescue_pre_repair_faithfulness",
        "rescue_post_repair_faithfulness",
        "rescue_post_repair_coverage",
    ]:
        if key in trace:
            metrics[key] = trace[key]
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "gold_answer": question["gold_answer"],
        "prediction": answer,
        "method": ablation_config,
        "domain": question.get("domain", ""),
        "dataset": question.get("dataset", "ultradomain"),
        "dataset_line_no": question.get("dataset_line_no"),
        "source_line_no": question.get("source_line_no"),
        "source_question_id": question.get("source_question_id"),
        "metrics": metrics,
        "trace": trace,
        "run_profile": run_profile,
    }


def error_record(
    question: Dict[str, Any],
    ablation_config: str,
    elapsed: float,
    error: Exception,
    run_profile: Dict[str, Any] = RUN_PROFILE,
) -> Dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "gold_answer": question["gold_answer"],
        "prediction": f"Error: {error}",
        "method": ablation_config,
        "domain": question.get("domain", ""),
        "dataset": question.get("dataset", "ultradomain"),
        "dataset_line_no": question.get("dataset_line_no"),
        "source_line_no": question.get("source_line_no"),
        "source_question_id": question.get("source_question_id"),
        "metrics": {
            "f1": 0,
            "rouge_l": 0,
            "exact_match": 0,
            "contains": 0,
            "runtime_seconds": elapsed,
            "faithfulness": 0,
            "repair_triggered": False,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
        },
        "error": str(error),
        "run_profile": run_profile,
    }


def save_config_progress(
    output_file: Path,
    ablation_config: str,
    result_map: Dict[str, Dict[str, Any]],
    target_questions: List[Dict[str, Any]],
    all_questions: List[Dict[str, Any]],
    question_timeout: int,
    run_profile: Dict[str, Any] = RUN_PROFILE,
) -> None:
    results = ordered_results(result_map, all_questions)
    valid_results = [row for row in results if "error" not in row]
    payload = {
        "ablation": ablation_config,
        "ablation_safe_name": safe_name(ablation_config),
        "dataset": ACTIVE_DATASET_NAME,
        "dataset_slug": ACTIVE_DATASET_SLUG,
        "dataset_file": str(ACTIVE_DATASET_FILE),
        "sample_size": len(results),
        "valid_sample_size": len(valid_results),
        "error_count": len(results) - len(valid_results),
        "target_sample_size": len(target_questions),
        "target_question_ids": [question["question_id"] for question in target_questions],
        "target_line_nos": target_line_nos(target_questions),
        "complete": len(valid_results) >= len(target_questions),
        "question_timeout_seconds": question_timeout or None,
        "run_profile": run_profile,
        "aggregated": aggregate_results(valid_results, ablation_config) if valid_results else {},
        "results": results,
    }
    write_json(output_file, payload)


async def run_one_question(
    question: Dict[str, Any],
    config: Dict[str, str],
    ablation_config: str,
    question_timeout: int,
) -> tuple[str, Dict[str, Any]]:
    coro = run_vsr_ablation(question["question"], question["context"], config, ablation_config)
    if question_timeout and question_timeout > 0:
        return await asyncio.wait_for(coro, timeout=question_timeout)
    return await coro


def extract_usage(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    extracted: Dict[str, Optional[int]] = {}
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        value = getattr(usage, key, None)
        if value is not None:
            extracted[key] = int(value)
    return extracted


def summarize_cost_accounting(cost_accounting: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    calls = list((cost_accounting or {}).get("calls") or [])
    by_stage: Dict[str, Dict[str, Any]] = {}
    totals = {
        "elapsed_seconds": 0.0,
        "prompt_chars": 0,
        "max_tokens_requested": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    usage_available = False
    for call in calls:
        stage = str(call.get("stage") or "unknown")
        bucket = by_stage.setdefault(
            stage,
            {
                "call_count": 0,
                "elapsed_seconds": 0.0,
                "prompt_chars": 0,
                "max_tokens_requested": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        bucket["call_count"] += 1
        for key in ["elapsed_seconds", "prompt_chars", "max_tokens_requested"]:
            value = float(call.get(key) or 0) if key == "elapsed_seconds" else int(call.get(key) or 0)
            bucket[key] += value
            totals[key] += value
        usage = call.get("usage") or {}
        if usage:
            usage_available = True
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            value = int(usage.get(key) or 0)
            bucket[key] += value
            totals[key] += value
    for bucket in by_stage.values():
        bucket["elapsed_seconds"] = round(bucket["elapsed_seconds"], 4)
    totals["elapsed_seconds"] = round(float(totals["elapsed_seconds"]), 4)
    return {
        "schema_version": "stage_cost_v1",
        "call_count": len(calls),
        "usage_available": usage_available,
        "totals": totals,
        "by_stage": by_stage,
        "calls": calls,
    }


async def chat_completion(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stage: str = "unknown",
    cost_accounting: Optional[Dict[str, Any]] = None,
):
    started = time.time()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        if cost_accounting is not None:
            cost_accounting.setdefault("calls", []).append(
                {
                    "stage": stage,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens_requested": max_tokens,
                    "prompt_chars": len(prompt),
                    "elapsed_seconds": round(time.time() - started, 4),
                    "error": str(exc),
                }
            )
        raise
    if cost_accounting is not None:
        cost_accounting.setdefault("calls", []).append(
            {
                "stage": stage,
                "model": model,
                "temperature": temperature,
                "max_tokens_requested": max_tokens,
                "prompt_chars": len(prompt),
                "elapsed_seconds": round(time.time() - started, 4),
                "usage": extract_usage(response),
            }
        )
    return response


def parse_verification_json(raw_text: str) -> Dict[str, Any]:
    verification = raw_text.strip()
    if "```json" in verification:
        verification = verification.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in verification:
        verification = verification.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        parsed = json.loads(verification)
    except Exception:
        parsed = {}
    return {
        "faithfulness": float(parsed.get("faithfulness_score", 0.7)),
        "supported_claims": int(parsed.get("supported_claims", 0)),
        "unsupported_claims": int(parsed.get("unsupported_claims", 0)),
        "contradicted_claims": int(parsed.get("contradicted_claims", 0)),
        "raw_verification": raw_text,
    }


def parse_verification_json_with_coverage(raw_text: str) -> Dict[str, Any]:
    parsed = parse_json_object(raw_text)
    return {
        "faithfulness": float(parsed.get("faithfulness_score", 0.0) or 0.0),
        "coverage": float(parsed.get("coverage_score", 0.0) or 0.0),
        "supported_claims": int(parsed.get("supported_claims", 0) or 0),
        "unsupported_claims": int(parsed.get("unsupported_claims", 0) or 0),
        "contradicted_claims": int(parsed.get("contradicted_claims", 0) or 0),
        "raw_verification": raw_text,
    }


async def repair_answer_v2(
    client: AsyncOpenAI,
    model: str,
    question_text: str,
    original_answer: str,
    evidence: str,
    verification: Dict[str, Any],
    cost_accounting: Optional[Dict[str, Any]] = None,
) -> str:
    unsupported = verification["unsupported_claims"]
    contradicted = verification["contradicted_claims"]
    raw_verification = verification.get("raw_verification", "")[:900]
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
{evidence[:active_context_limit('repair_context_chars')]}

Final answer only:"""
    repair_response = await chat_completion(
        client,
        model,
        repair_prompt,
        0.0,
        360,
        stage="repair_v2",
        cost_accounting=cost_accounting,
    )
    return normalize_final_answer(repair_response.choices[0].message.content)


async def postverify_answer(
    client: AsyncOpenAI,
    model: str,
    question_text: str,
    answer: str,
    evidence: str,
    cost_accounting: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    postverify_prompt = f"""Verify the final answer against the evidence. Output JSON only.

Question: {question_text}

Final answer: {answer[:1000]}

Evidence:
{evidence[:active_context_limit('verification_context_chars')]}

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
        0.0,
        260,
        stage="post_repair_verification",
        cost_accounting=cost_accounting,
    )
    raw_text = postverify_response.choices[0].message.content
    parsed = parse_json_object(raw_text)
    return {
        "faithfulness": float(parsed.get("faithfulness_score", 0.0) or 0.0),
        "coverage": float(parsed.get("coverage_score", 0.0) or 0.0),
        "supported_claims": int(parsed.get("supported_claims", 0) or 0),
        "unsupported_claims": int(parsed.get("unsupported_claims", 0) or 0),
        "contradicted_claims": int(parsed.get("contradicted_claims", 0) or 0),
        "raw_postverification": raw_text,
    }


async def run_full_v2_abstention_rescue(
    client: AsyncOpenAI,
    model: str,
    question_text: str,
    context: str,
    baseline_answer: str,
    baseline_trace: Dict[str, Any],
    cost_accounting: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = FULL_V2_RESCUE_PROFILE
    evidence_prompt = f"""Extract answer-bearing evidence snippets for the question.

Rules:
- Search the full provided context excerpt, not only the beginning.
- Return 8-12 snippets if available.
- Prefer snippets that directly contain the entities and facts needed for the gold-style answer.
- If the excerpt lacks the answer, say [NO_RELEVANT_EVIDENCE] and briefly name what is missing.
- Prefix snippets [E1], [E2], etc.

Context excerpt:
{context[:profile['rescue_evidence_context_chars']]}

Question:
{question_text}

Evidence snippets:"""
    evidence_response = await chat_completion(
        client,
        model,
        evidence_prompt,
        profile["temperature_rescue"],
        profile["rescue_evidence_max_tokens"],
        stage="rescue_evidence_extraction",
        cost_accounting=cost_accounting,
    )
    evidence = evidence_response.choices[0].message.content.strip()

    answer_prompt = f"""Answer the question using the context and extracted evidence.

Rules:
- Use only supported facts.
- Prefer a complete answer over a terse answer.
- If evidence is missing, give the best supported answer and state only the missing part concisely.
- Do not mention evidence IDs in the final answer.

Context excerpt:
{context[:profile['rescue_answer_context_chars']]}

Extracted evidence:
{evidence}

Question:
{question_text}

Answer:"""
    answer_response = await chat_completion(
        client,
        model,
        answer_prompt,
        profile["temperature_rescue"],
        profile["rescue_answer_max_tokens"],
        stage="rescue_answer_generation",
        cost_accounting=cost_accounting,
    )
    rescue_original_answer = normalize_final_answer(answer_response.choices[0].message.content)

    verify_prompt = f"""Verify the answer against the extracted evidence. Output JSON only.

Question: {question_text}

Answer: {rescue_original_answer[:1200]}

Evidence:
{evidence[:profile['rescue_verification_evidence_chars']]}

Return one JSON object with this schema. Replace placeholders with your assessed values:
{{
  "supported_claims": <integer count>,
  "unsupported_claims": <integer count>,
  "contradicted_claims": <integer count>,
  "faithfulness_score": <number from 0.0 to 1.0>,
  "coverage_score": <number from 0.0 to 1.0>
}}"""
    verify_response = await chat_completion(
        client,
        model,
        verify_prompt,
        profile["temperature_rescue"],
        profile["rescue_verification_max_tokens"],
        stage="rescue_verification",
        cost_accounting=cost_accounting,
    )
    verification = parse_verification_json_with_coverage(verify_response.choices[0].message.content)

    rescue_repair_triggered = verification["unsupported_claims"] > 0 or verification["contradicted_claims"] > 0
    final_answer = rescue_original_answer
    if rescue_repair_triggered:
        repair_prompt = f"""You are editing an answer for a retrieval QA benchmark.

Task: produce ONLY the final answer to the question.

Hard rules:
- Do not write a verification report.
- Do not mention unsupported claims, contradicted claims, correction steps, evidence IDs, or these instructions.
- Keep all answer content that is directly supported by the evidence.
- Remove or soften claims that are not supported by the evidence.
- If evidence is incomplete, give the best concise answer supported by the evidence; do not add a long refusal.
- Prefer 1 short paragraph. Use at most 4 sentences.

Question:
{question_text}

Original answer:
{rescue_original_answer[:1200]}

Previous verification summary:
unsupported_claims={verification['unsupported_claims']}; contradicted_claims={verification['contradicted_claims']}
{verification['raw_verification'][:900]}

Evidence:
{evidence[:profile['rescue_repair_evidence_chars']]}

Final answer only:"""
        repair_response = await chat_completion(
            client,
            model,
            repair_prompt,
            profile["temperature_rescue"],
            profile["rescue_repair_max_tokens"],
            stage="rescue_repair",
            cost_accounting=cost_accounting,
        )
        final_answer = normalize_final_answer(repair_response.choices[0].message.content)

    postverify_prompt = f"""Verify the final answer against the extracted evidence. Output JSON only.

Question: {question_text}

Final answer: {final_answer[:1200]}

Evidence:
{evidence[:profile['rescue_postverify_evidence_chars']]}

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
        profile["temperature_rescue"],
        profile["rescue_postverify_max_tokens"],
        stage="rescue_postverification",
        cost_accounting=cost_accounting,
    )
    postverification = parse_verification_json_with_coverage(postverify_response.choices[0].message.content)

    baseline_repair_triggered = bool(baseline_trace.get("repair_triggered", False))
    trace = {
        **baseline_trace,
        "faithfulness": postverification["faithfulness"],
        "coverage": postverification["coverage"],
        "supported_claims": verification["supported_claims"],
        "unsupported_claims": verification["unsupported_claims"],
        "contradicted_claims": verification["contradicted_claims"],
        "pre_repair_faithfulness": verification["faithfulness"],
        "pre_repair_coverage": verification["coverage"],
        "post_repair_faithfulness": postverification["faithfulness"],
        "post_repair_coverage": postverification["coverage"],
        "post_supported_claims": postverification["supported_claims"],
        "post_unsupported_claims": postverification["unsupported_claims"],
        "post_contradicted_claims": postverification["contradicted_claims"],
        "repair_triggered": baseline_repair_triggered or rescue_repair_triggered,
        "repair_performed": baseline_repair_triggered or rescue_repair_triggered,
        "repair_prompt_version": "v2_rescue" if rescue_repair_triggered else "v2_rescue_no_repair",
        "meta_audit_leak": detect_meta_audit_leak(final_answer),
        "abstention_language": detect_abstention_language(final_answer),
        "abstention_rescue_triggered": True,
        "abstention_rescue_applied": True,
        "baseline_full_v2_answer": baseline_answer,
        "baseline_full_v2_abstention_language": bool(baseline_trace.get("abstention_language", False)),
        "baseline_full_v2_faithfulness": baseline_trace.get("faithfulness", 0.0),
        "baseline_full_v2_repair_triggered": baseline_repair_triggered,
        "rescue_original_answer": rescue_original_answer,
        "rescue_evidence_count": evidence.count("[E"),
        "rescue_repair_triggered": rescue_repair_triggered,
        "rescue_pre_repair_faithfulness": verification["faithfulness"],
        "rescue_post_repair_faithfulness": postverification["faithfulness"],
        "rescue_post_repair_coverage": postverification["coverage"],
        "raw_rescue_verification": verification["raw_verification"],
        "raw_rescue_postverification": postverification["raw_verification"],
        "rescue_profile": profile,
    }
    return {"answer": final_answer, "trace": trace}


async def run_repair_ablation_pair_inner(
    question: Dict[str, Any],
    config: Dict[str, str],
    requested_configs: Optional[List[str]] = None,
    include_rescue: bool = False,
) -> Dict[str, Any]:
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    model = config["model"]
    context = question["context"]
    question_text = question["question"]
    start = time.time()
    requested = set(requested_configs or PAIRABLE_REPAIR_CONFIGS)
    cost_accounting: Dict[str, Any] = {"schema_version": "stage_cost_v1", "calls": []}
    evidence_context_chars = active_context_limit("evidence_context_chars")
    answer_context_chars = active_context_limit("answer_context_chars_with_evidence")
    verification_context_chars = active_context_limit("verification_context_chars")
    repair_context_chars = active_context_limit("repair_context_chars")

    evidence_prompt = f"""Extract key evidence snippets from the following context that are relevant to the question.

Context: {context[:evidence_context_chars]}

Question: {question_text}

List the 3-5 most relevant evidence snippets (one per line, prefixed with [E1], [E2], etc.):"""
    evidence_response = await chat_completion(
        client,
        model,
        evidence_prompt,
        0.3,
        500,
        stage="evidence_extraction",
        cost_accounting=cost_accounting,
    )
    evidence = evidence_response.choices[0].message.content

    answer_prompt = f"""Based on the following context and extracted evidence, answer the question.

Context: {context[:answer_context_chars]}

Extracted Evidence:
{evidence}

Question: {question_text}

Provide a comprehensive answer supported by the evidence:"""
    answer_response = await chat_completion(
        client,
        model,
        answer_prompt,
        0.3,
        400,
        stage="answer_generation",
        cost_accounting=cost_accounting,
    )
    original_answer = answer_response.choices[0].message.content

    verify_prompt = f"""Verify the following answer against the provided evidence.

Question: {question_text}

Answer: {original_answer[:1000]}

Evidence:
{evidence[:verification_context_chars]}

Check if every claim in the answer is supported by the evidence. Output:
1. supported_claims: number of claims supported
2. unsupported_claims: number not supported
3. contradicted_claims: number contradicted
4. faithfulness_score: 0.0 to 1.0

Output in JSON format only:"""
    verify_response = await chat_completion(
        client,
        model,
        verify_prompt,
        0.0,
        300,
        stage="verification",
        cost_accounting=cost_accounting,
    )
    verification = parse_verification_json(verify_response.choices[0].message.content)
    elapsed_without_repair = time.time() - start
    cost_without_repair = summarize_cost_accounting(cost_accounting)

    common_trace = {
        "faithfulness": verification["faithfulness"],
        "supported_claims": verification["supported_claims"],
        "unsupported_claims": verification["unsupported_claims"],
        "contradicted_claims": verification["contradicted_claims"],
        "evidence_extracted": True,
        "verification_performed": True,
        "evidence_count": evidence.count("[E"),
        "raw_verification": verification["raw_verification"],
        "paired_repair_ablation": True,
    }

    unsupported = verification["unsupported_claims"]
    contradicted = verification["contradicted_claims"]
    repair_triggered = unsupported > 0 or contradicted > 0
    repaired_answer = original_answer
    run_legacy_full_repair = repair_triggered and "full" in requested
    if run_legacy_full_repair:
        repair_prompt = f"""The following answer has {unsupported} unsupported claims and {contradicted} contradicted claims.

Question: {question_text}

Original Answer: {original_answer[:800]}

Evidence:
{evidence[:repair_context_chars]}

Provide a corrected answer that only includes claims supported by the evidence:"""
        repair_response = await chat_completion(
            client,
            model,
            repair_prompt,
            0.3,
            400,
            stage="repair_v1",
            cost_accounting=cost_accounting,
        )
        repaired_answer = repair_response.choices[0].message.content
    elapsed_full_v1 = time.time() - start
    cost_full_v1 = summarize_cost_accounting(cost_accounting)

    repaired_answer_v2 = original_answer
    full_v2_trace = {
        **common_trace,
        "pre_repair_faithfulness": verification["faithfulness"],
        "repair_triggered": repair_triggered,
        "repair_performed": repair_triggered,
        "repair_prompt_version": "v2_final_answer_only" if repair_triggered else "none",
        "meta_audit_leak": detect_meta_audit_leak(original_answer),
        "abstention_language": detect_abstention_language(original_answer),
    }
    if repair_triggered:
        repaired_answer_v2 = await repair_answer_v2(
            client,
            model,
            question_text,
            original_answer,
            evidence,
            verification,
            cost_accounting=cost_accounting,
        )
        postverification = await postverify_answer(
            client,
            model,
            question_text,
            repaired_answer_v2,
            evidence,
            cost_accounting=cost_accounting,
        )
        full_v2_trace.update(
            {
                "faithfulness": postverification["faithfulness"],
                "post_repair_faithfulness": postverification["faithfulness"],
                "post_repair_coverage": postverification["coverage"],
                "post_supported_claims": postverification["supported_claims"],
                "post_unsupported_claims": postverification["unsupported_claims"],
                "post_contradicted_claims": postverification["contradicted_claims"],
                "raw_postverification": postverification["raw_postverification"],
                "meta_audit_leak": detect_meta_audit_leak(repaired_answer_v2),
                "abstention_language": detect_abstention_language(repaired_answer_v2),
            }
        )
    elapsed_full_v2 = time.time() - start

    full_v2_rescue_answer = repaired_answer_v2
    full_v2_rescue_trace = {
        **full_v2_trace,
        "abstention_rescue_triggered": bool(full_v2_trace.get("abstention_language", False)),
        "abstention_rescue_applied": False,
        "baseline_full_v2_answer": repaired_answer_v2,
        "baseline_full_v2_abstention_language": bool(full_v2_trace.get("abstention_language", False)),
        "baseline_full_v2_faithfulness": full_v2_trace.get("faithfulness", 0.0),
        "baseline_full_v2_repair_triggered": bool(full_v2_trace.get("repair_triggered", False)),
        "rescue_trigger": FULL_V2_RESCUE_PROFILE["rescue_trigger"],
    }
    if include_rescue and full_v2_rescue_trace["abstention_rescue_triggered"]:
        rescue = await run_full_v2_abstention_rescue(
            client,
            model,
            question_text,
            context,
            repaired_answer_v2,
            full_v2_trace,
            cost_accounting=cost_accounting,
        )
        full_v2_rescue_answer = rescue["answer"]
        full_v2_rescue_trace = rescue["trace"]

    full_v2_calls = [
        call
        for call in cost_accounting.get("calls", [])
        if call.get("stage") != "repair_v1" and not str(call.get("stage", "")).startswith("rescue_")
    ]
    full_v2_rescue_calls = [
        call for call in cost_accounting.get("calls", []) if call.get("stage") != "repair_v1"
    ]
    full_v2_trace = {
        **full_v2_trace,
        "cost_accounting": summarize_cost_accounting({"calls": full_v2_calls}),
    }
    full_v2_rescue_trace = {
        **full_v2_rescue_trace,
        "cost_accounting": summarize_cost_accounting({"calls": full_v2_rescue_calls}),
    }

    return {
        "w/o_repair": {
            "answer": original_answer,
            "trace": {
                **common_trace,
                "repair_triggered": False,
                "repair_performed": False,
                "cost_accounting": cost_without_repair,
            },
            "elapsed": elapsed_without_repair,
        },
        "full": {
            "answer": repaired_answer,
            "trace": {
                **common_trace,
                "repair_triggered": repair_triggered,
                "repair_performed": run_legacy_full_repair,
                "cost_accounting": cost_full_v1,
            },
            "elapsed": elapsed_full_v1,
        },
        "full_v2": {
            "answer": repaired_answer_v2,
            "trace": full_v2_trace,
            "elapsed": elapsed_full_v2,
            "run_profile": run_profile_for_config("full_v2"),
        },
        "full_v2_rescue": {
            "answer": full_v2_rescue_answer,
            "trace": full_v2_rescue_trace,
            "elapsed": time.time() - start,
            "run_profile": run_profile_for_config("full_v2_rescue"),
        },
    }


async def run_repair_ablation_pair(
    question: Dict[str, Any],
    config: Dict[str, str],
    question_timeout: int,
    paired_configs: Optional[List[str]] = None,
    include_rescue: bool = False,
) -> Dict[str, Any]:
    coro = run_repair_ablation_pair_inner(
        question,
        config,
        requested_configs=paired_configs,
        include_rescue=include_rescue,
    )
    if question_timeout and question_timeout > 0:
        return await asyncio.wait_for(coro, timeout=question_timeout)
    return await coro


async def rerun_repair_pair(
    questions: List[Dict[str, Any]],
    target_questions: List[Dict[str, Any]],
    config: Dict[str, str],
    paired_configs: List[str],
    force: bool,
    retry_errors: bool,
    max_retries: int,
    question_timeout: int,
) -> None:
    outputs = {
        name: ALIGNED_OUTPUT_DIR / f"{safe_name(name)}_aligned_n{len(target_questions)}.json"
        for name in paired_configs
    }
    result_maps: Dict[str, Dict[str, Dict[str, Any]]] = {name: {} for name in paired_configs}

    if not force:
        for name, output_file in outputs.items():
            if output_file.exists():
                existing = load_json(output_file)
                for row in existing.get("results", []):
                    result_maps[name][row.get("question_id", "")] = row
                log(f"Loaded existing {name}: {len(result_maps[name])}/{len(target_questions)} rows")

    log("\n" + "=" * 70)
    log(f"Paired aligned rerun: {' + '.join(paired_configs)} ({len(target_questions)} questions)")
    log("=" * 70)

    quota_exhausted = False
    for index, question in enumerate(target_questions, 1):
        question_id = question["question_id"]
        existing_pair = [result_maps[name].get(question_id) for name in paired_configs]
        pair_complete = all(row and "error" not in row for row in existing_pair)
        pair_has_error = any(row and "error" in row for row in existing_pair)
        if pair_complete or (pair_has_error and not retry_errors):
            state = "already complete" if pair_complete else "existing error; use --retry-errors to rerun"
            log(f"[{index}/{len(target_questions)}] SKIP {question_id}: {state}")
            continue

        log(f"[{index}/{len(target_questions)}] {question_id} {question.get('domain', '')}: {question['question'][:72]}...")
        start = time.time()
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                pair = await run_repair_ablation_pair(
                    question,
                    config,
                    question_timeout,
                    paired_configs=paired_configs,
                    include_rescue="full_v2_rescue" in paired_configs,
                )
                for name in paired_configs:
                    item = pair[name]
                    row = result_record(
                        question,
                        item["answer"],
                        item["trace"],
                        name,
                        item["elapsed"],
                        run_profile=item.get("run_profile", run_profile_for_config(name)),
                    )
                    result_maps[name][question_id] = row
                summary_bits = []
                for name in paired_configs:
                    metrics = result_maps[name][question_id]["metrics"]
                    summary_bits.append(
                        f"{name} F1={metrics['f1']:.3f} Faith={metrics['faithfulness']:.2f}"
                    )
                repair_flag = result_maps[paired_configs[-1]][question_id]["metrics"].get("repair_triggered", False)
                rescue_flag = result_maps[paired_configs[-1]][question_id]["metrics"].get(
                    "abstention_rescue_applied", False
                )
                log(
                    f"  OK {'; '.join(summary_bits)} Repair={'Y' if repair_flag else 'N'} "
                    f"Rescue={'Y' if rescue_flag else 'N'} Time={time.time() - start:.1f}s"
                )
                break
            except Exception as exc:
                last_error = exc
                elapsed = time.time() - start
                log(f"  ERROR attempt {attempt}/{max_retries} after {elapsed:.1f}s: {exc}")
                if is_quota_error(exc):
                    quota_exhausted = True
                    log("  QUOTA exhausted; saving progress and stopping this run early.")
                    break
                if attempt < max_retries:
                    await asyncio.sleep(5 * attempt)
        else:
            assert last_error is not None
            elapsed = time.time() - start
            for name in paired_configs:
                result_maps[name][question_id] = error_record(
                    question,
                    name,
                    elapsed,
                    last_error,
                    run_profile=run_profile_for_config(name),
                )

        for name, output_file in outputs.items():
            save_config_progress(
                output_file,
                name,
                result_maps[name],
                target_questions,
                questions,
                question_timeout,
                run_profile=run_profile_for_config(name),
            )
        if quota_exhausted:
            break

    for name, output_file in outputs.items():
        save_config_progress(
            output_file,
            name,
            result_maps[name],
            target_questions,
            questions,
            question_timeout,
            run_profile=run_profile_for_config(name),
        )
        payload = load_json(output_file)
        agg = payload.get("aggregated", {})
        log(
            f"Saved {output_file}: n={payload.get('sample_size')} F1={agg.get('mean_f1', 0):.4f} "
            f"Faith={agg.get('mean_faithfulness', 0):.4f} Repair={agg.get('repair_rate', 0):.1%}"
        )


async def rerun_config(
    ablation_config: str,
    questions: List[Dict[str, Any]],
    target_questions: List[Dict[str, Any]],
    config: Dict[str, str],
    force: bool,
    retry_errors: bool,
    max_retries: int,
    question_timeout: int,
) -> None:
    output_file = ALIGNED_OUTPUT_DIR / f"{safe_name(ablation_config)}_aligned_n{len(target_questions)}.json"
    result_map: Dict[str, Dict[str, Any]] = {}

    if output_file.exists() and not force:
        existing = load_json(output_file)
        for row in existing.get("results", []):
            result_map[row.get("question_id", "")] = row
        log(f"Loaded existing {ablation_config}: {len(result_map)}/{len(target_questions)} rows")

    log("\n" + "=" * 70)
    log(f"Aligned ablation rerun: {ablation_config} ({len(target_questions)} questions)")
    log("=" * 70)

    quota_exhausted = False
    for index, question in enumerate(target_questions, 1):
        question_id = question["question_id"]
        existing = result_map.get(question_id)
        if existing and (retry_errors or "error" not in existing):
            if "error" not in existing or not retry_errors:
                log(f"[{index}/{len(target_questions)}] SKIP {question_id}: already complete")
                continue

        log(f"[{index}/{len(target_questions)}] {question_id} {question.get('domain', '')}: {question['question'][:72]}...")
        start = time.time()
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                answer, trace = await run_one_question(question, config, ablation_config, question_timeout)
                elapsed = time.time() - start
                row = result_record(question, answer, trace, ablation_config, elapsed)
                result_map[question_id] = row
                metrics = row["metrics"]
                log(
                    f"  OK F1={metrics['f1']:.3f} R-L={metrics['rouge_l']:.3f} "
                    f"Faith={metrics['faithfulness']:.2f} Repair={'Y' if metrics['repair_triggered'] else 'N'} "
                    f"Time={elapsed:.1f}s"
                )
                break
            except Exception as exc:  # Keep long experiments resumable.
                last_error = exc
                elapsed = time.time() - start
                log(f"  ERROR attempt {attempt}/{max_retries} after {elapsed:.1f}s: {exc}")
                if is_quota_error(exc):
                    quota_exhausted = True
                    log("  QUOTA exhausted; saving progress and stopping this run early.")
                    break
                if attempt < max_retries:
                    await asyncio.sleep(5 * attempt)
        else:
            assert last_error is not None
            elapsed = time.time() - start
            result_map[question_id] = error_record(question, ablation_config, elapsed, last_error)

        save_config_progress(output_file, ablation_config, result_map, target_questions, questions, question_timeout)
        if quota_exhausted:
            break

    save_config_progress(output_file, ablation_config, result_map, target_questions, questions, question_timeout)
    payload = load_json(output_file)
    agg = payload.get("aggregated", {})
    log(
        f"Saved {output_file}: n={payload.get('sample_size')} F1={agg.get('mean_f1', 0):.4f} "
        f"Faith={agg.get('mean_faithfulness', 0):.4f} Repair={agg.get('repair_rate', 0):.1%}"
    )


def load_config_results(config_name: str, sample_size: int, target_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target_keys = {paired_row_key(question) for question in target_questions}
    aligned = ALIGNED_OUTPUT_DIR / f"{safe_name(config_name)}_aligned_n{sample_size}.json"
    if aligned.exists():
        rows = load_json(aligned).get("results", [])
        return [row for row in rows if paired_row_key(row) in target_keys]
    if ACTIVE_DATASET_FILE.resolve() != DEFAULT_DATASET_FILE.resolve():
        return []
    source = SOURCE_ABLATION_DIR / f"{safe_name(config_name)}_n130.json"
    if source.exists():
        rows = load_json(source).get("results", [])
        return [row for row in rows if paired_row_key(row) in target_keys]
    return []


def result_source(config_name: str, sample_size: int) -> Optional[str]:
    aligned = ALIGNED_OUTPUT_DIR / f"{safe_name(config_name)}_aligned_n{sample_size}.json"
    if aligned.exists():
        return str(aligned)
    if ACTIVE_DATASET_FILE.resolve() == DEFAULT_DATASET_FILE.resolve():
        source = SOURCE_ABLATION_DIR / f"{safe_name(config_name)}_n130.json"
        if source.exists():
            return str(source)
    return None


def normalize_pair_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def paired_row_key(row: Dict[str, Any]) -> str:
    question = normalize_pair_text(row.get("question"))
    gold_answer = normalize_pair_text(row.get("gold_answer"))
    if question or gold_answer:
        return f"q={question}|a={gold_answer}"

    question_id = str(row.get("question_id") or "")
    match = re.search(r"(\d+)$", question_id)
    if match:
        return f"idx={match.group(1)}"
    return question_id


def aligned_scores(results: List[Dict[str, Any]], metric: str, pair_keys: List[str]) -> List[float]:
    by_key = {paired_row_key(row): row for row in results}
    return [by_key[key]["metrics"][metric] for key in pair_keys if key in by_key]


def add_answer_quality_rates(agg: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return agg
    agg = dict(agg)
    agg["meta_audit_leak_rate"] = statistics.mean(
        1.0
        if row.get("metrics", {}).get("meta_audit_leak", detect_meta_audit_leak(row.get("prediction", "")))
        else 0.0
        for row in rows
    )
    agg["abstention_language_rate"] = statistics.mean(
        1.0
        if row.get("metrics", {}).get("abstention_language", detect_abstention_language(row.get("prediction", "")))
        else 0.0
        for row in rows
    )
    post_faith = [
        row["metrics"]["post_repair_faithfulness"]
        for row in rows
        if "post_repair_faithfulness" in row.get("metrics", {})
    ]
    if post_faith:
        agg["mean_post_repair_faithfulness"] = statistics.mean(post_faith)
    if any("abstention_rescue_applied" in row.get("metrics", {}) for row in rows):
        agg["abstention_rescue_applied_rate"] = statistics.mean(
            1.0 if row.get("metrics", {}).get("abstention_rescue_applied") else 0.0
            for row in rows
        )
    if any("abstention_rescue_triggered" in row.get("metrics", {}) for row in rows):
        agg["abstention_rescue_triggered_rate"] = statistics.mean(
            1.0 if row.get("metrics", {}).get("abstention_rescue_triggered") else 0.0
            for row in rows
        )
    return agg


def rebuild_summary(questions: List[Dict[str, Any]], target_questions: List[Dict[str, Any]]) -> None:
    sample_size = len(target_questions)
    target_keys = [paired_row_key(q) for q in target_questions]
    all_results_raw = {name: load_config_results(name, sample_size, target_questions) for name in ALL_CONFIGS}
    all_error_counts = {
        name: sum(1 for row in rows if "error" in row)
        for name, rows in all_results_raw.items()
    }
    all_results = {
        name: [row for row in rows if "error" not in row]
        for name, rows in all_results_raw.items()
    }
    all_aggregated = {
        name: add_answer_quality_rates(aggregate_results(rows, name), rows) if rows else {}
        for name, rows in all_results.items()
    }

    stat_tests: Dict[str, List[Dict[str, Any]]] = {}
    full_results = all_results.get("full", [])
    full_keys = {paired_row_key(row) for row in full_results}
    for metric in ["f1", "rouge_l", "faithfulness"]:
        tests = []
        for config_name in ALL_CONFIGS:
            if config_name == "full":
                continue
            rows = all_results.get(config_name, [])
            row_keys = {paired_row_key(row) for row in rows}
            common_keys = [key for key in target_keys if key in full_keys and key in row_keys]
            if len(common_keys) < 5:
                continue
            scores_full = aligned_scores(full_results, metric, common_keys)
            scores_abl = aligned_scores(rows, metric, common_keys)
            t_stat, t_p = paired_t_test(scores_full, scores_abl)
            w_stat, w_p = wilcoxon_signed_rank_test(scores_full, scores_abl)
            d = cohens_d(scores_full, scores_abl)
            diffs = [left - right for left, right in zip(scores_full, scores_abl)]
            tests.append(
                {
                    "ablation": config_name,
                    "metric": metric,
                    "n_common": len(common_keys),
                    "mean_diff": statistics.mean(diffs),
                    "t_statistic": t_stat,
                    "t_p_value": t_p,
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_p_value": w_p,
                    "cohens_d": d,
                    "bootstrap_ci": bootstrap_ci(diffs),
                    "significant_at_0.05": min(t_p, w_p) < 0.05,
                }
            )
        stat_tests[metric] = tests

    summary_file = ALIGNED_OUTPUT_DIR / f"aligned_ablation_study_n{sample_size}.json"
    write_json(
        summary_file,
        {
            "experiment": "enhanced_ablation_study_aligned_context",
            "dataset": ACTIVE_DATASET_NAME,
            "dataset_slug": ACTIVE_DATASET_SLUG,
            "dataset_file": str(ACTIVE_DATASET_FILE),
            "sample_size": sample_size,
            "target_question_ids": [question["question_id"] for question in target_questions],
            "target_line_nos": target_line_nos(target_questions),
            "run_profile": RUN_PROFILE,
            "aggregated": all_aggregated,
            "error_counts": all_error_counts,
            "result_sources": {
                name: result_source(name, sample_size)
                for name in ALL_CONFIGS
            },
        },
    )

    stat_file = ALIGNED_OUTPUT_DIR / f"aligned_ablation_stat_tests_n{sample_size}.json"
    write_json(stat_file, stat_tests)
    write_markdown_report(
        ALIGNED_OUTPUT_DIR / f"aligned_ablation_report_n{sample_size}.md",
        sample_size,
        target_questions,
        all_aggregated,
        stat_tests,
    )
    log(f"Summary saved: {summary_file}")
    log(f"Stat tests saved: {stat_file}")


def write_markdown_report(
    path: Path,
    sample_size: int,
    target_questions: List[Dict[str, Any]],
    aggregated: Dict[str, Dict[str, Any]],
    stat_tests: Dict[str, List[Dict[str, Any]]],
) -> None:
    selected_lines = target_line_nos(target_questions)
    lines = [
        "# Aligned VSR-PathRAG Ablation Rerun",
        "",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Dataset: {ACTIVE_DATASET_NAME} ({ACTIVE_DATASET_FILE.name}), n={sample_size}",
        "> Profile: original ablation context lengths, not fast-mode CTX_LIMIT=2000",
    ]
    if selected_lines and selected_lines != list(range(sample_size)):
        lines.append(f"> Selected dataset line_nos: {','.join(str(value) for value in selected_lines)}")
    lines.extend([
        "",
        "## Aggregate Results",
        "",
        "| Config | Rows | Status | F1 | ROUGE-L | EM | Faithfulness | Repair% | MetaLeak% | Abstain% | Runtime(s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ALL_CONFIGS:
        agg = aggregated.get(name, {})
        if not agg:
            lines.append(f"| {name} | 0/{sample_size} | missing | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        rows = int(agg.get("sample_size", 0) or 0)
        status = "complete" if rows >= sample_size else "partial"
        lines.append(
            f"| {name} | {rows}/{sample_size} | {status} | "
            f"{agg.get('mean_f1', 0):.4f} | {agg.get('mean_rouge_l', 0):.4f} | "
            f"{agg.get('mean_em', 0):.4f} | {agg.get('mean_faithfulness', 0):.4f} | "
            f"{agg.get('repair_rate', 0):.1%} | {agg.get('meta_audit_leak_rate', 0):.1%} | "
            f"{agg.get('abstention_language_rate', 0):.1%} | {agg.get('mean_runtime_seconds', 0):.1f} |"
        )

    lines.extend(["", "## Full vs Ablations", ""])
    for metric, tests in stat_tests.items():
        lines.extend([
            f"### {metric}",
            "",
            "| Ablation | n | Mean Diff | t-p | Wilcoxon-p | Cohen's d |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in tests:
            lines.append(
                f"| {row['ablation']} | {row['n_common']} | {row['mean_diff']:+.4f} | "
                f"{row['t_p_value']:.4f} | {row['wilcoxon_p_value']:.4f} | {row['cohens_d']:+.2f} |"
            )
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- The paired w/o_repair/full rerun shares evidence extraction, initial answer generation, and verification; only the repair branch differs.",
        "- full_v2 uses the same shared evidence/answer/verification path but replaces the repair step with the final-answer-only repair_prompt_v2 and post-repair verification.",
        "- full_v2_rescue runs full_v2 first, then applies the 40k-context rescue branch only when the full_v2 answer contains abstention language.",
        "- Faithfulness for full and w/o_repair is the pre-repair verification score; full_v2 reports post-repair faithfulness on repaired rows and pre-repair faithfulness on passthrough rows.",
        "- Rows marked partial are interim progress only and should not be used as final paper evidence.",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    global ALIGNED_OUTPUT_DIR, ACTIVE_CONTEXT_LIMITS, ACTIVE_DATASET_FILE, ACTIVE_DATASET_NAME, ACTIVE_DATASET_SLUG

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_RERUN_CONFIGS, choices=ALL_CONFIGS)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions for smoke tests.")
    parser.add_argument(
        "--line-nos",
        default=None,
        help="Optional zero-based dataset line_no selection, e.g. 5-9 or 0,3,7-10. Applied before --limit.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="JSONL dataset file with input/question, answers/answer, and context fields.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset label to store in outputs. Defaults to the dataset filename stem.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for result files. Non-default datasets use a dataset-specific subdirectory by default.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing aligned outputs for the target sample size.")
    parser.add_argument("--retry-errors", action="store_true", help="Rerun rows already marked with errors.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--question-timeout",
        type=int,
        default=0,
        help="Optional timeout per question in seconds. Default 0 preserves no-timeout original behavior.",
    )
    parser.add_argument("--evidence-context-chars", type=int, default=RUN_PROFILE["evidence_context_chars"])
    parser.add_argument(
        "--answer-context-chars-with-evidence",
        type=int,
        default=RUN_PROFILE["answer_context_chars_with_evidence"],
    )
    parser.add_argument(
        "--answer-context-chars-without-evidence",
        type=int,
        default=RUN_PROFILE["answer_context_chars_without_evidence"],
    )
    parser.add_argument("--verification-context-chars", type=int, default=RUN_PROFILE["verification_context_chars"])
    parser.add_argument("--repair-context-chars", type=int, default=RUN_PROFILE["repair_context_chars"])
    parser.add_argument(
        "--separate-config-runs",
        action="store_true",
        help="Run configs independently instead of sharing common w/o_repair/full steps.",
    )
    args = parser.parse_args()

    dataset_file = args.dataset_file
    if not dataset_file.is_absolute():
        dataset_file = ROOT_DIR / dataset_file
    dataset_file = dataset_file.resolve()
    dataset_name = args.dataset_name or infer_dataset_name(dataset_file)
    dataset_slug = slugify(dataset_name)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ALIGNED_OUTPUT_DIR if dataset_file == DEFAULT_DATASET_FILE.resolve() else ALIGNED_OUTPUT_DIR / dataset_slug
    elif not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir

    ACTIVE_DATASET_FILE = dataset_file
    ACTIVE_DATASET_NAME = dataset_name
    ACTIVE_DATASET_SLUG = dataset_slug
    ALIGNED_OUTPUT_DIR = output_dir.resolve()
    ACTIVE_CONTEXT_LIMITS = {
        "evidence_context_chars": args.evidence_context_chars,
        "answer_context_chars_with_evidence": args.answer_context_chars_with_evidence,
        "answer_context_chars_without_evidence": args.answer_context_chars_without_evidence,
        "verification_context_chars": args.verification_context_chars,
        "repair_context_chars": args.repair_context_chars,
    }

    LOGFILE.write_text("", encoding="utf-8")
    log("=" * 70)
    log("ALIGNED ABLATION RERUN")
    log("=" * 70)

    config = get_api_config()
    log(f"Model: {config.get('model', '')}")
    log(f"API Base: {config.get('base_url', '')}")
    log(f"API key set: {bool(config.get('api_key'))}")
    log(f"Dataset: {ACTIVE_DATASET_NAME}")
    log(f"Dataset file: {ACTIVE_DATASET_FILE}")
    log(f"Output dir: {ALIGNED_OUTPUT_DIR}")
    log(f"Question timeout: {args.question_timeout or 'none'}")
    use_paired = (not args.separate_config_runs) and set(args.configs).issubset(PAIRABLE_REPAIR_CONFIGS)
    if use_paired and "full_v2_rescue" in args.configs:
        active_profile = run_profile_for_config("full_v2_rescue")
    elif use_paired and "full_v2" in args.configs:
        active_profile = run_profile_for_config("full_v2")
    else:
        active_profile = run_profile_for_config("full")
    log(f"Run profile: {active_profile if use_paired else run_profile_for_config('full')}")
    if not config.get("api_key"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    questions = load_ultradomain_jsonl(ACTIVE_DATASET_FILE, ACTIVE_DATASET_NAME)
    target_questions = select_target_questions(questions, args.limit, args.line_nos)
    sample_size = len(target_questions)
    ALIGNED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.line_nos:
        log(f"Selected line_nos: {','.join(str(value) for value in target_line_nos(target_questions))}")

    if use_paired:
        await rerun_repair_pair(
            questions=questions,
            target_questions=target_questions,
            config=config,
            paired_configs=args.configs,
            force=args.force,
            retry_errors=args.retry_errors,
            max_retries=args.max_retries,
            question_timeout=args.question_timeout,
        )
    else:
        for ablation_config in args.configs:
            await rerun_config(
                ablation_config=ablation_config,
                questions=questions,
                target_questions=target_questions,
                config=config,
                force=args.force,
                retry_errors=args.retry_errors,
                max_retries=args.max_retries,
                question_timeout=args.question_timeout,
            )

    rebuild_summary(questions, target_questions)

    log("\nAggregate snapshot:")
    summary = load_json(ALIGNED_OUTPUT_DIR / f"aligned_ablation_study_n{sample_size}.json")
    for name in ALL_CONFIGS:
        agg = summary["aggregated"].get(name, {})
        if agg:
            rescue = ""
            if "abstention_rescue_applied_rate" in agg:
                rescue = f" Rescue={agg.get('abstention_rescue_applied_rate', 0):.1%}"
            log(
                f"  {name:<24} F1={agg.get('mean_f1', 0):.4f} "
                f"Faith={agg.get('mean_faithfulness', 0):.4f} "
                f"Repair={agg.get('repair_rate', 0):.1%}{rescue} n={agg.get('sample_size', 0)}"
            )
        else:
            log(f"  {name:<24} missing")


if __name__ == "__main__":
    asyncio.run(main())
