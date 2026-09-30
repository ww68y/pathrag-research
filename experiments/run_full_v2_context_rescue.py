#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a long-context rescue pass for full_v2 quality-flagged rows."""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from experiments.run_enhanced_experiments import (  # noqa: E402
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
from experiments.run_repair_v2_postverify import (  # noqa: E402
    detect_abstention_language,
    detect_meta_audit_leak,
    normalize_final_answer,
    parse_json_object,
)


load_dotenv(ROOT_DIR / ".env")

ALIGNED_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
FULL_V2_FILE = ALIGNED_DIR / "full_v2_aligned_n130.json"
QUALITY_AUDIT_FILE = ALIGNED_DIR / "full_v2_quality_audit_flags_n18.json"
LOGFILE = ROOT_DIR / "logs" / "full_v2_context_rescue.log"

RUN_PROFILE = {
    "name": "full_v2_context_rescue",
    "evidence_context_chars": 40000,
    "answer_context_chars": 6000,
    "verification_evidence_chars": 3500,
    "repair_evidence_chars": 3500,
    "postverify_evidence_chars": 3500,
    "evidence_max_tokens": 900,
    "answer_max_tokens": 450,
    "verification_max_tokens": 320,
    "repair_max_tokens": 380,
    "postverify_max_tokens": 280,
    "temperature": 0.0,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


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


def parse_verification(raw_text: str) -> Dict[str, Any]:
    parsed = parse_json_object(raw_text)
    return {
        "supported_claims": int(parsed.get("supported_claims", 0) or 0),
        "unsupported_claims": int(parsed.get("unsupported_claims", 0) or 0),
        "contradicted_claims": int(parsed.get("contradicted_claims", 0) or 0),
        "faithfulness": float(parsed.get("faithfulness_score", 0.0) or 0.0),
        "coverage": float(parsed.get("coverage_score", 0.0) or 0.0),
        "raw": raw_text,
    }


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_name(target_mode: str, count: int, profile: Dict[str, Any]) -> str:
    return f"{target_mode}_n{count}_ctx{profile['evidence_context_chars']}"


def select_target_ids(target_mode: str, explicit_ids: List[str]) -> List[str]:
    full_v2_rows = load_json(FULL_V2_FILE)["results"]
    if target_mode == "ids":
        return explicit_ids
    low = [
        r["question_id"]
        for r in full_v2_rows
        if r["metrics"].get("repair_triggered")
        and r["metrics"].get("post_repair_faithfulness", r["metrics"].get("faithfulness", 1.0)) < 0.9
    ]
    abst = [r["question_id"] for r in full_v2_rows if r["metrics"].get("abstention_language")]
    if target_mode == "flags":
        return sorted(set(low) | set(abst), key=lambda qid: int(qid.split("_")[1]))
    if target_mode == "low-postfaith":
        return low
    if target_mode == "abstention":
        return abst
    if target_mode == "audit-actions":
        audit = load_json(QUALITY_AUDIT_FILE)["results"]
        target_actions = {"rerun_with_more_context", "improve_evidence_extraction"}
        return [
            row["question_id"]
            for row in audit
            if "error" not in row and row["audit"].get("recommended_action") in target_actions
        ]
    raise ValueError(f"Unsupported target mode: {target_mode}")


async def run_one(
    question: Dict[str, Any],
    baseline_row: Dict[str, Any],
    config: Dict[str, str],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    model = config["model"]
    qid = question["question_id"]
    qtext = question["question"]
    context = question["context"]
    start = time.time()

    evidence_prompt = f"""Extract answer-bearing evidence snippets for the question.

Rules:
- Search the full provided context excerpt, not only the beginning.
- Return 8-12 snippets if available.
- Prefer snippets that directly contain the entities and facts needed for the gold-style answer.
- If the excerpt lacks the answer, say [NO_RELEVANT_EVIDENCE] and briefly name what is missing.
- Prefix snippets [E1], [E2], etc.

Context excerpt:
{context[:profile['evidence_context_chars']]}

Question:
{qtext}

Evidence snippets:"""
    evidence_response = await chat_completion(
        client,
        model,
        evidence_prompt,
        profile["temperature"],
        profile["evidence_max_tokens"],
    )
    evidence = evidence_response.choices[0].message.content.strip()

    answer_prompt = f"""Answer the question using the context and extracted evidence.

Rules:
- Use only supported facts.
- Prefer a complete answer over a terse answer.
- If evidence is missing, give the best supported answer and state only the missing part concisely.
- Do not mention evidence IDs in the final answer.

Context excerpt:
{context[:profile['answer_context_chars']]}

Extracted evidence:
{evidence}

Question:
{qtext}

Answer:"""
    answer_response = await chat_completion(
        client,
        model,
        answer_prompt,
        profile["temperature"],
        profile["answer_max_tokens"],
    )
    original_answer = normalize_final_answer(answer_response.choices[0].message.content)

    verify_prompt = f"""Verify the answer against the extracted evidence. Output JSON only.

Question: {qtext}

Answer: {original_answer[:1200]}

Evidence:
{evidence[:profile['verification_evidence_chars']]}

Return JSON only:
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
        profile["temperature"],
        profile["verification_max_tokens"],
    )
    verification = parse_verification(verify_response.choices[0].message.content)

    repair_triggered = verification["unsupported_claims"] > 0 or verification["contradicted_claims"] > 0
    final_answer = original_answer
    if repair_triggered:
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
{qtext}

Original answer:
{original_answer[:1200]}

Previous verification summary:
unsupported_claims={verification['unsupported_claims']}; contradicted_claims={verification['contradicted_claims']}
{verification['raw'][:900]}

Evidence:
{evidence[:profile['repair_evidence_chars']]}

Final answer only:"""
        repair_response = await chat_completion(
            client,
            model,
            repair_prompt,
            profile["temperature"],
            profile["repair_max_tokens"],
        )
        final_answer = normalize_final_answer(repair_response.choices[0].message.content)

    postverify_prompt = f"""Verify the final answer against the extracted evidence. Output JSON only.

Question: {qtext}

Final answer: {final_answer[:1200]}

Evidence:
{evidence[:profile['postverify_evidence_chars']]}

Return JSON only:
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
        profile["temperature"],
        profile["postverify_max_tokens"],
    )
    postverification = parse_verification(postverify_response.choices[0].message.content)

    gold = question["gold_answer"]
    f1 = compute_f1(final_answer, gold)
    rouge_l = compute_rouge_l(final_answer, gold)
    return {
        "question_id": qid,
        "question": qtext,
        "gold_answer": gold,
        "domain": question.get("domain", ""),
        "dataset": question.get("dataset", "ultradomain"),
        "baseline_full_v2_prediction": baseline_row.get("prediction", ""),
        "prediction": final_answer,
        "evidence": evidence,
        "method": "full_v2_context_rescue",
        "metrics": {
            "f1": f1,
            "rouge_l": rouge_l,
            "exact_match": compute_exact_match(final_answer, gold),
            "contains": compute_contains(final_answer, gold),
            "runtime_seconds": time.time() - start,
            "faithfulness": postverification["faithfulness"],
            "coverage": postverification["coverage"],
            "repair_triggered": repair_triggered,
            "pre_repair_faithfulness": verification["faithfulness"],
            "pre_repair_coverage": verification["coverage"],
            "post_repair_faithfulness": postverification["faithfulness"],
            "post_repair_coverage": postverification["coverage"],
            "supported_claims": verification["supported_claims"],
            "unsupported_claims": verification["unsupported_claims"],
            "contradicted_claims": verification["contradicted_claims"],
            "post_supported_claims": postverification["supported_claims"],
            "post_unsupported_claims": postverification["unsupported_claims"],
            "post_contradicted_claims": postverification["contradicted_claims"],
            "meta_audit_leak": detect_meta_audit_leak(final_answer),
            "abstention_language": detect_abstention_language(final_answer),
            "baseline_full_v2_f1": baseline_row["metrics"]["f1"],
            "baseline_full_v2_rouge_l": baseline_row["metrics"]["rouge_l"],
            "baseline_full_v2_faithfulness": baseline_row["metrics"].get("faithfulness", 0.0),
            "delta_vs_baseline_f1": f1 - baseline_row["metrics"]["f1"],
            "delta_vs_baseline_rouge_l": rouge_l - baseline_row["metrics"]["rouge_l"],
        },
        "trace": {
            "raw_verification": verification["raw"],
            "raw_postverification": postverification["raw"],
            "profile": profile,
        },
        "run_profile": profile,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    if not valid:
        return {}
    metrics = [row["metrics"] for row in valid]
    return {
        "sample_size": len(rows),
        "valid_sample_size": len(valid),
        "error_count": len(rows) - len(valid),
        "mean_f1": mean([m["f1"] for m in metrics]),
        "mean_baseline_full_v2_f1": mean([m["baseline_full_v2_f1"] for m in metrics]),
        "mean_delta_vs_baseline_f1": mean([m["delta_vs_baseline_f1"] for m in metrics]),
        "mean_rouge_l": mean([m["rouge_l"] for m in metrics]),
        "mean_delta_vs_baseline_rouge_l": mean([m["delta_vs_baseline_rouge_l"] for m in metrics]),
        "mean_faithfulness": mean([m["faithfulness"] for m in metrics]),
        "mean_coverage": mean([m["coverage"] for m in metrics]),
        "repair_rate": mean([1.0 if m["repair_triggered"] else 0.0 for m in metrics]),
        "meta_audit_leak_rate": mean([1.0 if m["meta_audit_leak"] else 0.0 for m in metrics]),
        "abstention_language_rate": mean([1.0 if m["abstention_language"] else 0.0 for m in metrics]),
        "improved_vs_baseline": sum(1 for m in metrics if m["delta_vs_baseline_f1"] > 1e-9),
        "worse_vs_baseline": sum(1 for m in metrics if m["delta_vs_baseline_f1"] < -1e-9),
    }


def paired_stats(rows: List[Dict[str, Any]], metric: str, baseline_metric: str) -> Dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    if len(valid) < 2:
        return {}
    left = [row["metrics"][metric] for row in valid]
    right = [row["metrics"][baseline_metric] for row in valid]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "metric": metric,
        "baseline_metric": baseline_metric,
        "n": len(valid),
        "mean_diff": mean(diffs),
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload["results"]
    agg = payload["aggregated"]
    stats = payload["stat_tests"]
    lines = [
        "# Full V2 Long-Context Rescue",
        "",
        f"> Generated: {payload['created_at']}",
        f"> Target mode: {payload['target_mode']}; rows: {agg.get('valid_sample_size', 0)}/{payload['target_sample_size']}",
        f"> Evidence context chars: {payload['run_profile']['evidence_context_chars']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean rescue F1 | {agg.get('mean_f1', 0):.4f} |",
        f"| Mean baseline full_v2 F1 | {agg.get('mean_baseline_full_v2_f1', 0):.4f} |",
        f"| Delta vs baseline F1 | {agg.get('mean_delta_vs_baseline_f1', 0):+.4f} |",
        f"| Mean rescue ROUGE-L | {agg.get('mean_rouge_l', 0):.4f} |",
        f"| Delta vs baseline ROUGE-L | {agg.get('mean_delta_vs_baseline_rouge_l', 0):+.4f} |",
        f"| Faithfulness | {agg.get('mean_faithfulness', 0):.4f} |",
        f"| Coverage | {agg.get('mean_coverage', 0):.4f} |",
        f"| Repair rate | {agg.get('repair_rate', 0):.1%} |",
        f"| Meta-audit leak rate | {agg.get('meta_audit_leak_rate', 0):.1%} |",
        f"| Abstention-language rate | {agg.get('abstention_language_rate', 0):.1%} |",
        f"| Improved / worse vs baseline | {agg.get('improved_vs_baseline', 0)} / {agg.get('worse_vs_baseline', 0)} |",
        "",
        "## Paired Tests",
        "",
        "| Metric | Mean Diff | t-p | Wilcoxon-p | Cohen's d | 95% Bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in stats.values():
        if not row:
            continue
        ci = row["bootstrap_ci"]
        lines.append(
            f"| {row['metric']} | {row['mean_diff']:+.4f} | {row['t_p_value']:.4g} | "
            f"{row['wilcoxon_p_value']:.4g} | {row['cohens_d']:+.2f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )
    lines.extend([
        "",
        "## Rows",
        "",
        "| ID | Domain | Baseline F1 | Rescue F1 | Delta | Faith | Coverage | Repair | Abstain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(rows, key=lambda r: r.get("metrics", {}).get("delta_vs_baseline_f1", -999), reverse=True):
        if "error" in row:
            lines.append(f"| {row['question_id']} | error | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | False | False |")
            continue
        m = row["metrics"]
        lines.append(
            f"| {row['question_id']} | {row.get('domain', '')} | {m['baseline_full_v2_f1']:.4f} | "
            f"{m['f1']:.4f} | {m['delta_vs_baseline_f1']:+.4f} | {m['faithfulness']:.2f} | "
            f"{m['coverage']:.2f} | {m['repair_triggered']} | {m['abstention_language']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- This is a targeted diagnostic experiment, not a replacement for the full 130-row main result.",
        "- Positive deltas indicate the original quality flags were largely caused by context truncation or evidence extraction coverage.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> Path:
    profile = dict(RUN_PROFILE)
    profile["evidence_context_chars"] = args.evidence_context_chars
    profile["answer_context_chars"] = args.answer_context_chars
    ids = select_target_ids(args.target_mode, args.ids or [])
    if args.limit:
        ids = ids[: args.limit]

    target_name = safe_name(args.target_mode, len(ids), profile)
    output_file = ALIGNED_DIR / f"full_v2_context_rescue_{target_name}.json"
    questions = {q["question_id"]: q for q in load_ultradomain_full()}
    baseline_rows = {r["question_id"]: r for r in load_json(FULL_V2_FILE)["results"]}
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    if output_file.exists() and not args.force:
        existing = load_json(output_file)
        rows_by_id = {row["question_id"]: row for row in existing.get("results", [])}
        log(f"Loaded existing rows: {len(rows_by_id)}/{len(ids)}")

    config = get_api_config()
    if not config.get("api_key"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    log("=" * 70)
    log(f"FULL_V2 CONTEXT RESCUE: {target_name}")
    log("=" * 70)
    log(f"Model: {config.get('model')}")

    for index, qid in enumerate(ids, 1):
        existing = rows_by_id.get(qid)
        if existing and ("error" not in existing or not args.retry_errors):
            log(f"[{index}/{len(ids)}] SKIP {qid}: already complete")
            continue
        last_error: Optional[Exception] = None
        for attempt in range(1, args.max_retries + 1):
            try:
                log(f"[{index}/{len(ids)}] {qid} {questions[qid].get('domain', '')}: {questions[qid]['question'][:72]}...")
                start = time.time()
                coro = run_one(questions[qid], baseline_rows[qid], config, profile)
                row = await asyncio.wait_for(coro, timeout=args.question_timeout)
                rows_by_id[qid] = row
                m = row["metrics"]
                log(
                    f"  OK base={m['baseline_full_v2_f1']:.3f} rescue={m['f1']:.3f} "
                    f"delta={m['delta_vs_baseline_f1']:+.3f} faith={m['faithfulness']:.2f} "
                    f"abstain={'Y' if m['abstention_language'] else 'N'} time={time.time() - start:.1f}s"
                )
                break
            except Exception as exc:
                last_error = exc
                log(f"  ERROR attempt {attempt}/{args.max_retries}: {exc}")
                if attempt < args.max_retries:
                    await asyncio.sleep(5 * attempt)
        else:
            rows_by_id[qid] = {
                "question_id": qid,
                "question": questions[qid]["question"],
                "domain": questions[qid].get("domain", ""),
                "error": str(last_error),
                "metrics": {
                    "f1": 0.0,
                    "rouge_l": 0.0,
                    "faithfulness": 0.0,
                    "coverage": 0.0,
                    "baseline_full_v2_f1": baseline_rows[qid]["metrics"]["f1"],
                    "baseline_full_v2_rouge_l": baseline_rows[qid]["metrics"]["rouge_l"],
                    "delta_vs_baseline_f1": -baseline_rows[qid]["metrics"]["f1"],
                    "delta_vs_baseline_rouge_l": -baseline_rows[qid]["metrics"]["rouge_l"],
                },
                "run_profile": profile,
            }

        rows = [rows_by_id[qid] for qid in ids if qid in rows_by_id]
        payload = {
            "experiment": "full_v2_context_rescue",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_mode": args.target_mode,
            "target_sample_size": len(ids),
            "run_profile": profile,
            "aggregated": aggregate(rows),
            "stat_tests": {
                "f1": paired_stats(rows, "f1", "baseline_full_v2_f1"),
                "rouge_l": paired_stats(rows, "rouge_l", "baseline_full_v2_rouge_l"),
            },
            "results": rows,
        }
        write_json(output_file, payload)

    payload = load_json(output_file)
    report_file = output_file.with_suffix(".md")
    write_report(report_file, payload)
    agg = payload.get("aggregated", {})
    log(
        f"Saved {output_file}: n={agg.get('valid_sample_size', 0)} "
        f"deltaF1={agg.get('mean_delta_vs_baseline_f1', 0):+.4f}"
    )
    log(f"Report: {report_file}")
    return output_file


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-mode",
        choices=["audit-actions", "flags", "low-postfaith", "abstention", "ids"],
        default="audit-actions",
    )
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--question-timeout", type=int, default=240)
    parser.add_argument("--evidence-context-chars", type=int, default=40000)
    parser.add_argument("--answer-context-chars", type=int, default=6000)
    args = parser.parse_args()

    LOGFILE.write_text("", encoding="utf-8")
    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
