#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit final abstention and low-faithfulness rows from full_v2_rescue."""
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

from experiments.run_enhanced_experiments import get_api_config, load_ultradomain_full  # noqa: E402
from experiments.run_repair_v2_postverify import (  # noqa: E402
    detect_abstention_language,
    detect_meta_audit_leak,
    parse_json_object,
)


load_dotenv(ROOT_DIR / ".env")

ALIGNED_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
RESCUE_FILE = ALIGNED_DIR / "full_v2_rescue_aligned_n130.json"
FULL_V2_FILE = ALIGNED_DIR / "full_v2_aligned_n130.json"
FULL_V1_FILE = ALIGNED_DIR / "full_aligned_n130.json"
NO_REPAIR_FILE = ALIGNED_DIR / "w_o_repair_aligned_n130.json"
LOGFILE = ROOT_DIR / "logs" / "full_v2_rescue_quality_audit.log"

ISSUE_TYPES = {
    "verifier_false_negative",
    "evidence_gap",
    "answer_undercoverage",
    "overconservative_abstention",
    "reasonable_abstention",
    "answer_hallucination",
    "metric_mismatch",
    "rescue_gate_overreach",
    "mixed",
}

ACTION_TYPES = {
    "keep_full_v2_rescue",
    "fix_postverifier",
    "tighten_abstention_gate",
    "rerun_with_more_context",
    "improve_evidence_extraction",
    "improve_answer_coverage",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def compact(text: str, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:limit]


def bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, number))


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
        "gold_in_context_score": bounded_float(parsed.get("gold_in_context_score", 0.0)),
        "answer_support_score": bounded_float(parsed.get("answer_support_score", 0.0)),
        "answer_completeness_score": bounded_float(parsed.get("answer_completeness_score", 0.0)),
        "verifier_assessment_correct": bool(parsed.get("verifier_assessment_correct", False)),
        "abstention_is_justified": bool(parsed.get("abstention_is_justified", False)),
        "rescue_gate_was_appropriate": bool(parsed.get("rescue_gate_was_appropriate", True)),
        "short_reason": compact(parsed.get("short_reason", ""), 500),
        "raw_audit": raw_text,
    }


async def chat_completion(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 550,
):
    return await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def row_maps() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        "rescue": {r["question_id"]: r for r in load_json(RESCUE_FILE)["results"] if "error" not in r},
        "full_v2": {r["question_id"]: r for r in load_json(FULL_V2_FILE)["results"] if "error" not in r},
        "full_v1": {r["question_id"]: r for r in load_json(FULL_V1_FILE)["results"] if "error" not in r},
        "no_repair": {r["question_id"]: r for r in load_json(NO_REPAIR_FILE)["results"] if "error" not in r},
        "questions": {q["question_id"]: q for q in load_ultradomain_full()},
    }


def target_ids(maps: Dict[str, Dict[str, Dict[str, Any]]], mode: str, explicit_ids: List[str]) -> List[str]:
    rows = maps["rescue"]
    low_faith = [qid for qid, row in rows.items() if row["metrics"].get("faithfulness", 1.0) < 0.9]
    abstention = [qid for qid, row in rows.items() if detect_abstention_language(row.get("prediction", ""))]
    if mode == "low-faith":
        ids = low_faith
    elif mode == "abstention":
        ids = abstention
    elif mode == "flags":
        ids = sorted(set(low_faith) | set(abstention), key=lambda value: int(value.split("_")[1]))
    elif mode == "ids":
        ids = explicit_ids
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return ids


async def audit_one(qid: str, maps: Dict[str, Dict[str, Dict[str, Any]]], config: Dict[str, str]) -> Dict[str, Any]:
    row = maps["rescue"][qid]
    question = maps["questions"][qid]
    full_v2 = maps["full_v2"].get(qid, {})
    full_v1 = maps["full_v1"].get(qid, {})
    no_repair = maps["no_repair"].get(qid, {})
    metrics = row["metrics"]
    trace = row.get("trace", {})
    flags = []
    if metrics.get("faithfulness", 1.0) < 0.9:
        flags.append("low_faithfulness")
    if detect_abstention_language(row.get("prediction", "")):
        flags.append("final_abstention")
    if metrics.get("abstention_rescue_applied"):
        flags.append("rescue_applied")

    baseline_answer = trace.get("baseline_full_v2_answer", full_v2.get("prediction", ""))
    prompt = f"""You are auditing one retrieval-QA experiment result from the full_v2_rescue runner.

Classify the main residual issue. Use the context excerpt as the evidence available to the pipeline. Be strict, but do not penalize a correct answer merely because it is phrased differently from the gold answer.

Allowed issue_type values:
- verifier_false_negative: final answer is mostly supported and complete, but verifier marked it too low.
- evidence_gap: context excerpt lacks the gold-answer facts, so the model could not reasonably answer fully.
- answer_undercoverage: context has the needed facts, but final answer omitted important gold facts.
- overconservative_abstention: context has enough answer facts, but final answer refuses or says information is unavailable.
- reasonable_abstention: context genuinely lacks key answer facts and final answer appropriately refuses/hedges.
- answer_hallucination: final answer adds unsupported or contradicted claims.
- metric_mismatch: answer is substantially correct but local F1/ROUGE under-rewarded it.
- rescue_gate_overreach: rescue was triggered by broad abstention language but worsened a useful baseline answer.
- mixed: more than one issue is important.

Allowed recommended_action values:
- keep_full_v2_rescue
- fix_postverifier
- tighten_abstention_gate
- rerun_with_more_context
- improve_evidence_extraction
- improve_answer_coverage
- manual_review

Question ID: {qid}
Flags: {', '.join(flags) or 'none'}
Domain: {row.get('domain', '')}

Question:
{question['question']}

Gold answer:
{compact(question.get('gold_answer', row.get('gold_answer', '')), 1200)}

Context excerpt used by the normal aligned profile:
{compact(question.get('context', ''), 5000)}

No-repair answer, F1={no_repair.get('metrics', {}).get('f1', 0.0):.4f}:
{compact(no_repair.get('prediction', ''), 800)}

V1 full answer, F1={full_v1.get('metrics', {}).get('f1', 0.0):.4f}:
{compact(full_v1.get('prediction', ''), 800)}

Existing full_v2 answer, F1={full_v2.get('metrics', {}).get('f1', 0.0):.4f}:
{compact(full_v2.get('prediction', ''), 900)}

Same-run full_v2 baseline answer before rescue:
{compact(baseline_answer, 900)}

Final full_v2_rescue answer, F1={metrics.get('f1', 0.0):.4f}, faithfulness={metrics.get('faithfulness', 0.0):.4f}, rescue_applied={metrics.get('abstention_rescue_applied', False)}:
{compact(row.get('prediction', ''), 1200)}

Verifier output:
{compact(trace.get('raw_postverification', trace.get('raw_verification', '')), 900)}

Rescue verifier output:
{compact(trace.get('raw_rescue_postverification', ''), 900)}

Return JSON only:
{{
  "issue_type": "one allowed issue_type",
  "recommended_action": "one allowed recommended_action",
  "gold_in_context_score": 0.0,
  "answer_support_score": 0.0,
  "answer_completeness_score": 0.0,
  "verifier_assessment_correct": true,
  "abstention_is_justified": false,
  "rescue_gate_was_appropriate": true,
  "short_reason": "brief reason in one English sentence"
}}"""
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    response = await chat_completion(client, config["model"], prompt)
    audit = normalize_audit(response.choices[0].message.content)
    return {
        "question_id": qid,
        "question": row["question"],
        "domain": row.get("domain", ""),
        "flags": flags,
        "metrics": metrics,
        "prediction": row.get("prediction", ""),
        "gold_answer": row.get("gold_answer", ""),
        "audit": audit,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    issues = Counter(row["audit"]["issue_type"] for row in valid)
    actions = Counter(row["audit"]["recommended_action"] for row in valid)
    return {
        "sample_size": len(rows),
        "valid_sample_size": len(valid),
        "error_count": len(rows) - len(valid),
        "issue_counts": dict(issues),
        "action_counts": dict(actions),
        "mean_gold_in_context_score": statistics.mean(row["audit"]["gold_in_context_score"] for row in valid) if valid else 0.0,
        "mean_answer_support_score": statistics.mean(row["audit"]["answer_support_score"] for row in valid) if valid else 0.0,
        "mean_answer_completeness_score": statistics.mean(row["audit"]["answer_completeness_score"] for row in valid) if valid else 0.0,
        "verifier_assessment_correct_count": sum(1 for row in valid if row["audit"]["verifier_assessment_correct"]),
        "rescue_gate_appropriate_count": sum(1 for row in valid if row["audit"]["rescue_gate_was_appropriate"]),
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    agg = payload["aggregated"]
    lines = [
        "# Full V2 Rescue Quality Flag Audit",
        "",
        f"> Generated: {payload['created_at']}",
        f"> Target mode: {payload['target_mode']}; rows audited: {agg['valid_sample_size']}/{payload['target_sample_size']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Error count | {agg['error_count']} |",
        f"| Mean gold-in-context score | {agg['mean_gold_in_context_score']:.3f} |",
        f"| Mean answer support score | {agg['mean_answer_support_score']:.3f} |",
        f"| Mean answer completeness score | {agg['mean_answer_completeness_score']:.3f} |",
        f"| Verifier assessment correct | {agg['verifier_assessment_correct_count']} |",
        f"| Rescue gate appropriate | {agg['rescue_gate_appropriate_count']} |",
        "",
        "## Issue Counts",
        "",
        "| Issue | Count |",
        "|---|---:|",
    ]
    for issue, count in sorted(agg["issue_counts"].items()):
        lines.append(f"| {issue} | {count} |")
    lines.extend(["", "## Recommended Actions", "", "| Action | Count |", "|---|---:|"])
    for action, count in sorted(agg["action_counts"].items()):
        lines.append(f"| {action} | {count} |")
    lines.extend([
        "",
        "## Rows",
        "",
        "| ID | Flags | F1 | Faith | Rescue | Issue | Action | Reason |",
        "|---|---|---:|---:|---:|---|---|---|",
    ])
    for row in payload["results"]:
        if "error" in row:
            lines.append(f"| {row['question_id']} | error | 0.0000 | 0.00 | False | error | manual_review | {row['error']} |")
            continue
        metrics = row["metrics"]
        audit = row["audit"]
        reason = str(audit.get("short_reason", "")).replace("|", "/")
        lines.append(
            f"| {row['question_id']} | {', '.join(row['flags'])} | {metrics.get('f1', 0.0):.4f} | "
            f"{metrics.get('faithfulness', 0.0):.2f} | {metrics.get('abstention_rescue_applied', False)} | "
            f"{audit['issue_type']} | {audit['recommended_action']} | {reason} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> Path:
    maps = row_maps()
    ids = target_ids(maps, args.target_mode, args.ids or [])
    if args.limit:
        ids = ids[: args.limit]
    output = ALIGNED_DIR / f"full_v2_rescue_quality_audit_{args.target_mode}_n{len(ids)}.json"
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    if output.exists() and not args.force:
        existing = load_json(output)
        rows_by_id = {row["question_id"]: row for row in existing.get("results", [])}
        log(f"Loaded existing audit rows: {len(rows_by_id)}/{len(ids)}")

    config = get_api_config()
    if not config.get("api_key"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    log("=" * 70)
    log(f"FULL_V2_RESCUE QUALITY AUDIT: {args.target_mode} n={len(ids)}")
    log("=" * 70)

    for index, qid in enumerate(ids, 1):
        existing = rows_by_id.get(qid)
        if existing and ("error" not in existing or not args.retry_errors):
            log(f"[{index}/{len(ids)}] SKIP {qid}: already complete")
            continue
        last_error: Optional[Exception] = None
        for attempt in range(1, args.max_retries + 1):
            try:
                log(f"[{index}/{len(ids)}] {qid}: {maps['rescue'][qid]['question'][:72]}...")
                row = await asyncio.wait_for(audit_one(qid, maps, config), timeout=args.question_timeout)
                rows_by_id[qid] = row
                audit = row["audit"]
                log(f"  OK issue={audit['issue_type']} action={audit['recommended_action']}")
                break
            except Exception as exc:
                last_error = exc
                log(f"  ERROR attempt {attempt}/{args.max_retries}: {exc}")
                if attempt < args.max_retries:
                    await asyncio.sleep(5 * attempt)
        else:
            rows_by_id[qid] = {
                "question_id": qid,
                "question": maps["rescue"].get(qid, {}).get("question", ""),
                "error": str(last_error),
            }

        rows = [rows_by_id[qid] for qid in ids if qid in rows_by_id]
        payload = {
            "experiment": "full_v2_rescue_quality_audit",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_mode": args.target_mode,
            "target_sample_size": len(ids),
            "source_file": str(RESCUE_FILE.relative_to(ROOT_DIR)),
            "aggregated": aggregate(rows),
            "results": rows,
        }
        write_json(output, payload)

    payload = load_json(output)
    report = output.with_suffix(".md")
    write_report(report, payload)
    log(f"Saved {output}")
    log(f"Report {report}")
    return output


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-mode", choices=["flags", "low-faith", "abstention", "ids"], default="flags")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--question-timeout", type=int, default=180)
    args = parser.parse_args()

    LOGFILE.write_text("", encoding="utf-8")
    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
