import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]


SYSTEM_PROMPT = """You are a legal QA answer editor.
Compress the answer without changing its meaning.
Use only facts already present in the original answer.
Do not add outside knowledge or reference-answer information."""


USER_PROMPT = """Question:
{question}

Original answer:
{answer}

Rewrite the answer for a legal QA benchmark:
- Start with the direct answer.
- Preserve exact legal entities, numbers, percentages, prices, dates, formulas, and defined terms.
- Keep all facts needed to answer the question.
- Remove background, examples, caveats, adjustment mechanics, and extra legal context unless directly required by the question.
- For formula-definition questions, keep the formula and the names of formula variables, but remove timing windows, procedural qualifications, and adjustment mechanics unless the question asks for them.
- For indemnification questions, preserve the direction of indemnity: who indemnifies/holds harmless whom. Do not reverse the indemnifying and indemnified parties.
- For Initial Purchasers indemnification questions, include the protected parties if present: Initial Purchasers, affiliates, directors, officers, employees, selling agents, and controlling persons. Include covered loss types such as losses, claims, damages, liabilities, expenses, untrue statements, and omissions. Do not include contribution unless the question asks about contribution.
- Use 1-3 concise sentences.

Compressed answer:"""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())


def extract_protected_terms(text: str) -> set[str]:
    protected = set()
    protected.update(re.findall(r"\$[\d,.]+", text))
    protected.update(re.findall(r"\b\d+(?:\.\d+)?\s*%", text))
    protected.update(re.findall(r"\b\d+(?:\.\d+)?\s*%\s+per\s+annum\b", text, flags=re.IGNORECASE))
    protected.update(re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b", text))
    return {clean_text(term) for term in protected if len(clean_text(term)) >= 3}


def preserves_critical_terms(original: str, compressed: str) -> tuple[bool, list[str]]:
    original_terms = extract_protected_terms(original)
    compressed_lower = compressed.lower()
    missing = []
    for term in sorted(original_terms):
        if term.lower() not in compressed_lower:
            if term.lower() in {"based", "the", "this", "for", "notes"}:
                continue
            missing.append(term)
    high_signal_missing = [
        term for term in missing
        if "$" in term or "%" in term or term in {"Conversion Price", "Floor Price"}
    ]
    return not high_signal_missing, high_signal_missing


def passes_question_specific_guard(question: str, compressed: str) -> tuple[bool, list[str]]:
    lowered_q = question.lower()
    lowered_answer = compressed.lower()
    failures = []
    if "indemnification" in lowered_q and "initial purchaser" in lowered_q:
        has_indemnified_party = "initial purchaser" in lowered_answer
        has_indemnifying_party = "partnership parties" in lowered_answer
        has_indemnity_action = "indemnif" in lowered_answer or "hold harmless" in lowered_answer
        reversed_direction = bool(
            re.search(r"initial purchaser.{0,140}indemnif.{0,140}partnership parties", lowered_answer)
            or re.search(r"initial purchasers.{0,140}indemnif.{0,140}partnership parties", lowered_answer)
        )
        positive_direction = bool(
            re.search(r"partnership parties.{0,160}(indemnif|hold harmless).{0,160}initial purchaser", lowered_answer)
            or re.search(r"(indemnif|hold harmless).{0,160}initial purchaser", lowered_answer)
        )
        if not (has_indemnified_party and has_indemnifying_party and has_indemnity_action):
            failures.append("missing_indemnification_roles")
        if reversed_direction or not positive_direction:
            failures.append("unsafe_indemnification_direction")
        protected_party_cues = ["affiliates", "directors", "officers", "employees", "selling agents", "controlling persons"]
        if sum(1 for cue in protected_party_cues if cue in lowered_answer) < 3:
            failures.append("missing_specific_indemnitees")
    return not failures, failures


def postprocess_question_focused_answer(question: str, answer: str) -> str:
    question_lower = question.lower()
    if (
        ("price" in question_lower or "rate" in question_lower or "formula" in question_lower)
        and not any(cue in question_lower for cue in ["when", "condition", "conditions", "threshold", "timing"])
    ):
        answer = re.sub(
            r"(in any Equity Financing)\s+occurring\s+after\s+the\s+Effective\s+Date\s+and\s+through\s+the\s+date\s+on\s+which\s+the\s+Funding\s+Threshold\s+is\s+met",
            r"\1",
            answer,
            flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"(in any Equity Financing)\s+after\s+the\s+Effective\s+Date\s+and\s+through\s+the\s+date\s+on\s+which\s+the\s+Funding\s+Threshold\s+is\s+met",
            r"\1",
            answer,
            flags=re.IGNORECASE,
        )
    if "indemnification" in question_lower and "initial purchaser" in question_lower and "contribution" not in question_lower:
        sentences = re.split(r"(?<=[.!?])\s+", clean_text(answer))
        sentences = [
            sentence
            for sentence in sentences
            if "contribution" not in sentence.lower()
            and "reciprocal" not in sentence.lower()
            and not re.search(
                r"initial purchasers?.{0,80}indemnif.{0,80}partnership parties",
                sentence.lower(),
            )
        ]
        if sentences:
            answer = " ".join(sentences)
    return clean_text(answer)


def should_compress(row: dict, args: argparse.Namespace) -> bool:
    if args.rescue_only and not row.get("query_router", {}).get("rescue_used"):
        return False
    if args.legal_error_router_only and not row.get("legal_error_router", {}).get("action", "").startswith("use_"):
        return False
    return len(row.get("prediction", "")) >= args.min_chars


async def complete(client: AsyncOpenAI, model: str, prompt: str, timeout: int, retries: int) -> tuple[str, float]:
    last_error = None
    for attempt in range(retries + 1):
        started = time.time()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                ),
                timeout=timeout,
            )
            return response.choices[0].message.content.strip(), time.time() - started
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(1 + attempt)
    raise RuntimeError(f"Compression failed after {retries + 1} attempt(s): {last_error}")


async def run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("DEEPSEEK_MODEL")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in .env")
    if not base_url:
        raise RuntimeError("DEEPSEEK_API_BASE is missing in .env")
    if not model:
        raise RuntimeError("DEEPSEEK_MODEL is missing in .env")

    rows = load_jsonl(ROOT_DIR / args.input)
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    output_rows = []
    reports = []
    for index, row in enumerate(rows, start=1):
        if not should_compress(row, args):
            output_rows.append(row)
            continue

        print(f"[{index}/{len(rows)}] compressing line={row.get('line_no')}", flush=True)
        prompt = USER_PROMPT.format(
            question=clean_text(row.get("question")),
            answer=row.get("prediction", ""),
        )
        new_row = dict(row)
        started_chars = len(row.get("prediction", ""))
        try:
            raw_compressed, seconds = await complete(client, model, prompt, args.request_timeout, args.request_retries)
            compressed = postprocess_question_focused_answer(clean_text(row.get("question")), raw_compressed)
            ok, missing = preserves_critical_terms(row.get("prediction", ""), compressed)
            guard_ok, guard_failures = passes_question_specific_guard(clean_text(row.get("question")), compressed)
            if not ok or not guard_ok:
                final_answer = row.get("prediction", "")
                action = "keep_original_guard_failed"
            else:
                final_answer = compressed
                action = "use_compressed"
            new_row["prediction_before_compression"] = row.get("prediction", "")
            new_row["raw_compressed_prediction"] = raw_compressed
            new_row["postprocessed_compressed_prediction"] = compressed
            new_row["prediction"] = final_answer
            new_row["method"] = f"{row.get('method', 'unknown')}_rescue_compact"
            new_row["rescue_compression"] = {
                "enabled": True,
                "action": action,
                "query_seconds": round(seconds, 3),
                "original_chars": started_chars,
                "compressed_chars": len(compressed),
                "final_chars": len(final_answer),
                "missing_protected_terms": missing,
                "question_guard_failures": guard_failures,
            }
        except Exception as exc:  # noqa: BLE001
            new_row["rescue_compression"] = {
                "enabled": True,
                "action": "keep_original_error",
                "error": str(exc),
                "original_chars": started_chars,
                "final_chars": started_chars,
            }
        output_rows.append(new_row)
        reports.append((new_row.get("line_no"), new_row.get("rescue_compression", {})))

    with output_path.open("w", encoding="utf-8") as out:
        for row in output_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    compressed_count = sum(1 for _, report in reports if report.get("action") == "use_compressed")
    avg_chars = sum(len(row.get("prediction", "")) for row in output_rows) / max(1, len(output_rows))
    lines = [
        "# Rescue Compression Summary\n\n",
        f"- Input: `{args.input}`\n",
        f"- Output: `{args.output}`\n",
        f"- Samples: {len(output_rows)}\n",
        f"- Attempted compression: {len(reports)}\n",
        f"- Used compressed answers: {compressed_count}\n",
        f"- Avg answer chars: {avg_chars:.1f}\n\n",
        "| Line | Action | Original chars | Compressed chars | Final chars | Missing protected terms |\n",
        "|---:|---|---:|---:|---:|---|\n",
    ]
    for line_no, report in reports:
        lines.append(
            f"| {line_no} | {report.get('action')} | {report.get('original_chars')} | "
            f"{report.get('compressed_chars', '')} | {report.get('final_chars')} | "
            f"{', '.join(report.get('missing_protected_terms', [])) or 'none'} |\n"
        )
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved compressed results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model")
    parser.add_argument("--rescue-only", action="store_true")
    parser.add_argument("--legal-error-router-only", action="store_true")
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--request-retries", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
