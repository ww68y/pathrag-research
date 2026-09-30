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


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "through",
    "about",
    "which",
    "what",
    "when",
    "where",
    "how",
    "does",
    "are",
    "can",
    "could",
    "would",
    "should",
    "their",
    "there",
    "these",
    "those",
    "such",
    "also",
    "more",
    "most",
    "some",
    "than",
    "then",
    "been",
    "being",
    "have",
    "has",
    "had",
    "not",
    "but",
    "its",
    "our",
    "your",
    "use",
    "used",
    "using",
}


FACTOR_INTENT_TERMS = {
    "affect",
    "affecting",
    "affected",
    "effect",
    "effects",
    "differ",
    "differs",
    "different",
    "difference",
    "differences",
    "vary",
    "varies",
    "variation",
    "variations",
    "influence",
    "influences",
    "influenced",
    "why",
    "cause",
    "causes",
    "caused",
    "contribute",
    "contributes",
}


FACTOR_CUE_TERMS = {
    "associated",
    "because",
    "cause",
    "causes",
    "chronic",
    "consequence",
    "consequences",
    "contribute",
    "diet",
    "dietary",
    "driven",
    "environment",
    "environmental",
    "factor",
    "factors",
    "genetic",
    "genetics",
    "geography",
    "increase",
    "increases",
    "influence",
    "influences",
    "irreversible",
    "lead",
    "leads",
    "lifestyle",
    "long-term",
    "metabolism",
    "outcome",
    "outcomes",
    "risk",
    "risks",
}


SYSTEM_PROMPT = """You are an evidence-grounded answer editor.
Revise the draft answer using only the provided evidence.
Keep all question-critical facts, remove tangential background, and do not add unsupported claims.
Do not use reference answers or outside knowledge."""


USER_PROMPT = """Question:
{question}

Answer aspects that must be covered if present:
{answer_aspects}

Selected evidence:
{evidence}

Supporting snippets:
{snippets}

Evidence-supported draft facts to preserve if relevant:
{coverage_anchors}

Draft answer:
{draft}

Revision goals:
- Start with the direct answer.
- Preserve factual correctness and question coverage.
- Preserve the coverage anchors when they are relevant to the question and supported by evidence.
- Remove named researchers, broad history, reservoir speculation, extra examples, and caveats unless directly asked.
- For method/treatment questions, keep each distinct supported method.
- For transmission/spread questions, focus on direct contact routes and infected fluids/tissues.
- Use {min_sentences}-{max_sentences} sentences and preferably {min_words}-{max_words} words.

Final revised answer:"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_text(value: object, max_chars: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.replace("<SEP>", "; ").split())
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def extract_terms(text: str) -> set[str]:
    lowered = clean_text(text).lower()
    tokens = re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", lowered)
    return {token for token in tokens if token not in STOPWORDS}


def parse_line_nos(value: str | None) -> set[int] | None:
    if not value:
        return None
    selected = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return selected


def split_sentences(text: str) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def format_aspects(item: dict) -> str:
    aspects = item.get("answer_aspects") or item.get("evidence_trace", {}).get("answer_aspects", [])
    labels = [clean_text(aspect.get("label")) for aspect in aspects if clean_text(aspect.get("label"))]
    return "; ".join(labels) if labels else "none"


def aspect_labels(item: dict) -> list[str]:
    aspects = item.get("answer_aspects") or item.get("evidence_trace", {}).get("answer_aspects", [])
    return [clean_text(aspect.get("label")).lower() for aspect in aspects if clean_text(aspect.get("label"))]


def has_factor_intent(item: dict) -> bool:
    question_terms = extract_terms(item.get("question", ""))
    requirement_terms = set()
    for value in (item.get("requirements") or {}).get("expanded_terms", []):
        requirement_terms.update(extract_terms(value))
    return bool((question_terms | requirement_terms) & FACTOR_INTENT_TERMS)


def selected_evidence_text(item: dict) -> str:
    trace = item.get("evidence_trace") or {}
    chunks = trace.get("supporting_chunks") or []
    edges = trace.get("selected_edges") or []
    parts = []
    for edge in edges:
        parts.extend(
            [
                edge.get("path", ""),
                edge.get("keywords", ""),
                edge.get("description", ""),
            ]
        )
    for chunk in chunks:
        parts.append(chunk.get("snippet", ""))
    return clean_text(" ".join(parts))


def evidence_supported_anchor_sentences(item: dict, args: argparse.Namespace) -> list[str]:
    if not args.coverage_anchors:
        return []
    if args.selective_coverage_anchors and not has_factor_intent(item):
        return []

    draft = item.get("draft_prediction") or item.get("prediction", "")
    sentences = split_sentences(draft)
    if not sentences:
        return []

    evidence_terms = extract_terms(selected_evidence_text(item))
    question_terms = extract_terms(item.get("question", ""))
    requirement_terms = set()
    for value in (item.get("requirements") or {}).get("expanded_terms", []):
        requirement_terms.update(extract_terms(value))
    aspect_terms = set()
    for label in aspect_labels(item):
        aspect_terms.update(extract_terms(label))
    factor_terms = FACTOR_CUE_TERMS if args.factor_aware_anchors and has_factor_intent(item) else set()

    anchors = []
    for index, sentence in enumerate(sentences):
        sentence_terms = extract_terms(sentence)
        if not sentence_terms:
            continue
        evidence_hits = sentence_terms & evidence_terms
        question_hits = sentence_terms & question_terms
        requirement_hits = sentence_terms & requirement_terms
        aspect_hits = sentence_terms & aspect_terms
        factor_hits = sentence_terms & factor_terms

        if len(evidence_hits) < args.anchor_min_evidence_terms:
            continue

        score = (
            len(evidence_hits)
            + 1.5 * len(question_hits)
            + 0.75 * len(requirement_hits)
            + 2.0 * len(aspect_hits)
            + args.anchor_factor_boost * len(factor_hits)
        )
        if score < args.anchor_min_score:
            continue

        anchors.append(
            {
                "index": index,
                "sentence": clean_text(sentence, args.anchor_max_chars),
                "score": score,
                "evidence_hits": len(evidence_hits),
                "question_hits": len(question_hits),
                "requirement_hits": len(requirement_hits),
                "aspect_hits": len(aspect_hits),
                "factor_hits": len(factor_hits),
            }
        )

    anchors.sort(key=lambda item: (-item["score"], item["index"]))
    selected = sorted(anchors[: args.max_anchor_sentences], key=lambda item: item["index"])
    if factor_terms and anchors and not any(item["factor_hits"] for item in selected):
        factor_candidates = [item for item in anchors if item["factor_hits"]]
        if factor_candidates:
            selected = selected[: max(0, args.max_anchor_sentences - 1)] + [factor_candidates[0]]
            selected = sorted({item["index"]: item for item in selected}.values(), key=lambda item: item["index"])
    return [item["sentence"] for item in selected]


def format_coverage_anchors(item: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    anchors = evidence_supported_anchor_sentences(item, args)
    if not anchors:
        return "none", []
    return "\n".join(f"- {anchor}" for anchor in anchors), anchors


def supported_draft_terms(item: dict, draft: str) -> set[str]:
    evidence_terms = extract_terms(selected_evidence_text(item))
    draft_terms = extract_terms(draft)
    question_terms = extract_terms(item.get("question", ""))
    aspect_terms = set()
    for label in aspect_labels(item):
        aspect_terms.update(extract_terms(label))
    shared_terms = draft_terms & evidence_terms
    return shared_terms | (draft_terms & question_terms) | (draft_terms & aspect_terms)


def aspect_preservation_ratio(item: dict, answer: str) -> float:
    labels = aspect_labels(item)
    if not labels:
        return 1.0
    answer_l = clean_text(answer).lower()
    covered = 0
    for label in labels:
        label_terms = extract_terms(label)
        if label and label in answer_l:
            covered += 1
        elif label_terms and len(label_terms & extract_terms(answer_l)) / max(1, len(label_terms)) >= 0.67:
            covered += 1
    return covered / max(1, len(labels))


def guard_refinement(item: dict, refined: str, args: argparse.Namespace) -> tuple[str, dict]:
    draft = item.get("prediction", "")
    if not args.coverage_guard or not refined.strip():
        return refined, {"enabled": bool(args.coverage_guard), "triggered": False}

    draft_terms = supported_draft_terms(item, draft)
    refined_terms = extract_terms(refined)
    preserved = len(draft_terms & refined_terms) / max(1, len(draft_terms))
    aspect_ratio = aspect_preservation_ratio(item, refined)
    compression_ratio = len(refined) / max(1, len(draft))
    stats = item.get("mse_stats") or {}
    min_preserved = args.guard_min_supported_term_ratio
    if stats.get("risk_control_active"):
        min_preserved = min(min_preserved, args.guard_risk_min_supported_term_ratio)
    if stats.get("aspect_control_active"):
        min_preserved = max(min_preserved, args.guard_aspect_min_supported_term_ratio)

    reasons = []
    if preserved < min_preserved:
        reasons.append("low_supported_term_preservation")
    if aspect_ratio < args.guard_min_aspect_ratio:
        reasons.append("low_aspect_preservation")
    if (
        not stats.get("risk_control_active")
        and compression_ratio < args.guard_min_compression_ratio
        and preserved < args.guard_aggressive_compression_term_ratio
    ):
        reasons.append("over_aggressive_compression")

    if reasons:
        return draft, {
            "enabled": True,
            "triggered": True,
            "action": "fallback_to_draft",
            "reasons": reasons,
            "supported_term_preservation": round(preserved, 4),
            "aspect_preservation": round(aspect_ratio, 4),
            "compression_ratio": round(compression_ratio, 4),
            "draft_supported_terms": sorted(draft_terms)[:80],
        }
    return refined, {
        "enabled": True,
        "triggered": False,
        "action": "keep_refined",
        "supported_term_preservation": round(preserved, 4),
        "aspect_preservation": round(aspect_ratio, 4),
        "compression_ratio": round(compression_ratio, 4),
    }


def format_evidence(item: dict, max_edges: int) -> str:
    trace = item.get("evidence_trace") or {}
    edges = trace.get("selected_edges") or []
    prompt_edges = [edge for edge in edges if edge.get("used_in_prompt", True)]
    if not prompt_edges:
        prompt_edges = edges
    lines = []
    for edge in prompt_edges[:max_edges]:
        lines.append(
            "\n".join(
                [
                    f"- {clean_text(edge.get('path'))}",
                    f"  keywords: {clean_text(edge.get('keywords'), 180) or 'N/A'}",
                    f"  evidence: {clean_text(edge.get('description'), 260) or 'N/A'}",
                ]
            )
        )
    return "\n".join(lines) if lines else "No selected evidence."


def format_snippets(item: dict, max_chunks: int, max_chars: int) -> str:
    chunks = item.get("evidence_trace", {}).get("supporting_chunks") or []
    lines = []
    for chunk in chunks[:max_chunks]:
        lines.append(f"- {clean_text(chunk.get('snippet'), max_chars)}")
    return "\n".join(lines) if lines else "No supporting snippets."


def refinement_budget(item: dict, args: argparse.Namespace) -> tuple[int, int, int, int]:
    stats = item.get("mse_stats") or {}
    if stats.get("risk_control_active"):
        return 1, args.risk_max_sentences, args.risk_min_words, args.risk_max_words
    if stats.get("aspect_control_active"):
        return 2, args.aspect_max_sentences, args.aspect_min_words, args.aspect_max_words
    return 2, args.default_max_sentences, args.default_min_words, args.default_max_words


async def complete(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    timeout_seconds: int,
    retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                ),
                timeout=timeout_seconds,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Answer refinement failed after {retries + 1} attempt(s): {last_error}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--line-nos", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-original-on-failure", action="store_true")
    parser.add_argument("--apply-guard-only", action="store_true")
    parser.add_argument("--max-evidence-edges", type=int, default=10)
    parser.add_argument("--max-snippets", type=int, default=3)
    parser.add_argument("--snippet-chars", type=int, default=650)
    parser.add_argument("--default-min-words", type=int, default=75)
    parser.add_argument("--default-max-words", type=int, default=130)
    parser.add_argument("--default-max-sentences", type=int, default=3)
    parser.add_argument("--aspect-min-words", type=int, default=85)
    parser.add_argument("--aspect-max-words", type=int, default=145)
    parser.add_argument("--aspect-max-sentences", type=int, default=3)
    parser.add_argument("--risk-min-words", type=int, default=45)
    parser.add_argument("--risk-max-words", type=int, default=95)
    parser.add_argument("--risk-max-sentences", type=int, default=2)
    parser.add_argument("--coverage-guard", action="store_true")
    parser.add_argument("--guard-min-supported-term-ratio", type=float, default=0.48)
    parser.add_argument("--guard-aspect-min-supported-term-ratio", type=float, default=0.55)
    parser.add_argument("--guard-risk-min-supported-term-ratio", type=float, default=0.35)
    parser.add_argument("--guard-min-aspect-ratio", type=float, default=0.95)
    parser.add_argument("--guard-min-compression-ratio", type=float, default=0.45)
    parser.add_argument("--guard-aggressive-compression-term-ratio", type=float, default=0.62)
    parser.add_argument("--coverage-anchors", action="store_true")
    parser.add_argument("--selective-coverage-anchors", action="store_true")
    parser.add_argument("--factor-aware-anchors", action="store_true")
    parser.add_argument("--max-anchor-sentences", type=int, default=3)
    parser.add_argument("--anchor-min-evidence-terms", type=int, default=3)
    parser.add_argument("--anchor-min-score", type=float, default=5.0)
    parser.add_argument("--anchor-max-chars", type=int, default=260)
    parser.add_argument("--anchor-factor-boost", type=float, default=2.5)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("DEEPSEEK_MODEL")
    if not args.apply_guard_only:
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing in .env")
        if not base_url:
            raise RuntimeError("DEEPSEEK_API_BASE is missing in .env")
        if not model:
            raise RuntimeError("DEEPSEEK_MODEL is missing in .env")

    input_path = ROOT_DIR / args.input
    output_path = ROOT_DIR / args.output
    summary_path = ROOT_DIR / args.summary
    rows = load_jsonl(input_path)
    target_line_nos = parse_line_nos(args.line_nos)
    if target_line_nos is not None:
        rows = [row for row in rows if row.get("line_no") in target_line_nos]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    completed_lines = set()
    if args.resume and output_path.exists():
        existing_rows = load_jsonl(output_path)
        completed_lines = {row.get("line_no") for row in existing_rows}
        rows = [row for row in rows if row.get("line_no") not in completed_lines]

    client = None if args.apply_guard_only else AsyncOpenAI(api_key=api_key, base_url=base_url)
    refined_rows = list(existing_rows)
    started_all = time.perf_counter()
    mode = "a" if args.resume else "w"
    print(f"Selected {len(rows)} sample(s); skipped {len(completed_lines)} existing output sample(s)")
    with output_path.open(mode, encoding="utf-8") as out:
        for index, item in enumerate(rows, start=1):
            min_sentences, max_sentences, min_words, max_words = refinement_budget(item, args)
            prompt = USER_PROMPT.format(
                question=item.get("question", ""),
                answer_aspects=format_aspects(item),
                evidence=format_evidence(item, args.max_evidence_edges),
                snippets=format_snippets(item, args.max_snippets, args.snippet_chars),
                coverage_anchors=format_coverage_anchors(item, args)[0],
                draft=item.get("prediction", ""),
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                min_words=min_words,
                max_words=max_words,
            )
            _, coverage_anchors = format_coverage_anchors(item, args)
            print(f"[{index}/{len(rows)}] refining line={item.get('line_no')}", flush=True)
            started = time.perf_counter()
            failed = False
            failure_message = ""
            if args.apply_guard_only:
                refined = item.get("raw_refined_prediction") or item.get("prediction", "")
            else:
                try:
                    refined = await complete(client, model, prompt, args.request_timeout, args.request_retries)
                except Exception as exc:
                    if not args.keep_original_on_failure:
                        raise
                    failed = True
                    failure_message = str(exc)
                    refined = item.get("prediction", "")
            elapsed = time.perf_counter() - started
            guard_item = dict(item)
            if args.apply_guard_only and item.get("draft_prediction"):
                guard_item["prediction"] = item.get("draft_prediction", "")
            guarded_refined, guard_report = guard_refinement(guard_item, refined, args)

            new_item = dict(item)
            new_item["draft_prediction"] = item.get("draft_prediction", item.get("prediction", ""))
            new_item["raw_refined_prediction"] = refined
            new_item["prediction"] = guarded_refined
            new_item["method"] = f"{item.get('method', 'unknown')}_refined"
            new_item["answer_chars"] = len(guarded_refined)
            new_item["refiner_stats"] = {
                "enabled": True,
                "failed": failed,
                "failure_message": failure_message[:500],
                "coverage_guard": guard_report,
                "query_seconds": round(elapsed, 3),
                "prompt_chars": len(prompt),
                "draft_answer_chars": len(item.get("prediction", "")),
                "raw_refined_answer_chars": len(refined),
                "refined_answer_chars": len(guarded_refined),
                "coverage_anchors_enabled": bool(args.coverage_anchors),
                "coverage_anchor_count": len(coverage_anchors),
                "coverage_anchors": coverage_anchors,
            }
            refined_rows.append(new_item)
            out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            out.flush()

    avg_draft_chars = sum(row["refiner_stats"]["draft_answer_chars"] for row in refined_rows) / max(1, len(refined_rows))
    avg_refined_chars = sum(row["refiner_stats"]["refined_answer_chars"] for row in refined_rows) / max(1, len(refined_rows))
    avg_refine_seconds = sum(row["refiner_stats"]["query_seconds"] for row in refined_rows) / max(1, len(refined_rows))
    avg_anchor_count = sum(
        row["refiner_stats"].get("coverage_anchor_count", 0) for row in refined_rows
    ) / max(1, len(refined_rows))
    guard_triggered = sum(
        1 for row in refined_rows if row["refiner_stats"].get("coverage_guard", {}).get("triggered")
    )
    failed_count = sum(1 for row in refined_rows if row["refiner_stats"].get("failed"))
    summary = [
        "# Answer Refinement Summary\n\n",
        f"- Input: `{args.input}`\n",
        f"- Output: `{args.output}`\n",
        f"- Samples: {len(refined_rows)}\n",
        f"- Coverage guard enabled: {args.coverage_guard}\n",
        f"- Guard-triggered samples: {guard_triggered}\n",
        f"- Refinement-failed samples: {failed_count}\n",
        f"- Total seconds: {time.perf_counter() - started_all:.3f}\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Avg draft answer chars | {avg_draft_chars:.1f} |\n",
        f"| Avg refined answer chars | {avg_refined_chars:.1f} |\n",
        f"| Avg refine seconds | {avg_refine_seconds:.3f} |\n",
        f"| Avg coverage anchors | {avg_anchor_count:.2f} |\n",
    ]
    summary_path.write_text("".join(summary), encoding="utf-8")
    print(f"Saved refined results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
