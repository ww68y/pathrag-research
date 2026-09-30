import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from build_answer_contract_agent import clean_text, derive_answer_contract, load_jsonl
from build_claim_evidence_agent import (
    EvidenceIndex,
    as_float,
    claim_metrics,
    find_graph_dir,
    graph_evidence,
    index_key,
    load_full_source_rows,
    normalize_space,
    numbers,
    split_claims,
    split_windows,
    support_trace,
    write_jsonl,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
LENGTH_BUDGETS = {
    "general": (1000, 180),
    "comparison": (1300, 220),
    "compound": (1200, 210),
    "definition_and_purpose": (1000, 180),
    "enumeration": (1100, 190),
    "mechanism": (1100, 180),
    "numeric_lookup": (600, 110),
    "procedure": (1300, 220),
    "relation_lookup": (900, 160),
    "literal_and_metaphorical": (1100, 190),
}
REFUSAL_PATTERNS = (
    "cannot be determined", "cannot determine", "not described", "not provided",
    "does not contain", "no information", "insufficient information",
)


def answer_contract(row: dict) -> dict:
    return (
        (row.get("learned_dual_gate") or {}).get("contract")
        or (row.get("budget_meta_agent") or {}).get("contract")
        or derive_answer_contract(row.get("question", ""))
    )


def context_indexes(
    row: dict,
    source: dict,
    full_source: dict,
    data_root: Path,
    text_cache: dict[str, EvidenceIndex],
    graph_cache: dict[str, EvidenceIndex],
) -> tuple[EvidenceIndex, EvidenceIndex]:
    context = clean_text(full_source.get("context") or source.get("context", ""))
    key = index_key(context)
    if key not in text_cache:
        records = [
            {"kind": "full_text", "text": window, "source_id": f"full-window-{index}"}
            for index, window in enumerate(split_windows(context))
        ]
        text_cache[key] = EvidenceIndex.build(records)

    source_hash = normalize_space(source.get("source_context_hash"))
    graph_hash = normalize_space((row.get("context_budget_agent") or {}).get("context_hash"))
    graph_dir = find_graph_dir(data_root, int(row.get("line_no", 0)), [graph_hash, source_hash])
    graph_key = graph_dir.name if graph_dir else "missing"
    if graph_key not in graph_cache:
        graph_cache[graph_key] = EvidenceIndex.build(graph_evidence(graph_dir)) if graph_dir else EvidenceIndex.build([])
    return text_cache[key], graph_cache[graph_key]


def audit_answer(answer: str, text_index: EvidenceIndex, graph_index: EvidenceIndex) -> tuple[list[dict], dict]:
    traces = [support_trace(claim, text_index, graph_index) for claim in split_claims(answer)]
    metrics = claim_metrics(traces)
    metrics["unsupported_ratio"] = round(
        metrics["unsupported_claims"] / max(1, metrics["claims"]), 6
    )
    return traces, metrics


def protected_terms(answer: str) -> set[str]:
    acronyms = set(re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", answer))
    section_numbers = set(re.findall(r"\b(?:Section|Article)\s+\d+(?:\.\d+)*\b", answer))
    units = set(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|days?|months?|years?|hours?|acres?)\b", answer, re.I))
    return acronyms | section_numbers | units


def preservation_ratio(required: set[str], answer: str) -> float:
    if not required:
        return 1.0
    answer_lower = answer.lower()
    return sum(term.lower() in answer_lower for term in required) / len(required)


def semantic_claim_coverage(original_claims: list[str], compressed_claims: list[str]) -> tuple[float, list[float]]:
    if not original_claims:
        return 1.0, []
    if not compressed_claims:
        return 0.0, [0.0] * len(original_claims)
    texts = original_claims + compressed_claims
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        max_features=30000,
    )
    matrix = vectorizer.fit_transform(texts)
    original_matrix = matrix[:len(original_claims)]
    compressed_matrix = matrix[len(original_claims):]
    similarities = linear_kernel(original_matrix, compressed_matrix)
    maxima = similarities.max(axis=1).tolist()
    return float(np.mean(maxima)), [round(float(value), 6) for value in maxima]


def compression_route(row: dict, traces: list[dict], metrics: dict) -> dict:
    contract = answer_contract(row)
    contract_type = contract.get("type", "general")
    max_chars, target_words = LENGTH_BUDGETS.get(contract_type, LENGTH_BUDGETS["general"])
    answer = clean_text(row.get("prediction", ""))
    refusal = any(pattern in answer.lower() for pattern in REFUSAL_PATTERNS)
    supported_or_partial = metrics["supported_claims"] + metrics["partial_claims"]
    evidence_ready = metrics["mean_support"] >= 0.52 and supported_or_partial >= 2
    over_budget = len(answer) > max_chars
    candidate = over_budget and evidence_ready and not refusal and len(traces) >= 3
    reasons = []
    if over_budget:
        reasons.append("answer_exceeds_type_budget")
    if not evidence_ready:
        reasons.append("insufficient_grounding_for_safe_compression")
    if refusal:
        reasons.append("explicit_uncertainty_requires_retrieval_not_compression")
    if len(traces) < 3:
        reasons.append("too_few_claims_to_compress")
    return {
        "enabled": True,
        "contract": contract,
        "max_chars": max_chars,
        "target_words": target_words,
        "original_chars": len(answer),
        "original_claim_metrics": metrics,
        "compression_candidate": candidate,
        "compression_attempted": False,
        "compression_accepted": False,
        "route_reasons": reasons,
    }


COMPRESS_SYSTEM_PROMPT = """You are an evidence-preserving answer compression agent.
Rewrite the answer to be concise while preserving every distinct fact needed by the question.
Use only the supplied evidence. Preserve numbers, units, legal section references, named concepts,
conditions, exceptions, contrasts, and causal direction. Remove repetition and background that does
not answer the question. Give the direct answer first. Do not mention compression or reference answers."""


def compression_prompt(row: dict, agent: dict, traces: list[dict]) -> str:
    evidence = []
    for index, trace in enumerate(traces[:16], start=1):
        text_hit = (trace.get("text_evidence") or [{}])[0]
        graph_hit = (trace.get("graph_evidence") or [{}])[0]
        evidence.append(
            f"[Claim {index}; support={trace.get('support_score', 0)}]\n"
            f"{trace.get('claim', '')}\n"
            f"Text: {str(text_hit.get('text', ''))[:650]}\n"
            f"Graph: {str(graph_hit.get('text', ''))[:350]}"
        )
    return (
        f"Question:\n{row.get('question', '')}\n\n"
        f"Answer contract:\n{json.dumps(agent.get('contract', {}), ensure_ascii=False)}\n\n"
        f"Target: no more than about {agent.get('target_words')} words.\n\n"
        f"Current answer:\n{row.get('prediction', '')}\n\n"
        f"Claim-linked evidence:\n" + "\n\n".join(evidence)
    )


def verify_compression(
    original: str,
    compressed: str,
    original_traces: list[dict],
    text_index: EvidenceIndex,
    graph_index: EvidenceIndex,
    contract: dict,
) -> dict:
    original_claims = [trace.get("claim", "") for trace in original_traces]
    compressed_traces, post_metrics = audit_answer(compressed, text_index, graph_index)
    compressed_claims = [trace.get("claim", "") for trace in compressed_traces]
    semantic_coverage, claim_similarities = semantic_claim_coverage(original_claims, compressed_claims)

    required_numbers = numbers(" ".join(
        trace.get("claim", "") for trace in original_traces if trace.get("status") != "unsupported"
    ))
    number_ratio = preservation_ratio(required_numbers, compressed)
    required_terms = protected_terms(" ".join(original_claims))
    term_ratio = preservation_ratio(required_terms, compressed)
    original_metrics = claim_metrics(original_traces)
    length_reduction = 1 - len(compressed) / max(1, len(original))
    minimum_points = int(contract.get("minimum_distinct_points", 1))

    checks = {
        "meaning_preserved": semantic_coverage >= 0.66,
        "numbers_preserved": number_ratio >= 0.999,
        "protected_terms_preserved": term_ratio >= 0.75,
        "evidence_grounding_preserved": post_metrics["mean_support"] >= max(
            0.50, original_metrics["mean_support"] - 0.10
        ),
        "unsupported_claims_controlled": post_metrics["unsupported_ratio"] <= max(
            0.20,
            original_metrics["unsupported_claims"] / max(1, original_metrics["claims"]) + 0.10,
        ),
        "answer_contract_shape_preserved": len(compressed_claims) >= minimum_points,
        "meaningful_length_reduction": length_reduction >= 0.15,
        "minimum_answer_length": len(compressed) >= 80,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "semantic_claim_coverage": round(semantic_coverage, 6),
        "claim_similarities": claim_similarities,
        "number_preservation": round(number_ratio, 6),
        "protected_term_preservation": round(term_ratio, 6),
        "length_reduction": round(length_reduction, 6),
        "post_claim_metrics": post_metrics,
        "post_claim_traces": compressed_traces,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }


async def generate_compressions(
    candidates: list[tuple[dict, list[dict]]],
    model: str,
    concurrency: int,
) -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete in .env")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def generate(row: dict, traces: list[dict]) -> None:
        agent = row["evidence_compression_agent"]
        agent["compression_attempted"] = True
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
                        {"role": "user", "content": compression_prompt(row, agent, traces)},
                    ],
                )
                agent["compression_candidate_answer"] = clean_text(response.choices[0].message.content)
            except Exception as exc:
                agent["compression_error"] = f"{type(exc).__name__}: {exc}"[:300]

    await asyncio.gather(*(generate(row, traces) for row, traces in candidates))


def write_summary(path: Path, rows: list[dict], source_path: Path) -> None:
    agents = [row.get("evidence_compression_agent") or {} for row in rows]
    candidates = sum(bool(agent.get("compression_candidate")) for agent in agents)
    attempted = sum(bool(agent.get("compression_attempted")) for agent in agents)
    accepted = sum(bool(agent.get("compression_accepted")) for agent in agents)
    before_chars = sum(as_float(agent.get("original_chars")) for agent in agents if agent.get("compression_accepted"))
    after_chars = sum(len(row.get("prediction", "")) for row, agent in zip(rows, agents) if agent.get("compression_accepted"))
    rejection_counts = Counter(
        reason
        for agent in agents
        for reason in (agent.get("verification") or {}).get("rejection_reasons", [])
    )
    lines = [
        "# SAGE v26 Evidence-Preserving Compression Summary\n\n",
        f"- Source: `{source_path.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- References and evaluation scores are not used for routing, compression, or verification.\n\n",
        "## Routing\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Compression candidates | {candidates} |\n",
        f"| Compression attempted | {attempted} |\n",
        f"| Compression accepted | {accepted} |\n",
        f"| Compression rejected / original retained | {attempted - accepted} |\n",
        f"| Accepted answer characters | {int(before_chars)} -> {int(after_chars)} |\n",
        f"| Accepted character reduction | {((1 - after_chars / before_chars) * 100 if before_chars else 0):.2f}% |\n\n",
        "## Guard Rejections\n\n",
        "| Reason | Count |\n",
        "|---|---:|\n",
    ]
    if rejection_counts:
        for reason, count in sorted(rejection_counts.items()):
            lines.append(f"| {reason} | {count} |\n")
    else:
        lines.append("| none | 0 |\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-context-root", required=True)
    parser.add_argument("--data-root", default="data_repro")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    input_path = ROOT_DIR / args.input
    dataset_path = ROOT_DIR / args.dataset
    rows = load_jsonl(input_path)
    sources = load_jsonl(dataset_path)
    source_by_line = {
        int(source.get("_subset_line_no", index)): source for index, source in enumerate(sources)
    }
    full_sources = load_full_source_rows(sources, ROOT_DIR / args.full_context_root)
    text_cache: dict[str, EvidenceIndex] = {}
    graph_cache: dict[str, EvidenceIndex] = {}
    runtime: dict[int, tuple[list[dict], EvidenceIndex, EvidenceIndex]] = {}
    output_rows = []

    for row in rows:
        line_no = int(row.get("line_no", 0))
        source = source_by_line[line_no]
        full_source = full_sources.get((
            normalize_space(source.get("source_dataset")),
            int(source.get("source_line_no", -1)),
        ), {})
        text_index, graph_index = context_indexes(
            row, source, full_source, ROOT_DIR / args.data_root, text_cache, graph_cache
        )
        traces, metrics = audit_answer(row.get("prediction", ""), text_index, graph_index)
        new_row = dict(row)
        new_row["evidence_compression_agent"] = compression_route(row, traces, metrics)
        output_rows.append(new_row)
        runtime[line_no] = (traces, text_index, graph_index)

    if args.compress:
        load_dotenv(ROOT_DIR / ".env")
        model = args.model or os.getenv("DEEPSEEK_MODEL")
        candidates = [
            (row, runtime[int(row.get("line_no", 0))][0])
            for row in output_rows
            if row["evidence_compression_agent"]["compression_candidate"]
        ]
        asyncio.run(generate_compressions(candidates, model, args.concurrency))

        for row, original_traces in candidates:
            agent = row["evidence_compression_agent"]
            compressed = clean_text(agent.pop("compression_candidate_answer", ""))
            if not compressed:
                continue
            line_no = int(row.get("line_no", 0))
            _, text_index, graph_index = runtime[line_no]
            verification = verify_compression(
                row.get("prediction", ""),
                compressed,
                original_traces,
                text_index,
                graph_index,
                agent["contract"],
            )
            agent["verification"] = verification
            agent["compressed_chars"] = len(compressed)
            if verification["accepted"]:
                row["prediction_before_evidence_compression"] = row.get("prediction", "")
                row["prediction"] = compressed
                row["method"] = "sage_v26_evidence_preserving_compression"
                agent["compression_accepted"] = True
            else:
                row["rejected_compressed_prediction"] = compressed

    output_path = ROOT_DIR / args.output
    summary_path = ROOT_DIR / args.summary
    write_jsonl(output_rows, output_path)
    write_summary(summary_path, output_rows, input_path)
    print(f"Saved SAGE v26 results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
