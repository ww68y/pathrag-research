from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from build_answer_contract_agent import clean_text, load_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]
STOP_WORDS = {
    "about", "after", "again", "against", "also", "among", "and", "are", "based",
    "because", "been", "before", "being", "between", "both", "but", "cannot", "could",
    "does", "each", "from", "have", "having", "into", "only", "other", "over", "provided",
    "should", "such", "than", "that", "their", "there", "these", "they", "this", "those",
    "through", "under", "using", "very", "what", "when", "where", "which", "while", "with",
    "would", "your",
}
REFUSAL_PATTERNS = (
    "not described", "not provided", "not specified", "cannot be determined",
    "cannot determine", "does not contain", "no information", "insufficient information",
)


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def split_windows(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind(". ", start + size // 2, end), text.rfind("; ", start + size // 2, end))
            if boundary > start:
                end = boundary + 1
        window = text[start:end].strip()
        if window:
            windows.append(window)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return windows


def split_claims(answer: str, limit: int = 16) -> list[str]:
    answer = clean_text(answer)
    answer = re.sub(r"^\s{0,3}#{1,6}\s*", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", answer, flags=re.MULTILINE)
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    claims = []
    for part in parts:
        part = normalize_space(part).strip(" -*")
        if not part or len(part) < 18:
            continue
        if len(part) > 420:
            subparts = [item.strip() for item in re.split(r";\s+", part) if item.strip()]
        else:
            subparts = [part]
        claims.extend(subparts)
    if not claims and normalize_space(answer):
        claims = [normalize_space(answer)]
    return claims[:limit]


def content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}|\d+(?:\.\d+)?%?", text)
        if token.lower() not in STOP_WORDS
    }


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)*(?:%|\s*(?:days?|years?|months?|hours?))?\b", text.lower()))


def clean_graph_value(value: object) -> str:
    return normalize_space(str(value)).strip('"')


def graph_evidence(graph_dir: Path) -> list[dict]:
    graph_path = graph_dir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        return []
    graph = nx.read_graphml(graph_path)
    records = []
    for node, attrs in graph.nodes(data=True):
        description = clean_graph_value(attrs.get("description", ""))
        if description:
            records.append({
                "kind": "node",
                "text": f"{clean_graph_value(node)}: {description}",
                "source_id": clean_graph_value(attrs.get("source_id", "")),
            })
    for source, target, attrs in graph.edges(data=True):
        description = clean_graph_value(attrs.get("description", ""))
        keywords = clean_graph_value(attrs.get("keywords", ""))
        relation = "; ".join(item for item in (keywords, description) if item)
        if relation:
            records.append({
                "kind": "edge",
                "text": f"{clean_graph_value(source)} -> {clean_graph_value(target)}: {relation}",
                "source_id": clean_graph_value(attrs.get("source_id", "")),
            })
    return records


@dataclass
class EvidenceIndex:
    texts: list[str]
    records: list[dict]
    vectorizer: TfidfVectorizer | None
    matrix: object | None

    @classmethod
    def build(cls, records: list[dict]) -> "EvidenceIndex":
        texts = [normalize_space(record.get("text", "")) for record in records]
        valid = [(text, record) for text, record in zip(texts, records) if text]
        if not valid:
            return cls([], [], None, None)
        texts = [item[0] for item in valid]
        records = [item[1] for item in valid]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=40000,
        )
        matrix = vectorizer.fit_transform(texts)
        return cls(texts, records, vectorizer, matrix)

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self.vectorizer or self.matrix is None or not normalize_space(query):
            return []
        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.matrix).ravel()
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in indices:
            if scores[index] <= 0:
                continue
            record = dict(self.records[int(index)])
            record["score"] = round(float(scores[index]), 6)
            record["text"] = self.texts[int(index)][:900]
            results.append(record)
        return results


def lexical_coverage(claim: str, evidence: list[dict]) -> float:
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 1.0
    evidence_tokens = content_tokens(" ".join(item.get("text", "") for item in evidence))
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def support_trace(claim: str, text_index: EvidenceIndex, graph_index: EvidenceIndex) -> dict:
    text_hits = text_index.search(claim, 2)
    graph_hits = graph_index.search(claim, 2)
    text_score = as_float(text_hits[0].get("score")) if text_hits else 0.0
    graph_score = as_float(graph_hits[0].get("score")) if graph_hits else 0.0
    lexical = lexical_coverage(claim, text_hits + graph_hits)
    score = 0.55 * min(1.0, text_score / 0.22) + 0.20 * min(1.0, graph_score / 0.30) + 0.25 * lexical

    claim_numbers = numbers(claim)
    evidence_text = " ".join(item.get("text", "") for item in text_hits + graph_hits).lower()
    missing_numbers = sorted(number for number in claim_numbers if number not in evidence_text)
    if missing_numbers:
        score -= min(0.30, 0.12 * len(missing_numbers))

    lower_claim = claim.lower()
    negative_claim = any(pattern in lower_claim for pattern in REFUSAL_PATTERNS)
    if negative_claim and text_score >= 0.12:
        score -= 0.18

    score = max(0.0, min(1.0, score))
    if score >= 0.58:
        status = "supported"
    elif score >= 0.38:
        status = "partial"
    else:
        status = "unsupported"
    return {
        "claim": claim,
        "status": status,
        "support_score": round(score, 6),
        "text_score": round(text_score, 6),
        "graph_score": round(graph_score, 6),
        "lexical_coverage": round(lexical, 6),
        "missing_numbers": missing_numbers,
        "text_evidence": text_hits,
        "graph_evidence": graph_hits,
    }


def find_graph_dir(data_root: Path, line_no: int, hashes: list[str]) -> Path | None:
    for context_hash in hashes:
        if not context_hash:
            continue
        exact = data_root / f"ctx_{line_no}_{context_hash}"
        if exact.exists():
            return exact
        matches = list(data_root.glob(f"*_{context_hash}"))
        if matches:
            return matches[0]
    return None


def load_full_source_rows(dataset_rows: list[dict], root: Path) -> dict[tuple[str, int], dict]:
    targets: dict[str, set[int]] = {}
    for source in dataset_rows:
        dataset = normalize_space(source.get("source_dataset"))
        line_no = int(source.get("source_line_no", -1))
        if dataset and line_no >= 0:
            targets.setdefault(dataset, set()).add(line_no)

    loaded = {}
    for dataset, line_numbers in targets.items():
        path = root / f"{dataset}.jsonl"
        if not path.exists():
            continue
        last_needed = max(line_numbers)
        with path.open("r", encoding="utf-8") as source_file:
            for line_no, line in enumerate(source_file):
                if line_no in line_numbers:
                    loaded[(dataset, line_no)] = json.loads(line)
                if line_no >= last_needed:
                    break
    return loaded


def index_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def claim_metrics(claim_traces: list[dict]) -> dict:
    counts = Counter(item.get("status") for item in claim_traces)
    return {
        "claims": len(claim_traces),
        "supported_claims": counts["supported"],
        "partial_claims": counts["partial"],
        "unsupported_claims": counts["unsupported"],
        "mean_support": round(
            float(np.mean([as_float(item.get("support_score")) for item in claim_traces]))
            if claim_traces else 0.0,
            6,
        ),
    }


def expand_full_context_evidence(
    row: dict,
    trace: dict,
    full_source: dict,
    data_root: Path,
    index_cache: dict[str, EvidenceIndex],
) -> None:
    full_context = clean_text(full_source.get("context", ""))
    if not full_context:
        return
    key = index_key(full_context)
    if key not in index_cache:
        records = [
            {"kind": "full_text", "text": window, "source_id": f"full-window-{index}"}
            for index, window in enumerate(split_windows(full_context))
        ]
        index_cache[key] = EvidenceIndex.build(records)
    text_index = index_cache[key]

    graph_dir_name = trace.get("graph_dir")
    graph_dir = data_root / graph_dir_name if graph_dir_name else None
    graph_index = EvidenceIndex.build(graph_evidence(graph_dir)) if graph_dir and graph_dir.exists() else EvidenceIndex.build([])
    full_hits = text_index.search(row.get("question", ""), 6)
    old_hits = trace.get("question_text_evidence", [])
    old_score = as_float(old_hits[0].get("score")) if old_hits else 0.0
    full_score = as_float(full_hits[0].get("score")) if full_hits else 0.0

    claims = [item.get("claim", "") for item in trace.get("claim_traces", [])]
    expanded_claims = [support_trace(claim, text_index, graph_index) for claim in claims]
    trace["window_question_text_evidence"] = old_hits
    trace["question_text_evidence"] = full_hits
    trace["expanded_claim_traces"] = expanded_claims
    trace["context_expansion"] = {
        "enabled": True,
        "full_context_chars": len(full_context),
        "full_text_windows": len(text_index.records),
        "window_question_score": round(old_score, 6),
        "full_question_score": round(full_score, 6),
        "score_gain": round(full_score - old_score, 6),
        **claim_metrics(expanded_claims),
    }


def build_trace(row: dict, source: dict, data_root: Path) -> tuple[dict, list[dict]]:
    line_no = int(row.get("line_no", 0))
    source_hash = normalize_space(source.get("source_context_hash"))
    graph_context_hash = normalize_space((row.get("context_budget_agent") or {}).get("context_hash"))
    graph_dir = find_graph_dir(data_root, line_no, [graph_context_hash, source_hash])
    text_records = [
        {"kind": "text", "text": window, "source_id": f"window-{index}"}
        for index, window in enumerate(split_windows(source.get("context", "")))
    ]
    graph_records = graph_evidence(graph_dir) if graph_dir else []
    text_index = EvidenceIndex.build(text_records)
    graph_index = EvidenceIndex.build(graph_records)
    claims = split_claims(row.get("prediction", ""))
    claim_traces = [support_trace(claim, text_index, graph_index) for claim in claims]
    status_counts = Counter(item["status"] for item in claim_traces)
    mean_support = float(np.mean([item["support_score"] for item in claim_traces])) if claim_traces else 0.0
    unsupported_ratio = status_counts["unsupported"] / max(1, len(claim_traces))
    question_hits = text_index.search(row.get("question", ""), 5)
    graph_question_hits = graph_index.search(row.get("question", ""), 4)
    question_score = as_float(question_hits[0].get("score")) if question_hits else 0.0
    answer_lower = clean_text(row.get("prediction", "")).lower()
    refusal = any(pattern in answer_lower for pattern in REFUSAL_PATTERNS)
    missing_number_count = sum(len(item["missing_numbers"]) for item in claim_traces)

    reasons = []
    if refusal and question_score >= 0.10:
        reasons.append("retrieval_gap_after_refusal")
    if unsupported_ratio >= 0.34:
        reasons.append("high_unsupported_claim_ratio")
    if mean_support < 0.46 and len(claim_traces) >= 2:
        reasons.append("low_mean_claim_support")
    if missing_number_count:
        reasons.append("unverified_numeric_claim")

    repair_needed = any(
        reason in {
            "retrieval_gap_after_refusal",
            "high_unsupported_claim_ratio",
            "low_mean_claim_support",
        }
        for reason in reasons
    )
    trace = {
        "enabled": True,
        "version": "v25",
        "source_context_hash": source_hash,
        "graph_context_hash": graph_context_hash,
        "graph_dir": graph_dir.name if graph_dir else None,
        "text_windows": len(text_records),
        "graph_evidence_records": len(graph_records),
        "claims": len(claim_traces),
        "supported_claims": status_counts["supported"],
        "partial_claims": status_counts["partial"],
        "unsupported_claims": status_counts["unsupported"],
        "mean_support": round(mean_support, 6),
        "unsupported_ratio": round(unsupported_ratio, 6),
        "question_evidence_score": round(question_score, 6),
        "repair_needed": repair_needed,
        "repair_reasons": reasons,
        "repair_attempted": False,
        "repair_applied": False,
        "question_text_evidence": question_hits,
        "question_graph_evidence": graph_question_hits,
        "claim_traces": claim_traces,
    }
    return trace, text_records


REPAIR_SYSTEM_PROMPT = """You are an evidence-bounded knowledge-graph QA editor.
Answer the question using only the supplied text and graph evidence.
Correct unsupported statements in the current answer, but preserve supported useful content.
Give the direct answer first, cover the requested conditions/steps/actors, and stay concise.
If the evidence is genuinely insufficient, state exactly what is missing.
Do not mention this editing instruction, confidence scores, or reference answers."""


def repair_prompt(row: dict, trace: dict) -> str:
    contract = (row.get("learned_dual_gate") or {}).get("contract") or {}
    text_hits = trace.get("question_text_evidence", [])
    graph_hits = trace.get("question_graph_evidence", [])
    evidence_lines = []
    for index, item in enumerate(text_hits, start=1):
        evidence_lines.append(f"[Text {index}] {item.get('text', '')}")
    for index, item in enumerate(graph_hits, start=1):
        evidence_lines.append(f"[Graph {index}] {item.get('text', '')}")
    verification_claims = trace.get("expanded_claim_traces") or trace.get("claim_traces", [])
    weak_claims = [item["claim"] for item in verification_claims if item.get("status") != "supported"]
    return (
        f"Question:\n{row.get('question', '')}\n\n"
        f"Requested answer contract:\n{json.dumps(contract, ensure_ascii=False)}\n\n"
        f"Current answer:\n{row.get('prediction', '')}\n\n"
        f"Claims requiring verification:\n" + "\n".join(f"- {claim}" for claim in weak_claims[:10]) + "\n\n"
        f"Retrieved evidence:\n" + "\n\n".join(evidence_lines)
    )


async def repair_rows(rows: list[dict], model: str, concurrency: int) -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete in .env")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
    semaphore = asyncio.Semaphore(concurrency)

    async def repair(row: dict) -> None:
        trace = row["claim_evidence_agent"]
        trace["repair_attempted"] = True
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                        {"role": "user", "content": repair_prompt(row, trace)},
                    ],
                )
                repaired = clean_text(response.choices[0].message.content)
            except Exception as exc:  # Keep the original answer when the optional repair call fails.
                trace["repair_error"] = f"{type(exc).__name__}: {exc}"[:300]
                return
        if len(repaired) >= 40 and repaired != clean_text(row.get("prediction", "")):
            row["prediction_before_claim_guard"] = row.get("prediction", "")
            row["prediction"] = repaired
            row["method"] = "sage_v25_claim_evidence_repaired"
            trace["repair_applied"] = True

    await asyncio.gather(*(repair(row) for row in rows))


def write_summary(path: Path, rows: list[dict], source_path: Path) -> None:
    traces = [row.get("claim_evidence_agent") or {} for row in rows]
    total_claims = sum(int(trace.get("claims", 0)) for trace in traces)
    supported = sum(int(trace.get("supported_claims", 0)) for trace in traces)
    partial = sum(int(trace.get("partial_claims", 0)) for trace in traces)
    unsupported = sum(int(trace.get("unsupported_claims", 0)) for trace in traces)
    repair_needed = sum(bool(trace.get("repair_needed")) for trace in traces)
    repair_attempted = sum(bool(trace.get("repair_attempted")) for trace in traces)
    repair_applied = sum(bool(trace.get("repair_applied")) for trace in traces)
    graph_mapped = sum(bool(trace.get("graph_dir")) for trace in traces)
    expansions = [trace.get("context_expansion") for trace in traces if trace.get("context_expansion")]
    mean_support = np.mean([as_float(trace.get("mean_support")) for trace in traces]) if traces else 0.0
    reason_counts = Counter(reason for trace in traces for reason in trace.get("repair_reasons", []))

    lines = [
        "# SAGE v25 Claim-Evidence Agent Summary\n\n",
        f"- Source: `{source_path.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        f"- Graph contexts mapped: {graph_mapped} / {len(rows)}\n",
        "- Reference answers and evaluation scores are not used for evidence retrieval or repair routing.\n\n",
        "## Claim Grounding\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Claims | {total_claims} |\n",
        f"| Supported | {supported} ({supported / max(1, total_claims) * 100:.2f}%) |\n",
        f"| Partial | {partial} ({partial / max(1, total_claims) * 100:.2f}%) |\n",
        f"| Unsupported | {unsupported} ({unsupported / max(1, total_claims) * 100:.2f}%) |\n",
        f"| Mean support | {mean_support:.3f} |\n\n",
        "## Adaptive Context Expansion\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Expanded samples | {len(expansions)} / {len(rows)} |\n",
        f"| Mean window question score | {np.mean([as_float(item.get('window_question_score')) for item in expansions]) if expansions else 0:.3f} |\n",
        f"| Mean full-context question score | {np.mean([as_float(item.get('full_question_score')) for item in expansions]) if expansions else 0:.3f} |\n",
        f"| Mean post-expansion claim support | {np.mean([as_float(item.get('mean_support')) for item in expansions]) if expansions else 0:.3f} |\n\n",
        "## Conservative Repair\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Repair needed | {repair_needed} |\n",
        f"| Repair attempted | {repair_attempted} |\n",
        f"| Repair applied | {repair_applied} |\n",
    ]
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"| Trigger `{reason}` | {count} |\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="data_repro")
    parser.add_argument("--full-context-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=0)
    args = parser.parse_args()

    input_path = ROOT_DIR / args.input
    source_path = ROOT_DIR / args.dataset
    rows = load_jsonl(input_path)
    source_rows = load_jsonl(source_path)
    source_by_line = {
        int(source.get("_subset_line_no", index)): source for index, source in enumerate(source_rows)
    }
    full_source_rows = (
        load_full_source_rows(source_rows, ROOT_DIR / args.full_context_root)
        if args.full_context_root else {}
    )
    full_index_cache: dict[str, EvidenceIndex] = {}

    output_rows = []
    for row in rows:
        line_no = int(row.get("line_no", 0))
        source = source_by_line.get(line_no)
        if source is None:
            raise KeyError(f"No source context for line {line_no}")
        trace, _ = build_trace(row, source, ROOT_DIR / args.data_root)
        if trace["repair_needed"] and full_source_rows:
            full_source = full_source_rows.get((
                normalize_space(source.get("source_dataset")),
                int(source.get("source_line_no", -1)),
            ))
            if full_source:
                expand_full_context_evidence(
                    row,
                    trace,
                    full_source,
                    ROOT_DIR / args.data_root,
                    full_index_cache,
                )
        new_row = dict(row)
        new_row["method"] = "sage_v25_claim_evidence_analyzed"
        new_row["claim_evidence_agent"] = trace
        output_rows.append(new_row)

    if args.repair:
        load_dotenv(ROOT_DIR / ".env")
        model = args.model or os.getenv("DEEPSEEK_MODEL")
        candidates = [row for row in output_rows if row["claim_evidence_agent"]["repair_needed"]]
        candidates.sort(
            key=lambda row: (
                (row["claim_evidence_agent"]["unsupported_ratio"]),
                -row["claim_evidence_agent"]["mean_support"],
            ),
            reverse=True,
        )
        if args.max_repairs > 0:
            candidates = candidates[:args.max_repairs]
        asyncio.run(repair_rows(candidates, model, max(1, args.concurrency)))

    output_path = ROOT_DIR / args.output
    summary_path = ROOT_DIR / args.summary
    write_jsonl(output_rows, output_path)
    write_summary(summary_path, output_rows, input_path)
    print(f"Saved SAGE v25 results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
