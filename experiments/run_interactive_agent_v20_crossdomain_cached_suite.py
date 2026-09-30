import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from PathRAG import PathRAG
from PathRAG.llm import openai_complete, siliconcloud_embedding
from PathRAG.prompt import PROMPTS
from PathRAG.utils import EmbeddingFunc
from api.features.chats.agent_controller import run_interactive_agent


DATASET_PATH = ROOT_DIR / "datasets" / "UltraDomain" / "crossdomain_multiquery_condensed_12.jsonl"
OUT_JSONL = ROOT_DIR / "results" / "interactive_agent_v20_crossdomain_cached_suite_12.jsonl"
OUT_SUMMARY = ROOT_DIR / "results" / "interactive_agent_v20_crossdomain_cached_suite_12_summary.md"
SAMPLE_TIMEOUT_SECONDS = 180
REQUESTED_MODE = "agent"
SUMMARY_TITLE = "# Interactive Agent v20 Cross-Domain Cached Suite"
RESUME_EXISTING = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def find_graph_dir(sample: dict[str, Any]) -> Path:
    h = context_hash(sample.get("context", ""))
    line_no = int(sample["_line_no"])
    exact = ROOT_DIR / "data_repro" / f"ctx_{line_no}_{h}"
    if (exact / "graph_chunk_entity_relation.graphml").exists():
        return exact
    candidates = sorted((ROOT_DIR / "data_repro").glob(f"ctx_*_{h}"))
    for candidate in candidates:
        if (candidate / "graph_chunk_entity_relation.graphml").exists():
            return candidate
    raise FileNotFoundError(f"No cached graph found for line={line_no}, context_hash={h}")


def build_rag_for_dir(graph_dir: Path) -> PathRAG:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    PROMPTS["process_tickers"] = ["-", "\\", "|", "/"]
    return PathRAG(
        working_dir=str(graph_dir),
        llm_model_func=openai_complete,
        llm_model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        llm_model_kwargs={
            "api_key": api_key,
            "base_url": os.getenv("DEEPSEEK_API_BASE", "https://api.siliconflow.cn/v1"),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
            func=partial(
                siliconcloud_embedding,
                model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                base_url=os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1/embeddings"),
                api_key=api_key,
                max_token_size=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
            ),
        ),
        embedding_func_max_async=int(os.getenv("EMBEDDING_ASYNC", "1")),
        llm_model_max_async=int(os.getenv("LLM_ASYNC", "1")),
        chunk_token_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP", "100")),
        entity_extract_max_gleaning=int(os.getenv("ENTITY_EXTRACT_MAX_GLEANING", "0")),
    )


def summarize(rows: list[dict[str, Any]]) -> str:
    ok_rows = [row for row in rows if row.get("ok")]
    domain_counts = Counter(row.get("dataset", "unknown") for row in ok_rows)
    profile_counts = Counter(row.get("selected_profile", "unknown") for row in ok_rows)
    policy_counts = Counter(row.get("final_policy", "unknown") for row in ok_rows)
    budget_counts = Counter((row.get("trace") or {}).get("budget_policy", "unknown") for row in ok_rows)
    audit_risks = Counter(((row.get("trace") or {}).get("evidence_audit") or {}).get("risk_level", "missing") for row in ok_rows)
    audit_recs = Counter(((row.get("trace") or {}).get("evidence_audit") or {}).get("recommendation", "missing") for row in ok_rows)
    avg_runtime = sum(row.get("runtime_seconds", 0) for row in ok_rows) / max(1, len(ok_rows))
    avg_candidates = sum(row.get("candidate_count", 0) for row in ok_rows) / max(1, len(ok_rows))
    avg_claim_support = sum(row.get("claim_support_ratio", 0) for row in ok_rows) / max(1, len(ok_rows))
    total_claims = sum(row.get("claim_count", 0) for row in ok_rows)
    total_unsupported = sum(row.get("unsupported_claim_count", 0) for row in ok_rows)
    avg_facet = sum(((row.get("trace") or {}).get("facet_coverage") or {}).get("coverage_ratio", 0) for row in ok_rows) / max(1, len(ok_rows))
    direct_exits = sum(1 for row in ok_rows if (row.get("trace") or {}).get("budget_policy") == "structured_hardcase_direct")

    lines = [
        SUMMARY_TITLE,
        "",
        f"Dataset: `{DATASET_PATH.relative_to(ROOT_DIR)}`",
        "",
        "## Aggregate",
        "",
        f"- Samples: {len(rows)}",
        f"- Successful: {len(ok_rows)}",
        f"- Domains: {dict(domain_counts)}",
        f"- Avg runtime seconds: {avg_runtime:.2f}",
        f"- Avg candidate count: {avg_candidates:.2f}",
        f"- Avg claim support ratio: {avg_claim_support:.3f}",
        f"- Total checked claims: {total_claims}",
        f"- Total unsupported claims: {total_unsupported}",
        f"- Unsupported claim rate: {total_unsupported / max(1, total_claims):.3f}",
        f"- Avg supported-facet coverage: {avg_facet:.3f}",
        f"- Structured direct exits: {direct_exits}",
        f"- Budget policies: {dict(budget_counts)}",
        f"- Selected profiles: {dict(profile_counts)}",
        f"- Final policies: {dict(policy_counts)}",
        f"- Audit risks: {dict(audit_risks)}",
        f"- Audit recommendations: {dict(audit_recs)}",
        "",
        "## Per Sample",
        "",
        "| # | Domain | Question | Profile | Budget | Candidates | Claims | Unsupported | Facets | Audit | Runtime |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        trace = row.get("trace") or {}
        facet = trace.get("facet_coverage") or {}
        audit = trace.get("evidence_audit") or {}
        question = row.get("question", "").replace("|", "/")
        audit_text = f"{audit.get('support_level', 'n/a')}/{audit.get('risk_level', 'n/a')}"
        lines.append(
            f"| {row.get('index')} | {row.get('dataset')} | {question} | "
            f"{row.get('selected_profile', 'error')} | {trace.get('budget_policy', 'error')} | "
            f"{row.get('candidate_count', 0)} | {row.get('claim_supported_count', 0)}/{row.get('claim_count', 0)} | "
            f"{row.get('unsupported_claim_count', 0)} | {facet.get('covered_count', 0)}/{facet.get('facet_count', 0)} | "
            f"{audit_text} | {row.get('runtime_seconds', 0):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    samples = load_jsonl(DATASET_PATH)
    rag_cache: dict[Path, PathRAG] = {}
    completed_by_line: dict[int, dict[str, Any]] = {}
    if RESUME_EXISTING and OUT_JSONL.exists():
        with OUT_JSONL.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("ok") and row.get("line_no") is not None:
                    completed_by_line[int(row["line_no"])] = row
    rows: list[dict[str, Any]] = sorted(
        completed_by_line.values(),
        key=lambda row: int(row.get("line_no", 0)),
    )
    for index, sample in enumerate(samples, start=1):
        line_no = int(sample["_line_no"])
        if line_no in completed_by_line:
            print(
                f"[{index}/{len(samples)}] skip completed "
                f"line={line_no}",
                flush=True,
            )
            continue
        question = sample.get("input") or sample.get("question") or ""
        started = time.perf_counter()
        try:
            graph_dir = find_graph_dir(sample)
            rag = rag_cache.get(graph_dir)
            if rag is None:
                rag = build_rag_for_dir(graph_dir)
                rag_cache[graph_dir] = rag
            result = await asyncio.wait_for(
                run_interactive_agent(rag, question, requested_mode=REQUESTED_MODE),
                timeout=SAMPLE_TIMEOUT_SECONDS,
            )
            trace = result["trace"]
            evidence_summary = trace.get("evidence_summary") or {}
            claim_verification = trace.get("claim_verification") or {}
            row = {
                "index": index,
                "line_no": sample["_line_no"],
                "ok": True,
                "dataset": sample.get("dataset") or sample.get("label"),
                "question": question,
                "reference_answers": sample.get("answers", []),
                "answer": result["answer"],
                "version": trace.get("version"),
                "working_dir": str(graph_dir.relative_to(ROOT_DIR)),
                "context_hash": context_hash(sample.get("context", "")),
                "question_type": trace.get("question_type"),
                "selected_mode": trace.get("selected_mode"),
                "selected_profile": trace.get("selected_profile"),
                "candidate_count": trace.get("candidate_count", 0),
                "evidence_edges": len(evidence_summary.get("relationships", [])),
                "evidence_entities": len(evidence_summary.get("entities", [])),
                "final_policy": trace.get("final_policy"),
                "claim_count": claim_verification.get("claim_count", 0),
                "claim_supported_count": claim_verification.get("supported_count", 0),
                "unsupported_claim_count": claim_verification.get("unsupported_count", 0),
                "claim_support_ratio": claim_verification.get("support_ratio", 0),
                "runtime_seconds": trace.get("runtime_seconds", time.perf_counter() - started),
                "trace": trace,
            }
        except Exception as exc:
            row = {
                "index": index,
                "line_no": sample.get("_line_no"),
                "ok": False,
                "dataset": sample.get("dataset") or sample.get("label"),
                "question": question,
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_seconds": time.perf_counter() - started,
            }
        rows = [
            existing
            for existing in rows
            if int(existing.get("line_no", -1)) != line_no
        ]
        rows.append(row)
        rows.sort(key=lambda item: int(item.get("line_no", 0)))
        write_jsonl(OUT_JSONL, rows)
        OUT_SUMMARY.write_text(summarize(rows), encoding="utf-8")
        print(
            f"[{index}/{len(samples)}] ok={row.get('ok')} domain={row.get('dataset')} "
            f"profile={row.get('selected_profile')} claims={row.get('claim_supported_count', 0)}/"
            f"{row.get('claim_count', 0)} unsupported={row.get('unsupported_claim_count', 0)} "
            f"runtime={row.get('runtime_seconds', 0):.2f}s"
        )

    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    asyncio.run(main())
