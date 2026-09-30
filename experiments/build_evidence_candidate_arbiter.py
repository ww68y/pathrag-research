import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_answer_contract_agent import parse_json_object


ROOT_DIR = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You are an evidence-calibrated answer candidate arbiter.
Use only the supplied evidence. Do not use outside knowledge or a reference answer.
Candidates are anonymous and may come from different retrieval methods. Evaluate factual grounding,
question coverage, relevance, and conciseness. Prefer the shorter candidate when quality is tied.
Return JSON only."""

USER_PROMPT = """Question:
{question}

Question-derived answer contract:
{contract}

Shared evidence for every candidate:
{evidence}

Anonymous candidates:
{candidates}

Score every candidate from 0.0 to 1.0 for grounding, coverage, relevance, and conciseness.
Select one winner only when it has a concrete evidence-supported advantage. If candidates are
substantively tied, select the shorter one. Do not infer candidate identity from writing style.

Return this exact JSON shape:
{{
  "winner": "A",
  "confidence": 0.0,
  "scores": {{
    "A": {{"grounding": 0.0, "coverage": 0.0, "relevance": 0.0, "conciseness": 0.0}},
    "B": {{"grounding": 0.0, "coverage": 0.0, "relevance": 0.0, "conciseness": 0.0}}
  }},
  "decisive_facts": ["..."],
  "reason": "..."
}}"""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def by_line(rows: list[dict]) -> dict[int, dict]:
    return {int(row.get("line_no", 0)): row for row in rows}


def clamp(value: object) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 6)
    except (TypeError, ValueError):
        return 0.0


def evidence_text(agent: dict) -> str:
    parts = []
    for index, item in enumerate((agent.get("question_text_evidence") or [])[:5], start=1):
        parts.append(f"[Text {index}] {str(item.get('text', ''))[:900]}")
    for index, item in enumerate((agent.get("question_graph_evidence") or [])[:5], start=1):
        parts.append(f"[Graph {index}] {str(item.get('text', ''))[:600]}")
    return "\n\n".join(parts)


def candidate_evidence(agent: dict, limit: int = 8) -> str:
    traces = (
        agent.get("expanded_claim_traces")
        or agent.get("claim_traces")
        or agent.get("post_claim_traces")
        or []
    )
    parts = []
    for index, trace in enumerate(traces[:limit], start=1):
        evidence = (trace.get("text_evidence") or []) + (trace.get("graph_evidence") or [])
        snippet = str(evidence[0].get("text", ""))[:420] if evidence else "no retrieved support"
        parts.append(
            f"Claim {index} (support={trace.get('support_score', 0)}, status={trace.get('status', 'unknown')}): "
            f"{trace.get('claim', '')}\nEvidence: {snippet}"
        )
    return "\n".join(parts) or "No candidate-conditioned evidence was retrieved."


def candidate_order(line_no: int, candidates: dict[str, str]) -> list[str]:
    return sorted(
        candidates,
        key=lambda name: hashlib.sha1(f"{line_no}:{name}".encode("utf-8")).hexdigest(),
    )


def normalize_result(raw: dict, labels: list[str]) -> dict:
    winner = str(raw.get("winner", "")).strip().upper()
    if winner not in labels:
        winner = labels[0]
    scores = {}
    raw_scores = raw.get("scores") or {}
    for label in labels:
        item = raw_scores.get(label) or {}
        scores[label] = {
            "grounding": clamp(item.get("grounding")),
            "coverage": clamp(item.get("coverage")),
            "relevance": clamp(item.get("relevance")),
            "conciseness": clamp(item.get("conciseness")),
        }
    return {
        "winner": winner,
        "confidence": clamp(raw.get("confidence")),
        "scores": scores,
        "decisive_facts": [str(item).strip() for item in raw.get("decisive_facts", []) if str(item).strip()],
        "reason": str(raw.get("reason", "")).strip(),
    }


def candidate_utility(scores: dict) -> float:
    return (
        0.32 * scores.get("grounding", 0.0)
        + 0.38 * scores.get("coverage", 0.0)
        + 0.20 * scores.get("relevance", 0.0)
        + 0.10 * scores.get("conciseness", 0.0)
    )


def select_action(result: dict, label_to_name: dict[str, str], candidates: dict[str, str], min_gain: float) -> tuple[str, dict]:
    name_to_label = {name: label for label, name in label_to_name.items()}
    text_label = name_to_label["text"]
    winner_label = result["winner"]
    winner_name = label_to_name[winner_label]
    text_utility = candidate_utility(result["scores"][text_label])
    winner_utility = candidate_utility(result["scores"][winner_label])
    utility_gain = winner_utility - text_utility
    accepted = bool(
        winner_name == "text"
        or (
            result["confidence"] >= 0.55
            and utility_gain >= min_gain
            and result["scores"][winner_label]["grounding"] >= 0.60
        )
    )
    selected = winner_name if accepted else "text"
    return selected, {
        "raw_winner": winner_name,
        "selected_candidate": selected,
        "winner_accepted": accepted,
        "text_utility": round(text_utility, 6),
        "winner_utility": round(winner_utility, 6),
        "utility_gain": round(utility_gain, 6),
        "minimum_utility_gain": min_gain,
    }


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("DEEPSEEK_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete")

    text_rows = by_line(load_jsonl(ROOT_DIR / args.text))
    graph_rows = by_line(load_jsonl(ROOT_DIR / args.graph))
    completion_rows = by_line(load_jsonl(ROOT_DIR / args.completion))
    text_audits = by_line(load_jsonl(ROOT_DIR / args.text_audit)) if args.text_audit else {}
    graph_audits = by_line(load_jsonl(ROOT_DIR / args.graph_audit)) if args.graph_audit else {}
    common_lines = sorted(set(text_rows) & set(graph_rows) & set(completion_rows))
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=150.0, max_retries=2)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def arbitrate(line_no: int) -> dict:
        text_row = text_rows[line_no]
        graph_row = graph_rows[line_no]
        completion_row = completion_rows[line_no]
        completion_agent = completion_row.get("contract_completion_agent") or {}
        evidence_delta_guard = completion_row.get("evidence_delta_guard") or {}
        candidates = {
            "text": str(text_row.get("prediction", "")),
            "graph": str(graph_row.get("prediction", "")),
        }
        repair_allowed = bool(completion_agent.get("repair_accepted"))
        if evidence_delta_guard.get("enabled"):
            repair_allowed = repair_allowed and evidence_delta_guard.get("action") == "retain_repair"
        if repair_allowed:
            candidates["repair"] = str(completion_row.get("prediction", ""))
        order = candidate_order(line_no, candidates)
        labels = [chr(ord("A") + index) for index in range(len(order))]
        label_to_name = dict(zip(labels, order))
        audit_agents = {
            "text": (text_audits.get(line_no, {}).get("claim_evidence_agent") or {}),
            "graph": (graph_audits.get(line_no, {}).get("claim_evidence_agent") or {}),
            "repair": (completion_agent.get("repair_guard") or {}),
        }
        candidate_prompt = "\n\n".join(
            f"Candidate {label}:\n{candidates[name]}\n\n"
            f"Candidate {label} claim-conditioned evidence:\n{candidate_evidence(audit_agents[name])}"
            for label, name in label_to_name.items()
        )
        prompt = USER_PROMPT.format(
            question=text_row.get("question", ""),
            contract=json.dumps(completion_agent.get("contract") or {}, ensure_ascii=False),
            evidence=evidence_text(completion_agent),
            candidates=candidate_prompt,
        )
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
            result = normalize_result(parse_json_object(response.choices[0].message.content or ""), labels)
            selected, decision = select_action(result, label_to_name, candidates, args.min_utility_gain)
            error = None
        except Exception as exc:
            result = {}
            selected = "text"
            decision = {
                "raw_winner": "unknown",
                "selected_candidate": "text",
                "winner_accepted": False,
                "utility_gain": 0.0,
                "minimum_utility_gain": args.min_utility_gain,
            }
            error = f"{type(exc).__name__}: {exc}"[:300]
        row = dict(text_row)
        row["prediction_before_candidate_arbitration"] = row.get("prediction", "")
        row["prediction"] = candidates[selected]
        row["method"] = f"sage_v35_evidence_arbiter_{selected}"
        row["evidence_candidate_arbiter"] = {
            "enabled": True,
            "version": "v35",
            "candidate_order": label_to_name,
            "candidate_lengths": {name: len(value) for name, value in candidates.items()},
            "repair_allowed": repair_allowed,
            "evidence_delta_action": evidence_delta_guard.get("action"),
            "result": result,
            "decision": decision,
            "error": error,
            "reference_answer_used": False,
        }
        if selected != "text":
            row[f"{selected}_prediction_before_candidate_arbitration"] = candidates[selected]
        return row

    rows = await asyncio.gather(*(arbitrate(line_no) for line_no in common_lines))
    output_path = ROOT_DIR / args.output
    write_jsonl(rows, output_path)
    counts: dict[str, int] = {}
    errors = 0
    for row in rows:
        agent = row["evidence_candidate_arbiter"]
        selected = agent["decision"]["selected_candidate"]
        counts[selected] = counts.get(selected, 0) + 1
        errors += bool(agent.get("error"))
    lines = [
        "# SAGE v35 Evidence-Calibrated Candidate Arbiter Summary\n\n",
        f"- Samples: {len(rows)}\n",
        f"- API errors: {errors}\n",
        "- Candidates are anonymized and deterministically reordered per sample.\n",
        "- Reference answers and judge scores are not arbitration inputs.\n\n",
        "| Selected candidate | Count |\n",
        "|---|---:|\n",
    ]
    for name, count in sorted(counts.items()):
        lines.append(f"| {name} | {count} |\n")
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved v35 candidate arbitration results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--text-audit", default=None)
    parser.add_argument("--graph-audit", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--min-utility-gain", type=float, default=0.05)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
