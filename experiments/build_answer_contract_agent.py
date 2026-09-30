import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]


SYSTEM_PROMPT = """You are an evidence-sufficiency verifier for retrieval-augmented QA.
You must not use outside knowledge or any reference answer.
Judge only whether the draft answer fulfills the question-derived answer contract and is supported by the supplied evidence.
Return JSON only."""


USER_PROMPT = """Question:
{question}

Question-derived answer contract:
{contract}

Current draft answer:
{answer}

Retrieved evidence excerpts:
{evidence}

Runtime signals:
{signals}

Check the draft without using outside knowledge:
1. Determine which required facets are clearly answered.
2. List required facets that are missing or only vaguely answered.
3. Identify claims that are unsupported by the supplied evidence or drift away from the question.
4. Estimate answer-contract coverage and evidence grounding from 0.0 to 1.0.
5. Recommend one action:
   - keep: the answer is sufficiently complete and grounded;
   - graph_rescue: broader relational/path evidence is needed for comparison, mechanism, multiple aspects, or missing relations;
   - text_rescue: a narrow factual lookup needs a more direct text passage.

Return this exact JSON shape:
{{
  "covered_facets": ["..."],
  "missing_facets": ["..."],
  "unsupported_claims": ["..."],
  "off_topic_claims": ["..."],
  "contract_coverage": 0.0,
  "grounding_score": 0.0,
  "risk_level": "low",
  "recommended_action": "keep",
  "reason": "..."
}}"""


UNCERTAINTY_PATTERNS = [
    r"\bcannot be determined\b",
    r"\bcannot determine\b",
    r"\bcannot be answered\b",
    r"\bcannot answer\b",
    r"\bnot specified\b",
    r"\bdoes not specify\b",
    r"\bdoes not contain\b",
    r"\bdoes not mention\b",
    r"\bno specific information\b",
    r"\bno information\b",
    r"\bnot provided\b",
    r"\binsufficient evidence\b",
]


def clean_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())


def clamp_score(value: object, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_map(path: Path) -> dict[int, dict]:
    return {row.get("line_no"): row for row in load_jsonl(path)}


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"Could not parse verifier JSON: {text[:500]}")


def derive_answer_contract(question: str) -> dict:
    lowered = clean_text(question).lower()
    facets = ["direct_answer"]
    contract_type = "general"
    minimum_points = 1

    comparison = bool(
        re.search(r"\b(differ|differs|differed|differing|compare|contrast)\b", lowered)
        or re.search(r"\bdifferences?\s+between\b", lowered)
    )
    compound = " and " in lowered and bool(re.search(r"\b(what|how|why|where|when|who)\b", lowered))
    purpose = bool(re.search(r"\b(aim|purpose|objective|achieve)\b", lowered))
    enumeration = bool(
        re.search(
            r"\b(methods?|measures?|ways?|factors?|reasons?|challenges?|solutions?|steps?|types?|examples?)\b",
            lowered,
        )
    )
    numeric = bool(re.search(r"\b(rate|percentage|percent|how many|how much|number|amount|price)\b", lowered))
    mechanism = bool(
        re.search(r"\b(how does|how do|how can|respond|affect|influence|impact|role|cause|mechanism)\b", lowered)
    )
    relation_lookup = bool(
        re.search(r"\b(what is the role|what role|spread from|transmission of|transmission from)\b", lowered)
    )
    metaphor = bool(re.search(r"\b(metaphor|metaphorically|symbol|symbolic)\b", lowered))
    procedure = bool(re.search(r"\b(procedure|process|steps?|requirements?|conditions?)\b", lowered))

    if comparison:
        contract_type = "comparison"
        facets.extend(["comparison_basis", "contrasting_facts"])
        minimum_points = 2
    elif purpose and compound:
        contract_type = "definition_and_purpose"
        facets.extend(["definition", "specific_objective", "scope_or_method_if_supported"])
        minimum_points = 2
    elif metaphor and compound:
        contract_type = "literal_and_metaphorical"
        facets.extend(["literal_concept", "metaphorical_use", "supported_example_or_interpretation"])
        minimum_points = 2
    elif enumeration:
        contract_type = "enumeration"
        facets.extend(["multiple_distinct_items", "item_relevance"])
        minimum_points = 3
    elif procedure:
        contract_type = "procedure"
        facets.extend(["ordered_or_conditional_steps", "actors_and_actions"])
        minimum_points = 2
    elif numeric:
        contract_type = "numeric_lookup"
        facets.extend(["numeric_value", "range_or_condition_if_supported"])
    elif relation_lookup:
        contract_type = "relation_lookup"
        facets.extend(["direct_relation", "supported_relation_detail"])
    elif mechanism:
        contract_type = "mechanism"
        facets.extend(["mechanism_or_relation", "effect_or_outcome"])
        minimum_points = 2
    elif compound:
        contract_type = "compound"
        facets.extend(["first_question_part", "second_question_part"])
        minimum_points = 2

    return {
        "type": contract_type,
        "required_facets": list(dict.fromkeys(facets)),
        "minimum_distinct_points": minimum_points,
        "comparison_required": comparison,
        "compound_question": compound,
        "numeric_answer_expected": numeric,
    }


def answer_shape_signals(row: dict, contract: dict) -> dict:
    answer = clean_text(row.get("prediction", ""))
    lowered = answer.lower()
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", answer) if part]
    list_markers = len(re.findall(r"(?:^|\s)(?:\d+[.)]|[-*])\s+", answer))
    conjunction_points = len(re.findall(r"\b(?:and|also|additionally|furthermore|while|whereas|but)\b", lowered))
    contrast_markers = len(re.findall(r"\b(?:while|whereas|unlike|however|in contrast|differ)\b", lowered))
    causal_markers = len(re.findall(r"\b(?:because|therefore|through|by|which leads|result|affect|influence|cause)\b", lowered))
    numeric_mentions = len(re.findall(r"(?:\$?\d[\d,.]*\s*%?)", answer))
    uncertainty = [pattern for pattern in UNCERTAINTY_PATTERNS if re.search(pattern, lowered)]

    mse_stats = row.get("mse_stats") or {}
    refiner_stats = row.get("refiner_stats") or {}
    coverage_guard = refiner_stats.get("coverage_guard") or {}
    return {
        "answer_chars": len(answer),
        "sentence_count": len(sentences),
        "estimated_distinct_points": max(len(sentences), list_markers + 1, conjunction_points + 1),
        "contrast_markers": contrast_markers,
        "causal_markers": causal_markers,
        "numeric_mentions": numeric_mentions,
        "uncertainty_patterns": uncertainty,
        "term_coverage": clamp_score(mse_stats.get("term_coverage"), 1.0),
        "sufficiency_status": clean_text(mse_stats.get("sufficiency_status")),
        "sufficiency_score": clamp_score(mse_stats.get("sufficiency_score"), 1.0),
        "repair_triggered": bool(mse_stats.get("repair_triggered")),
        "risk_control_active": bool(mse_stats.get("risk_control_active")),
        "refiner_failed": bool(refiner_stats.get("failed")),
        "supported_term_preservation": clamp_score(coverage_guard.get("supported_term_preservation"), 1.0),
        "contract_type": contract.get("type"),
    }


def deterministic_verification(contract: dict, signals: dict) -> dict:
    contract_type = contract.get("type")
    missing = []
    reasons = []

    if contract_type == "comparison" and (
        signals.get("term_coverage", 1.0) < 0.70
        or signals.get("sufficiency_status") == "repair_needed"
    ):
        missing.append("complete_comparison_dimensions")
        reasons.append("comparison evidence coverage is below the safe threshold")

    if contract_type == "definition_and_purpose" and (
        signals.get("repair_triggered") and signals.get("answer_chars", 0) < 320
    ):
        missing.append("specific_objective_or_scope")
        reasons.append("repaired evidence produced only a short compound answer")

    if contract_type == "literal_and_metaphorical" and (
        signals.get("supported_term_preservation", 1.0) < 0.65
        or signals.get("estimated_distinct_points", 0) < contract.get("minimum_distinct_points", 2)
    ):
        missing.append("supported_literal_and_metaphorical_facets")
        reasons.append("compound answer lost too many supported terms")

    if contract_type in {"enumeration", "procedure"} and signals.get("refiner_failed"):
        missing.append("complete_distinct_items_or_steps")
        reasons.append("answer refinement failed for a multi-item contract")

    if contract_type == "numeric_lookup" and (
        signals.get("numeric_mentions", 0) == 0
        or signals.get("sufficiency_status") == "repair_needed"
        or signals.get("term_coverage", 1.0) < 0.55
    ):
        missing.append("verified_numeric_range_or_condition")
        reasons.append("numeric answer is backed by low-sufficiency evidence")

    if contract_type == "mechanism" and (
        signals.get("risk_control_active")
        and signals.get("answer_chars", 0) < 360
        and signals.get("sentence_count", 0) <= 2
    ):
        missing.append("complete_mechanism_and_outcome")
        reasons.append("risk-gated mechanism answer is unusually short")

    if signals.get("uncertainty_patterns"):
        missing.append("confident_evidence_grounded_answer")
        reasons.append("draft contains an explicit uncertainty phrase")

    if signals.get("refiner_failed") and not missing:
        missing.append("successful_answer_refinement")
        reasons.append("answer refinement failed")

    required_count = max(1, len(contract.get("required_facets") or []))
    coverage = max(0.0, 1.0 - len(missing) / required_count)
    grounding = min(
        signals.get("sufficiency_score", 1.0),
        signals.get("supported_term_preservation", 1.0),
    )
    high_risk = bool(missing)
    return {
        "covered_facets": [
            facet for facet in contract.get("required_facets", [])
            if facet not in missing
        ],
        "missing_facets": missing,
        "unsupported_claims": [],
        "off_topic_claims": [],
        "contract_coverage": round(coverage, 4),
        "grounding_score": round(grounding, 4),
        "risk_level": "high" if high_risk else "low",
        "recommended_action": "graph_rescue" if high_risk else "keep",
        "reason": "; ".join(reasons) if reasons else "deterministic answer-contract checks passed",
        "verification_mode": "deterministic_runtime_guard",
    }


def evidence_excerpt(row: dict, max_chars: int) -> str:
    trace = row.get("evidence_trace") or {}
    parts = []
    for edge in (trace.get("selected_edges") or [])[:10]:
        path = clean_text(edge.get("path"))
        description = clean_text(edge.get("description"))
        if path or description:
            parts.append(f"PATH: {path}\nEVIDENCE: {description}")
    for chunk in (trace.get("supporting_chunks") or row.get("supporting_chunks") or [])[:4]:
        snippet = clean_text(chunk.get("snippet") or chunk.get("content"))
        if snippet:
            parts.append(f"TEXT CHUNK: {snippet}")
    excerpt = "\n\n".join(parts)
    return excerpt[:max_chars] if excerpt else "No evidence excerpt was recorded."


def normalize_verification(raw: dict) -> dict:
    action = clean_text(raw.get("recommended_action")).lower()
    if action not in {"keep", "graph_rescue", "text_rescue"}:
        action = "keep"
    risk = clean_text(raw.get("risk_level")).lower()
    if risk not in {"low", "medium", "high"}:
        risk = "medium"
    return {
        "covered_facets": [clean_text(item) for item in raw.get("covered_facets", []) if clean_text(item)],
        "missing_facets": [clean_text(item) for item in raw.get("missing_facets", []) if clean_text(item)],
        "unsupported_claims": [clean_text(item) for item in raw.get("unsupported_claims", []) if clean_text(item)],
        "off_topic_claims": [clean_text(item) for item in raw.get("off_topic_claims", []) if clean_text(item)],
        "contract_coverage": clamp_score(raw.get("contract_coverage"), 0.5),
        "grounding_score": clamp_score(raw.get("grounding_score"), 0.5),
        "risk_level": risk,
        "recommended_action": action,
        "reason": clean_text(raw.get("reason")),
    }


def select_agent_action(contract: dict, signals: dict, verification: dict, threshold: float) -> tuple[str, list[str]]:
    reasons = []
    if signals.get("uncertainty_patterns"):
        reasons.append("explicit_uncertainty")
    if signals.get("refiner_failed"):
        reasons.append("refiner_failed")
    if signals.get("sufficiency_status") == "repair_needed" and signals.get("term_coverage", 1.0) < 0.65:
        reasons.append("low_evidence_sufficiency")
    if verification.get("contract_coverage", 1.0) < threshold:
        reasons.append(f"low_contract_coverage:{verification.get('contract_coverage', 0):.3f}")
    if verification.get("grounding_score", 1.0) < threshold:
        reasons.append(f"low_grounding:{verification.get('grounding_score', 0):.3f}")
    if verification.get("risk_level") == "high":
        reasons.append("high_verifier_risk")
    if verification.get("missing_facets"):
        reasons.append("missing_facets:" + ",".join(verification["missing_facets"][:4]))
    if verification.get("unsupported_claims"):
        reasons.append("unsupported_claims")
    if verification.get("off_topic_claims"):
        reasons.append("off_topic_drift")

    requested = verification.get("recommended_action", "keep")
    high_risk = (
        signals.get("refiner_failed")
        or verification.get("risk_level") == "high"
        or verification.get("contract_coverage", 1.0) < threshold
        or verification.get("grounding_score", 1.0) < threshold
    )
    structural_risk = contract.get("type") in {
        "comparison",
        "definition_and_purpose",
        "literal_and_metaphorical",
        "compound",
        "enumeration",
        "procedure",
        "mechanism",
    } and bool(verification.get("missing_facets"))

    if requested == "graph_rescue" and (high_risk or structural_risk):
        return "graph_rescue", reasons + ["verifier_requested_graph_rescue"]
    if requested == "text_rescue" and high_risk:
        return "text_rescue", reasons + ["verifier_requested_text_rescue"]
    if signals.get("refiner_failed") and structural_risk:
        return "graph_rescue", reasons + ["failed_refinement_with_structural_risk"]
    if signals.get("uncertainty_patterns") and requested != "keep":
        return requested, reasons + ["uncertainty_rescue"]
    return "keep", reasons or ["answer_contract_satisfied"]


async def verify_item(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    contract: dict,
    signals: dict,
    max_evidence_chars: int,
    timeout: int,
    retries: int,
) -> tuple[dict, float]:
    prompt = USER_PROMPT.format(
        question=clean_text(row.get("question")),
        contract=json.dumps(contract, ensure_ascii=False),
        answer=clean_text(row.get("prediction")),
        evidence=evidence_excerpt(row, max_evidence_chars),
        signals=json.dumps(signals, ensure_ascii=False),
    )
    last_error = None
    for attempt in range(retries + 1):
        started = time.time()
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
                timeout=timeout,
            )
            raw = parse_json_object(response.choices[0].message.content or "")
            return normalize_verification(raw), time.time() - started
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(1 + attempt)
    raise RuntimeError(f"Answer-contract verification failed after {retries + 1} attempt(s): {last_error}")


def write_summary(rows: list[dict], path: Path, args: argparse.Namespace) -> None:
    actions = Counter((row.get("answer_contract_agent") or {}).get("action", "unknown") for row in rows)
    contracts = Counter((row.get("answer_contract_agent") or {}).get("contract", {}).get("type", "unknown") for row in rows)
    verifier_seconds = sum(float((row.get("answer_contract_agent") or {}).get("verifier_seconds", 0) or 0) for row in rows)
    rescue_seconds = sum(float((row.get("answer_contract_agent") or {}).get("rescue_query_seconds", 0) or 0) for row in rows)
    graph_rescues = sum(1 for row in rows if (row.get("answer_contract_agent") or {}).get("action") == "graph_rescue")

    lines = [
        "# SAGE v19 Answer-Contract Agent Summary\n\n",
        f"- Base: `{args.base}`\n",
        f"- Graph rescue: `{args.graph_rescue}`\n",
        f"- Text rescue: `{args.text_rescue or 'none'}`\n",
        f"- Output: `{args.output}`\n",
        f"- Samples: {len(rows)}\n",
        "- Routing inputs: question, current prediction, retrieved evidence trace, and runtime signals only\n",
        "- Reference answers are not supplied to the verifier or routing policy\n\n",
        "## Action Counts\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")

    lines.extend(["\n## Contract Types\n\n", "| Contract | Count |\n", "|---|---:|\n"])
    for contract, count in sorted(contracts.items()):
        lines.append(f"| {contract} | {count} |\n")

    lines.extend([
        "\n## Runtime\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Verifier API time | {verifier_seconds:.3f} s |\n",
        f"| Added rescue query time | {rescue_seconds:.3f} s |\n",
        f"| Graph rescue rate | {graph_rescues}/{len(rows)} |\n",
        "\n## Per-Sample Decisions\n\n",
        "| Line | Contract | Action | Coverage | Grounding | Risk | Missing facets | Reason |\n",
        "|---:|---|---|---:|---:|---|---|---|\n",
    ])
    for row in sorted(rows, key=lambda item: item.get("line_no", -1)):
        agent = row.get("answer_contract_agent") or {}
        verification = agent.get("verification") or {}
        reason = "; ".join(agent.get("decision_reasons") or []).replace("|", "/")
        missing = ", ".join(verification.get("missing_facets") or []).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {agent.get('contract', {}).get('type')} | {agent.get('action')} | "
            f"{verification.get('contract_coverage')} | {verification.get('grounding_score')} | "
            f"{verification.get('risk_level')} | {missing or 'none'} | {reason} |\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("DEEPSEEK_MODEL")
    if not args.deterministic_only and not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in .env")
    if not args.deterministic_only and not base_url:
        raise RuntimeError("DEEPSEEK_API_BASE is missing in .env")
    if not args.deterministic_only and not model:
        raise RuntimeError("DEEPSEEK_MODEL is missing in .env")

    base_rows = load_jsonl(ROOT_DIR / args.base)
    if args.max_samples is not None:
        base_rows = base_rows[: args.max_samples]
    graph_rescue = row_map(ROOT_DIR / args.graph_rescue)
    text_rescue = row_map(ROOT_DIR / args.text_rescue) if args.text_rescue else {}
    output_path = ROOT_DIR / args.output
    summary_path = ROOT_DIR / args.summary
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if args.resume and output_path.exists():
        existing = row_map(output_path)

    client = None if args.deterministic_only else AsyncOpenAI(api_key=api_key, base_url=base_url)
    output_rows = []
    for index, row in enumerate(base_rows, start=1):
        line_no = row.get("line_no")
        if line_no in existing:
            output_rows.append(existing[line_no])
            print(f"[{index}/{len(base_rows)}] line={line_no} cached", flush=True)
            continue

        mode = "guarding" if args.deterministic_only else "verifying"
        print(f"[{index}/{len(base_rows)}] {mode} line={line_no}", flush=True)
        contract = derive_answer_contract(clean_text(row.get("question")))
        signals = answer_shape_signals(row, contract)
        new_row = dict(row)
        try:
            if args.deterministic_only:
                verification = deterministic_verification(contract, signals)
                verifier_seconds = 0.0
            else:
                verification, verifier_seconds = await verify_item(
                    client,
                    model,
                    row,
                    contract,
                    signals,
                    args.max_evidence_chars,
                    args.request_timeout,
                    args.request_retries,
                )
            action, reasons = select_agent_action(contract, signals, verification, args.contract_threshold)
        except Exception as exc:  # noqa: BLE001
            verification = {
                "covered_facets": [],
                "missing_facets": [],
                "unsupported_claims": [],
                "off_topic_claims": [],
                "contract_coverage": 0.0,
                "grounding_score": 0.0,
                "risk_level": "high",
                "recommended_action": "keep",
                "reason": clean_text(exc),
            }
            verifier_seconds = 0.0
            action = "keep"
            reasons = ["verifier_error_keep_base"]

        rescue_row = None
        if action == "graph_rescue":
            rescue_row = graph_rescue.get(line_no)
        elif action == "text_rescue":
            rescue_row = text_rescue.get(line_no)

        if rescue_row and clean_text(rescue_row.get("prediction")):
            new_row["prediction_before_answer_contract_rescue"] = row.get("prediction", "")
            new_row["prediction"] = rescue_row.get("prediction", "")
            new_row["method"] = f"{row.get('method', 'base')}_answer_contract_{action}"
            rescue_query_seconds = float(rescue_row.get("query_seconds", 0) or 0)
        else:
            if action != "keep":
                reasons.append("missing_rescue_prediction_keep_base")
            action = "keep"
            rescue_query_seconds = 0.0

        new_row["answer_contract_agent"] = {
            "enabled": True,
            "action": action,
            "contract": contract,
            "signals": signals,
            "verification": verification,
            "decision_reasons": reasons,
            "verifier_seconds": round(verifier_seconds, 3),
            "rescue_query_seconds": round(rescue_query_seconds, 3),
            "base_answer_chars": len(clean_text(row.get("prediction", ""))),
            "final_answer_chars": len(clean_text(new_row.get("prediction", ""))),
        }
        output_rows.append(new_row)

        with output_path.open("w", encoding="utf-8") as out:
            for output_row in sorted(output_rows, key=lambda item: item.get("line_no", -1)):
                out.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    write_summary(output_rows, summary_path, args)
    print(f"Saved SAGE v19 results to {output_path}")
    print(f"Saved summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--graph-rescue", required=True)
    parser.add_argument("--text-rescue")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model")
    parser.add_argument("--contract-threshold", type=float, default=0.72)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Use question contracts and runtime signals without an extra verifier API call",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
