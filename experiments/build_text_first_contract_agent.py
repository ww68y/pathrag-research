import argparse
import json
import re
from collections import Counter
from pathlib import Path

from build_answer_contract_agent import (
    UNCERTAINTY_PATTERNS,
    answer_shape_signals,
    clean_text,
    derive_answer_contract,
    load_jsonl,
    row_map,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def has_explicit_uncertainty(answer: str) -> list[str]:
    lowered = clean_text(answer).lower()
    return [pattern for pattern in UNCERTAINTY_PATTERNS if re.search(pattern, lowered)]


def route_text_answer(row: dict) -> tuple[str, dict, dict, list[str]]:
    question = clean_text(row.get("question"))
    answer = clean_text(row.get("prediction"))
    contract = derive_answer_contract(question)
    signals = answer_shape_signals(row, contract)
    uncertainty = has_explicit_uncertainty(answer)
    contract_type = contract.get("type")
    reasons = []

    if not answer:
        reasons.append("empty_text_answer")
    elif signals.get("answer_chars", 0) < 100:
        reasons.append("answer_too_short_for_reliable_use")
    if uncertainty:
        reasons.append("explicit_uncertainty:" + ",".join(uncertainty[:3]))

    if contract_type == "comparison" and (
        signals.get("contrast_markers", 0) < 1
        or signals.get("estimated_distinct_points", 0) < contract.get("minimum_distinct_points", 2)
    ):
        reasons.append("comparison_contract_incomplete")

    if contract_type == "definition_and_purpose" and (
        signals.get("sentence_count", 0) < 2 or signals.get("answer_chars", 0) < 260
    ):
        reasons.append("definition_purpose_contract_incomplete")

    if contract_type in {"enumeration", "procedure"} and (
        signals.get("estimated_distinct_points", 0) < contract.get("minimum_distinct_points", 2)
    ):
        reasons.append("multi_item_contract_incomplete")

    if contract_type == "numeric_lookup" and signals.get("numeric_mentions", 0) == 0:
        reasons.append("numeric_value_missing")

    if contract_type == "mechanism" and (
        signals.get("answer_chars", 0) < 180
        or (signals.get("sentence_count", 0) < 2 and signals.get("causal_markers", 0) < 1)
    ):
        reasons.append("mechanism_contract_too_thin")

    if contract_type == "literal_and_metaphorical" and (
        signals.get("sentence_count", 0) < 2 or signals.get("answer_chars", 0) < 260
    ):
        reasons.append("literal_metaphorical_contract_incomplete")

    action = "graph_rescue" if reasons else "keep_text"
    return action, contract, signals, reasons or ["text_answer_contract_satisfied"]


def rescue_utility_score(signals: dict, contract_type: str) -> float:
    score = 0.0
    if signals.get("answer_chars", 0) >= 100:
        score += 1.0
    score += min(3.0, float(signals.get("estimated_distinct_points", 0))) * 0.35
    if contract_type == "comparison":
        score += min(2.0, float(signals.get("contrast_markers", 0)))
    elif contract_type == "numeric_lookup":
        score += min(2.0, float(signals.get("numeric_mentions", 0)))
    elif contract_type == "mechanism":
        score += min(2.0, float(signals.get("causal_markers", 0))) * 0.7
    elif contract_type in {"enumeration", "procedure", "compound", "literal_and_metaphorical"}:
        score += min(3.0, float(signals.get("sentence_count", 0))) * 0.25
    return score


def accept_graph_rescue(
    text_row: dict,
    graph_row: dict,
    contract: dict,
    text_signals: dict,
    trigger_reasons: list[str],
) -> tuple[bool, dict]:
    text_answer = clean_text(text_row.get("prediction"))
    graph_answer = clean_text(graph_row.get("prediction"))
    graph_signals = answer_shape_signals(graph_row, contract)
    text_uncertainty = has_explicit_uncertainty(text_answer)
    graph_uncertainty = has_explicit_uncertainty(graph_answer)
    contract_type = contract.get("type", "general")
    text_utility = rescue_utility_score(text_signals, contract_type)
    graph_utility = rescue_utility_score(graph_signals, contract_type)
    utility_gain = graph_utility - text_utility
    length_ratio = len(graph_answer) / max(1, len(text_answer))
    reasons = []

    if not graph_answer or graph_uncertainty:
        reasons.append("reject_empty_or_uncertain_graph_answer")
        accepted = False
    elif text_uncertainty and not graph_uncertainty:
        reasons.append("accept_graph_over_uncertain_text")
        accepted = True
    elif len(text_answer) < 100 and len(graph_answer) > len(text_answer):
        reasons.append("accept_graph_over_too_short_text")
        accepted = True
    else:
        required_gain = 1.5 if length_ratio > 3.0 else 0.5
        accepted = utility_gain >= required_gain
        reasons.append(
            f"{'accept' if accepted else 'reject'}_utility_gain:{utility_gain:.3f}>={required_gain:.3f}"
        )
        if length_ratio > 3.0:
            reasons.append(f"verbosity_penalty:length_ratio={length_ratio:.3f}")

    return accepted, {
        "enabled": True,
        "accepted": accepted,
        "trigger_reasons": trigger_reasons,
        "decision_reasons": reasons,
        "text_utility": round(text_utility, 4),
        "graph_utility": round(graph_utility, 4),
        "utility_gain": round(utility_gain, 4),
        "length_ratio": round(length_ratio, 4),
        "graph_signals": graph_signals,
    }


def write_summary(rows: list[dict], path: Path, args: argparse.Namespace, pathrag_rows: dict[int, dict]) -> None:
    actions = Counter((row.get("text_first_contract_agent") or {}).get("action", "unknown") for row in rows)
    contracts = Counter(
        (row.get("text_first_contract_agent") or {}).get("contract", {}).get("type", "unknown")
        for row in rows
    )
    text_seconds = sum(
        as_float((row.get("text_first_contract_agent") or {}).get("text_query_seconds")) for row in rows
    )
    rescue_seconds = sum(
        as_float((row.get("text_first_contract_agent") or {}).get("rescue_query_seconds")) for row in rows
    )
    rescue_attempts = sum(
        1 for row in rows if (row.get("text_first_contract_agent") or {}).get("graph_rescue_attempted")
    )
    rescue_accepts = sum(
        1 for row in rows if (row.get("text_first_contract_agent") or {}).get("graph_rescue_accepted")
    )

    all_graph_contexts = {}
    selected_graph_contexts = {}
    selected_lines = {row.get("line_no") for row in rows}
    for line_no, pathrag_row in pathrag_rows.items():
        if line_no not in selected_lines:
            continue
        context_hash = pathrag_row.get("context_hash")
        if not context_hash:
            continue
        all_graph_contexts.setdefault(context_hash, as_float(pathrag_row.get("insert_seconds")))

    for row in rows:
        agent = row.get("text_first_contract_agent") or {}
        if not agent.get("graph_rescue_attempted"):
            continue
        pathrag_row = pathrag_rows.get(row.get("line_no"), {})
        context_hash = pathrag_row.get("context_hash")
        if context_hash:
            selected_graph_contexts.setdefault(context_hash, as_float(pathrag_row.get("insert_seconds")))

    full_pathrag_query_seconds = sum(
        as_float(pathrag_rows.get(line_no, {}).get("query_seconds")) for line_no in selected_lines
    )
    full_index_seconds = sum(all_graph_contexts.values())
    selected_index_seconds = sum(selected_graph_contexts.values())
    index_reduction = 1 - selected_index_seconds / max(1.0, full_index_seconds)
    rescue_query_reduction = 1 - rescue_seconds / max(1.0, full_pathrag_query_seconds)

    lines = [
        "# SAGE v21 Text-First Contract and Rescue-Utility Agent Summary\n\n",
        f"- Text base: `{args.text_base}`\n",
        f"- Graph rescue: `{args.graph_rescue}`\n",
        f"- Output: `{args.output}`\n",
        f"- Line interval: {args.start_line} to {args.end_line if args.end_line is not None else 'end'}\n",
        f"- Samples: {len(rows)}\n",
        "- Initial routing inputs: question and current text answer only\n",
        "- A graph answer is read only after the initial contract triggers graph rescue\n",
        "- Reference answers and evaluation scores are never read by routing or utility decisions\n\n",
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
        "\n## Replay Cost\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Text-stage query time | {text_seconds:.3f} s |\n",
        f"| Graph rescue attempts | {rescue_attempts} |\n",
        f"| Graph rescue answers accepted | {rescue_accepts} |\n",
        f"| Graph rescue answers rejected | {rescue_attempts - rescue_accepts} |\n",
        f"| Added graph-rescue query time | {rescue_seconds:.3f} s |\n",
        f"| Full PathRAG query time on the same samples | {full_pathrag_query_seconds:.3f} s |\n",
        f"| Graph-rescue query reduction vs full PathRAG | {rescue_query_reduction * 100:.2f}% |\n",
        f"| Selected graph contexts | {len(selected_graph_contexts)} / {len(all_graph_contexts)} |\n",
        f"| Selected graph indexing time | {selected_index_seconds:.3f} s |\n",
        f"| Full graph indexing time | {full_index_seconds:.3f} s |\n",
        f"| Graph indexing reduction | {index_reduction * 100:.2f}% |\n",
        "\n## Per-Sample Routing\n\n",
        "| Line | Contract | Action | Text chars | Utility decision | Reasons |\n",
        "|---:|---|---|---:|---|---|\n",
    ])
    for row in sorted(rows, key=lambda item: item.get("line_no", -1)):
        agent = row.get("text_first_contract_agent") or {}
        reasons = "; ".join(agent.get("reasons") or []).replace("|", "/")
        utility = "; ".join(
            (agent.get("rescue_utility_guard") or {}).get("decision_reasons") or []
        ).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {agent.get('contract', {}).get('type')} | "
            f"{agent.get('action')} | {agent.get('signals', {}).get('answer_chars')} | "
            f"{utility} | {reasons} |\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-base", required=True)
    parser.add_argument("--graph-rescue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--start-line", type=int, default=0)
    parser.add_argument("--end-line", type=int)
    args = parser.parse_args()

    text_rows = load_jsonl(ROOT_DIR / args.text_base)
    pathrag_rows = row_map(ROOT_DIR / args.graph_rescue)
    output_rows = []
    for row in text_rows:
        line_no = row.get("line_no")
        if line_no is None or line_no < args.start_line:
            continue
        if args.end_line is not None and line_no > args.end_line:
            continue

        action, contract, signals, reasons = route_text_answer(row)
        new_row = dict(row)
        rescue = pathrag_rows.get(line_no)
        rescue_seconds = 0.0
        rescue_attempted = action == "graph_rescue"
        rescue_accepted = False
        utility_guard = {"enabled": True, "accepted": False, "decision_reasons": ["rescue_not_triggered"]}
        if rescue_attempted and rescue and clean_text(rescue.get("prediction")):
            rescue_seconds = as_float(rescue.get("query_seconds"))
            rescue_accepted, utility_guard = accept_graph_rescue(row, rescue, contract, signals, reasons)
            if rescue_accepted:
                new_row["prediction_before_text_first_graph_rescue"] = row.get("prediction", "")
                new_row["prediction"] = rescue.get("prediction", "")
                new_row["method"] = "sage_v21_text_first_contract_utility_guard"
                action = "graph_rescue"
            else:
                new_row["rejected_graph_rescue_prediction"] = rescue.get("prediction", "")
                action = "keep_text_after_utility_guard"
        elif rescue_attempted:
            reasons.append("missing_graph_rescue_prediction_keep_text")
            action = "keep_text"

        new_row["text_first_contract_agent"] = {
            "enabled": True,
            "action": action,
            "contract": contract,
            "signals": signals,
            "reasons": reasons,
            "text_query_seconds": round(as_float(row.get("query_seconds")), 3),
            "rescue_query_seconds": round(rescue_seconds, 3),
            "requires_graph": rescue_attempted,
            "graph_rescue_attempted": rescue_attempted,
            "graph_rescue_accepted": rescue_accepted,
            "rescue_utility_guard": utility_guard,
            "base_answer_chars": len(clean_text(row.get("prediction", ""))),
            "final_answer_chars": len(clean_text(new_row.get("prediction", ""))),
        }
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in output_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_summary(output_rows, ROOT_DIR / args.summary, args, pathrag_rows)
    print(f"Saved SAGE v21 results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
