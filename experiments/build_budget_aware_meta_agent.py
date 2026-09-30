import argparse
import json
from collections import Counter
from pathlib import Path

from build_answer_contract_agent import clean_text, derive_answer_contract, load_jsonl, row_map


ROOT_DIR = Path(__file__).resolve().parents[1]
BALANCED_GRAPH_FIRST_CONTRACTS = {"enumeration", "numeric_lookup"}


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def graph_policy_query_seconds(row: dict) -> float:
    answer_contract = row.get("answer_contract_agent") or {}
    return as_float(row.get("query_seconds")) + as_float(answer_contract.get("rescue_query_seconds"))


def text_policy_query_seconds(row: dict) -> float:
    agent = row.get("text_first_contract_agent") or {}
    if agent:
        return as_float(agent.get("text_query_seconds")) + as_float(agent.get("rescue_query_seconds"))
    return as_float(row.get("query_seconds"))


def choose_policy(
    budget: str,
    text_row: dict,
    graph_row: dict | None,
) -> tuple[str, str, dict]:
    contract = derive_answer_contract(clean_text(text_row.get("question")))
    text_agent = text_row.get("text_first_contract_agent") or {}
    graph_available = bool(graph_row and clean_text(graph_row.get("prediction")))

    if budget == "economy" or not graph_available:
        reason = "budget:economy_text_first" if graph_available else "graph_policy_unavailable"
        return "text_first", reason, contract

    if budget == "quality":
        return "graph_first", "budget:quality_graph_first", contract

    if text_agent.get("graph_rescue_attempted"):
        return "graph_first", "balanced:text_contract_escalation", contract

    if contract.get("type") in BALANCED_GRAPH_FIRST_CONTRACTS:
        return "graph_first", f"balanced:high_risk_contract:{contract.get('type')}", contract

    text_signals = text_agent.get("signals") or {}
    if contract.get("type") == "mechanism" and int(text_signals.get("sentence_count", 0) or 0) < 4:
        return "graph_first", "balanced:thin_mechanism_answer", contract

    return "text_first", "balanced:simple_contract_text_first", contract


def build_rows(args: argparse.Namespace) -> tuple[list[dict], dict[int, dict]]:
    text_rows = load_jsonl(ROOT_DIR / args.text_policy)
    graph_rows = row_map(ROOT_DIR / args.graph_policy)
    output_rows = []

    for text_row in text_rows:
        line_no = text_row.get("line_no")
        if line_no is None or line_no < args.start_line:
            continue
        if args.end_line is not None and line_no > args.end_line:
            continue

        graph_row = graph_rows.get(line_no)
        chosen_policy, reason, contract = choose_policy(args.budget, text_row, graph_row)
        if chosen_policy == "graph_first" and graph_row:
            new_row = dict(graph_row)
            new_row["prediction_before_budget_meta_route"] = text_row.get("prediction", "")
            selected_query_seconds = graph_policy_query_seconds(graph_row)
            text_stage_seconds = 0.0
            if reason == "balanced:text_contract_escalation":
                text_stage_seconds = as_float(
                    (text_row.get("text_first_contract_agent") or {}).get("text_query_seconds")
                )
                selected_query_seconds += text_stage_seconds
            graph_required = True
        else:
            new_row = dict(text_row)
            selected_query_seconds = text_policy_query_seconds(text_row)
            text_stage_seconds = as_float(
                (text_row.get("text_first_contract_agent") or {}).get("text_query_seconds")
            )
            graph_required = bool(
                (text_row.get("text_first_contract_agent") or {}).get("graph_rescue_attempted")
            )

        new_row["method"] = f"sage_v22_budget_meta_{args.budget}_{chosen_policy}"
        new_row["budget_meta_agent"] = {
            "enabled": True,
            "budget": args.budget,
            "chosen_policy": chosen_policy,
            "reason": reason,
            "contract": contract,
            "graph_required": graph_required,
            "text_stage_seconds": round(text_stage_seconds, 3),
            "estimated_query_seconds": round(selected_query_seconds, 3),
            "text_policy_answer_chars": len(clean_text(text_row.get("prediction"))),
            "graph_policy_answer_chars": len(clean_text((graph_row or {}).get("prediction"))),
            "final_answer_chars": len(clean_text(new_row.get("prediction"))),
        }
        output_rows.append(new_row)

    return output_rows, graph_rows


def write_summary(
    rows: list[dict],
    graph_rows: dict[int, dict],
    pathrag_rows: dict[int, dict],
    path: Path,
    args: argparse.Namespace,
) -> None:
    policies = Counter((row.get("budget_meta_agent") or {}).get("chosen_policy") for row in rows)
    reasons = Counter((row.get("budget_meta_agent") or {}).get("reason") for row in rows)
    selected_lines = {row.get("line_no") for row in rows}
    query_seconds = sum(
        as_float((row.get("budget_meta_agent") or {}).get("estimated_query_seconds")) for row in rows
    )
    full_graph_policy_seconds = sum(
        graph_policy_query_seconds(graph_rows.get(line_no, {})) for line_no in selected_lines
    )

    all_contexts = {}
    selected_contexts = {}
    for line_no in selected_lines:
        pathrag_row = pathrag_rows.get(line_no, {})
        context_hash = pathrag_row.get("context_hash")
        if context_hash:
            all_contexts.setdefault(context_hash, as_float(pathrag_row.get("insert_seconds")))

    for row in rows:
        agent = row.get("budget_meta_agent") or {}
        if not agent.get("graph_required"):
            continue
        pathrag_row = pathrag_rows.get(row.get("line_no"), {})
        context_hash = pathrag_row.get("context_hash")
        if context_hash:
            selected_contexts.setdefault(context_hash, as_float(pathrag_row.get("insert_seconds")))

    full_index_seconds = sum(all_contexts.values())
    selected_index_seconds = sum(selected_contexts.values())
    query_reduction = 1 - query_seconds / max(1.0, full_graph_policy_seconds)
    index_reduction = 1 - selected_index_seconds / max(1.0, full_index_seconds)

    lines = [
        "# SAGE v22 Budget-Aware Meta Agent Summary\n\n",
        f"- Budget mode: `{args.budget}`\n",
        f"- Text-first policy: `{args.text_policy}`\n",
        f"- Graph-first policy: `{args.graph_policy}`\n",
        f"- Samples: {len(rows)}\n",
        "- Routing inputs: budget, question contract, and text-policy runtime trigger state\n",
        "- Reference answers and evaluation scores are not used by the meta router\n\n",
        "## Policy Counts\n\n",
        "| Policy | Count |\n",
        "|---|---:|\n",
    ]
    for policy, count in sorted(policies.items()):
        lines.append(f"| {policy} | {count} |\n")

    lines.extend(["\n## Route Reasons\n\n", "| Reason | Count |\n", "|---|---:|\n"])
    for reason, count in sorted(reasons.items()):
        lines.append(f"| {reason} | {count} |\n")

    lines.extend([
        "\n## Replay Cost\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Selected policy query time | {query_seconds:.3f} s |\n",
        f"| Full graph-first policy query time | {full_graph_policy_seconds:.3f} s |\n",
        f"| Query-time reduction vs graph-first | {query_reduction * 100:.2f}% |\n",
        f"| Selected graph contexts | {len(selected_contexts)} / {len(all_contexts)} |\n",
        f"| Selected graph indexing time | {selected_index_seconds:.3f} s |\n",
        f"| Full graph indexing time | {full_index_seconds:.3f} s |\n",
        f"| Graph indexing reduction | {index_reduction * 100:.2f}% |\n",
        "\n## Per-Sample Meta Routes\n\n",
        "| Line | Contract | Policy | Graph | Query seconds | Reason |\n",
        "|---:|---|---|---|---:|---|\n",
    ])
    for row in sorted(rows, key=lambda item: item.get("line_no", -1)):
        agent = row.get("budget_meta_agent") or {}
        lines.append(
            f"| {row.get('line_no')} | {agent.get('contract', {}).get('type')} | "
            f"{agent.get('chosen_policy')} | {agent.get('graph_required')} | "
            f"{agent.get('estimated_query_seconds')} | {agent.get('reason')} |\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-policy", required=True)
    parser.add_argument("--graph-policy", required=True)
    parser.add_argument("--pathrag", required=True)
    parser.add_argument("--budget", choices=["economy", "balanced", "quality"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--start-line", type=int, default=0)
    parser.add_argument("--end-line", type=int)
    args = parser.parse_args()

    rows, graph_rows = build_rows(args)
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_summary(
        rows,
        graph_rows,
        row_map(ROOT_DIR / args.pathrag),
        ROOT_DIR / args.summary,
        args,
    )
    print(f"Saved SAGE v22 {args.budget} results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
