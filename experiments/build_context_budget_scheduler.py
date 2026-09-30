import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_answer_contract_agent import clean_text, load_jsonl, row_map


ROOT_DIR = Path(__file__).resolve().parents[1]
BUDGET_RATIOS = {
    "economy": 0.30,
    "balanced": 0.60,
    "quality": 1.00,
}


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def context_value(rows: list[dict]) -> float:
    value = 0.0
    for row in rows:
        gate = row.get("learned_dual_gate") or {}
        pre_probability = as_float(gate.get("pre_graph_probability"))
        rule_risk = 1.0 if gate.get("rule_graph_attempt") else 0.0
        value += pre_probability + 0.35 * rule_risk
    return value


def select_contexts(contexts: list[dict], budget_seconds: float) -> set[str]:
    if not contexts:
        return set()
    if budget_seconds >= sum(item["cost"] for item in contexts):
        return {item["context_hash"] for item in contexts}

    if len(contexts) <= 20:
        best_value = -1.0
        best_cost = 0.0
        best_selection: tuple[dict, ...] = ()
        for size in range(1, len(contexts) + 1):
            for selection in itertools.combinations(contexts, size):
                cost = sum(item["cost"] for item in selection)
                if cost > budget_seconds + 1e-9:
                    continue
                value = sum(item["value"] for item in selection)
                if value > best_value or (value == best_value and cost < best_cost):
                    best_value = value
                    best_cost = cost
                    best_selection = selection
        return {item["context_hash"] for item in best_selection}

    ranked = sorted(
        contexts,
        key=lambda item: (item["value"] / max(1.0, item["cost"]), item["value"]),
        reverse=True,
    )
    selected = set()
    used = 0.0
    for item in ranked:
        if used + item["cost"] <= budget_seconds + 1e-9:
            selected.add(item["context_hash"])
            used += item["cost"]
    return selected


def build_context_table(candidate_rows: list[dict], graph_rows: dict[int, dict]) -> list[dict]:
    rows_by_context = defaultdict(list)
    context_costs = {}
    for candidate in candidate_rows:
        line_no = candidate.get("line_no")
        graph_row = graph_rows.get(line_no, {})
        context_hash = clean_text(graph_row.get("context_hash")) or f"line-{line_no}"
        rows_by_context[context_hash].append(candidate)
        context_costs.setdefault(context_hash, as_float(graph_row.get("insert_seconds")))

    table = []
    for context_hash, rows in rows_by_context.items():
        value = context_value(rows)
        cost = context_costs.get(context_hash, 0.0)
        attempts = sum(1 for row in rows if (row.get("learned_dual_gate") or {}).get("graph_attempted"))
        table.append({
            "context_hash": context_hash,
            "samples": len(rows),
            "candidate_attempts": attempts,
            "value": round(value, 6),
            "cost": round(cost, 3),
            "density": round(value / max(1.0, cost), 8),
        })
    return table


def write_summary(
    path: Path,
    rows: list[dict],
    contexts: list[dict],
    selected_contexts: set[str],
    args: argparse.Namespace,
) -> None:
    actions = Counter((row.get("context_budget_agent") or {}).get("final_policy") for row in rows)
    text_query_seconds = sum(
        as_float((row.get("context_budget_agent") or {}).get("text_query_seconds")) for row in rows
    )
    graph_query_seconds = sum(
        as_float((row.get("context_budget_agent") or {}).get("graph_query_seconds")) for row in rows
    )
    selected_index_seconds = sum(item["cost"] for item in contexts if item["context_hash"] in selected_contexts)
    full_index_seconds = sum(item["cost"] for item in contexts)
    budget_seconds = full_index_seconds * BUDGET_RATIOS[args.budget]
    attempts = sum(1 for row in rows if (row.get("context_budget_agent") or {}).get("graph_attempted"))
    accepts = sum(1 for row in rows if (row.get("context_budget_agent") or {}).get("graph_accepted"))

    lines = [
        "# SAGE v24 Context-Budget Scheduler Summary\n\n",
        f"- Budget: `{args.budget}`\n",
        f"- Context budget ratio: {BUDGET_RATIOS[args.budget]:.2f}\n",
        f"- Samples: {len(rows)}\n",
        f"- Selected contexts: {len(selected_contexts)} / {len(contexts)}\n",
        "- Context selection inputs: pre-route probabilities, rule-risk flags, query count, and indexing cost\n",
        "- Graph answers, post-gate probabilities, references, and evaluation scores are not used for context selection\n\n",
        "## Routing\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Graph attempts after scheduling | {attempts} |\n",
        f"| Graph answers accepted | {accepts} |\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| Final {action} answers | {count} |\n")

    lines.extend([
        "\n## Replay Cost\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Text query time | {text_query_seconds:.3f} s |\n",
        f"| Added graph query time | {graph_query_seconds:.3f} s |\n",
        f"| Context indexing budget | {budget_seconds:.3f} s |\n",
        f"| Selected indexing time | {selected_index_seconds:.3f} s |\n",
        f"| Full indexing time | {full_index_seconds:.3f} s |\n",
        f"| Indexing reduction | {(1 - selected_index_seconds / max(1.0, full_index_seconds)) * 100:.2f}% |\n",
        "\n## Context Schedule\n\n",
        "| Context | Selected | Samples | Candidate attempts | Value | Cost | Value / cost |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ])
    for item in sorted(contexts, key=lambda row: row["density"], reverse=True):
        lines.append(
            f"| {item['context_hash']} | {item['context_hash'] in selected_contexts} | "
            f"{item['samples']} | {item['candidate_attempts']} | {item['value']} | "
            f"{item['cost']} | {item['density']} |\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--text-source", required=True)
    parser.add_argument("--graph-source", required=True)
    parser.add_argument("--budget", choices=list(BUDGET_RATIOS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    candidate_rows = load_jsonl(ROOT_DIR / args.candidate)
    text_rows = row_map(ROOT_DIR / args.text_source)
    graph_rows = row_map(ROOT_DIR / args.graph_source)
    contexts = build_context_table(candidate_rows, graph_rows)
    full_index_seconds = sum(item["cost"] for item in contexts)
    selected_contexts = select_contexts(
        contexts,
        full_index_seconds * BUDGET_RATIOS[args.budget],
    )
    context_by_hash = {item["context_hash"]: item for item in contexts}

    output_rows = []
    for candidate in candidate_rows:
        line_no = candidate.get("line_no")
        text_row = text_rows.get(line_no, {})
        graph_row = graph_rows.get(line_no, {})
        context_hash = clean_text(graph_row.get("context_hash")) or f"line-{line_no}"
        context_selected = context_hash in selected_contexts
        gate = candidate.get("learned_dual_gate") or {}
        candidate_attempted = bool(gate.get("graph_attempted"))
        candidate_accepted = bool(gate.get("graph_accepted"))
        actual_attempted = context_selected and candidate_attempted
        actual_accepted = context_selected and candidate_accepted

        if actual_accepted:
            new_row = dict(candidate)
            final_policy = "graph"
        else:
            new_row = dict(text_row)
            new_row["learned_dual_gate"] = gate
            final_policy = "text"
            if candidate_accepted:
                new_row["suppressed_graph_prediction_by_context_budget"] = candidate.get("prediction", "")

        context_item = context_by_hash[context_hash]
        new_row["method"] = f"sage_v24_context_budget_{args.budget}_{final_policy}"
        new_row["context_budget_agent"] = {
            "enabled": True,
            "budget": args.budget,
            "budget_ratio": BUDGET_RATIOS[args.budget],
            "context_hash": context_hash,
            "context_selected": context_selected,
            "context_value": context_item["value"],
            "context_cost": context_item["cost"],
            "context_density": context_item["density"],
            "candidate_graph_attempted": candidate_attempted,
            "candidate_graph_accepted": candidate_accepted,
            "graph_attempted": actual_attempted,
            "graph_accepted": actual_accepted,
            "final_policy": final_policy,
            "text_query_seconds": round(as_float(text_row.get("query_seconds")), 3),
            "graph_query_seconds": round(
                as_float(graph_row.get("query_seconds")) if actual_attempted else 0.0,
                3,
            ),
        }
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in output_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_summary(
        ROOT_DIR / args.summary,
        output_rows,
        contexts,
        selected_contexts,
        args,
    )
    print(f"Saved SAGE v24 {args.budget} results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
