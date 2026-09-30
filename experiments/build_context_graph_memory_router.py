import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arbiter", required=True)
    parser.add_argument("--text-first", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--minimum-seeds", type=int, default=2)
    args = parser.parse_args()

    arbiter = by_line(load_jsonl(ROOT_DIR / args.arbiter))
    text_first = by_line(load_jsonl(ROOT_DIR / args.text_first))
    text = by_line(load_jsonl(ROOT_DIR / args.text))
    graph = by_line(load_jsonl(ROOT_DIR / args.graph))
    sources = {
        int(row.get("_subset_line_no", index)): row
        for index, row in enumerate(load_jsonl(ROOT_DIR / args.dataset))
    }

    groups: dict[str, list[int]] = defaultdict(list)
    for line_no, source in sources.items():
        groups[str(source.get("source_context_hash"))].append(line_no)

    context_state = {}
    for context_hash, line_nos in groups.items():
        seed_lines = []
        seed_reasons = {}
        for line_no in line_nos:
            reasons = []
            arbiter_choice = (
                (arbiter[line_no].get("evidence_candidate_arbiter") or {})
                .get("decision", {})
                .get("selected_candidate")
            )
            if arbiter_choice in {"graph", "repair"}:
                reasons.append(f"arbiter_{arbiter_choice}")
            text_first_action = (
                (text_first[line_no].get("text_first_contract_agent") or {})
                .get("action")
            )
            if text_first_action == "graph_rescue":
                reasons.append("deterministic_graph_rescue")
            if reasons:
                seed_lines.append(line_no)
                seed_reasons[line_no] = reasons
        context_state[context_hash] = {
            "context_hash": context_hash,
            "sample_count": len(line_nos),
            "seed_lines": sorted(seed_lines),
            "seed_reasons": seed_reasons,
            "seed_count": len(seed_lines),
            "graph_memory_active": len(seed_lines) >= args.minimum_seeds,
            "minimum_seeds": args.minimum_seeds,
        }

    output_rows = []
    for line_no in sorted(arbiter):
        source = sources[line_no]
        context_hash = str(source.get("source_context_hash"))
        state = context_state[context_hash]
        base_row = arbiter[line_no]
        arbiter_choice = (
            (base_row.get("evidence_candidate_arbiter") or {})
            .get("decision", {})
            .get("selected_candidate", "text")
        )
        if arbiter_choice == "repair":
            selected = "repair"
            action = "preserve_guarded_repair"
            prediction = base_row.get("prediction", "")
        elif state["graph_memory_active"]:
            selected = "graph"
            action = "use_context_graph_memory"
            prediction = graph[line_no].get("prediction", "")
        else:
            selected = arbiter_choice
            action = "retain_arbiter_decision"
            prediction = base_row.get("prediction", "")

        row = dict(base_row)
        row["prediction_before_context_graph_memory"] = row.get("prediction", "")
        row["prediction"] = prediction
        row["method"] = f"sage_v37_context_graph_memory_{selected}"
        row["context_graph_memory_agent"] = {
            "enabled": True,
            "version": "v37",
            "action": action,
            "selected_candidate": selected,
            "arbiter_candidate": arbiter_choice,
            **state,
            "reference_answer_used": False,
        }
        output_rows.append(row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    active_contexts = sum(state["graph_memory_active"] for state in context_state.values())
    selected_counts: dict[str, int] = {}
    for row in output_rows:
        selected = row["context_graph_memory_agent"]["selected_candidate"]
        selected_counts[selected] = selected_counts.get(selected, 0) + 1
    lines = [
        "# SAGE v37 Context-Level Graph Utility Memory Summary\n\n",
        f"- Samples: {len(output_rows)}\n",
        f"- Contexts: {len(context_state)}\n",
        f"- Active graph-memory contexts: {active_contexts}\n",
        f"- Minimum independent seed lines: {args.minimum_seeds}\n",
        "- Reference answers and evaluation scores are not routing inputs.\n\n",
        "| Selected candidate | Count |\n",
        "|---|---:|\n",
    ]
    for name, count in sorted(selected_counts.items()):
        lines.append(f"| {name} | {count} |\n")
    lines.extend([
        "\n## Context States\n\n",
        "| Context | Samples | Seeds | Active | Seed lines |\n",
        "|---|---:|---:|---|---|\n",
    ])
    for context_hash, state in sorted(context_state.items()):
        lines.append(
            f"| `{context_hash}` | {state['sample_count']} | {state['seed_count']} | "
            f"{state['graph_memory_active']} | {state['seed_lines']} |\n"
        )
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved v37 context graph-memory results to {output_path}")


if __name__ == "__main__":
    main()
