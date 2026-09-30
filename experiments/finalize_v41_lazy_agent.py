import argparse
import json
from pathlib import Path

from build_evidence_calibrated_lazy_agent import by_line, load_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_optional(path_arg: str | None) -> dict[int, dict]:
    if not path_arg:
        return {}
    path = ROOT_DIR / path_arg
    if not path.exists():
        return {}
    return by_line(load_jsonl(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--arbiter", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--full-context-count", type=int, default=3)
    parser.add_argument(
        "--allow-missing-graph",
        action="store_true",
        help="Keep text with an explicit pending-graph marker when graph-triggered lines are unavailable.",
    )
    args = parser.parse_args()

    plan_rows = by_line(load_jsonl(ROOT_DIR / args.plan))
    arbiter_rows = load_optional(args.arbiter)
    graph_rows = load_optional(args.graph)
    output_rows = []
    graph_attempts = 0
    graph_selections = 0

    for line_no in sorted(plan_rows):
        plan_row = plan_rows[line_no]
        plan = plan_row.get("v41_pregraph_plan") or {}
        action = str(plan.get("action", "keep_text"))
        row = dict(plan_row)
        selected = "repair" if action == "use_text_repair" else "text"
        final_action = action

        if action == "request_graph":
            graph_attempts += 1
            arbiter_row = arbiter_rows.get(line_no)
            if not arbiter_row:
                if not args.allow_missing_graph:
                    raise ValueError(f"Missing selective arbiter result for graph trigger line {line_no}")
                selected = "text"
                final_action = "graph_pending_text_fallback"
            else:
                selected = str(
                    ((arbiter_row.get("evidence_candidate_arbiter") or {}).get("decision") or {}).get(
                        "selected_candidate", "text"
                    )
                )
                row["prediction"] = arbiter_row.get("prediction", row.get("prediction", ""))
                final_action = f"selective_arbiter_{selected}"
                if selected == "graph":
                    graph_selections += 1

        row["method"] = f"sage_v41_{final_action}"
        row["evidence_calibrated_lazy_agent"] = {
            "enabled": True,
            "version": "v41",
            "action": final_action,
            "selected_candidate": selected,
            "graph_attempted": action == "request_graph" and final_action != "graph_pending_text_fallback",
            "graph_requested": action == "request_graph",
            "graph_available": final_action != "graph_pending_text_fallback",
            "graph_selected": selected == "graph",
            "pregraph_plan": plan,
            "reference_answer_used": False,
        }
        output_rows.append(row)

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    graph_query_seconds = sum(float(row.get("query_seconds", 0.0)) for row in graph_rows.values())
    graph_index_seconds = sum(float(row.get("insert_seconds", 0.0)) for row in graph_rows.values())
    graph_contexts = {row.get("context_hash") for row in graph_rows.values() if row.get("context_hash")}
    repairs = sum(row["evidence_calibrated_lazy_agent"]["selected_candidate"] == "repair" for row in output_rows)
    vetoes = sum(
        row["evidence_calibrated_lazy_agent"]["pregraph_plan"].get("action") == "keep_text_verifier_veto"
        for row in output_rows
    )
    action_counts: dict[str, int] = {}
    for row in output_rows:
        action_name = row["evidence_calibrated_lazy_agent"]["action"]
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

    lines = [
        "# SAGE v41 Verifier-Calibrated Lazy-Agent Summary\n\n",
        f"- Samples: {len(output_rows)}\n",
        f"- Pre-graph text repairs selected: {repairs}\n",
        f"- Pre-graph verifier vetoes: {vetoes}\n",
        f"- Graph attempts: {graph_attempts} / {len(output_rows)}\n",
        f"- Graph answers selected: {graph_selections}\n",
        f"- Graph-query reduction vs running PathRAG for every question: {(1 - graph_attempts / max(1, len(output_rows))) * 100:.2f}%\n",
        f"- Graph contexts constructed: {len(graph_contexts)} / {args.full_context_count}\n",
        f"- Triggered graph indexing time: {graph_index_seconds:.3f} s\n",
        f"- Triggered graph query time: {graph_query_seconds:.3f} s\n",
        "- Reference answers and evaluation scores are not runtime inputs.\n\n",
        "## Actions\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action_name, count in sorted(action_counts.items()):
        lines.append(f"| {action_name} | {count} |\n")
    lines.extend([
        "\n## Per-Sample Decisions\n\n",
        "| Line | Dataset | Action | Selected | Graph Attempted | Reasons |\n",
        "|---:|---|---|---|---|---|\n",
    ])
    for row in output_rows:
        agent = row["evidence_calibrated_lazy_agent"]
        reasons = "; ".join((agent.get("pregraph_plan") or {}).get("reasons", [])).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('selected_candidate')} | {agent.get('graph_attempted')} | {reasons} |\n"
        )
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved v41 lazy results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
