import argparse
import json
from pathlib import Path

from build_evidence_calibrated_lazy_agent import by_line, load_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--arbiter", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    plan_rows = by_line(load_jsonl(ROOT_DIR / args.plan))
    arbiter_rows = by_line(load_jsonl(ROOT_DIR / args.arbiter))
    graph_rows = by_line(load_jsonl(ROOT_DIR / args.graph))
    output_rows = []
    graph_attempts = 0
    graph_selections = 0

    for line_no in sorted(plan_rows):
        plan_row = plan_rows[line_no]
        plan = plan_row.get("v39_pregraph_plan") or {}
        action = str(plan.get("action", "keep_text"))
        row = dict(plan_row)
        selected = "repair" if action == "use_text_repair" else "text"
        final_action = action

        if action == "request_graph":
            graph_attempts += 1
            arbiter_row = arbiter_rows.get(line_no)
            if not arbiter_row:
                raise ValueError(f"Missing selective arbiter result for graph trigger line {line_no}")
            selected = str(
                ((arbiter_row.get("evidence_candidate_arbiter") or {}).get("decision") or {}).get(
                    "selected_candidate", "text"
                )
            )
            row["prediction"] = arbiter_row.get("prediction", row.get("prediction", ""))
            final_action = f"selective_arbiter_{selected}"
            if selected == "graph":
                graph_selections += 1

        row["method"] = f"sage_v39_{final_action}"
        row["evidence_calibrated_lazy_agent"] = {
            "enabled": True,
            "version": "v39",
            "action": final_action,
            "selected_candidate": selected,
            "graph_attempted": action == "request_graph",
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
    repairs = sum(row["evidence_calibrated_lazy_agent"]["selected_candidate"] == "repair" for row in output_rows)
    lines = [
        "# SAGE v39 Frozen Lazy-Agent Summary\n\n",
        f"- Samples: {len(output_rows)}\n",
        f"- Pre-graph text repairs selected: {repairs}\n",
        f"- Graph attempts: {graph_attempts} / {len(output_rows)}\n",
        f"- Graph answers selected: {graph_selections}\n",
        f"- Graph-query reduction vs running PathRAG for every question: {(1 - graph_attempts / max(1, len(output_rows))) * 100:.2f}%\n",
        f"- Graph contexts constructed: {len({row.get('context_hash') for row in graph_rows.values()})} / 3\n",
        f"- Triggered graph indexing time: {graph_index_seconds:.3f} s\n",
        f"- Triggered graph query time: {graph_query_seconds:.3f} s\n",
        "- Reference answers and evaluation scores are not runtime inputs.\n\n",
        "| Line | Dataset | Action | Selected | Graph Attempted |\n",
        "|---:|---|---|---|---|\n",
    ]
    for row in output_rows:
        agent = row["evidence_calibrated_lazy_agent"]
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('selected_candidate')} | {agent.get('graph_attempted')} |\n"
        )
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved frozen v39 results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
