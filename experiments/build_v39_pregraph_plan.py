import argparse
import json
from pathlib import Path

from build_evidence_calibrated_lazy_agent import (
    by_line,
    load_jsonl,
    sanitize_evidence_labels,
    supported_text_repair,
)
from build_text_first_contract_agent import route_text_answer


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    text_rows = by_line(load_jsonl(ROOT_DIR / args.text))
    completion_rows = by_line(load_jsonl(ROOT_DIR / args.completion))
    common_lines = sorted(set(text_rows) & set(completion_rows))
    output_rows = []
    trigger_lines = []

    for line_no in common_lines:
        text_row = text_rows[line_no]
        completion_row = completion_rows[line_no]
        dominance, dominance_checks = supported_text_repair(completion_row)
        route_action, contract, signals, reasons = route_text_answer(text_row)
        row = dict(text_row)

        if dominance:
            action = "use_text_repair"
            requires_graph = False
            prediction = sanitize_evidence_labels(str(completion_row.get("prediction", "")))
            reasons = ["supported_text_repair_dominates_explicit_uncertainty"]
        elif route_action == "graph_rescue":
            action = "request_graph"
            requires_graph = True
            prediction = str(text_row.get("prediction", ""))
            trigger_lines.append(line_no)
        else:
            action = "keep_text"
            requires_graph = False
            prediction = str(text_row.get("prediction", ""))

        row["prediction_before_v39_pregraph_plan"] = row.get("prediction", "")
        row["prediction"] = prediction
        row["method"] = f"sage_v39_pregraph_{action}"
        row["v39_pregraph_plan"] = {
            "enabled": True,
            "version": "v39",
            "action": action,
            "requires_graph": requires_graph,
            "contract": contract,
            "signals": signals,
            "repair_dominance": dominance,
            "repair_dominance_checks": dominance_checks,
            "reasons": reasons,
            "reference_answer_used": False,
        }
        output_rows.append(row)

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    repairs = sum(row["v39_pregraph_plan"]["action"] == "use_text_repair" for row in output_rows)
    lines = [
        "# SAGE v39 Pre-Graph Plan\n\n",
        f"- Samples: {len(output_rows)}\n",
        f"- Text repairs accepted before graph: {repairs}\n",
        f"- Graph trigger lines: {trigger_lines}\n",
        f"- Graph attempts: {len(trigger_lines)} / {len(output_rows)}\n",
        f"- Planned graph-query reduction: {(1 - len(trigger_lines) / max(1, len(output_rows))) * 100:.2f}%\n",
        "- Reference answers and evaluation scores are not routing inputs.\n\n",
        "| Line | Dataset | Action | Reasons |\n",
        "|---:|---|---|---|\n",
    ]
    for row in output_rows:
        plan = row["v39_pregraph_plan"]
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {plan.get('action')} | "
            f"{'; '.join(plan.get('reasons', []))} |\n"
        )
    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved v39 pre-graph plan to {output_path}")
    print(f"Graph trigger lines: {','.join(str(line) for line in trigger_lines) or 'none'}")


if __name__ == "__main__":
    main()
