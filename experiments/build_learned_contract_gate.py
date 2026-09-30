import argparse
import json
from pathlib import Path

from build_answer_contract_agent import load_jsonl
from build_claim_evidence_agent import write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]


def by_line(rows: list[dict]) -> dict[int, dict]:
    return {int(row.get("line_no", 0)): row for row in rows}


def write_summary(
    path: Path,
    rows: list[dict],
    dataset_name: str,
    source_path: Path,
    version: str,
) -> None:
    agents = [row.get("learned_contract_gap_router") or {} for row in rows]
    eligible = [agent for agent in agents if agent.get("enabled")]
    selected = sum(bool(agent.get("verification_selected")) for agent in eligible)
    skipped = len(eligible) - selected
    repairs = sum(bool(agent.get("repair_used")) for agent in eligible)
    lines = [
        f"# SAGE {version} Learned Contract-Gap Gate Summary\n\n",
        f"- Dataset: `{dataset_name}`\n",
        f"- Source completion results: `{source_path.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Probabilities are context-grouped out-of-fold predictions.\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Eligible risk candidates | {len(eligible)} |\n",
        f"| Contract verifications selected | {selected} |\n",
        f"| Contract verifications skipped | {skipped} |\n",
        f"| Verification reduction | {skipped / max(1, len(eligible)) * 100:.2f}% |\n",
        f"| Repairs used | {repairs} |\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--version", default="v31")
    args = parser.parse_args()

    baseline_path = ROOT_DIR / args.baseline
    completion_path = ROOT_DIR / args.completion
    baseline = by_line(load_jsonl(baseline_path))
    completion = by_line(load_jsonl(completion_path))
    predictions = {
        int(row.get("line_no", 0)): row
        for row in load_jsonl(ROOT_DIR / args.oof)
        if row.get("source_dataset") == args.dataset_name
    }

    output_rows = []
    for line_no in sorted(baseline):
        base_row = baseline[line_no]
        completion_row = completion.get(line_no, base_row)
        prediction = predictions.get(line_no)
        enabled = prediction is not None
        selected = bool(prediction and prediction.get("predicted_severe_gap"))
        if selected:
            row = dict(completion_row)
            action = "run_contract_verifier"
            repair_used = bool((completion_row.get("contract_completion_agent") or {}).get("repair_accepted"))
        else:
            row = dict(base_row)
            action = "skip_contract_verifier" if enabled else "not_eligible"
            repair_used = False
            if enabled:
                row["suppressed_contract_completion_prediction"] = completion_row.get("prediction", "")
                row["contract_completion_agent"] = completion_row.get("contract_completion_agent", {})
        row["method"] = f"sage_{args.version}_learned_contract_gap_{action}"
        row["learned_contract_gap_router"] = {
            "enabled": enabled,
            "dataset_name": args.dataset_name,
            "oof_probability": prediction.get("oof_probability") if prediction else None,
            "threshold": prediction.get("threshold") if prediction else None,
            "verification_selected": selected,
            "action": action,
            "repair_used": repair_used,
            "fold": prediction.get("fold") if prediction else None,
        }
        output_rows.append(row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(
        ROOT_DIR / args.summary,
        output_rows,
        args.dataset_name,
        completion_path,
        args.version,
    )
    print(f"Saved learned contract-gap results to {output_path}")


if __name__ == "__main__":
    main()
