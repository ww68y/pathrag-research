import argparse
from pathlib import Path

from build_answer_contract_agent import load_jsonl
from build_claim_evidence_agent import write_jsonl
from build_contract_completion_agent import write_summary


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-coverage", type=float, default=0.60)
    parser.add_argument("--max-grounding", type=float, default=0.70)
    args = parser.parse_args()

    rows = load_jsonl(ROOT_DIR / args.input)
    for row in rows:
        agent = row.get("contract_completion_agent") or {}
        verification = agent.get("verification") or {}
        severe_gap = (
            float(verification.get("contract_coverage", 1.0)) <= args.max_coverage
            and float(verification.get("evidence_grounding", 1.0)) <= args.max_grounding
        )
        agent["conservative_policy"] = {
            "max_contract_coverage": args.max_coverage,
            "max_evidence_grounding": args.max_grounding,
            "severe_gap": severe_gap,
        }
        if agent.get("repair_accepted") and not severe_gap:
            agent["broad_policy_repair_attempted"] = bool(agent.get("repair_attempted"))
            agent["repair_attempted"] = False
            row["rejected_contract_completion_prediction"] = row.get("prediction", "")
            row["prediction"] = row.pop("prediction_before_contract_completion")
            row["method"] = "sage_v27_contract_completion_reverted_by_conservative_policy"
            agent["repair_accepted"] = False
            agent["repair_needed"] = False
            agent["conservative_policy"]["action"] = "retain_original"
        elif agent.get("repair_attempted") and not severe_gap:
            agent["broad_policy_repair_attempted"] = True
            agent["repair_attempted"] = False
            agent["repair_needed"] = False
            agent["conservative_policy"]["action"] = "retain_original"
        elif agent.get("repair_accepted"):
            row["method"] = "sage_v27_conservative_contract_completion"
            agent["conservative_policy"]["action"] = "accept_repair"
        else:
            agent["conservative_policy"]["action"] = "retain_original"

    output_path = ROOT_DIR / args.output
    write_jsonl(rows, output_path)
    write_summary(ROOT_DIR / args.summary, rows, output_path)
    print(f"Saved conservative SAGE v27 results to {output_path}")


if __name__ == "__main__":
    main()
