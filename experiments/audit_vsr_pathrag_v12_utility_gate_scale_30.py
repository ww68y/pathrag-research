import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    audit_supported_facet_retention,
    build_facet_activation_policy,
    build_supported_facet_plan,
    rank_source_excerpts,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="results/historical_vsr/vsr_pathrag_v10_facet_retention_scale_30_final.jsonl",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/UltraDomain/vsr_v6_cached_scale_30.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/historical_vsr/vsr_pathrag_v12_utility_gate_scale_30_audit.jsonl",
    )
    args = parser.parse_args()

    predictions = {
        int(row["line_no"]): row
        for row in load_jsonl(ROOT_DIR / args.predictions)
    }
    dataset_rows = load_jsonl(ROOT_DIR / args.dataset)
    output_path = ROOT_DIR / args.output

    audits = []
    for fallback_line_no, dataset_row in enumerate(dataset_rows):
        line_no = int(dataset_row.get("line_no", fallback_line_no))
        prediction = predictions[line_no]
        question = prediction["question"]
        excerpts = rank_source_excerpts(
            question,
            dataset_row.get("context", ""),
        )
        plan = build_supported_facet_plan(
            question,
            {},
            excerpts,
            candidate_answer=prediction.get("original_answer", ""),
        )
        policy = build_facet_activation_policy(
            prediction.get("original_answer", ""),
            plan,
        )
        active_facets = (
            plan["facets"]
            if policy["action"] in {"enforce", "monitor"}
            else []
        )
        retention = audit_supported_facet_retention(
            prediction.get("answer", ""),
            active_facets,
        )
        minimum_retention = 1.0 if len(active_facets) <= 3 else 0.8
        hard_gate_passed = bool(
            not policy["hard_gate_required"]
            or retention["coverage_ratio"] >= minimum_retention
        )
        audits.append(
            {
                "line_no": line_no,
                "question": question,
                "query_family": plan["query_family"],
                "planner_enabled": plan["enabled"],
                "facet_count": plan["facet_count"],
                "facet_plan_stop_reason": plan["stop_reason"],
                "policy": policy,
                "active_facets": active_facets,
                "retention": retention,
                "minimum_retention": minimum_retention,
                "hard_gate_passed": hard_gate_passed,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits),
        encoding="utf-8",
    )

    action_counts = Counter(row["policy"]["action"] for row in audits)
    planned = sum(bool(row["facet_count"]) for row in audits)
    hard_failures = sum(not row["hard_gate_passed"] for row in audits)
    print(f"Audited rows: {len(audits)}")
    print(f"High-confidence facet plans: {planned}")
    print(
        "Policy actions: "
        + ", ".join(
            f"{action}={action_counts.get(action, 0)}"
            for action in ["enforce", "monitor", "skip"]
        )
    )
    print(f"Hard-gate retention failures: {hard_failures}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
