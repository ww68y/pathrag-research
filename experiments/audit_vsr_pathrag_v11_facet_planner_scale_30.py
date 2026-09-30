import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import (
    audit_supported_facet_retention,
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
        default="results/historical_vsr/vsr_pathrag_v11_facet_planner_scale_30_audit.jsonl",
    )
    args = parser.parse_args()

    prediction_path = ROOT_DIR / args.predictions
    dataset_path = ROOT_DIR / args.dataset
    output_path = ROOT_DIR / args.output
    predictions = {
        int(row["line_no"]): row
        for row in load_jsonl(prediction_path)
    }
    dataset_rows = load_jsonl(dataset_path)

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
        retention = audit_supported_facet_retention(
            prediction.get("answer", ""),
            plan["facets"],
        )
        minimum_retention = 1.0 if len(plan["facets"]) <= 3 else 0.8
        audits.append(
            {
                "line_no": line_no,
                "question": question,
                "query_family": plan["query_family"],
                "enabled": plan["enabled"],
                "activation_reason": plan["activation_reason"],
                "activation_confidence": plan["activation_confidence"],
                "stop_reason": plan["stop_reason"],
                "facet_count": plan["facet_count"],
                "facets": plan["facets"],
                "retention": retention,
                "minimum_retention": minimum_retention,
                "accepted": bool(
                    not plan["facets"]
                    or retention["coverage_ratio"] >= minimum_retention
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits),
        encoding="utf-8",
    )

    enabled = sum(bool(row["enabled"]) for row in audits)
    planned = sum(bool(row["facet_count"]) for row in audits)
    rejected = sum(not row["accepted"] for row in audits)
    print(f"Audited rows: {len(audits)}")
    print(f"Planner enabled: {enabled}")
    print(f"High-confidence facet plans: {planned}")
    print(f"Existing-answer retention failures: {rejected}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
