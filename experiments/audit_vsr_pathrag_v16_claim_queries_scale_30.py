import argparse
import copy
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.claim_active_verification import (
    plan_claim_verification_queries,
)
from api.features.chats.vsr_pathrag import lexical_terms


LEGACY_QUERY_MARKERS = (
    "verify this claim",
    "source evidence",
    "original question",
)
META_MARKERS = (
    "provided data",
    "provided information",
    "not explicitly stated",
    "does not contain specific",
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def marker_count(queries: list[str], markers: tuple[str, ...]) -> int:
    return sum(
        any(marker in query.lower() for marker in markers)
        for query in queries
    )


def mean_pairwise_diversity(queries: list[str]) -> float:
    term_sets = [lexical_terms(query) for query in queries]
    distances = []
    for left_index, left in enumerate(term_sets):
        for right in term_sets[left_index + 1 :]:
            union = left | right
            similarity = len(left & right) / max(1, len(union))
            distances.append(1 - similarity)
    if not distances:
        return 0.0
    return round(sum(distances) / len(distances), 4)


def reconstruct_targets(verification: dict) -> list[dict]:
    targets = []
    seen = set()
    for row in verification.get("rounds", []):
        claim_id = str(row.get("claim_id", ""))
        if not claim_id or claim_id in seen:
            continue
        seen.add(claim_id)
        status = str(row.get("before_status", "unknown"))
        targets.append(
            {
                "claim_index": int(row.get("claim_index", 0) or 0),
                "claim_id": claim_id,
                "claim": str(row.get("claim_preview", "")),
                "status": status,
                "confidence": 0.0,
                "priority": 1.0 if status == "contradicted" else 0.75,
            }
        )
    targets.sort(
        key=lambda row: (
            -row["priority"],
            row["claim_index"],
        )
    )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="results/historical_vsr/vsr_pathrag_v10_facet_retention_scale_30_final.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "vsr_pathrag_v16_claim_query_scale_30_audit.jsonl"
        ),
    )
    args = parser.parse_args()

    rows = []
    for prediction in load_jsonl(ROOT_DIR / args.predictions):
        trace = prediction.get("trace") or {}
        verification = trace.get("claim_active_verification") or {}
        targets = reconstruct_targets(verification)
        budget = copy.deepcopy(
            trace.get("compute_budget") or {"tier": "low"}
        )
        budget["requested_fast"] = False
        plans = plan_claim_verification_queries(
            prediction["question"],
            targets,
            budget,
        )
        legacy_queries = [
            str(row.get("query", ""))
            for row in verification.get("rounds", [])
        ]
        v16_queries = [
            str(plan["query_plan"]["query"])
            for plan in plans
        ]
        rows.append(
            {
                "line_no": int(prediction["line_no"]),
                "question": prediction["question"],
                "answer_held_fixed": True,
                "historical_target_claim_count": int(
                    verification.get("target_claim_count", 0) or 0
                ),
                "reconstructed_target_count": len(targets),
                "legacy_query_count": len(legacy_queries),
                "v16_planned_query_count": len(v16_queries),
                "v16_planned_claim_count": len(
                    {plan["claim_id"] for plan in plans}
                ),
                "v16_strategy_count": len(
                    {plan["query_strategy"] for plan in plans}
                ),
                "legacy_scaffolding_contamination_count": marker_count(
                    legacy_queries,
                    LEGACY_QUERY_MARKERS,
                ),
                "v16_scaffolding_contamination_count": marker_count(
                    v16_queries,
                    LEGACY_QUERY_MARKERS,
                ),
                "legacy_meta_contamination_count": marker_count(
                    legacy_queries,
                    META_MARKERS,
                ),
                "v16_meta_contamination_count": marker_count(
                    v16_queries,
                    META_MARKERS,
                ),
                "v16_mean_pairwise_query_diversity": (
                    mean_pairwise_diversity(v16_queries)
                ),
                "v16_meta_sanitized_plan_count": sum(
                    bool(plan["meta_claim_sanitized"])
                    for plan in plans
                ),
                "v16_plans": [
                    {
                        "claim_id": plan["claim_id"],
                        "strategy": plan["query_strategy"],
                        "variant_index": plan[
                            "query_variant_index"
                        ],
                        "meta_claim_sanitized": plan[
                            "meta_claim_sanitized"
                        ],
                        "query": plan["query_plan"]["query"],
                    }
                    for plan in plans
                ],
            }
        )

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    target_rows = [
        row for row in rows
        if row["reconstructed_target_count"] > 0
    ]
    print(f"Audited rows: {len(rows)}")
    print(f"Rows with reconstructable attempted claims: {len(target_rows)}")
    print(
        "Reconstructed targets: "
        f"{sum(row['reconstructed_target_count'] for row in rows)}"
    )
    print(
        "Queries legacy -> v16 planned: "
        f"{sum(row['legacy_query_count'] for row in rows)} -> "
        f"{sum(row['v16_planned_query_count'] for row in rows)}"
    )
    print(
        "Scaffolding contamination legacy -> v16: "
        f"{sum(row['legacy_scaffolding_contamination_count'] for row in rows)} -> "
        f"{sum(row['v16_scaffolding_contamination_count'] for row in rows)}"
    )
    print(
        "Meta contamination legacy -> v16: "
        f"{sum(row['legacy_meta_contamination_count'] for row in rows)} -> "
        f"{sum(row['v16_meta_contamination_count'] for row in rows)}"
    )
    print(
        "Mean v16 pairwise diversity on target rows: "
        f"{sum(row['v16_mean_pairwise_query_diversity'] for row in target_rows) / max(1, len(target_rows)):.4f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
