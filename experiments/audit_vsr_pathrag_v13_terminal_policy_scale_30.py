import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    assess_terminal_answer,
    build_fast_extractive_rescue_answer,
    build_risk_bounded_terminal_policy,
    build_terminal_claim_pruned_answer,
    terminal_assessment_score,
    terminal_hard_contradicted_claims,
)
from api.features.chats.answer_realization import rank_source_excerpts


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compact_assessment(assessment: dict) -> dict:
    audit = assessment["evidence_audit"]
    graph = assessment["claim_evidence_graph"]
    return {
        "evidence_audit": audit,
        "signed_status_counts": graph.get("status_counts", {}),
        "hard_contradicted_claims": terminal_hard_contradicted_claims(graph),
        "terminal_score": terminal_assessment_score(assessment),
    }


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
        default="results/historical_vsr/vsr_pathrag_v13_terminal_policy_scale_30_audit.jsonl",
    )
    args = parser.parse_args()

    predictions = load_jsonl(ROOT_DIR / args.predictions)
    dataset_rows = load_jsonl(ROOT_DIR / args.dataset)
    dataset_by_line = {
        int(row.get("line_no", index)): row
        for index, row in enumerate(dataset_rows)
    }
    output_path = ROOT_DIR / args.output

    rows = []
    for prediction in predictions:
        line_no = int(prediction["line_no"])
        question = prediction["question"]
        evidence_summary = copy.deepcopy(
            prediction.get("trace", {}).get("evidence_summary")
            or {"entities": [], "relationships": []}
        )
        evidence_summary["source_excerpts"] = rank_source_excerpts(
            question,
            dataset_by_line[line_no].get("context", ""),
        )
        structured_confidence = prediction.get("trace", {}).get(
            "structured_confidence"
        )
        current = assess_terminal_answer(
            question,
            prediction["answer"],
            evidence_summary,
            structured_confidence,
        )
        policy = build_risk_bounded_terminal_policy(current)
        fallback_source = None
        fallback = None

        if policy["action"] == "abstain":
            fallback_candidates = []
            pruned = build_terminal_claim_pruned_answer(current)
            if pruned:
                fallback_candidates.append(
                    (
                        "terminal_claim_pruning",
                        assess_terminal_answer(
                            question,
                            pruned,
                            evidence_summary,
                            structured_confidence,
                        ),
                    )
                )
            original_answer = prediction.get("original_answer", "")
            if (
                original_answer
                and original_answer.strip() != prediction["answer"].strip()
            ):
                fallback_candidates.append(
                    (
                        "pre_realization_candidate",
                        assess_terminal_answer(
                            question,
                            original_answer,
                            evidence_summary,
                            structured_confidence,
                        ),
                    )
                )
            extractive = build_fast_extractive_rescue_answer(
                question,
                evidence_summary,
            )
            if extractive and extractive.strip() != prediction["answer"].strip():
                fallback_candidates.append(
                    (
                        "verified_extractive_rescue",
                        assess_terminal_answer(
                            question,
                            extractive,
                            evidence_summary,
                            structured_confidence,
                        ),
                    )
                )
            safe_fallbacks = [
                (source, assessment)
                for source, assessment in fallback_candidates
                if build_risk_bounded_terminal_policy(
                    current,
                    assessment,
                )["action"]
                == "rollback_to_verified_fallback"
            ]
            if safe_fallbacks:
                fallback_source, fallback = max(
                    safe_fallbacks,
                    key=lambda item: (
                        terminal_assessment_score(item[1])
                        + (
                            0.25
                            if item[0] == "terminal_claim_pruning"
                            else 0.0
                        )
                    ),
                )
                policy = build_risk_bounded_terminal_policy(current, fallback)

        policy["fallback_source"] = fallback_source
        if policy["action"] == "rollback_to_verified_fallback":
            terminal_answer = fallback["answer"]
        elif policy["action"] == "abstain":
            terminal_answer = (
                "The available verified evidence is insufficient to provide a "
                "reliable answer to this question."
            )
        else:
            terminal_answer = prediction["answer"]
        rows.append(
            {
                "line_no": line_no,
                "question": question,
                "current_answer": prediction["answer"],
                "terminal_answer": terminal_answer,
                "terminal_policy": policy,
                "current_assessment": compact_assessment(current),
                "fallback_assessment": (
                    compact_assessment(fallback) if fallback else None
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    actions = Counter(row["terminal_policy"]["action"] for row in rows)
    fallback_sources = Counter(
        row["terminal_policy"].get("fallback_source")
        for row in rows
        if row["terminal_policy"].get("fallback_source")
    )
    print(f"Audited rows: {len(rows)}")
    print(
        "Terminal actions: "
        + ", ".join(
            f"{action}={actions.get(action, 0)}"
            for action in [
                "commit",
                "commit_with_monitoring",
                "rollback_to_verified_fallback",
                "abstain",
            ]
        )
    )
    print(f"Fallback sources: {dict(fallback_sources)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
