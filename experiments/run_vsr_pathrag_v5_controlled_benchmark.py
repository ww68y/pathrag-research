import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.active_evidence import (
    admit_active_relations,
    build_evidence_obligations,
    plan_active_acquisition,
)
from api.features.chats.vsr_pathrag import build_query_graph, diagnose_knowledge_gaps


OUT_JSON = ROOT_DIR / "results" / "vsr_pathrag_v5_controlled_benchmark.json"
OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v5_controlled_benchmark_summary.md"


def main() -> None:
    checks = []
    question = "How did the Aurora project improve battery safety?"
    sparse = {
        "entities": [
            {
                "name": "AURORA PROJECT",
                "description": "A battery project.",
                "source_ids": ["base-1"],
            }
        ],
        "relationships": [],
    }
    query_graph = build_query_graph(
        question,
        "process_or_explanation",
        {"type": "explanation"},
        sparse,
    )
    gap = diagnose_knowledge_gaps(query_graph, sparse)
    obligations = build_evidence_obligations(query_graph, gap, sparse)

    for index in range(10):
        tier = ["low", "medium", "high"][index % 3]
        plan = plan_active_acquisition(
            question,
            query_graph,
            gap,
            obligations,
            {"tier": tier},
        )
        checks.append(
            {
                "case": f"high-value-plan-{index}",
                "passed": bool(plan["should_run"] and plan["planned_queries"]),
                "detail": plan,
            }
        )

    obligation = next(row for row in obligations if row["id"] == "intent:mechanism")
    for index in range(10):
        has_source = index % 2 == 0
        admission = admit_active_relations(
            [
                {
                    "src": "AURORA PROJECT",
                    "tgt": f"TEMPERATURE MONITORING {index}",
                    "description": "Aurora improves safety through a monitoring mechanism.",
                    "score": 0.9,
                    "source_ids": [f"source-{index}"] if has_source else [],
                }
            ],
            obligation,
            [],
        )
        checks.append(
            {
                "case": f"provenance-gate-{index}",
                "passed": (admission["admitted_count"] == 1) == has_source,
                "detail": admission,
            }
        )

    sufficient = {
        "entities": sparse["entities"],
        "relationships": [
            {
                "src": "AURORA PROJECT",
                "tgt": "TEMPERATURE MONITORING",
                "keywords": "mechanism improve through monitoring cause effect impact result",
                "description": (
                    "Real-time temperature monitoring caused improved Aurora battery safety; "
                    "this mechanism reduced heat risk and produced a safer result."
                ),
                "source_ids": ["source-complete"],
            }
        ],
    }
    sufficient_gap = diagnose_knowledge_gaps(query_graph, sufficient)
    sufficient_obligations = build_evidence_obligations(
        query_graph,
        sufficient_gap,
        sufficient,
    )
    for index in range(10):
        plan = plan_active_acquisition(
            question,
            query_graph,
            sufficient_gap,
            sufficient_obligations,
            {"tier": "low"},
        )
        checks.append(
            {
                "case": f"sufficient-stop-{index}",
                "passed": not plan["should_run"],
                "detail": plan,
            }
        )

    passed = sum(bool(row["passed"]) for row in checks)
    payload = {"method": "active_evidence_acquisition_v1", "passed": passed, "total": len(checks), "checks": checks}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_SUMMARY.write_text(
        "\n".join(
            [
                "# VSR-PathRAG v5 Controlled Active Evidence Benchmark",
                "",
                f"- Passed: {passed}/{len(checks)}",
                "- High-value acquisition planning: 10 cases",
                "- Provenance admission/rejection: 10 cases",
                "- Evidence-sufficient early stopping: 10 cases",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"passed": passed, "total": len(checks)}))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
