import asyncio
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.claim_active_verification import run_claim_level_active_verification


OUT_JSON = ROOT_DIR / "results" / "vsr_pathrag_v6_controlled_benchmark.json"
OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v6_controlled_benchmark_summary.md"


class VectorStore:
    def __init__(self, rows):
        self.rows = rows

    async def query(self, _query, top_k=5):
        return self.rows[:top_k]


class GraphStore:
    def __init__(self, target, keywords, description):
        self.target = target
        self.edge = {
            "keywords": keywords,
            "description": description,
            "source_id": "controlled-source",
            "weight": 1.0,
        }

    async def get_edge(self, source, target):
        return self.edge if (source, target) == ("AURORA", self.target) else None

    async def get_node(self, _name):
        return None


class RAG:
    def __init__(self, target, keywords, description):
        self.chunk_entity_relation_graph = GraphStore(target, keywords, description)
        self.relationships_vdb = VectorStore(
            [{"src_id": "AURORA", "tgt_id": target, "distance": 0.95}]
        )
        self.entities_vdb = VectorStore([])


async def main() -> None:
    question = "How does Aurora prevent overheating?"
    answer = (
        "Aurora uses plasma shielding to prevent battery overheating during "
        "high-load charging operation in summer."
    )
    sparse = {
        "entities": [],
        "relationships": [
            {
                "src": "AURORA",
                "tgt": "BATTERY PROJECT",
                "keywords": "project description",
                "description": "Aurora is an experimental battery project.",
                "source_ids": ["base-source"],
            }
        ],
    }
    complete = {
        "entities": [],
        "relationships": [
            {
                "src": "AURORA",
                "tgt": "PLASMA SHIELDING",
                "keywords": "uses plasma shielding prevent overheating",
                "description": answer,
                "source_ids": ["complete-source"],
            }
        ],
    }
    checks = []
    for index in range(10):
        _expanded, result = await run_claim_level_active_verification(
            RAG("PLASMA SHIELDING", "uses plasma shielding prevent overheating", answer),
            question,
            answer,
            sparse,
            {"tier": "medium"},
        )
        checks.append(
            {
                "case": f"unknown-to-supported-{index}",
                "passed": result["resolved_claim_count"] == 1
                and result["counterfactual_committed_count"] == 1,
                "result": result,
            }
        )
    for index in range(10):
        _expanded, result = await run_claim_level_active_verification(
            RAG(
                "SAFETY REPORT",
                "documented preliminary report",
                "Aurora documented a general safety process during preliminary research.",
            ),
            question,
            answer,
            sparse,
            {"tier": "medium"},
        )
        checks.append(
            {
                "case": f"no-gain-rejected-{index}",
                "passed": result["resolved_claim_count"] == 0
                and result["counterfactual_committed_count"] == 0
                and result["counterfactual_rejected_count"] >= 1,
                "result": result,
            }
        )
    for index in range(10):
        _expanded, result = await run_claim_level_active_verification(
            RAG("PLASMA SHIELDING", "uses plasma shielding prevent overheating", answer),
            question,
            answer,
            complete,
            {"tier": "medium"},
        )
        checks.append(
            {
                "case": f"supported-skip-{index}",
                "passed": not result["triggered"] and result["query_count"] == 0,
                "result": result,
            }
        )
    passed = sum(bool(row["passed"]) for row in checks)
    payload = {"method": "counterfactual_claim_active_verification_v1", "passed": passed, "total": len(checks), "checks": checks}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_SUMMARY.write_text(
        "\n".join(
            [
                "# VSR-PathRAG v6 Controlled Claim Verification Benchmark",
                "",
                f"- Passed: {passed}/{len(checks)}",
                "- Unknown-to-supported active verification: 10 cases",
                "- Relevant but non-resolving evidence rejection: 10 cases",
                "- Already-supported zero-query stopping: 10 cases",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"passed": passed, "total": len(checks)}))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    asyncio.run(main())

