import os
from collections import Counter
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.environ["VSR_MEMORY_WRITE_ENABLED"] = "0"

import experiments.run_vsr_pathrag_v3_crossdomain_suite as parent


parent.base.OUT_JSONL = ROOT_DIR / "results" / "vsr_pathrag_v4_crossdomain_suite_12.jsonl"
parent.base.OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v4_crossdomain_suite_12_summary.md"
parent.base.SUMMARY_TITLE = "# VSR-PathRAG v4 Experience-Guided Cross-Domain Suite"
_v3_summarize = parent.base.summarize


def summarize(rows: list[dict]) -> str:
    text = _v3_summarize(rows)
    ok_rows = [row for row in rows if row.get("ok")]
    retrieved = 0
    recommendations = 0
    overrides = 0
    route_counts = Counter()
    gate_counts = Counter()
    memory_sizes = []
    failure_tags = Counter()
    for row in ok_rows:
        experience = (row.get("trace") or {}).get("experience_memory") or {}
        recall = experience.get("recall") or {}
        decision = experience.get("router_decision") or {}
        retrieved += int(bool(recall.get("retrieved_count", 0)))
        recommendations += int(bool(recall.get("recommended_route")))
        overrides += int(bool(decision.get("override")))
        route_counts[str(recall.get("recommended_route") or "none")] += 1
        gate_counts[str(decision.get("safety_gate", "unknown"))] += 1
        memory_sizes.append(int(recall.get("memory_size", 0) or 0))
        failure_tags.update(recall.get("failure_counts") or {})
    denominator = max(1, len(ok_rows))
    return text + "\n" + "\n".join(
        [
            "## VSR-PathRAG v4 Experience Metrics",
            "",
            f"- Samples with recalled neighbors: {retrieved}/{len(ok_rows)}",
            f"- Samples with route recommendations: {recommendations}/{len(ok_rows)}",
            f"- Safety-approved memory overrides: {overrides}",
            f"- Recommended routes: {dict(route_counts)}",
            f"- Safety gates: {dict(gate_counts)}",
            f"- Avg memory size: {sum(memory_sizes) / denominator:.2f}",
            f"- Recalled failure tags: {dict(failure_tags)}",
            "- Raw question persistence: disabled (hash and structural context only)",
            "- Experiment memory writes: disabled (read-only evaluation)",
            "",
        ]
    )


parent.base.summarize = summarize
base = parent.base


if __name__ == "__main__":
    import asyncio

    asyncio.run(base.main())
