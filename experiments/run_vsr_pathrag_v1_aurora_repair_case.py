import asyncio
import json
from pathlib import Path
import sys

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from api.features.chats.agent_controller import run_interactive_agent
from api.features.rag_manager import rag


QUESTION = "How did the Aurora battery system improve battery safety?"
OUT_JSON = ROOT_DIR / "results" / "vsr_pathrag_v1_aurora_repair_case.json"
OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v1_aurora_repair_case_summary.md"


async def main() -> None:
    result = await run_interactive_agent(rag, QUESTION, requested_mode="agent_vsr_fast")
    trace = result["trace"]
    repair = trace.get("graph_repair") or {}
    validation = repair.get("validation") or {}
    payload = {
        "question": QUESTION,
        "answer": result["answer"],
        "trace": trace,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    accepted = validation.get("accepted", [])
    claim_graph = trace.get("claim_evidence_graph") or {}
    rollback = trace.get("counterfactual_rollback") or {}
    lines = [
        "# VSR-PathRAG v1 Aurora Repair Case",
        "",
        f"Question: `{QUESTION}`",
        "",
        "## Result",
        "",
        f"- Version: `{trace.get('version')}`",
        f"- Gap score: `{(trace.get('knowledge_gap') or {}).get('gap_score')}`",
        f"- Repair required: `{(trace.get('knowledge_gap') or {}).get('repair_required')}`",
        f"- Accepted temporary relations: `{validation.get('accepted_count', 0)}`",
        f"- Rejected candidate relations: `{validation.get('rejected_count', 0)}`",
        f"- Base graph unchanged: `{(repair.get('overlay') or {}).get('base_graph_unchanged')}`",
        f"- Signed claim statuses: `{claim_graph.get('status_counts')}`",
        f"- Counterfactual rollback: `{rollback.get('triggered')}`",
        f"- Final policy: `{trace.get('final_policy')}`",
        f"- Runtime seconds: `{trace.get('runtime_seconds'):.3f}`",
        "",
        "## Accepted Repair",
        "",
    ]
    for row in accepted:
        lines.append(
            f"- `{row.get('head')} --{row.get('relation')}--> {row.get('tail')}` "
            f"(confidence `{row.get('confidence')}`, source `{row.get('source_id')}`)"
        )
        lines.append(f"  Evidence: {row.get('evidence_quote')}")
    lines.extend(["", "## Final Answer", "", result["answer"], ""])
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    asyncio.run(main())
