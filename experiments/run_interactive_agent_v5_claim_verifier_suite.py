import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from api.features.chats.agent_controller import run_interactive_agent
from api.features.rag_manager import get_rag_instance


QUESTIONS = [
    "Who directed the Aurora Battery Pilot Project, and what organization was responsible for system integration?",
    "How did the Aurora battery system improve bus operating range and charging time?",
    "Why did two buses report abnormal temperature warnings during the first month of testing?",
    "What did Sunvale University evaluate, and what improvements did it report?",
    "How did the expansion plan change the project from June 2025 onward?",
    "What was the main supply-chain risk, and how did the project reduce that risk?",
    "Which organizations were involved in the battery supply chain, and what role did each one play?",
    "Compare the roles of BlueLake Materials and SilverPeak Cells in the Aurora project.",
    "What happened after GreenVolt Energy replaced the faulty cooling valve?",
    "Which company produced the faulty cooling valve, and who traced the problem?",
    "What software architecture did the Aurora system use for predictive maintenance?",
    "List only the supported facts about the October 2025 delivery requirement and the backup supplier.",
    "What percentage carbon-emission reduction did Sunvale University report?",
    "Did Mark Chen supply the lithium iron phosphate battery cells? Answer only with supported facts.",
]


OUT_JSONL = ROOT_DIR / "results" / "interactive_agent_v5_claim_verifier_suite_14.jsonl"
OUT_SUMMARY = ROOT_DIR / "results" / "interactive_agent_v5_claim_verifier_suite_14_summary.md"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict]) -> str:
    ok_rows = [row for row in rows if row.get("ok")]
    profile_counts = Counter(row.get("selected_profile", "unknown") for row in ok_rows)
    policy_counts = Counter(row.get("final_policy", "unknown") for row in ok_rows)
    type_counts = Counter(row.get("question_type", "unknown") for row in ok_rows)
    risk_counts = Counter(flag for row in ok_rows for flag in row.get("risk_flags", []))
    repair_triggered = sum(1 for row in ok_rows if row.get("repair_triggered"))
    repair_accepted = sum(1 for row in ok_rows if row.get("repair_accepted"))
    avg_runtime = sum(row.get("runtime_seconds", 0) for row in ok_rows) / max(1, len(ok_rows))
    avg_candidates = sum(row.get("candidate_count", 0) for row in ok_rows) / max(1, len(ok_rows))
    avg_evidence_edges = sum(row.get("evidence_edges", 0) for row in ok_rows) / max(1, len(ok_rows))
    avg_claim_support = sum(row.get("claim_support_ratio", 0) for row in ok_rows) / max(1, len(ok_rows))
    total_claims = sum(row.get("claim_count", 0) for row in ok_rows)
    total_unsupported = sum(row.get("unsupported_claim_count", 0) for row in ok_rows)

    lines = [
        "# Interactive Agent v5 Claim Verifier Suite",
        "",
        "Dataset: current uploaded Aurora Battery Pilot Project demo document.",
        "",
        "## Aggregate",
        "",
        f"- Samples: {len(rows)}",
        f"- Successful: {len(ok_rows)}",
        f"- Avg runtime seconds: {avg_runtime:.2f}",
        f"- Avg candidate count: {avg_candidates:.2f}",
        f"- Avg evidence edges: {avg_evidence_edges:.2f}",
        f"- Avg claim support ratio: {avg_claim_support:.3f}",
        f"- Total checked claims: {total_claims}",
        f"- Total unsupported claims: {total_unsupported}",
        f"- Question types: {dict(type_counts)}",
        f"- Selected profiles: {dict(profile_counts)}",
        f"- Final policies: {dict(policy_counts)}",
        f"- Repair triggered: {repair_triggered}",
        f"- Repair accepted: {repair_accepted}",
        f"- Risk flags: {dict(risk_counts)}",
        "",
        "## Per Sample",
        "",
        "| # | Question | Type | Profile | Candidates | Repair | Claims | Unsupported | Policy | Runtime | Risks |",
        "|---:|---|---|---|---:|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        risks = ", ".join(row.get("risk_flags", [])) or "none"
        question = row["question"].replace("|", "/")
        repair = "accepted" if row.get("repair_accepted") else ("triggered" if row.get("repair_triggered") else "skipped")
        lines.append(
            f"| {row['index']} | {question} | {row.get('question_type', 'error')} | "
            f"{row.get('selected_profile', 'error')} | {row.get('candidate_count', 0)} | "
            f"{repair} | {row.get('claim_supported_count', 0)}/{row.get('claim_count', 0)} | "
            f"{row.get('unsupported_claim_count', 0)} | {row.get('final_policy', 'error')} | "
            f"{row.get('runtime_seconds', 0):.2f} | {risks} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    rag = get_rag_instance()
    rows: list[dict] = []
    for index, question in enumerate(QUESTIONS, start=1):
        started = time.perf_counter()
        try:
            result = await run_interactive_agent(rag, question, requested_mode="agent")
            trace = result["trace"]
            evidence_summary = trace.get("evidence_summary") or {}
            claim_verification = trace.get("claim_verification") or {}
            row = {
                "index": index,
                "ok": True,
                "question": question,
                "answer": result["answer"],
                "version": trace.get("version"),
                "question_type": trace.get("question_type"),
                "selected_mode": trace.get("selected_mode"),
                "selected_profile": trace.get("selected_profile"),
                "candidate_count": trace.get("candidate_count", 0),
                "evidence_edges": len(evidence_summary.get("relationships", [])),
                "evidence_entities": len(evidence_summary.get("entities", [])),
                "final_policy": trace.get("final_policy"),
                "repair_triggered": trace.get("repair_triggered", False),
                "repair_reasons": trace.get("repair_reasons", []),
                "repair_accepted": trace.get("repair_accepted", False),
                "claim_count": claim_verification.get("claim_count", 0),
                "claim_supported_count": claim_verification.get("supported_count", 0),
                "unsupported_claim_count": claim_verification.get("unsupported_count", 0),
                "claim_support_ratio": claim_verification.get("support_ratio", 0),
                "risk_flags": trace.get("risk_flags", []),
                "runtime_seconds": trace.get("runtime_seconds", time.perf_counter() - started),
                "trace": trace,
            }
        except Exception as exc:
            row = {
                "index": index,
                "ok": False,
                "question": question,
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_seconds": time.perf_counter() - started,
            }
        rows.append(row)
        write_jsonl(OUT_JSONL, rows)
        OUT_SUMMARY.write_text(summarize(rows), encoding="utf-8")
        print(
            f"[{index}/{len(QUESTIONS)}] ok={row.get('ok')} "
            f"type={row.get('question_type')} profile={row.get('selected_profile')} "
            f"claims={row.get('claim_supported_count', 0)}/{row.get('claim_count', 0)} "
            f"unsupported={row.get('unsupported_claim_count', 0)} "
            f"repair={row.get('repair_accepted')} runtime={row.get('runtime_seconds', 0):.2f}s"
        )

    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    asyncio.run(main())
