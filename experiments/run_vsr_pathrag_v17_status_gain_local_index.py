import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from api.features.chats.claim_active_verification import (
    run_claim_level_active_verification,
)
from experiments.run_interactive_agent_v20_crossdomain_cached_suite import (
    build_rag_for_dir,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/historical_vsr/vsr_pathrag_v6_cached_scale_30.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "vsr_pathrag_v17_status_gain_local_index.jsonl"
        ),
    )
    parser.add_argument(
        "--line-nos",
        default="0,7,13",
        help="Comma-separated cached-suite line numbers.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    requested_lines = {
        int(value.strip())
        for value in args.line_nos.split(",")
        if value.strip()
    }
    source_rows = {
        int(row["line_no"]): row
        for row in load_jsonl(ROOT_DIR / args.input)
        if int(row["line_no"]) in requested_lines
    }
    missing = requested_lines - set(source_rows)
    if missing:
        raise RuntimeError(f"Missing input rows: {sorted(missing)}")

    output_rows = []
    for line_no in sorted(requested_lines):
        row = source_rows[line_no]
        trace = row.get("trace") or {}
        graph_dir = ROOT_DIR / row["working_dir"]
        budget = copy.deepcopy(trace.get("compute_budget") or {})
        budget["requested_fast"] = False
        started = time.perf_counter()
        try:
            rag = build_rag_for_dir(graph_dir)
            base_evidence = copy.deepcopy(
                trace.get("evidence_summary")
                or {"entities": [], "relationships": []}
            )
            expanded, verification = await asyncio.wait_for(
                run_claim_level_active_verification(
                    rag,
                    row["question"],
                    row["answer"],
                    base_evidence,
                    budget,
                ),
                timeout=args.timeout_seconds,
            )
            output_row = {
                "line_no": line_no,
                "ok": True,
                "question": row["question"],
                "draft_answer": row["answer"],
                "answer_held_fixed": True,
                "working_dir": row["working_dir"],
                "compute_tier": budget.get("tier", "unknown"),
                "base_relationship_count": len(
                    base_evidence.get("relationships", [])
                ),
                "base_entity_count": len(
                    base_evidence.get("entities", [])
                ),
                "expanded_relationship_count": len(
                    expanded.get("relationships", [])
                ),
                "expanded_entity_count": len(
                    expanded.get("entities", [])
                ),
                "claim_active_verification": verification,
                "wall_clock_seconds": round(
                    time.perf_counter() - started,
                    4,
                ),
                "reference_answer_used": False,
                "generator_used": False,
                "judge_used": False,
            }
        except Exception as exc:
            output_row = {
                "line_no": line_no,
                "ok": False,
                "question": row["question"],
                "answer_held_fixed": True,
                "working_dir": row["working_dir"],
                "error": f"{type(exc).__name__}: {exc}",
                "wall_clock_seconds": round(
                    time.perf_counter() - started,
                    4,
                ),
                "reference_answer_used": False,
                "generator_used": False,
                "judge_used": False,
            }
        output_rows.append(output_row)
        print(
            f"line={line_no} ok={output_row['ok']} "
            f"queries={(output_row.get('claim_active_verification') or {}).get('query_count', 0)} "
            f"gain={(output_row.get('claim_active_verification') or {}).get('verified_status_gain', 0)} "
            f"diagnosed={(output_row.get('claim_active_verification') or {}).get('diagnosed_claim_count', 0)} "
            f"seconds={output_row['wall_clock_seconds']}",
            flush=True,
        )

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in output_rows
        ),
        encoding="utf-8",
    )
    successful = [row for row in output_rows if row["ok"]]
    verifications = [
        row["claim_active_verification"] for row in successful
    ]
    print(f"Successful rows: {len(successful)}/{len(output_rows)}")
    print(
        "Queries/status gain: "
        f"{sum(row.get('query_count', 0) for row in verifications)}/"
        f"{sum(row.get('verified_status_gain', 0) for row in verifications)}"
    )
    print(
        "Resolved claims: "
        f"{sum(row.get('resolved_claim_count', 0) for row in verifications)}"
    )
    print(
        "Diagnosed claims: "
        f"{sum(row.get('diagnosed_claim_count', 0) for row in verifications)}"
    )
    print(
        "Skipped query variants: "
        f"{sum(row.get('skipped_query_variant_count', 0) for row in verifications)}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
