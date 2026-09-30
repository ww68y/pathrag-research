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

from api.features.chats.agent_controller import (
    assess_terminal_answer,
    build_risk_bounded_terminal_policy,
    build_terminal_claim_pruned_answer,
)
from api.features.chats.claim_active_verification import (
    apply_diagnosed_claim_correction,
    build_delayed_verification_credit,
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


def settle_answer(
    question: str,
    answer: str,
    evidence_summary: dict,
) -> tuple[dict, dict, str]:
    current = assess_terminal_answer(
        question,
        answer,
        evidence_summary,
    )
    policy = build_risk_bounded_terminal_policy(current)
    final = current
    fallback_source = None
    if policy["action"] == "abstain":
        pruned_answer = build_terminal_claim_pruned_answer(current)
        if pruned_answer:
            fallback = assess_terminal_answer(
                question,
                pruned_answer,
                evidence_summary,
            )
            candidate_policy = build_risk_bounded_terminal_policy(
                current,
                fallback,
            )
            if candidate_policy["action"] == "rollback_to_verified_fallback":
                final = fallback
                policy = candidate_policy
                fallback_source = "terminal_claim_pruning"
    if policy["action"] == "abstain":
        final = assess_terminal_answer(
            question,
            (
                "The available verified evidence is insufficient to provide "
                "a reliable answer to this question."
            ),
            evidence_summary,
        )
    policy["fallback_source"] = fallback_source
    return final, policy, fallback_source or "none"


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
            "vsr_pathrag_v19_diagnosis_correction_local_index.jsonl"
        ),
    )
    parser.add_argument("--line-nos", default="0,7,13")
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
            corrected_answer, correction = (
                apply_diagnosed_claim_correction(
                    row["question"],
                    row["answer"],
                    verification,
                    expanded,
                )
            )
            final, terminal_policy, fallback_source = settle_answer(
                row["question"],
                corrected_answer,
                expanded,
            )
            delayed_credit = build_delayed_verification_credit(
                verification,
                {
                    "enabled": True,
                    "triggered": False,
                    "committed": False,
                    "used_source_ids": [],
                    "stop_reason": "fixed_draft_no_generator",
                },
                final["claim_evidence_graph"],
                final["evidence_audit"],
                terminal_policy,
                expanded,
            )
            output_row = {
                "line_no": line_no,
                "ok": True,
                "question": row["question"],
                "draft_answer": row["answer"],
                "corrected_answer": corrected_answer,
                "draft_held_fixed_except_diagnosis_correction": True,
                "working_dir": row["working_dir"],
                "compute_tier": budget.get("tier", "unknown"),
                "claim_active_verification": verification,
                "diagnosis_answer_correction": correction,
                "verification_outcome_credit": delayed_credit,
                "terminal_policy": terminal_policy,
                "terminal_fallback_source": fallback_source,
                "final_status_counts": final[
                    "claim_evidence_graph"
                ].get("status_counts", {}),
                "final_risk_level": final["evidence_audit"].get(
                    "risk_level",
                    "unknown",
                ),
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
        verification = output_row.get("claim_active_verification") or {}
        correction = output_row.get("diagnosis_answer_correction") or {}
        credit = output_row.get("verification_outcome_credit") or {}
        print(
            f"line={line_no} ok={output_row['ok']} "
            f"queries={verification.get('query_count', 0)} "
            f"gain={verification.get('verified_status_gain', 0)} "
            f"corrected={correction.get('accepted', False)} "
            f"credit={credit.get('mean_delayed_credit', 0)} "
            f"terminal={(output_row.get('terminal_policy') or {}).get('action', 'n/a')}",
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
    corrections = [
        row["diagnosis_answer_correction"] for row in successful
    ]
    credits = [
        row["verification_outcome_credit"] for row in successful
    ]
    print(f"Successful rows: {len(successful)}/{len(output_rows)}")
    print(
        "Accepted corrections: "
        f"{sum(row.get('accepted', False) for row in corrections)}"
    )
    print(
        "Credited/retained/partial/revoked queries: "
        f"{sum(row.get('credited_query_count', 0) for row in credits)}/"
        f"{sum(row.get('retained_query_count', 0) for row in credits)}/"
        f"{sum(row.get('partially_retained_query_count', 0) for row in credits)}/"
        f"{sum(row.get('revoked_query_count', 0) for row in credits)}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
