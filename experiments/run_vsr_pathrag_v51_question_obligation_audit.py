import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    answer_contract,
    assess_question_obligation_alignment,
    assess_terminal_answer,
    build_question_obligation_focused_answer,
    build_risk_bounded_terminal_policy,
    classify_question,
    terminal_assessment_score,
)
from api.features.chats.answer_realization import (
    prepare_retrieved_excerpts,
)
from api.features.chats.vsr_pathrag import (
    build_query_graph,
    collect_source_chunks,
)


async def main_async(args) -> dict:
    rows = [
        json.loads(line)
        for line in (ROOT_DIR / args.input)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    cases = []
    eligible_count = 0
    triggered_count = 0
    character_reductions = []
    coverage_deltas = []
    for row in rows:
        question = row["question"]
        answer = row["answer"]
        trace = row.get("trace") or {}
        evidence = dict(trace.get("evidence_summary") or {})
        chunk_store_path = (
            ROOT_DIR
            / row["working_dir"]
            / "kv_store_text_chunks.json"
        )
        raw_chunks = json.loads(
            chunk_store_path.read_text(encoding="utf-8")
        )
        fake_rag = SimpleNamespace(
            text_chunks=SimpleNamespace(_data=raw_chunks)
        )
        question_type = classify_question(question)
        chunks = await collect_source_chunks(
            fake_rag,
            question,
            build_query_graph(
                question,
                question_type,
                answer_contract(question, question_type),
                evidence,
            ),
            evidence,
            max_chunks=8,
        )
        evidence["source_excerpts"] = prepare_retrieved_excerpts(
            chunks,
            max_excerpts=8,
        )
        alignment = assess_question_obligation_alignment(
            question,
            answer,
        )
        triggered_count += int(alignment["triggered"])
        focused = build_question_obligation_focused_answer(
            question,
            evidence,
        )
        current = assess_terminal_answer(
            question,
            answer,
            evidence,
        )
        focused_assessment = (
            assess_terminal_answer(question, focused, evidence)
            if focused
            else None
        )
        focused_policy = (
            build_risk_bounded_terminal_policy(focused_assessment)
            if focused_assessment
            else {"action": "abstain"}
        )
        current_state = current["answer_state"]
        focused_state = (
            focused_assessment["answer_state"]
            if focused_assessment
            else {}
        )
        eligible = bool(
            alignment["triggered"]
            and focused_assessment
            and focused_policy["action"]
            in {"commit", "commit_with_monitoring"}
            and focused_state.get("question_term_coverage", 0)
            >= current_state.get("question_term_coverage", 0)
            and len(focused) < len(answer)
            and terminal_assessment_score(focused_assessment)
            >= terminal_assessment_score(current)
        )
        eligible_count += int(eligible)
        if eligible:
            character_reductions.append(
                1.0 - len(focused) / max(1, len(answer))
            )
            coverage_deltas.append(
                focused_state["question_term_coverage"]
                - current_state["question_term_coverage"]
            )
        cases.append(
            {
                "line_no": row["line_no"],
                "dataset": row.get("dataset"),
                "alignment": alignment,
                "focused_answer": focused,
                "source_excerpt_count": len(
                    evidence["source_excerpts"]
                ),
                "current_chars": len(answer),
                "focused_chars": len(focused or ""),
                "current_question_coverage": current_state.get(
                    "question_term_coverage"
                ),
                "focused_question_coverage": focused_state.get(
                    "question_term_coverage"
                ),
                "current_terminal_score": (
                    terminal_assessment_score(current)
                ),
                "focused_terminal_score": (
                    terminal_assessment_score(focused_assessment)
                    if focused_assessment
                    else None
                ),
                "focused_policy": focused_policy["action"],
                "eligible": eligible,
            }
        )

    return {
        "agent_version": "vsr-pathrag-v51",
        "protocol": "question_obligation_focused_answer_audit",
        "sample_count": len(rows),
        "triggered_count": triggered_count,
        "eligible_count": eligible_count,
        "mean_character_reduction": round(
            statistics.mean(character_reductions)
            if character_reductions
            else 0.0,
            4,
        ),
        "mean_question_coverage_delta": round(
            statistics.mean(coverage_deltas)
            if coverage_deltas
            else 0.0,
            4,
        ),
        "reference_answer_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/historical_vsr/vsr_pathrag_v50_zero_overlap_30.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "vsr_pathrag_v51_question_obligation_audit.json"
        ),
    )
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    (ROOT_DIR / args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "samples={sample_count} triggered={triggered_count} "
        "eligible={eligible_count} reduction={mean_character_reduction} "
        "coverage_delta={mean_question_coverage_delta}".format(**result)
    )


if __name__ == "__main__":
    main()
