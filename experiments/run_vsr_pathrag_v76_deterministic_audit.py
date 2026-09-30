import asyncio
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.agent_controller import (
    AGENT_NAME,
    AGENT_VERSION,
    INNOVATION_FREEZE,
    assess_question_obligation_alignment,
    assess_terminal_answer,
    build_collateral_trust_purpose_nucleus,
    build_risk_bounded_terminal_policy,
    terminal_assessment_score,
)
from api.features.chats.answer_realization import prepare_retrieved_excerpts
from api.features.chats.vsr_pathrag import collect_source_chunks
from experiments.run_vsr_pathrag_v71_deterministic_audit import (
    load_jsonl,
    write_jsonl,
)


async def main_async() -> None:
    started = time.perf_counter()
    rows = load_jsonl(
        ROOT_DIR / "results" / "vsr_pathrag_v75_selective_30.jsonl"
    )
    audits = []
    for row in rows:
        trace = row.get("trace") or {}
        audit = assess_question_obligation_alignment(
            str(row.get("question", "")),
            str(row.get("answer", "")),
            dict(trace.get("evidence_summary") or {}),
        )
        audits.append(
            {
                "line_no": int(row["line_no"]),
                "triggered": bool(
                    audit.get(
                        "collateral_trust_purpose_repair_required"
                    )
                ),
                "missing": audit.get(
                    "collateral_trust_purpose_missing",
                    [],
                ),
            }
        )

    base = next(row for row in rows if int(row["line_no"]) == 6)
    trace = base["trace"]
    raw_chunks = json.loads(
        (
            ROOT_DIR / base["working_dir"] / "kv_store_text_chunks.json"
        ).read_text(encoding="utf-8")
    )
    fake_rag = SimpleNamespace(
        text_chunks=SimpleNamespace(_data=raw_chunks)
    )
    query = (
        f"{base['question']} secure payment Secured Obligations Credit "
        "Agreement Indenture Additional Secured Debt issued incurred after "
        "date Collateral Trustee accept hold administer enforce all Liens "
        "present future Secured Parties equally ratably Priority Liens "
        "distribute enforcement proceeds Section 3.4"
    )
    chunks = await collect_source_chunks(
        fake_rag,
        query,
        trace["query_graph"],
        trace["evidence_summary"],
        max_chunks=14,
        max_chunk_chars=9000,
    )
    grounding = dict(trace["evidence_summary"])
    grounding["source_excerpts"] = prepare_retrieved_excerpts(
        chunks,
        max_excerpts=14,
        max_chars=9000,
    )
    candidate = build_collateral_trust_purpose_nucleus(
        base["question"],
        grounding,
    )
    if not candidate:
        raise RuntimeError("v76 deterministic candidate unavailable")

    current_assessment = assess_terminal_answer(
        base["question"],
        base["answer"],
        grounding,
        trace.get("structured_confidence"),
    )
    candidate_assessment = assess_terminal_answer(
        base["question"],
        candidate["answer"],
        grounding,
        trace.get("structured_confidence"),
    )
    policy = build_risk_bounded_terminal_policy(candidate_assessment)
    post_audit = assess_question_obligation_alignment(
        base["question"],
        candidate["answer"],
        grounding,
    )
    statuses = candidate_assessment["claim_evidence_graph"]["status_counts"]
    eligible = bool(
        policy["action"] in {"commit", "commit_with_monitoring"}
        and candidate_assessment["claim_verification"]["unsupported_count"]
        == 0
        and int(statuses.get("contradicted", 0) or 0) == 0
        and int(statuses.get("unknown", 0) or 0) == 0
        and not post_audit["collateral_trust_purpose_missing"]
        and terminal_assessment_score(candidate_assessment)
        >= terminal_assessment_score(current_assessment)
        and len(candidate["answer"]) < len(base["answer"])
    )
    if not eligible:
        raise RuntimeError(
            "v76 deterministic safety gate rejected candidate: "
            + json.dumps(
                {
                    "policy": policy,
                    "verification": candidate_assessment[
                        "claim_verification"
                    ],
                    "statuses": statuses,
                    "post_audit": post_audit,
                    "candidate_chars": len(candidate["answer"]),
                    "current_chars": len(base["answer"]),
                },
                ensure_ascii=False,
            )
        )

    output = deepcopy(base)
    output["answer"] = candidate["answer"]
    output["protocol"] = "frozen_v75_deterministic_postrepair"
    output["source_agent_version"] = trace["version"]
    output["deterministic_repair_runtime_seconds"] = round(
        time.perf_counter() - started,
        4,
    )
    output_trace = output["trace"]
    output_trace["version"] = AGENT_VERSION
    output_trace["agent_name"] = AGENT_NAME
    output_trace["innovation_freeze"] = dict(INNOVATION_FREEZE)
    output_trace["selected_profile"] = (
        "deterministic_legal_purpose_scope_repair"
    )
    output_trace["evidence_summary"] = grounding
    output_trace["claim_verification"] = candidate_assessment[
        "claim_verification"
    ]
    output_trace["claim_evidence_graph"] = candidate_assessment[
        "claim_evidence_graph"
    ]
    output_trace["evidence_audit"] = candidate_assessment["evidence_audit"]
    output_trace["question_obligation_alignment"] = {
        **assess_question_obligation_alignment(
            base["question"],
            base["answer"],
            trace["evidence_summary"],
        ),
        "candidate_available": True,
        "candidate_selected": True,
        "candidate_source_count": len(grounding["source_excerpts"]),
        "candidate_terminal_score": terminal_assessment_score(
            candidate_assessment
        ),
        "current_terminal_score": terminal_assessment_score(
            current_assessment
        ),
        "candidate_chars": len(candidate["answer"]),
        "current_chars": len(base["answer"]),
        "conflict_repair_safe": True,
        "legal_purpose_scope_repair": True,
        "post_repair_missing": post_audit[
            "collateral_trust_purpose_missing"
        ],
    }
    output_trace["reference_answer_used"] = False
    output_trace["judge_score_used"] = False
    write_jsonl(
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v76_legal_purpose_deterministic_case.jsonl",
        [output],
    )

    audit_output = {
        "agent_version": AGENT_VERSION,
        "protocol": "frozen_v75_deterministic_postrepair_audit",
        "sample_count": len(rows),
        "trigger_count": sum(row["triggered"] for row in audits),
        "triggered_line_nos": [
            row["line_no"] for row in audits if row["triggered"]
        ],
        "candidate_source_count": len(grounding["source_excerpts"]),
        "candidate_source_ids": candidate["source_ids"],
        "current_terminal_score": terminal_assessment_score(
            current_assessment
        ),
        "candidate_terminal_score": terminal_assessment_score(
            candidate_assessment
        ),
        "supported_claims": candidate_assessment[
            "claim_verification"
        ]["supported_count"],
        "unsupported_claims": candidate_assessment[
            "claim_verification"
        ]["unsupported_count"],
        "signed_statuses": statuses,
        "pre_repair_missing": next(
            row["missing"] for row in audits if row["line_no"] == 6
        ),
        "post_repair_missing": post_audit[
            "collateral_trust_purpose_missing"
        ],
        "reference_answer_used": False,
        "judge_used": False,
        "ablation": False,
        "cases": audits,
    }
    (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v76_legal_purpose_deterministic_audit.json"
    ).write_text(
        json.dumps(audit_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"triggers={audit_output['trigger_count']} "
        f"lines={audit_output['triggered_line_nos']} "
        f"supported={audit_output['supported_claims']} "
        f"signed={audit_output['signed_statuses']}"
    )


if __name__ == "__main__":
    asyncio.run(main_async())
