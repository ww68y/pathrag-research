import argparse
import json
import re
from collections import Counter
from pathlib import Path


LEGAL_TERMS = {
    "agreement",
    "collateral",
    "lien",
    "liens",
    "secured",
    "obligations",
    "proceeds",
    "grantor",
    "indenture",
    "clause",
    "section",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", (text or "").lower())
        if token
        not in {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "from",
            "what",
            "which",
            "does",
            "how",
            "why",
            "are",
            "was",
            "were",
            "has",
            "have",
            "into",
            "about",
            "between",
            "based",
            "provided",
            "context",
        }
    }


def classify_question(question: str, dataset: str | None) -> str:
    terms = content_terms(question)
    lowered = question.lower()
    if dataset == "legal" or terms & LEGAL_TERMS:
        return "legal_clause_lookup"
    if any(word in lowered for word in ["differ", "difference", "compare", "between"]):
        return "comparison"
    if lowered.startswith("why") or any(word in lowered for word in ["reason", "cause", "lead to", "affect"]):
        return "causal_explanation"
    if any(word in lowered for word in ["what role", "what are", "conditions", "factors", "ways"]):
        return "enumeration"
    if lowered.startswith("how"):
        return "process_or_explanation"
    return "open_domain_qa"


def answer_state(answer: str, question: str) -> dict:
    answer_clean = clean_text(answer)
    q_terms = content_terms(question)
    a_terms = content_terms(answer)
    q_cov = len(q_terms & a_terms) / max(1, len(q_terms))
    refusal_like = any(
        phrase in answer_clean.lower()
        for phrase in [
            "not enough information",
            "cannot determine",
            "provided context does not",
            "insufficient context",
            "i don't know",
        ]
    )
    return {
        "chars": len(answer_clean),
        "question_term_coverage": round(q_cov, 6),
        "refusal_like": refusal_like,
        "short_answer": len(answer_clean) < 240,
        "long_answer": len(answer_clean) > 900,
    }


def action(tool: str, decision: str, observation: str, **extra) -> dict:
    payload = {
        "tool": tool,
        "decision": decision,
        "observation": observation,
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def module_enabled(row: dict, name: str) -> bool:
    module = row.get(name) or {}
    return bool(module.get("enabled") or module.get("action") or module.get("accepted"))


def summarize_existing_modules(row: dict) -> list[dict]:
    steps = []
    module_specs = [
        ("question_anchored_repair", "question_anchor_repair"),
        ("question_chunk_completion", "question_chunk_completion"),
        ("query_expanded_refusal_repair", "query_expanded_refusal_repair"),
        ("substance_refusal_repair", "substance_refusal_repair"),
        ("query_expanded_contract_gap_repair", "query_expanded_contract_gap_repair"),
        ("supported_facet_guard", "supported_facet_guard"),
        ("supported_facet_fusion", "supported_facet_fusion"),
        ("facet_balanced_compaction", "facet_balanced_compaction"),
    ]
    for field, tool in module_specs:
        module = row.get(field) or {}
        if not module_enabled(row, field):
            continue
        decision = module.get("action") or ("accepted" if module.get("accepted") else "checked")
        observation = module.get("reason") or module.get("decision") or "runtime module executed"
        steps.append(action(tool, str(decision), str(observation), accepted=module.get("accepted")))
    return steps


def build_agent(row: dict) -> dict:
    question = row.get("question", "")
    prediction = row.get("prediction", "")
    dataset = row.get("dataset")
    q_type = classify_question(question, dataset)
    state = answer_state(prediction, question)
    graph_edges = row.get("graph_edges") or 0
    graph_nodes = row.get("graph_nodes") or 0
    residual = row.get("residual_hardcase_editor") or {}
    dual_gate = row.get("dual_objective_gate") or {}
    evidence_gate = row.get("evidence_risk_gate") or {}

    risk_flags = []
    if state["refusal_like"]:
        risk_flags.append("refusal_like_answer")
    if state["short_answer"]:
        risk_flags.append("short_answer")
    if state["long_answer"]:
        risk_flags.append("long_answer")
    if q_type == "legal_clause_lookup":
        risk_flags.append("legal_precision_required")
    if residual.get("enabled"):
        risk_flags.append("residual_hardcase_candidate")
    if dual_gate.get("enabled") and not dual_gate.get("accepted"):
        risk_flags.append("candidate_reverted_by_dual_gate")

    selected_tools = ["question_analyzer", "evidence_policy_router", "answer_state_observer"]
    if graph_edges or graph_nodes:
        selected_tools.append("graph_evidence_reader")
    selected_tools.extend(step["tool"] for step in summarize_existing_modules(row))
    if residual.get("enabled"):
        selected_tools.append("residual_hardcase_editor")
    if evidence_gate.get("enabled"):
        selected_tools.append("evidence_risk_gate")
    if dual_gate.get("enabled"):
        selected_tools.append("dual_objective_gate")

    actions = [
        action(
            "question_analyzer",
            q_type,
            f"classified question as {q_type}",
            dataset=dataset,
            question_terms=sorted(content_terms(question))[:20],
        ),
        action(
            "evidence_policy_router",
            "graph_aware" if graph_edges or graph_nodes else "text_first",
            f"available graph nodes={graph_nodes}, edges={graph_edges}",
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        ),
        action(
            "answer_state_observer",
            "inspect_current_answer",
            "computed answer length, refusal signal, and question-term coverage",
            state=state,
            risk_flags=risk_flags,
        ),
    ]
    actions.extend(summarize_existing_modules(row))

    if residual.get("enabled"):
        guard = residual.get("guard") or {}
        decision = residual.get("decision") or {}
        actions.append(
            action(
                "residual_hardcase_editor",
                residual.get("action") or ("accept" if residual.get("accepted") else "reject"),
                decision.get("reason") or "edited residual hard case with local full-text evidence",
                edit_type=decision.get("edit_type"),
                accepted=residual.get("accepted"),
                missing_fact_coverage=guard.get("missing_fact_coverage"),
                current_chars=guard.get("current_chars"),
                candidate_chars=guard.get("candidate_chars"),
            )
        )

    if evidence_gate.get("enabled"):
        actions.append(
            action(
                "evidence_risk_gate",
                evidence_gate.get("action"),
                "checked question-term preservation, support delta, unsupported ratio, and supported-facet retention",
                accepted=evidence_gate.get("accepted"),
                rejection_reasons=evidence_gate.get("rejection_reasons", []),
            )
        )

    if dual_gate.get("enabled"):
        actions.append(
            action(
                "dual_objective_gate",
                dual_gate.get("action"),
                "checked evidence risk plus high-compression facet retention",
                accepted=dual_gate.get("accepted"),
                rejection_reasons=dual_gate.get("rejection_reasons", []),
                compression_ratio=dual_gate.get("compression_ratio"),
                supported_facet_retention=dual_gate.get("supported_facet_retention"),
            )
        )

    if dual_gate.get("enabled"):
        final_policy = "use_edited_answer" if dual_gate.get("accepted") else "rollback_to_safer_answer"
    elif residual.get("enabled") and residual.get("accepted"):
        final_policy = "use_residual_hardcase_edit"
    else:
        final_policy = "use_current_answer"

    actions.append(
        action(
            "final_answer_controller",
            final_policy,
            "selected final answer after planner actions and runtime safety gates",
            final_answer_chars=len(clean_text(prediction)),
        )
    )

    return {
        "enabled": True,
        "version": "v61",
        "agent_name": "Evidence-Aware KG-RAG Agent",
        "agent_loop": "observe -> plan -> act -> verify -> rollback_or_answer",
        "planner": {
            "goal": "answer the question with grounded text/graph evidence while minimizing hallucination and unnecessary graph cost",
            "question_type": q_type,
            "selected_tools": list(dict.fromkeys(selected_tools)),
            "stop_condition": "final answer passes the available evidence gates or the candidate is rolled back",
        },
        "initial_observation": {
            "dataset": dataset,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "answer_state": state,
            "risk_flags": risk_flags,
        },
        "actions": actions,
        "final_decision": {
            "policy": final_policy,
            "answer_chars": len(clean_text(prediction)),
            "used_repair_candidate": final_policy in {"use_edited_answer", "use_residual_hardcase_edit"},
            "rolled_back": final_policy == "rollback_to_safer_answer",
        },
        "reference_answer_used": False,
        "judge_score_used": False,
    }


def build_summary(rows: list[dict], source: Path) -> str:
    agents = [r.get("kg_rag_agent") or {} for r in rows]
    decisions = Counter((a.get("final_decision") or {}).get("policy", "none") for a in agents)
    qtypes = Counter((a.get("planner") or {}).get("question_type", "unknown") for a in agents)
    tool_counts = Counter()
    for agent in agents:
        for tool in (agent.get("planner") or {}).get("selected_tools", []):
            tool_counts[tool] += 1

    lines = [
        "# SAGE v61 Evidence-Aware KG-RAG Agent Controller Summary\n\n",
        f"- Source: `{source}`\n",
        "- Purpose: make the previous adaptive KG-RAG pipeline explicit as an agent controller.\n",
        "- Agent loop: observe -> plan -> act -> verify -> rollback_or_answer.\n",
        "- Reference answers and judge scores are not used at runtime.\n\n",
        "| Metric | Count |\n|---|---:|\n",
        f"| Samples | {len(rows)} |\n",
        f"| Agent traces | {len(agents)} |\n\n",
        "## Final Decisions\n\n",
        "| Decision | Count |\n|---|---:|\n",
    ]
    for decision, count in sorted(decisions.items()):
        lines.append(f"| {decision} | {count} |\n")

    lines.extend(["\n## Question Types\n\n", "| Question Type | Count |\n|---|---:|\n"])
    for qtype, count in sorted(qtypes.items()):
        lines.append(f"| {qtype} | {count} |\n")

    lines.extend(["\n## Tool Usage\n\n", "| Tool | Samples |\n|---|---:|\n"])
    for tool, count in tool_counts.most_common():
        lines.append(f"| {tool} | {count} |\n")

    lines.extend(
        [
            "\n## Per-Sample Agent Decisions\n\n",
            "| Line | Dataset | Question Type | Final Policy | Actions | Risk Flags |\n",
            "|---:|---|---|---|---:|---|\n",
        ]
    )
    for row in rows:
        agent = row.get("kg_rag_agent") or {}
        planner = agent.get("planner") or {}
        final_decision = agent.get("final_decision") or {}
        obs = agent.get("initial_observation") or {}
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {planner.get('question_type')} | "
            f"{final_decision.get('policy')} | {len(agent.get('actions') or [])} | "
            f"{', '.join(obs.get('risk_flags') or []) or 'none'} |\n"
        )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    for row in rows:
        row["prediction_before_kg_rag_agent"] = row.get("prediction", "")
        row["kg_rag_agent"] = build_agent(row)
        row["method"] = "mse-pathrag-sage-v61-kg-rag-agent-controller"

    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(build_summary(rows, args.input), encoding="utf-8")
    print(f"Saved v61 agent results to {args.output}")
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
