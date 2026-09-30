import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
UNCERTAINTY_PATTERNS = (
    r"\bcannot\b",
    r"\bdoes not (?:contain|provide|describe)\b",
    r"\bnot (?:explicitly )?(?:contained|provided|described)\b",
    r"\binsufficient (?:information|evidence)\b",
    r"\bbased solely\b",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_line(rows: list[dict]) -> dict[int, dict]:
    return {int(row.get("line_no", index)): row for index, row in enumerate(rows)}


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def sanitize_evidence_labels(answer: str) -> str:
    answer = re.sub(r"\s*\((?:Text|Graph)\s+\d+\)", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\b(?:Text|Graph)\s+\d+\b", "the evidence", answer, flags=re.IGNORECASE)
    return clean_text(answer)


def has_explicit_uncertainty(answer: str) -> bool:
    lowered = clean_text(answer).lower()
    return any(re.search(pattern, lowered) for pattern in UNCERTAINTY_PATTERNS)


def supported_text_repair(completion_row: dict) -> tuple[bool, dict]:
    agent = completion_row.get("contract_completion_agent") or {}
    guard = agent.get("repair_guard") or {}
    post = guard.get("post_claim_metrics") or {}
    delta = completion_row.get("evidence_delta_guard") or {}
    original = completion_row.get("prediction_before_contract_completion", completion_row.get("prediction", ""))
    checks = {
        "text_only_evidence": agent.get("evidence_mode") == "text-only",
        "explicit_uncertainty": has_explicit_uncertainty(str(original)),
        "repair_accepted": bool(agent.get("repair_accepted") and guard.get("accepted")),
        "missing_fact_verified": bool((agent.get("verification") or {}).get("missing_supported_facts")),
        "supported_claim_present": int(post.get("supported_claims", 0)) >= 1,
        "no_unsupported_claim": int(post.get("unsupported_claims", 0)) == 0,
        "minimum_text_support": float(post.get("mean_support", 0.0)) >= 0.55,
        "evidence_delta_retained": not delta.get("enabled") or delta.get("action") == "retain_repair",
    }
    return all(checks.values()), checks


def unique_context_cost(rows: list[dict]) -> tuple[int, float]:
    contexts = {}
    for row in rows:
        key = row.get("context_hash") or row.get("working_dir") or f"line-{row.get('line_no')}"
        contexts.setdefault(str(key), float(row.get("insert_seconds", 0.0)))
    return len(contexts), sum(contexts.values())


def write_summary(path: Path, rows: list[dict], text_rows: dict[int, dict], graph_rows: dict[int, dict]) -> None:
    agents = [row.get("evidence_calibrated_lazy_agent") or {} for row in rows]
    actions = Counter(agent.get("action", "unknown") for agent in agents)
    graph_attempt_lines = [int(row["line_no"]) for row in rows if row["evidence_calibrated_lazy_agent"].get("graph_attempted")]
    attempted_graph_rows = [graph_rows[line] for line in graph_attempt_lines]
    full_contexts, full_index_seconds = unique_context_cost(list(graph_rows.values()))
    attempted_contexts, attempted_index_seconds = unique_context_cost(attempted_graph_rows)
    text_seconds = sum(float(row.get("query_seconds", 0.0)) for row in text_rows.values())
    graph_query_seconds = sum(float(row.get("query_seconds", 0.0)) for row in attempted_graph_rows)
    graph_reduction = 1 - len(graph_attempt_lines) / max(1, len(rows))
    index_reduction = 1 - attempted_index_seconds / max(1e-9, full_index_seconds)
    lines = [
        "# SAGE v39 Evidence-Calibrated Lazy Agent Summary\n\n",
        f"- Samples: {len(rows)}\n",
        "- Repair dominance uses text-only evidence and runs before graph construction.\n",
        "- Candidate arbitration is replayed only after a text-first graph trigger.\n",
        "- Reference answers and evaluation scores are not runtime inputs.\n\n",
        "## Actions\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")
    lines.extend([
        "\n## Replay Cost\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Text query time | {text_seconds:.3f} s |\n",
        f"| Text-only contract verifications | {sum(bool(agent.get('contract_verified')) for agent in agents)} |\n",
        f"| Pre-graph repairs accepted | {sum(agent.get('action') == 'use_text_repair' for agent in agents)} |\n",
        f"| Graph attempts | {len(graph_attempt_lines)} / {len(rows)} |\n",
        f"| Selective arbiter calls | {len(graph_attempt_lines)} / {len(rows)} |\n",
        f"| Added graph query time | {graph_query_seconds:.3f} s |\n",
        f"| Graph query reduction vs full PathRAG | {graph_reduction * 100:.2f}% |\n",
        f"| Attempted graph contexts | {attempted_contexts} / {full_contexts} |\n",
        f"| Attempted graph indexing time | {attempted_index_seconds:.3f} s |\n",
        f"| Full graph indexing time | {full_index_seconds:.3f} s |\n",
        f"| Graph indexing reduction | {index_reduction * 100:.2f}% |\n",
        "\n## Per-Sample Decisions\n\n",
        "| Line | Dataset | Action | Graph Attempted | Final Source | Reasons |\n",
        "|---:|---|---|---|---|---|\n",
    ])
    for row in sorted(rows, key=lambda item: int(item.get("line_no", 0))):
        agent = row["evidence_calibrated_lazy_agent"]
        reasons = "; ".join(agent.get("reasons", []))
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('graph_attempted')} | {agent.get('selected_candidate')} | {reasons} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--text-first", required=True)
    parser.add_argument("--arbiter", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    text_rows = by_line(load_jsonl(ROOT_DIR / args.text))
    graph_rows = by_line(load_jsonl(ROOT_DIR / args.graph))
    text_first_rows = by_line(load_jsonl(ROOT_DIR / args.text_first))
    arbiter_rows = by_line(load_jsonl(ROOT_DIR / args.arbiter))
    completion_rows = by_line(load_jsonl(ROOT_DIR / args.completion))
    common_lines = sorted(set(text_rows) & set(graph_rows) & set(text_first_rows) & set(arbiter_rows) & set(completion_rows))

    output_rows = []
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line_no in common_lines:
            text_row = text_rows[line_no]
            completion_row = completion_rows[line_no]
            text_first_agent = text_first_rows[line_no].get("text_first_contract_agent") or {}
            arbiter_row = arbiter_rows[line_no]
            dominance, dominance_checks = supported_text_repair(completion_row)
            graph_attempted = bool(text_first_agent.get("graph_rescue_attempted"))
            contract_agent = completion_row.get("contract_completion_agent") or {}

            if dominance:
                selected = "repair"
                prediction = sanitize_evidence_labels(str(completion_row.get("prediction", "")))
                action = "use_text_repair"
                graph_attempted = False
                reasons = ["supported_text_repair_dominates_explicit_uncertainty"]
            elif graph_attempted:
                selected = str(
                    ((arbiter_row.get("evidence_candidate_arbiter") or {}).get("decision") or {}).get(
                        "selected_candidate", "text"
                    )
                )
                if selected == "graph":
                    prediction = str(graph_rows[line_no].get("prediction", ""))
                elif selected == "repair":
                    prediction = str(arbiter_row.get("prediction", text_row.get("prediction", "")))
                else:
                    selected = "text"
                    prediction = str(text_row.get("prediction", ""))
                action = f"selective_arbiter_{selected}"
                reasons = ["text_contract_triggered_graph", "candidate_arbiter_after_graph_trigger"]
            else:
                selected = "text"
                prediction = str(text_row.get("prediction", ""))
                action = "keep_text"
                reasons = ["text_contract_satisfied", "graph_not_constructed"]

            row = dict(text_row)
            row["prediction_before_evidence_calibrated_lazy_agent"] = row.get("prediction", "")
            row["prediction"] = prediction
            row["method"] = f"sage_v39_{action}"
            row["evidence_calibrated_lazy_agent"] = {
                "enabled": True,
                "version": "v39",
                "action": action,
                "selected_candidate": selected,
                "graph_attempted": graph_attempted,
                "contract_verified": bool(contract_agent.get("verification_attempted")),
                "repair_dominance": dominance,
                "repair_dominance_checks": dominance_checks,
                "repair_sanitized": dominance and prediction != str(completion_row.get("prediction", "")),
                "reasons": reasons,
                "reference_answer_used": False,
            }
            output_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_summary(ROOT_DIR / args.summary, output_rows, text_rows, graph_rows)
    print(f"Saved v39 evidence-calibrated lazy results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
