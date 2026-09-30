import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_answer_contract_agent import clean_text
from build_claim_evidence_agent import content_tokens, write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def supported_facet_stats(before: str, after: str, evidence: list[dict]) -> dict:
    evidence_text = " ".join(str(item.get("text") or "") for item in evidence)
    before_tokens = content_tokens(before)
    after_tokens = content_tokens(after)
    evidence_tokens = content_tokens(evidence_text)
    supported_before = before_tokens & evidence_tokens
    retained = supported_before & after_tokens
    dropped = supported_before - after_tokens
    retention = len(retained) / max(1, len(supported_before))
    return {
        "before_chars": len(clean_text(before)),
        "after_chars": len(clean_text(after)),
        "supported_before_terms": len(supported_before),
        "retained_supported_terms": len(retained),
        "dropped_supported_terms": len(dropped),
        "supported_facet_retention": round(retention, 6),
        "compression_ratio": round(len(clean_text(after)) / max(1, len(clean_text(before))), 6),
        "dropped_supported_term_examples": sorted(dropped)[:60],
    }


def should_revert(stats: dict) -> bool:
    return (
        stats["before_chars"] >= 900
        and stats["after_chars"] < stats["before_chars"] * 0.60
        and stats["supported_before_terms"] >= 40
        and stats["supported_facet_retention"] < 0.22
    )


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    guards = [row.get("supported_facet_guard") or {} for row in rows]
    enabled = [guard for guard in guards if guard.get("enabled")]
    reverted = [guard for guard in enabled if guard.get("action") == "revert_to_pre_repair_answer"]
    actions = Counter(guard.get("action", "not_applicable") for guard in guards)
    lines = [
        "# SAGE v55 Supported-Facet Retention Guard Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Purpose: reject repairs that over-compress the answer and drop many evidence-supported facets from the original answer.\n",
        "- Reference answers and judge scores are not runtime inputs.\n\n",
        "| Metric | Count |\n",
        "|---|---:|\n",
        f"| Enabled accepted v54 repairs | {len(enabled)} |\n",
        f"| Reverted by v55 guard | {len(reverted)} |\n\n",
        "## Actions\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")

    lines.extend(
        [
            "\n## Per-Sample Guard Decisions\n\n",
            "| Line | Dataset | Action | Retention | Compression | Supported Terms | Reason |\n",
            "|---:|---|---|---:|---:|---:|---|\n",
        ]
    )
    for row in rows:
        guard = row.get("supported_facet_guard") or {}
        if not guard.get("enabled"):
            continue
        stats = guard.get("stats") or {}
        reason = clean_text(guard.get("reason", "")).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {guard.get('action')} | "
            f"{stats.get('supported_facet_retention')} | {stats.get('compression_ratio')} | "
            f"{stats.get('supported_before_terms')} | {reason} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    rows = load_jsonl(ROOT_DIR / args.input)
    output_rows = []
    for row in rows:
        new_row = dict(row)
        agent = row.get("query_expanded_contract_gap_repair") or {}
        before = row.get("prediction_before_query_expanded_contract_gap_repair")
        if not (agent.get("accepted") and before):
            new_row["supported_facet_guard"] = {"enabled": False}
            output_rows.append(new_row)
            continue

        stats = supported_facet_stats(before, row.get("prediction", ""), agent.get("selected_evidence") or [])
        revert = should_revert(stats)
        guard = {
            "enabled": True,
            "version": "v55",
            "stats": stats,
            "action": "revert_to_pre_repair_answer" if revert else "keep_repair",
            "reason": (
                "candidate compressed a long answer while retaining too few evidence-supported original facets"
                if revert
                else "candidate retained enough evidence-supported original facets"
            ),
        }
        if revert:
            new_row["prediction_before_supported_facet_guard"] = row.get("prediction", "")
            new_row["prediction"] = before
            new_row["method"] = "sage_v55_supported_facet_guard_revert"
        elif row.get("method") == "sage_v54_query_expanded_contract_gap_repair":
            new_row["method"] = "sage_v55_supported_facet_guard"
        new_row["supported_facet_guard"] = guard
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v55 guarded results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
