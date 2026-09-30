import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


METRICS = (
    "correctness",
    "coverage",
    "relevance",
    "conciseness",
    "overall",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def keyed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["line_no"]): row for row in rows}


def main() -> None:
    base_results_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v51_anchor_preserving_30.jsonl"
    )
    repair_results_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v52_refusal_guard_case.jsonl"
    )
    audit_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v52_refusal_guard_audit.json"
    )
    base_eval_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v51_anchor_preserving_30_eval.jsonl"
    )
    repair_eval_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v52_refusal_guard_case_eval.jsonl"
    )
    output_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v52_selective_replay_30.jsonl"
    )
    eval_output_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v52_selective_replay_30_eval.jsonl"
    )
    manifest_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v52_selective_replay_manifest.json"
    )
    summary_path = (
        ROOT_DIR
        / "results"
        / "vsr_pathrag_v52_selective_replay_30_eval_summary.md"
    )

    base_results = keyed(load_jsonl(base_results_path))
    repair_results = keyed(load_jsonl(repair_results_path))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    conflict_lines = {
        int(line_no)
        for line_no in audit["conflict_line_nos"]
    }
    if set(repair_results) != conflict_lines:
        raise ValueError(
            "actual v52 repair cases do not match audited conflicts"
        )
    if set(base_results) != set(range(30)):
        raise ValueError("v51 base result is not a complete 30-row run")

    promoted_results = []
    for line_no in range(30):
        if line_no in conflict_lines:
            row = dict(repair_results[line_no])
            row["promotion_replay"] = {
                "triggered": True,
                "source_agent_version": "vsr-pathrag-v52",
                "source_result": str(repair_results_path),
            }
        else:
            row = dict(base_results[line_no])
            row["promotion_replay"] = {
                "triggered": False,
                "source_agent_version": "vsr-pathrag-v51",
                "source_result": str(base_results_path),
            }
        promoted_results.append(row)
    write_jsonl(output_path, promoted_results)

    base_eval = keyed(load_jsonl(base_eval_path))
    repair_eval = keyed(load_jsonl(repair_eval_path))
    promoted_eval = [
        repair_eval[line_no]
        if line_no in conflict_lines
        else base_eval[line_no]
        for line_no in range(30)
    ]
    write_jsonl(eval_output_path, promoted_eval)

    means = {
        metric: round(
            statistics.mean(
                float(row["scores"][metric])
                for row in promoted_eval
            ),
            3,
        )
        for metric in METRICS
    }
    base_means = {
        metric: round(
            statistics.mean(
                float(row["scores"][metric])
                for row in base_eval.values()
            ),
            3,
        )
        for metric in METRICS
    }
    deltas = {
        metric: round(means[metric] - base_means[metric], 3)
        for metric in METRICS
    }
    verdicts = Counter(
        str(row["scores"]["verdict"]) for row in promoted_eval
    )
    manifest = {
        "protocol": "deterministic_gate_selective_replay",
        "agent_version": "vsr-pathrag-v52",
        "sample_count": 30,
        "audited_trigger_count": len(conflict_lines),
        "actual_v52_rerun_count": len(repair_results),
        "reused_v51_nontrigger_count": 30 - len(conflict_lines),
        "triggered_line_nos": sorted(conflict_lines),
        "means": means,
        "v51_means": base_means,
        "mean_deltas": deltas,
        "verdict_counts": dict(verdicts),
        "runtime_reference_answer_used": False,
        "judge_reuse": (
            "Reused completed per-answer judgments; the changed v52 "
            "answer was judged independently."
        ),
        "full_stochastic_rerun": False,
        "ablation": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# VSR-PathRAG v52 Selective-Replay Evaluation",
        "",
        "The deterministic refusal gate was audited over all 30 v51 "
        "outputs. Only the triggered row was rerun with v52; unchanged "
        "rows and their completed judgments were reused.",
        "",
        "| Metric | v51 | v52 selective replay | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {base_means[metric]:.3f} | "
            f"{means[metric]:.3f} | {deltas[metric]:+.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Samples: {manifest['sample_count']}",
            f"- Audited triggers: {manifest['audited_trigger_count']}",
            f"- Actual v52 reruns: {manifest['actual_v52_rerun_count']}",
            (
                "- Reused non-triggered v51 rows: "
                f"{manifest['reused_v51_nontrigger_count']}"
            ),
            f"- Triggered lines: {manifest['triggered_line_nos']}",
            "- Full stochastic rerun: false",
            "- Runtime reference answers: not used",
            "- Formal ablation: false",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"samples=30 triggers={len(conflict_lines)} "
        f"v51={base_means['overall']:.3f} "
        f"v52={means['overall']:.3f} "
        f"delta={deltas['overall']:+.3f}"
    )


if __name__ == "__main__":
    main()
