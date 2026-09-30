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
        return [json.loads(line) for line in source if line.strip()]


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


def metric_means(
    rows: list[dict[str, Any]] | dict[int, dict[str, Any]],
) -> dict[str, float]:
    values = rows.values() if isinstance(rows, dict) else rows
    materialized = list(values)
    return {
        metric: round(
            statistics.mean(
                float(row["scores"][metric])
                for row in materialized
            ),
            3,
        )
        for metric in METRICS
    }


def main() -> None:
    results_dir = ROOT_DIR / "results"
    base_results_path = (
        results_dir / "vsr_pathrag_v52_selective_replay_30.jsonl"
    )
    repair_results_path = (
        results_dir / "vsr_pathrag_v53_definition_nucleus_case.jsonl"
    )
    audit_path = (
        results_dir / "vsr_pathrag_v53_definition_nucleus_audit.json"
    )
    base_eval_path = (
        results_dir
        / "vsr_pathrag_v52_selective_replay_30_eval.jsonl"
    )
    repair_eval_path = (
        results_dir
        / "vsr_pathrag_v53_definition_nucleus_case_eval.jsonl"
    )
    output_path = (
        results_dir / "vsr_pathrag_v53_selective_replay_30.jsonl"
    )
    eval_output_path = (
        results_dir
        / "vsr_pathrag_v53_selective_replay_30_eval.jsonl"
    )
    manifest_path = (
        results_dir
        / "vsr_pathrag_v53_selective_replay_manifest.json"
    )
    summary_path = (
        results_dir
        / "vsr_pathrag_v53_selective_replay_30_eval_summary.md"
    )

    base_results = keyed(load_jsonl(base_results_path))
    repair_results = keyed(load_jsonl(repair_results_path))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    trigger_lines = {
        int(line_no)
        for line_no in audit["triggered_line_nos"]
    }
    if set(repair_results) != trigger_lines:
        raise ValueError(
            "actual v53 repair cases do not match audited triggers"
        )
    if set(base_results) != set(range(30)):
        raise ValueError("v52 base result is not a complete 30-row replay")

    promoted_results = []
    for line_no in range(30):
        if line_no in trigger_lines:
            row = dict(repair_results[line_no])
            row["promotion_replay"] = {
                "triggered": True,
                "source_agent_version": "vsr-pathrag-v53",
                "source_result": str(repair_results_path),
            }
        else:
            row = dict(base_results[line_no])
            row["promotion_replay"] = {
                "triggered": False,
                "source_agent_version": "vsr-pathrag-v52",
                "source_result": str(base_results_path),
            }
        promoted_results.append(row)
    write_jsonl(output_path, promoted_results)

    base_eval = keyed(load_jsonl(base_eval_path))
    repair_eval = keyed(load_jsonl(repair_eval_path))
    promoted_eval = [
        repair_eval[line_no]
        if line_no in trigger_lines
        else base_eval[line_no]
        for line_no in range(30)
    ]
    write_jsonl(eval_output_path, promoted_eval)

    means = metric_means(promoted_eval)
    base_means = metric_means(base_eval)
    deltas = {
        metric: round(means[metric] - base_means[metric], 3)
        for metric in METRICS
    }
    verdicts = Counter(
        str(row["scores"]["verdict"]) for row in promoted_eval
    )
    manifest = {
        "protocol": "deterministic_gate_selective_replay",
        "agent_version": "vsr-pathrag-v53",
        "sample_count": 30,
        "audited_trigger_count": len(trigger_lines),
        "actual_v53_rerun_count": len(repair_results),
        "reused_v52_nontrigger_count": 30 - len(trigger_lines),
        "triggered_line_nos": sorted(trigger_lines),
        "means": means,
        "v52_means": base_means,
        "mean_deltas": deltas,
        "verdict_counts": dict(verdicts),
        "runtime_reference_answer_used": False,
        "judge_reuse": (
            "Reused completed per-answer judgments; the changed v53 "
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
        "# VSR-PathRAG v53 Selective-Replay Evaluation",
        "",
        "The deterministic definition-nucleus gate was audited over "
        "all 30 v52 replay outputs. Only the triggered row was rerun "
        "with v53; unchanged rows and their judgments were reused.",
        "",
        "| Metric | v52 | v53 selective replay | Delta |",
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
            f"- Actual v53 reruns: {manifest['actual_v53_rerun_count']}",
            (
                "- Reused non-triggered v52 rows: "
                f"{manifest['reused_v52_nontrigger_count']}"
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
        f"samples=30 triggers={len(trigger_lines)} "
        f"v52={base_means['overall']:.3f} "
        f"v53={means['overall']:.3f} "
        f"delta={deltas['overall']:+.3f}"
    )


if __name__ == "__main__":
    main()
