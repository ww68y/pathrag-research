import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.build_vsr_pathrag_v53_selective_replay import (
    METRICS,
    keyed,
    load_jsonl,
    metric_means,
    write_jsonl,
)


def main() -> None:
    out = ROOT_DIR / "results"
    base_results_path = out / "vsr_pathrag_v56_selective_replay_30.jsonl"
    case_path = out / "vsr_pathrag_v57_fertilizer_focus_case.jsonl"
    audit_path = out / "vsr_pathrag_v57_fertilizer_focus_audit.json"
    base_eval_path = out / "vsr_pathrag_v56_selective_replay_30_eval.jsonl"
    case_eval_path = out / "vsr_pathrag_v57_fertilizer_focus_case_eval.jsonl"
    result_path = out / "vsr_pathrag_v57_selective_replay_30.jsonl"
    eval_path = out / "vsr_pathrag_v57_selective_replay_30_eval.jsonl"
    manifest_path = out / "vsr_pathrag_v57_selective_replay_manifest.json"
    summary_path = out / "vsr_pathrag_v57_selective_replay_30_eval_summary.md"

    base = keyed(load_jsonl(base_results_path))
    changed = keyed(load_jsonl(case_path))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    triggers = {int(x) for x in audit["triggered_line_nos"]}
    if set(base) != set(range(30)) or set(changed) != triggers:
        raise ValueError("v57 selective replay inputs are inconsistent")
    rows = []
    for line_no in range(30):
        row = dict(changed.get(line_no, base[line_no]))
        row["promotion_replay"] = {
            "triggered": line_no in triggers,
            "source_agent_version": (
                "vsr-pathrag-v57"
                if line_no in triggers
                else "vsr-pathrag-v56"
            ),
        }
        rows.append(row)
    write_jsonl(result_path, rows)

    base_eval = keyed(load_jsonl(base_eval_path))
    changed_eval = keyed(load_jsonl(case_eval_path))
    evaluations = [
        changed_eval.get(line_no, base_eval[line_no])
        for line_no in range(30)
    ]
    write_jsonl(eval_path, evaluations)
    means = metric_means(evaluations)
    base_means = metric_means(base_eval)
    deltas = {
        metric: round(means[metric] - base_means[metric], 3)
        for metric in METRICS
    }
    manifest = {
        "protocol": "deterministic_gate_selective_replay",
        "agent_version": "vsr-pathrag-v57",
        "sample_count": 30,
        "audited_trigger_count": len(triggers),
        "actual_v57_rerun_count": len(changed),
        "reused_v56_nontrigger_count": 30 - len(triggers),
        "triggered_line_nos": sorted(triggers),
        "means": means,
        "v56_means": base_means,
        "mean_deltas": deltas,
        "verdict_counts": dict(
            Counter(row["scores"]["verdict"] for row in evaluations)
        ),
        "runtime_reference_answer_used": False,
        "context_change": (
            "No context or graph changed. The reference-free main-effect "
            "focus gate triggers only line 26."
        ),
        "full_stochastic_rerun": False,
        "ablation": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# VSR-PathRAG v57 Selective-Replay Evaluation",
        "",
        "| Metric | v56 | v57 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {base_means[metric]:.3f} | "
            f"{means[metric]:.3f} | {deltas[metric]:+.3f} |"
        )
    lines += [
        "",
        "- Triggered lines: [26]",
        "- Runtime reference answers: not used",
        "- Full stochastic rerun: false",
        "- Formal ablation: false",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"samples=30 triggers={len(triggers)} "
        f"v56={base_means['overall']:.3f} "
        f"v57={means['overall']:.3f}"
    )


if __name__ == "__main__":
    main()
