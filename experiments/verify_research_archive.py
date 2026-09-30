"""Check archival claims against stored outputs; never call a model endpoint."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from evaluate_impact_aware_validation import load_jsonl, metric_for
from summarize_risk_only_pooled_evidence import PRIMARY, bootstrap_mean
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_archive" / "20260920"


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, abs_tol=1e-10):
        raise ValueError(f"Metric mismatch: {actual} != {expected}")


def main() -> None:
    pooled = json.loads((ROOT / "results/risk_only_stability/RISK_ONLY_POOLED_EVIDENCE.json").read_text(encoding="utf-8"))
    checked = []
    all_deltas = []
    for item in pooled["primary_holdouts"]:
        paths = PRIMARY[item["holdout"]]
        base = load_jsonl(paths["base"])
        final = {int(row["line_no"]): row for row in load_jsonl(paths["policy"])}
        if len(base) != len(final) or len(base) != item["n"]:
            raise ValueError("Holdout row count mismatch")
        delta = [metric_for(row, final[int(row["line_no"])]["prediction"])[0]
                 - metric_for(row, row["prediction"])[0] for row in base]
        close(statistics.mean(delta), item["mean_f1_delta"])
        all_deltas.extend(delta)
        checked.append(f"{item['holdout']}: N={len(delta)}, F1差={statistics.mean(delta):+.6f}")
    mean, low, high = bootstrap_mean(np.asarray(all_deltas))
    close(mean, pooled["pooled"]["mean_f1_delta"])
    close(low, pooled["pooled"]["f1_delta_ci95"][0])
    close(high, pooled["pooled"]["f1_delta_ci95"][1])
    checked.append(f"N83配对bootstrap复算：{mean:+.6f}, 95% CI [{low:+.6f}, {high:+.6f}]")

    legal_dir = ROOT / "results/risk_guard_validation"
    legal = json.loads((legal_dir / "RISK_GUARD_LEGAL20_RESULTS.json").read_text(encoding="utf-8"))
    base = load_jsonl(legal_dir / "legal20_direct_context.jsonl")
    final = {int(row["line_no"]): row for row in load_jsonl(legal_dir / "legal20_risk_only.jsonl")}
    close(statistics.mean(metric_for(row, row["prediction"])[0] for row in base), legal["base_f1"])
    close(statistics.mean(metric_for(row, final[int(row["line_no"])]["prediction"])[0] for row in base), legal["final_f1"])
    checked.append(f"新Legal20原始预测复算：{legal['base_f1']:.6f} -> {legal['final_f1']:.6f}")

    meta = json.loads((ROOT / "results/sage_locked_unseen_meta/sage_locked_unseen_meta_summary.json").read_text(encoding="utf-8"))
    total = sum(row["expected_n"] for row in meta["repeats"])
    if total != 326:
        raise ValueError("Locked meta sample count mismatch")
    for method, aggregate in meta["weighted_methods_all_repeats"].items():
        weighted = sum(row["expected_n"] * row["method_aggregates"][method]["mean_f1"] for row in meta["repeats"]) / total
        close(weighted, aggregate["mean_f1"])
        checked.append(f"326题分批摘要加权核对 {method}: F1={weighted:.6f}")

    windows = []
    window_paths = sorted((ROOT / "datasets/UltraDomain").glob("sage_locked_*windowed.jsonl"))
    window_paths += [ROOT / "datasets/UltraDomain" / name for name in (
        "impact_aware_locked_new48_windowed.jsonl", "impact_aware_v2_locked_new35_windowed.jsonl",
        "risk_guard_locked_legal20_windowed.jsonl")]
    for path in window_paths:
        rows = load_jsonl(path)
        counts = Counter(row.get("window_reason", "missing") for row in rows)
        windows.append({"dataset": path.name, "n": len(rows), "window_reasons": dict(counts)})

    light = load_jsonl(ROOT / "results/historical_baselines/lightrag_current.jsonl")
    latest = {int(row["line_no"]): row for row in light}
    valid = sum(bool(row.get("prediction")) and not row.get("error") for row in latest.values())
    if valid != 130:
        raise ValueError(f"Unexpected latest LightRAG coverage: {valid}")
    checked.append(f"LightRAG追加文件原始行{len(light)}，最新唯一有效预测{valid}")

    inventory = json.loads((OUT / "experiment_source_inventory.json").read_text(encoding="utf-8"))
    changed = []
    for item in inventory["files"] + inventory["templates"]:
        path = ROOT / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise ValueError(f"Source files changed: {changed}")
    checked.append(f"{len(inventory['files'])}份来源材料及{len(inventory['templates'])}份模板哈希保持一致")

    broken = []
    for path in OUT.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (path.parent / unquote(target.split("#")[0])).resolve()
            if not resolved.exists() and resolved != OUT / "ARCHIVE_VERIFICATION.md":
                broken.append(f"{path.name}: {target}")
    if broken:
        raise ValueError(f"Broken links: {broken}")
    checked.append("归档Markdown相对文件链接检查通过（不验证锚点）")
    patent_path = OUT / "PATENT_DISCLOSURE_DRAFT.md"
    if patent_path.exists():
        patent = patent_path.read_text(encoding="utf-8")
        claim_numbers = [int(value) for value in re.findall(r"(?m)^\*\*(\d+)\.\*\*", patent)]
        if claim_numbers != list(range(1, 13)):
            raise ValueError("Patent claim numbering mismatch")
        abstract = patent.split("## 十、说明书摘要", 1)[1].split("摘要附图", 1)[0].strip()
        if len(abstract) > 300:
            raise ValueError("Patent abstract exceeds 300 characters")
        if patent.count("```mermaid") != 3 or patent.count("```") % 2:
            raise ValueError("Patent figure or code fence mismatch")
        checked.append(f"专利草案：12项连续编号权利要求、3幅Mermaid源稿、摘要{len(abstract)}字符；未执行Word版式验收")
    payload = {"status": "PASS", "checks": checked, "window_audit": windows,
               "scope": "Existing-output recomputation and file checks, no model calls or new experiment runs"}
    (OUT / "archive_verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 归档核验记录", "", "核验日期：2026-09-20。状态：PASS。", "",
             "仅对既有产物离线复算与检查，没有新模型调用，也没有修改历史结果或两个专利模板。", "",
             "## 指标和文件检查", ""] + [f"- {check}" for check in checked]
    lines.extend(["", "## 窗口预处理审计", "", "下表按问题行计数，不是去重后的文档数。构建器存在参考答案匹配分支，但本次检查的实际记录均采用问题密度窗口，不能把分支存在写成已经发生答案辅助截窗。固定窗口评估仍不等于全语料端到端检索。", "",
                  "| 数据文件 | N | window_reason分布 |", "|---|---:|---|"])
    for item in windows:
        lines.append(f"| `{item['dataset']}` | {item['n']} | {item['window_reasons']} |")
    lines.extend(["", "## 核验边界", "",
                  "N83和新Legal20从原始预测复算词面F1；N326核对分批汇总加权，未逐题重做语义judge。语法与哈希正确不代表模型答案事实正确。未重新验证官方数据基准、模型服务版本、全部历史单元测试或全部来源去重逻辑。",
                  "本次交付为Markdown内容稿，不包含已经完成Word分页和附图版式验收的DOCX。"])
    (OUT / "ARCHIVE_VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
