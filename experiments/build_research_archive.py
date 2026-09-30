"""Inventory historical experiment artifacts without changing or rerunning them."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_archive" / "20260920"


def inspect_file(path: Path) -> dict:
    raw = path.read_bytes()
    record = {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if path.suffix == ".md":
        text = raw.decode("utf-8-sig", errors="replace")
        headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.M)
        record.update(title=headings[0] if headings else path.stem,
                      headings=headings, lines=len(text.splitlines()))
        # Retain original conclusion sections for later source inspection.
        sections = re.split(r"(?m)(?=^#{1,3} )", text)
        record["conclusion_sections"] = [
            section for section in sections
            if re.search(r"conclusion|interpretation|takeaway|caveat|limitation|结论|局限|总结|注意",
                         section.splitlines()[0] if section.splitlines() else "", re.I)
        ]
    elif path.suffix in {".json", ".jsonl"}:
        try:
            if path.suffix == ".json":
                value = json.loads(raw.decode("utf-8-sig"))
                record["container_type"] = type(value).__name__
                record["top_level_count"] = len(value) if isinstance(value, (dict, list)) else None
            else:
                rows = 0
                errors = []
                for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        continue
                    rows += 1
                    try:
                        json.loads(line)
                    except ValueError:
                        errors.append(number)
                record["raw_nonempty_rows"] = rows
                record["invalid_json_lines"] = errors
            record["parse_ok"] = not record.get("invalid_json_lines")
        except (UnicodeError, ValueError) as exc:
            record["parse_ok"] = False
            record["parse_error_type"] = type(exc).__name__
    return record


def template_structure(path: Path) -> dict:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        tree = ElementTree.fromstring(archive.read("word/document.xml"))
        paragraphs = ["".join(p.itertext()) for p in tree.findall(".//w:t", ns)]
        text_nodes = "\n".join(paragraphs)
        headings = [term for term in ("发明名称", "技术领域", "背景技术", "发明目的",
                                     "技术解决方案", "技术效果", "附图说明", "具体实施方式")
                    if term in text_nodes]
        sections = [dict(node.attrib) for node in tree.findall(".//w:sectPr/w:pgSz", ns)]
        return {"path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "section_page_sizes": sections, "section_headings_present": headings,
                "tables": len(tree.findall(".//w:tbl", ns)),
                "images": len([name for name in archive.namelist() if name.startswith("word/media/")])}


def main() -> None:
    paths = set()
    for folder in ("results", "paper", "experiments"):
        paths.update((ROOT / folder).rglob("*.md"))
    paths.update((ROOT / "docs" / "research_notes").glob("*.md"))
    paths.update((ROOT / "results").rglob("*.json"))
    paths.update((ROOT / "results").rglob("*.jsonl"))
    records = [inspect_file(path) for path in sorted(paths)]
    counts = Counter(Path(row["path"]).suffix for row in records)
    reports = [row for row in records if row["path"].endswith(".md")]
    failures = [row["path"] for row in records if row.get("parse_ok") is False]
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Full text scan of report corpus; structural JSON audit, not a rerun or per-answer semantic audit",
               "counts": dict(counts), "parse_failures": failures,
               "templates": [
                   template_structure(p)
                   for p in sorted((ROOT / "paper" / "patent_materials").glob("*.docx"))
               ],
               "files": records}
    (OUT / "experiment_source_inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    groups = defaultdict(list)
    for record in reports:
        path = Path(record["path"])
        group = path.parts[1] if path.parts[0] == "results" and len(path.parts) > 2 else path.parts[0] if len(path.parts) > 1 else "root"
        groups[group].append(record)
    lines = ["# 实验资料全量索引", "", "归档日期：2026-09-20。", "",
             f"共扫描 {len(reports)} 份 Markdown，{counts['.json']} 个 JSON，{counts['.jsonl']} 个 JSONL。",
             "扫描包括报告全文、标题、结论段、文件哈希和 JSON 语法；不等于逐条人工复核所有模型答案。",
             "同一实验可能有多版报告、快照和缓存，文件数与原始行数均不等于独立实验数或独立样本数。",
             "按时间和证据口径整合后的结论见 [全过程总结](EXPERIMENT_HISTORY_AND_SYNTHESIS.md)。",
             "完整 SHA-256、结论原文及结构检查记录见 `experiment_source_inventory.json`。", "",
             f"JSON/JSONL 语法异常文件数：{len(failures)}。", ""]
    for failure in failures:
        lines.append(f"- `{failure}`")
    for group, items in sorted(groups.items()):
        lines.extend(["", f"## {group} ({len(items)})", "", "| 报告 | 行数 | SHA-256 前16位 |", "|---|---:|---|"])
        for row in items:
            label = row["path"].replace("|", "\\|")
            target = quote("../../" + row["path"], safe="/._-")
            lines.append(f"| [{label}]({target}) | {row['lines']} | `{row['sha256'][:16]}` |")
    (OUT / "EXPERIMENT_SOURCE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "parse_failures": failures,
                      "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
