"""Dump all questions from held-out sets, classified by family."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = ROOT / "datasets" / "UltraDomain"
FILES = [
    ("v46_zero_overlap_validation_30.jsonl", "v46"),
    ("crossdomain_multiquery_heldout_12.jsonl", "cross12"),
    ("vsr_v55_order_preserving_validation_30.jsonl", "vsr_v55"),
]

from api.features.chats.obligation_type_compiler import (
    classify_question_family,
)


def load_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main():
    for fname, tag in FILES:
        print(f"=== {tag} ({fname}) ===")
        rows = load_jsonl(BASE / fname)
        for i, row in enumerate(rows):
            q = row.get("input", "")
            family = classify_question_family(q)
            print(f"  Q{i+1:02d} [{family}]: {q[:80]}")
        from collections import Counter
        cnt = Counter(
            classify_question_family(row.get("input", ""))
            for row in rows
        )
        print(f"  -> {dict(cnt)}")
        print()


if __name__ == "__main__":
    main()
