import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--updates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="line_no")
    parser.add_argument("--line-nos", default=None)
    parser.add_argument("--append-missing", action="store_true")
    args = parser.parse_args()

    base = load_jsonl(ROOT_DIR / args.base)
    updates = load_jsonl(ROOT_DIR / args.updates)
    selected = None
    if args.line_nos:
        selected = {int(value.strip()) for value in args.line_nos.split(",") if value.strip()}
    update_by_key = {
        row.get(args.key): row
        for row in updates
        if selected is None or int(row.get("line_no", -1)) in selected
    }
    merged = [update_by_key.get(row.get(args.key), row) for row in base]
    if args.append_missing:
        base_keys = {row.get(args.key) for row in base}
        merged.extend(row for key, row in update_by_key.items() if key not in base_keys)
        merged.sort(key=lambda row: row.get(args.key))
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in merged:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Merged {len(update_by_key)} rows into {len(merged)} rows: {output_path}")


if __name__ == "__main__":
    main()
