import argparse
import json
from pathlib import Path

from evaluate_repro_results import load_items, write_summary


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--updates", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--line-nos", default=None)
    args = parser.parse_args()

    baseline = {item.get("line_no"): item for item in load_items(ROOT_DIR / args.baseline)}
    updates = {item.get("line_no"): item for item in load_items(ROOT_DIR / args.updates)}
    if args.line_nos:
        selected = {int(value.strip()) for value in args.line_nos.split(",") if value.strip()}
        updates = {line_no: item for line_no, item in updates.items() if line_no in selected}
    baseline.update(updates)
    evaluations = [baseline[line_no] for line_no in sorted(baseline)]

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for evaluation in evaluations:
            out.write(json.dumps(evaluation, ensure_ascii=False) + "\n")

    write_summary(evaluations, ROOT_DIR / args.summary, ROOT_DIR / args.source)
    print(f"Merged {len(updates)} updated evaluations into {len(evaluations)} rows")
    print(f"Saved merged evaluations to {output_path}")


if __name__ == "__main__":
    main()
