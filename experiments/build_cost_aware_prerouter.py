import argparse
import json
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_map(path: Path) -> dict[int, dict]:
    return {row.get("line_no"): row for row in load_jsonl(path)}


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def choose_route(row: dict) -> tuple[str, list[str]]:
    error_router = row.get("legal_error_router") or {}
    error_action = error_router.get("action", "")
    if error_action == "use_naive":
        return "naive", ["pre_route:missing_financial_specificity"]
    if error_action == "use_pathrag":
        return "pathrag", ["pre_route:indemnification_direction"]

    query_router = row.get("query_router") or {}
    if query_router.get("rescue_used"):
        reasons = ["pre_route:legal_lookup_to_hyde", "runtime_rescue:hyde_uncertain_to_pathrag"]
        if (row.get("rescue_compression") or {}).get("action") == "use_compressed":
            reasons.append("post_route:formula_aware_compression")
            return "hyde_pathrag_rescue_compress", reasons
        return "hyde_pathrag_rescue", reasons

    return "hyde", ["pre_route:legal_clause_lookup"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--pathrag", required=True)
    parser.add_argument("--naive", required=True)
    parser.add_argument("--hyde", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT_DIR / args.base)
    pathrag = row_map(ROOT_DIR / args.pathrag)
    naive = row_map(ROOT_DIR / args.naive)
    hyde = row_map(ROOT_DIR / args.hyde)

    output_rows = []
    reports = []
    selected_graph_contexts: dict[str, float] = {}
    all_graph_contexts: dict[str, float] = {}
    for source_row in pathrag.values():
        context_hash = source_row.get("context_hash")
        if context_hash:
            all_graph_contexts.setdefault(context_hash, as_float(source_row.get("insert_seconds")))

    for row in base_rows:
        line_no = row.get("line_no")
        route, reasons = choose_route(row)
        pathrag_row = pathrag.get(line_no, {})
        naive_row = naive.get(line_no, {})
        hyde_row = hyde.get(line_no, {})

        query_seconds = 0.0
        if route == "naive":
            query_seconds = as_float(naive_row.get("query_seconds"))
        elif route == "pathrag":
            query_seconds = as_float(pathrag_row.get("query_seconds"))
        elif route == "hyde":
            query_seconds = as_float(hyde_row.get("query_seconds"))
        elif route.startswith("hyde_pathrag_rescue"):
            query_seconds = as_float(hyde_row.get("query_seconds")) + as_float(pathrag_row.get("query_seconds"))
            if route.endswith("compress"):
                query_seconds += as_float((row.get("rescue_compression") or {}).get("query_seconds"))

        if route == "pathrag" or route.startswith("hyde_pathrag_rescue"):
            context_hash = pathrag_row.get("context_hash")
            if context_hash:
                selected_graph_contexts.setdefault(context_hash, as_float(pathrag_row.get("insert_seconds")))

        new_row = dict(row)
        new_row["cost_aware_prerouter"] = {
            "enabled": True,
            "route": route,
            "reasons": reasons,
            "estimated_query_seconds": round(query_seconds, 3),
            "requires_graph": route == "pathrag" or route.startswith("hyde_pathrag_rescue"),
        }
        output_rows.append(new_row)
        reports.append((line_no, new_row["cost_aware_prerouter"]))

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in output_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    route_counts = Counter(report["route"] for _, report in reports)
    routed_query_seconds = sum(report["estimated_query_seconds"] for _, report in reports)
    pathrag_query_seconds = sum(as_float(row.get("query_seconds")) for row in pathrag.values())
    naive_query_seconds = sum(as_float(row.get("query_seconds")) for row in naive.values())
    hyde_query_seconds = sum(as_float(row.get("query_seconds")) for row in hyde.values())
    compression_seconds = sum(
        as_float((row.get("rescue_compression") or {}).get("query_seconds"))
        for row in base_rows
        if (row.get("rescue_compression") or {}).get("action") == "use_compressed"
    )
    run_all_query_seconds = pathrag_query_seconds + naive_query_seconds + hyde_query_seconds + compression_seconds
    selected_index_seconds = sum(selected_graph_contexts.values())
    full_index_seconds = sum(all_graph_contexts.values())

    query_reduction = 1 - routed_query_seconds / max(1.0, run_all_query_seconds)
    pathrag_query_reduction = 1 - routed_query_seconds / max(1.0, pathrag_query_seconds)
    index_reduction = 1 - selected_index_seconds / max(1.0, full_index_seconds)

    lines = [
        "# Cost-Aware Pre-Router Summary\n\n",
        f"- Base answers: `{args.base}`\n",
        f"- Output: `{args.output}`\n",
        f"- Samples: {len(output_rows)}\n",
        f"- Selected graph contexts: {len(selected_graph_contexts)} / {len(all_graph_contexts)}\n\n",
        "## Route Counts\n\n",
        "| Route | Count |\n",
        "|---|---:|\n",
    ]
    for route, count in sorted(route_counts.items()):
        lines.append(f"| {route} | {count} |\n")

    lines.extend([
        "\n## Estimated Runtime\n\n",
        "| Metric | Seconds | Reduction |\n",
        "|---|---:|---:|\n",
        f"| Run-all candidates query time | {run_all_query_seconds:.3f} | baseline |\n",
        f"| Full PathRAG query time | {pathrag_query_seconds:.3f} | baseline |\n",
        f"| Cost-aware routed query time | {routed_query_seconds:.3f} | {query_reduction * 100:.2f}% vs run-all |\n",
        f"| Cost-aware routed query time vs full PathRAG | {routed_query_seconds:.3f} | {pathrag_query_reduction * 100:.2f}% |\n",
        f"| Full PathRAG graph indexing | {full_index_seconds:.3f} | baseline |\n",
        f"| Selected-context graph indexing | {selected_index_seconds:.3f} | {index_reduction * 100:.2f}% |\n",
        "\n## Per-Sample Plan\n\n",
        "| Line | Route | Estimated query seconds | Requires graph | Reasons |\n",
        "|---:|---|---:|---|---|\n",
    ])
    for line_no, report in reports:
        lines.append(
            f"| {line_no} | {report.get('route')} | {report.get('estimated_query_seconds')} | "
            f"{report.get('requires_graph')} | {', '.join(report.get('reasons', []))} |\n"
        )

    (ROOT_DIR / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Saved pre-routed results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
