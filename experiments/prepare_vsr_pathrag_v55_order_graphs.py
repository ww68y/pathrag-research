import asyncio
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
import experiments.prepare_vsr_pathrag_anchor_preserving_graphs as base


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--dataset",
        (
            "datasets/UltraDomain/"
            "vsr_v55_order_preserving_validation_30.jsonl"
        ),
        "--base-dataset",
        (
            "datasets/UltraDomain/"
            "vsr_v52_anchor_preserving_validation_30.jsonl"
        ),
        "--output",
        "results/historical_vsr/vsr_pathrag_v55_order_graph_preparation.jsonl",
        "--added-indices-field",
        "order_preserving_added_chunk_indices",
        "--staging-prefix",
        "v55",
    ]
    asyncio.run(base.main())
