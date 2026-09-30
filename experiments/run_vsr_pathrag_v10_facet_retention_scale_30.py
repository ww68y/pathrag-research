import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNNER = ROOT_DIR / "experiments" / "run_vsr_pathrag_v7_answer_realization_scale_30.py"


if __name__ == "__main__":
    sys.argv = [
        str(RUNNER),
        "--output",
        "results/historical_vsr/vsr_pathrag_v10_facet_retention_changed.jsonl",
        "--version",
        "vsr-pathrag-v10-facet-retention",
        "--line-nos",
        "0",
        "--concurrency",
        "1",
        *sys.argv[1:],
    ]
    runpy.run_path(str(RUNNER), run_name="__main__")
