import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNNER = ROOT_DIR / "experiments" / "run_vsr_pathrag_v7_answer_realization_scale_30.py"


if __name__ == "__main__":
    sys.argv = [
        str(RUNNER),
        "--output",
        "results/historical_vsr/vsr_pathrag_v9_counterfactual_realization_scale_30.jsonl",
        "--version",
        "vsr-pathrag-v9-counterfactual-realization",
        *sys.argv[1:],
    ]
    runpy.run_path(str(RUNNER), run_name="__main__")
