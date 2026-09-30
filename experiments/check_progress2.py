"""Quick status check - no complex syntax."""
import os, json

base = r"d:\GitHubProjects\PathRAG"

print("=== Enhanced Comparison Results ===")
enh_dir = os.path.join(base, "results", "enhanced")
if os.path.exists(enh_dir):
    for f in sorted(os.listdir(enh_dir)):
        if f.endswith(".json"):
            fpath = os.path.join(enh_dir, f)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                n = len(data.get("results", []))
                agg = data.get("aggregated", {})
                f1 = agg.get("mean_f1", "?")
                method = data.get("method", data.get("ablation", "?"))
                print(f"  {f}: method={method}, n_results={n}, F1={f1}")
            except Exception as e:
                print(f"  {f}: ERROR {e}")
else:
    print("  DIR NOT FOUND")

print("\n=== Enhanced Ablation Results ===")
abl_dir = os.path.join(base, "results", "enhanced_ablation")
if os.path.exists(abl_dir):
    for f in sorted(os.listdir(abl_dir)):
        if f.endswith(".json"):
            fpath = os.path.join(abl_dir, f)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                n = len(data.get("results", []))
                agg = data.get("aggregated", {})
                f1 = agg.get("mean_f1", "?")
                print(f"  {f}: n_results={n}, F1={f1}")
            except Exception as e:
                print(f"  {f}: ERROR {e}")
else:
    print("  DIR NOT FOUND")

print("\n=== Benchmark Results ===")
bench_dir = os.path.join(base, "results", "benchmark")
if os.path.exists(bench_dir):
    for f in sorted(os.listdir(bench_dir)):
        if f.endswith(".json"):
            fpath = os.path.join(bench_dir, f)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    n = len(data.get("results", []))
                    agg = data.get("aggregated", {})
                    f1 = agg.get("mean_f1", "?")
                    print(f"  {f}: n_results={n}, F1={f1}")
                elif isinstance(data, list):
                    print(f"  {f}: list with {len(data)} entries")
            except Exception as e:
                print(f"  {f}: ERROR {e}")
else:
    print("  DIR NOT FOUND")

print("\n=== Old Ablation Results ===")
old_abl = os.path.join(base, "results", "ablation")
if os.path.exists(old_abl):
    for f in sorted(os.listdir(old_abl)):
        if f.endswith(".json"):
            fpath = os.path.join(old_abl, f)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    n = len(data.get("results", data.get("ablation_results", [])))
                    print(f"  {f}: n={n}")
                elif isinstance(data, list):
                    print(f"  {f}: list with {len(data)} entries")
            except Exception as e:
                print(f"  {f}: ERROR {e}")
else:
    print("  DIR NOT FOUND")
