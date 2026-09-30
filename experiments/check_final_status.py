"""Check all experiment results."""
import os, json

base = r"d:\GitHubProjects\PathRAG"

print("=" * 70)
print("COMPREHENSIVE EXPERIMENT STATUS")
print("=" * 70)

# Enhanced comparison
print("\n[1] Enhanced Comparison Results:")
enh_dir = os.path.join(base, "results", "enhanced")
for f in sorted(os.listdir(enh_dir)):
    if f.endswith(".json"):
        fpath = os.path.join(enh_dir, f)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                n = len(data.get("results", []))
                agg = data.get("aggregated", {})
                f1 = agg.get("mean_f1", "?")
                rl = agg.get("mean_rouge_l", "?")
                em = agg.get("mean_em", "?")
                faith = agg.get("mean_faithfulness", "?")
                repair = agg.get("repair_rate", "?")
                t = agg.get("mean_runtime_seconds", "?")
                method = data.get("method", data.get("ablation", data.get("experiment", "?")))
                if n > 0:
                    print(f"  {f}: method={method}, n={n}, F1={f1:.4f}, RL={rl:.4f}, EM={em}, Faith={faith:.4f}, Repair={repair}, Time={t:.1f}s" if isinstance(f1, float) else f"  {f}: method={method}, n={n}")
                else:
                    print(f"  {f}: (summary file, no per-question results)")
        except Exception as e:
            print(f"  {f}: ERROR {e}")

# Enhanced ablation
print("\n[2] Enhanced Ablation Results:")
abl_dir = os.path.join(base, "results", "enhanced_ablation")
if os.path.exists(abl_dir):
    files = [f for f in os.listdir(abl_dir) if f.endswith(".json")]
    if files:
        for f in sorted(files):
            fpath = os.path.join(abl_dir, f)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    n = len(data.get("results", []))
                    agg = data.get("aggregated", {})
                    f1 = agg.get("mean_f1", "?")
                    print(f"  {f}: n={n}, F1={f1}")
            except Exception as e:
                print(f"  {f}: ERROR {e}")
    else:
        print("  (no files yet - ablation not started or still running)")
else:
    print("  DIR NOT FOUND")

# Check log tail
print("\n[3] Last 10 lines of experiment log:")
logfile = os.path.join(base, "logs", "remaining_experiments.log")
if os.path.exists(logfile):
    try:
        with open(logfile, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"  Total log lines: {len(lines)}")
        for line in lines[-10:]:
            print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")
else:
    print("  LOG NOT FOUND")
