"""Check experiment status: what results exist, what's still running, what needs to be done."""
import os
import json
import glob

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base, "results")

print("=" * 70)
print("EXPERIMENT STATUS CHECK")
print("=" * 70)

# Check all result directories
for subdir in ['enhanced', 'ablation', 'benchmark']:
    d = os.path.join(results_dir, subdir)
    if not os.path.exists(d):
        print(f"\n[{subdir}] Directory does not exist")
        continue
    
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print(f"\n[{subdir}] No JSON result files found")
        continue
    
    print(f"\n[{subdir}] {len(files)} result file(s):")
    for f in files:
        fname = os.path.basename(f)
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            
            # Try to extract summary info
            if isinstance(data, dict):
                method = data.get('method', data.get('config', ''))
                n_q = data.get('n_questions', data.get('total_questions', '?'))
                n_done = data.get('n_completed', data.get('completed', '?'))
                # Check if it has per-question results
                results = data.get('results', data.get('questions', []))
                if isinstance(results, list):
                    n_actual = len(results)
                else:
                    n_actual = '?'
                
                # Extract metrics if available
                metrics = data.get('metrics', data.get('aggregate_metrics', {}))
                if isinstance(metrics, dict):
                    f1 = metrics.get('f1', metrics.get('avg_f1', '?'))
                    em = metrics.get('exact_match', metrics.get('avg_em', '?'))
                    faith = metrics.get('faithfulness', metrics.get('avg_faithfulness', '?'))
                else:
                    f1 = em = faith = '?'
                
                print(f"  - {fname}: method={method}, n_q={n_q}, n_done={n_done}, actual_results={n_actual}, F1={f1}, EM={em}, Faith={faith}")
            elif isinstance(data, list):
                print(f"  - {fname}: list with {len(data)} entries")
        except Exception as e:
            print(f"  - {fname}: ERROR reading: {e}")

# Check for running processes (look for lock files or temp files)
print("\n" + "=" * 70)
print("RUNNING PROCESSES / TEMP FILES")
print("=" * 70)

# Check for log files
log_dir = os.path.join(base, "logs")
if os.path.exists(log_dir):
    logs = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    for l in logs[-5:]:  # last 5 logs
        fname = os.path.basename(l)
        mtime = os.path.getmtime(l)
        import time
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        size = os.path.getsize(l)
        # Read last few lines
        try:
            with open(l, 'r', encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()
            last_line = lines[-1].strip() if lines else "(empty)"
            print(f"  {fname} ({size}B, modified {mtime_str}): last line: {last_line[:100]}")
        except:
            print(f"  {fname} ({size}B, modified {mtime_str})")

# Check experiment scripts
print("\n" + "=" * 70)
print("EXPERIMENT SCRIPTS AVAILABLE")
print("=" * 70)

exp_dir = os.path.join(base, "experiments")
for f in sorted(os.listdir(exp_dir)):
    if f.endswith('.py'):
        fpath = os.path.join(exp_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size}B)")

# Check what's still needed
print("\n" + "=" * 70)
print("WHAT'S NEEDED FOR SCI-TIER COMPLETION")
print("=" * 70)

# Enhanced comparison: need all 6 methods
enhanced_dir = os.path.join(results_dir, "enhanced")
enhanced_files = []
if os.path.exists(enhanced_dir):
    enhanced_files = [f for f in os.listdir(enhanced_dir) if f.endswith('.json')]
print(f"Enhanced comparison: {len(enhanced_files)} files found")
print(f"  Files: {enhanced_files}")

# Need: naive_rag, hyde, graphrag_sim, self_rag_sim, vsr_pathrag, pathrag_baseline
expected_methods = ['naive_rag', 'hyde', 'graphrag_sim', 'self_rag_sim', 'vsr_pathrag', 'pathrag_baseline']
for m in expected_methods:
    found = any(m in f for f in enhanced_files)
    print(f"  {m}: {'DONE' if found else 'MISSING'}")

# Enhanced ablation
ablation_dir = os.path.join(results_dir, "ablation")
ablation_files = []
if os.path.exists(ablation_dir):
    ablation_files = [f for f in os.listdir(ablation_dir) if f.endswith('.json')]
print(f"\nEnhanced ablation: {len(ablation_files)} files found")
print(f"  Files: {ablation_files}")
