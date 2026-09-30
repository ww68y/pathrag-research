import json

# Check comparison experiment (R2 - current JSON)
with open(r'd:\GitHubProjects\PathRAG\results\benchmark\ultradomain_full_comparison_n30.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print("=== Comparison Experiment (R2) ===")
for method, agg in d['aggregated'].items():
    f1 = agg.get('mean_f1', 0)
    faith = agg.get('mean_faithfulness', 'N/A')
    repair = agg.get('repair_rate', 'N/A')
    print(f"  {method}: F1={f1:.3f}, Faith={faith}, Repair={repair}")

# Check ablation experiment
with open(r'd:\GitHubProjects\PathRAG\results\ablation\ablation_study_n15.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print("\n=== Ablation Study ===")
for config, agg in d['aggregated'].items():
    f1 = agg.get('mean_f1', 0)
    faith = agg.get('mean_faithfulness', 0)
    repair = agg.get('repair_rate', 0)
    print(f"  {config}: F1={f1:.3f}, Faith={faith:.3f}, Repair={repair:.1%}")

# Check optimized full
with open(r'd:\GitHubProjects\PathRAG\results\ablation\full_optimized_n15.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print("\n=== Optimized Full Config ===")
agg = d['aggregated']
print(f"  F1={agg['mean_f1']:.3f}, Faith={agg['mean_faithfulness']:.3f}, Repair={agg['repair_rate']:.1%}")

# Check baselines
with open(r'd:\GitHubProjects\PathRAG\results\benchmark\ultradomain_baselines_n30.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print("\n=== Baselines (Phase 1) ===")
for method, agg in d['aggregated'].items():
    f1 = agg.get('mean_f1', 0)
    print(f"  {method}: F1={f1:.3f}")

# Check VSR R1 from log
print("\n=== VSR R1 (from log - vsr_ctx_out2.txt) ===")
import re
with open(r'd:\GitHubProjects\PathRAG\experiments\vsr_ctx_out2.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    # Extract aggregated stats
    match = re.search(r'mean_f1:\s*([\d.]+)', content)
    if match:
        print(f"  mean_f1: {match.group(1)}")
    match = re.search(r'mean_faithfulness:\s*([\d.]+)', content)
    if match:
        print(f"  mean_faithfulness: {match.group(1)}")
    match = re.search(r'repair_rate:\s*([\d.]+)', content)
    if match:
        print(f"  repair_rate: {match.group(1)}")

print("\n=== ALL EXPERIMENTS STATUS ===")
print("1. Naive RAG baseline:      DONE (n=30, F1=0.329)")
print("2. HyDE baseline:            DONE (n=30, F1=0.329)")
print("3. VSR-PathRAG R1 (original): DONE (n=30, F1=0.267, Faith=0.871)")
print("4. VSR-PathRAG R2 (optimized):DONE (n=30, F1=0.255, Faith=0.807)")
print("5. Ablation w/o_all:          DONE (n=15, F1=0.303)")
print("6. Ablation w/o_evidence:     DONE (n=15, F1=0.232)")
print("7. Ablation w/o_verification: DONE (n=15, F1=0.224)")
print("8. Ablation w/o_repair:       DONE (n=15, F1=0.224)")
print("9. Ablation full:             DONE (n=15, F1=0.264, Faith=0.912)")
print("10. Full optimized:           DONE (n=15, F1=0.228, Faith=0.926)")
