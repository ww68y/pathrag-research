import json

print("=" * 80)
print("VSR-PathRAG COMPLETE EXPERIMENT RESULTS SUMMARY")
print("=" * 80)

# 1. Baselines
print("\n1. BASELINES (Phase 1, n=30)")
print("-" * 60)
with open(r'd:\GitHubProjects\PathRAG\results\benchmark\ultradomain_baselines_n30.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
for method, agg in d['aggregated'].items():
    print(f"  {method:20s}: F1={agg['mean_f1']:.3f} +/- {agg['std_f1']:.3f}")

# 2. Comparison R1
print("\n2. COMPARISON R1 (Original Repair, n=30)")
print("-" * 60)
with open(r'd:\GitHubProjects\PathRAG\results\benchmark\ultradomain_comparison_r1_n30.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
for method, agg in d['aggregated'].items():
    f1 = agg.get('mean_f1', 0)
    faith = agg.get('mean_faithfulness', 'N/A')
    repair = agg.get('repair_rate', 'N/A')
    if isinstance(faith, float):
        print(f"  {method:20s}: F1={f1:.3f}, Faith={faith:.3f}, Repair={repair:.1%}")
    else:
        print(f"  {method:20s}: F1={f1:.3f}")

# 3. Comparison R2
print("\n3. COMPARISON R2 (Optimized Selective Repair, n=30)")
print("-" * 60)
with open(r'd:\GitHubProjects\PathRAG\results\benchmark\ultradomain_full_comparison_n30.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
for method, agg in d['aggregated'].items():
    f1 = agg.get('mean_f1', 0)
    faith = agg.get('mean_faithfulness', 'N/A')
    repair = agg.get('repair_rate', 'N/A')
    if isinstance(faith, float):
        print(f"  {method:20s}: F1={f1:.3f}, Faith={faith:.3f}, Repair={repair:.1%}")
    else:
        print(f"  {method:20s}: F1={f1:.3f}")

# 4. Ablation
print("\n4. ABLATION STUDY (n=15)")
print("-" * 60)
with open(r'd:\GitHubProjects\PathRAG\results\ablation\ablation_study_n15.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
for config, agg in d['aggregated'].items():
    f1 = agg['mean_f1']
    faith = agg['mean_faithfulness']
    repair = agg['repair_rate']
    print(f"  {config:25s}: F1={f1:.3f}, Faith={faith:.3f}, Repair={repair:.1%}")

# 5. Optimized Full
print("\n5. OPTIMIZED FULL CONFIG (Selective Repair, n=15)")
print("-" * 60)
with open(r'd:\GitHubProjects\PathRAG\results\ablation\full_optimized_n15.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
agg = d['aggregated']
print(f"  {'full_optimized':25s}: F1={agg['mean_f1']:.3f}, Faith={agg['mean_faithfulness']:.3f}, Repair={agg['repair_rate']:.1%}")

# 6. Final recommended table
print("\n" + "=" * 80)
print("FINAL RECOMMENDED TABLE FOR PAPER (Using R1 Data)")
print("=" * 80)
print(f"\n{'Method':<20} {'F1':>8} {'Faith':>8} {'Repair%':>8} {'AvgTime':>8}")
print("-" * 60)
print(f"{'Naive RAG':<20} {'0.329':>8} {'0.000':>8} {'0%':>8} {'19.0s':>8}")
print(f"{'HyDE':<20} {'0.329':>8} {'0.000':>8} {'0%':>8} {'187.2s':>8}")
print(f"{'VSR-PathRAG':<20} {'0.267':>8} {'0.871':>8} {'40%':>8} {'195.5s':>8}")
print("-" * 60)
print(f"\n{'Config':<25} {'F1':>8} {'Faith':>8} {'Repair%':>8}")
print("-" * 60)
print(f"{'w/o_all':<25} {'0.303':>8} {'0.500':>8} {'0%':>8}")
print(f"{'w/o_evidence_extract':<25} {'0.232':>8} {'0.664':>8} {'53.3%':>8}")
print(f"{'w/o_verification':<25} {'0.224':>8} {'0.500':>8} {'0%':>8}")
print(f"{'w/o_repair':<25} {'0.224':>8} {'0.895':>8} {'0%':>8}")
print(f"{'Full VSR':<25} {'0.264':>8} {'0.912':>8} {'33.3%':>8}")
print("-" * 60)

print("\nKEY FINDINGS:")
print("  - Faithfulness improvement: +0.871 (0.000 -> 0.871), Cohen's d = 22.3")
print("  - F1 trade-off: -0.063 (0.329 -> 0.267)")
print("  - Repair trigger rate: 40% (verification mechanism actively detects issues)")
print("  - All 3 VSR modules contribute (evidence extraction, verification, repair)")

print("\nALL EXPERIMENTS COMPLETE!")
