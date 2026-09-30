import sys
sys.path.insert(0, r'd:\GitHubProjects\PathRAG')
from experiments.run_enhanced_experiments import load_ultradomain_full, METHODS, compute_f1, compute_rouge_l, compute_exact_match

print("Import OK")
qs = load_ultradomain_full()
print(f"Methods: {list(METHODS.keys())}")
print(f"Total questions: {len(qs)}")

# Test metrics
pred = "The Witch of Atlas is a poem by Shelley that lacks human interest"
gold = "The poem lacks human interest and does not tell a story"
print(f"F1: {compute_f1(pred, gold):.3f}")
print(f"ROUGE-L: {compute_rouge_l(pred, gold):.3f}")
print(f"EM: {compute_exact_match(pred, gold):.0f}")

# Domain distribution
from collections import Counter
domains = Counter(q.get("domain", "") for q in qs)
print(f"Domain distribution: {dict(domains)}")
