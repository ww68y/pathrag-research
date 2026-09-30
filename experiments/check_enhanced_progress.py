import json
import statistics

# Check Naive RAG results
with open(r'd:\GitHubProjects\PathRAG\results\enhanced\naive_rag_n130.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print("=== Naive RAG (n=130) ===")
agg = d['aggregated']
print(f"  F1: {agg['mean_f1']:.3f} +/- {agg['std_f1']:.3f}")
print(f"  ROUGE-L: {agg['mean_rouge_l']:.3f}")
print(f"  EM: {agg['mean_em']:.1%}")
print(f"  Faith: {agg['mean_faithfulness']:.3f}")
print(f"  Time: {agg['mean_runtime_seconds']:.1f}s")

# Check HyDE results
with open(r'd:\GitHubProjects\PathRAG\results\enhanced\hyde_n130.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print("\n=== HyDE (n=130) ===")
agg = d['aggregated']
print(f"  F1: {agg['mean_f1']:.3f} +/- {agg['std_f1']:.3f}")
print(f"  ROUGE-L: {agg['mean_rouge_l']:.3f}")
print(f"  EM: {agg['mean_em']:.1%}")
print(f"  Faith: {agg['mean_faithfulness']:.3f}")
print(f"  Time: {agg['mean_runtime_seconds']:.1f}s")

# Per-domain analysis for Naive RAG
print("\n=== Per-Domain F1 (Naive RAG) ===")
from collections import defaultdict
domain_scores = defaultdict(list)
for r in d['results']:
    domain = r.get('domain', 'unknown')
    domain_scores[domain].append(r['metrics']['f1'])
for domain, scores in sorted(domain_scores.items()):
    print(f"  {domain}: {statistics.mean(scores):.3f} (n={len(scores)})")
