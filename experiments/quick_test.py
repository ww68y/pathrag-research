"""Quick test to verify dataset loading."""
import json

lines = open('datasets/UltraDomain/mix.jsonl', 'r', encoding='utf-8').readlines()
print(f"Total questions: {len(lines)}")

d = json.loads(lines[0])
print(f"Keys: {list(d.keys())}")
q = d.get("input", "")
print(f"First question: {q[:80]}")

# Check domains
domains = set()
for line in lines:
    item = json.loads(line)
    domains.add(item.get("dataset", item.get("label", "")))
print(f"Domains: {sorted(domains)}")
