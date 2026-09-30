import json

# Check UltraDomain dataset
f = open(r'd:\GitHubProjects\PathRAG\datasets\UltraDomain\mix.jsonl', 'r', encoding='utf-8')
lines = f.readlines()
f.close()
print(f"UltraDomain mix.jsonl: {len(lines)} questions")
d = json.loads(lines[0])
print(f"Fields: {list(d.keys())}")
print(f"Context length: {len(d.get('context',''))}")
print(f"Num answers: {len(d.get('answers',[]))}")
print(f"Q: {d.get('input','')[:100]}")
print(f"Gold answer: {d.get('answers',[''])[0][:100]}")

# Check per-domain datasets
import os
domain_dir = r'd:\GitHubProjects\PathRAG\datasets\UltraDomain'
for fname in ['agriculture.jsonl', 'biology.jsonl', 'history.jsonl', 'legal.jsonl', 'physics.jsonl']:
    fpath = os.path.join(domain_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            n = len(f.readlines())
        print(f"  {fname}: {n} questions")
