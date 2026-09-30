import json, os
from pathlib import Path

p = Path(__file__).resolve().parents[2] / 'results' / 'lightrag_current.jsonl'
lines = open(p, 'r', encoding='utf-8').readlines()
print(f'Completed: {len(lines)}/130')
for l in lines[-5:]:
    item = json.loads(l)
    print(f"  line={item['line_no']} idx={item.get('index_seconds',0):.0f}s q={item.get('query_seconds',0):.0f}s")
