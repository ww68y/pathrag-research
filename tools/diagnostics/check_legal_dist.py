import json
from collections import Counter
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[2] / 'datasets' / 'UltraDomain'

lines = open(DATASETS_DIR / 'legal.jsonl', 'r', encoding='utf-8').readlines()
ctx_count = Counter()
ctx_lens = {}
for l in lines:
    item = json.loads(l)
    key = item.get('context', '')[:100]
    ctx_count[key] += 1
    if key not in ctx_lens:
        ctx_lens[key] = len(item.get('context', ''))

print(f'Legal: {len(lines)} samples, {len(ctx_count)} unique contexts')
print('\nTop 10 contexts by sample count:')
for i, (k, c) in enumerate(ctx_count.most_common(10)):
    print(f'  {i+1}. samples={c} ctx_len={ctx_lens[k]}')

# Also check biology and history
for fname in ['biology.jsonl', 'history.jsonl']:
    lines2 = open(DATASETS_DIR / fname, 'r', encoding='utf-8').readlines()
    ctx_count2 = Counter()
    ctx_lens2 = {}
    for l in lines2:
        item = json.loads(l)
        key = item.get('context', '')[:100]
        ctx_count2[key] += 1
        if key not in ctx_lens2:
            ctx_lens2[key] = len(item.get('context', ''))
    print(f'\n{fname}: {len(lines2)} samples, {len(ctx_count2)} unique contexts')
    print('Top 5 contexts by sample count:')
    for i, (k, c) in enumerate(ctx_count2.most_common(5)):
        print(f'  {i+1}. samples={c} ctx_len={ctx_lens2[k]}')
