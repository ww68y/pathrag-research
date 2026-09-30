import json
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[2] / 'datasets' / 'UltraDomain'

for fname in ['legal.jsonl', 'history.jsonl', 'biology.jsonl']:
    fpath = DATASETS_DIR / fname
    lines = open(fpath, 'r', encoding='utf-8').readlines()
    ctx_lens = [len(json.loads(l).get('context', '')) for l in lines]
    ctx_lens.sort()
    print(f'\n=== {fname} ({len(lines)} samples) ===')
    print(f'  Context length: min={min(ctx_lens)} median={ctx_lens[len(ctx_lens)//2]} max={max(ctx_lens)}')
    under_50k = sum(1 for c in ctx_lens if c <= 50000)
    under_80k = sum(1 for c in ctx_lens if c <= 80000)
    print(f'  Under 50k chars: {under_50k}/{len(lines)}')
    print(f'  Under 80k chars: {under_80k}/{len(lines)}')
    # Check unique contexts
    unique = len(set(json.loads(l).get('context', '')[:100] for l in lines))
    print(f'  Unique contexts (approx): {unique}')
    # Show first sample structure
    item = json.loads(lines[0])
    print(f'  Keys: {list(item.keys())}')
