import json, os
from collections import Counter
from pathlib import Path

datasets_dir = Path(__file__).resolve().parents[2] / 'datasets' / 'UltraDomain'
for fname in sorted(os.listdir(datasets_dir)):
    if not fname.endswith('.jsonl'):
        continue
    fpath = datasets_dir / fname
    lines = open(fpath, 'r', encoding='utf-8').readlines()
    ctx_count = Counter()
    ctx_lens = {}
    for l in lines:
        item = json.loads(l)
        key = item.get('context', '')[:100]
        ctx_count[key] += 1
        if key not in ctx_lens:
            ctx_lens[key] = len(item.get('context', ''))
    
    all_ctx_lens = sorted(ctx_lens.values())
    min_ctx = all_ctx_lens[0] if all_ctx_lens else 0
    med_ctx = all_ctx_lens[len(all_ctx_lens)//2] if all_ctx_lens else 0
    max_ctx = all_ctx_lens[-1] if all_ctx_lens else 0
    
    # Estimate indexing time (Mix median 37K chars ≈ 14 min per context)
    # Rough: 1K chars ≈ 0.38 min indexing
    est_min_per_ctx = min_ctx / 1000 * 0.38
    est_total_hours = sum(l / 1000 * 0.38 for l in all_ctx_lens) / 60
    
    print(f'{fname:20s} | {len(lines):4d} samples | {len(ctx_count):3d} unique ctx | '
          f'min={min_ctx//1000}K med={med_ctx//1000}K max={max_ctx//1000}K chars | '
          f'est_index={est_total_hours:.1f}h')
