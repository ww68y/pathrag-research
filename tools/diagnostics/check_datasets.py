import json, os
from pathlib import Path

datasets_dir = Path(__file__).resolve().parents[2] / 'datasets' / 'UltraDomain'
for fname in sorted(os.listdir(datasets_dir)):
    if fname.endswith('.jsonl'):
        fpath = datasets_dir / fname
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        lines = open(fpath, 'r', encoding='utf-8').readlines()
        n = len(lines)
        # Check sample structure
        item = json.loads(lines[0])
        ctx_len = len(item.get('context', ''))
        datasets_in_file = set()
        for l in lines:
            datasets_in_file.add(json.loads(l).get('dataset', '?'))
        print(f'{fname:20s} | {n:4d} samples | {size_mb:6.1f} MB | ctx_len_first={ctx_len} | sub_datasets={datasets_in_file}')
