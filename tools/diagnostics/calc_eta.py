import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / 'results' / 'lightrag_current.jsonl'
rows = [json.loads(line) for line in p.open('r', encoding='utf-8') if line.strip()]
latest_rows = {}
for item in rows:
    if item.get('line_no') is not None:
        latest_rows[int(item['line_no'])] = item
valid_rows = [
    item for item in latest_rows.values()
    if not item.get('error')
    and str(item.get('prediction') or '').strip()
    and '[no-context]' not in str(item.get('prediction') or '').lower()
]
error_lines = sorted(
    line_no for line_no, item in latest_rows.items()
    if item.get('error')
    or not str(item.get('prediction') or '').strip()
    or '[no-context]' in str(item.get('prediction') or '').lower()
)
completed = len(valid_rows)

# Calculate timing stats
total_idx = 0
total_q = 0
unique_ctx = 0
for item in valid_rows:
    idx = item.get('index_seconds', 0)
    q = item.get('query_seconds', 0)
    total_idx += idx
    total_q += q
    if idx > 0:
        unique_ctx += 1

avg_per_sample = (total_idx + total_q) / completed if completed else 0
remaining = 130 - completed

# There are 61 unique contexts total
remaining_ctx = 61 - unique_ctx
avg_idx_per_ctx = total_idx / unique_ctx if unique_ctx else 0

# Estimate remaining time
# Some remaining samples will reuse contexts (no index time)
# Some will need new indexing
est_remaining_idx = remaining_ctx * avg_idx_per_ctx
est_remaining_q = remaining * 30  # avg 30s per query
est_total_sec = est_remaining_idx + est_remaining_q
est_hours = est_total_sec / 3600

print(f'=== LightRAG Progress ===')
print(f'Completed: {completed}/130 ({completed/130*100:.1f}%)')
print(f'Recorded rows: {len(rows)}/130')
print(f'Unique recorded lines: {len(latest_rows)}/130')
print(f'Error / retry rows: {error_lines}')
print(f'Unique contexts indexed: {unique_ctx}/61')
print(f'Total index time so far: {total_idx:.0f}s ({total_idx/3600:.1f}h)')
print(f'Total query time so far: {total_q:.0f}s ({total_q/60:.1f}min)')
print(f'Average index per context: {avg_idx_per_ctx:.0f}s ({avg_idx_per_ctx/60:.1f}min)')
print(f'Average query per valid sample: {total_q/completed:.0f}s')
print(f'')
print(f'=== ETA Estimate ===')
print(f'Remaining samples: {remaining}')
print(f'Remaining new contexts: {remaining_ctx}')
print(f'Estimated remaining index time: {est_remaining_idx/3600:.1f}h')
print(f'Estimated remaining query time: {est_remaining_q/3600:.1f}h')
print(f'Total estimated remaining: {est_hours:.1f}h')
