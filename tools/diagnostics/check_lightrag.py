import json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 1. Check if lightrag process is still running
result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                       capture_output=True, text=True, shell=True)
python_procs = [l for l in result.stdout.split('\n') if 'python' in l.lower()]
print(f'=== Python processes running: {len(python_procs)} ===')
for p in python_procs:
    print(f'  {p.strip()}')

# 2. Check LightRAG output file
p = ROOT / 'results' / 'lightrag_current.jsonl'
if os.path.exists(p):
    lines = open(p, 'r', encoding='utf-8').readlines()
    print(f'\n=== LightRAG output: {len(lines)} samples completed ===')
    for l in lines:
        item = json.loads(l)
        idx = item.get('index_seconds', 0)
        q = item.get('query_seconds', 0)
        err = item.get('error', '')
        print(f"  line={item['line_no']} idx={idx:.0f}s q={q:.0f}s {'ERROR:'+err if err else 'ok'}")
else:
    print('\n=== LightRAG output file does not exist ===')

# 3. Check log tail
logp = ROOT / 'logs' / 'lightrag_full_run.log'
if os.path.exists(logp):
    data = open(logp, 'r', encoding='utf-8', errors='replace').read()
    print(f'\n=== LightRAG log (last 600 chars) ===')
    print(data[-600:] if len(data) > 600 else data)
else:
    print('\n=== No log file found ===')
