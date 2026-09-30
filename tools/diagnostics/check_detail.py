import json, os, subprocess, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Check process details
result = subprocess.run(['wmic', 'process', 'where', 'name="python.exe"', 'get', 'ProcessId,CommandLine,WorkingSetSize', '/FORMAT:LIST'],
                       capture_output=True, text=True, shell=True)
print('=== Python process details ===')
print(result.stdout[:2000])

# Check log file modification time
logp = ROOT / 'logs' / 'lightrag_full_run.log'
if os.path.exists(logp):
    mtime = os.path.getmtime(logp)
    size = os.path.getsize(logp)
    print(f'\n=== Log file ===')
    print(f'Size: {size} bytes')
    print(f'Last modified: {datetime.fromtimestamp(mtime).strftime("%H:%M:%S")}')
    print(f'Current time:  {datetime.now().strftime("%H:%M:%S")}')
    data = open(logp, 'r', encoding='utf-8', errors='replace').read()
    print(f'Last 400 chars:\n{data[-400:]}')

# Check output
p = ROOT / 'results' / 'lightrag_current.jsonl'
if os.path.exists(p):
    lines = open(p, 'r', encoding='utf-8').readlines()
    print(f'\n=== Output: {len(lines)} samples ===')
    if lines:
        last = json.loads(lines[-1])
        print(f'Last sample: line={last["line_no"]}')
