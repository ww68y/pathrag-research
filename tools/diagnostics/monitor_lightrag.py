import json, os, time, subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def check_status():
    # Check process
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                           capture_output=True, text=True, shell=True)
    procs = [l for l in result.stdout.split('\n') if 'python' in l.lower()]
    
    # Check output
    p = ROOT / 'results' / 'lightrag_current.jsonl'
    lines = open(p, 'r', encoding='utf-8').readlines() if os.path.exists(p) else []
    completed = len(lines)
    
    # Check log tail
    logp = ROOT / 'logs' / 'lightrag_full_run.log'
    log_tail = ''
    if os.path.exists(logp):
        data = open(logp, 'r', encoding='utf-8', errors='replace').read()
        log_tail = data[-300:]
    
    now = datetime.now().strftime('%H:%M:%S')
    print(f'\n[{now}] Processes: {len(procs)} | LightRAG completed: {completed}/130')
    print(f'Log tail: {log_tail[-200:]}')
    return completed

if __name__ == '__main__':
    check_status()
