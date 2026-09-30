@echo off
cd /d d:\GitHubProjects\PathRAG
D:\Anaconda\envs\lw\python.exe experiments\run_lightrag_baseline.py --output results/historical_baselines/lightrag_current.jsonl --resume --query-mode hybrid --chunk-token-size 1200 --top-k 10 --llm-max-async 1 --embedding-max-async 1
