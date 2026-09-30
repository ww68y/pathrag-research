@echo off
cd /d d:\GitHubProjects\PathRAG
python experiments\run_remaining_experiments.py > logs\remaining_experiments.log 2>&1
echo DONE >> logs\remaining_experiments.log
