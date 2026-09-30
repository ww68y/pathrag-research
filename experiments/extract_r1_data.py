import re
import json

# Extract R1 data from log file
log_path = r'd:\GitHubProjects\PathRAG\experiments\vsr_ctx_out2.txt'
with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Parse per-question results from R1 log
questions = []
for match in re.finditer(r'\[(\d+)/30\] Q: (.+?)\n\s+F1=([\d.]+)\s+Cont=(\d+)\s+Time=([\d.]+)s\s+Faith=([\d.]+)\s+Supp=(\d+)\s+Unsupp=(\d+)\s+Repair=(\w+)', content):
    q_idx = int(match.group(1))
    question = match.group(2).strip()
    f1 = float(match.group(3))
    contains = int(match.group(4))
    runtime = float(match.group(5))
    faith = float(match.group(6))
    supported = int(match.group(7))
    unsupported = int(match.group(8))
    repair = match.group(9) == 'Y'
    questions.append({
        "question_id": f"ultra_{q_idx-1}",
        "question": question,
        "method": "vsr_pathrag",
        "metrics": {
            "f1": f1,
            "contains": contains,
            "runtime_seconds": runtime,
            "faithfulness": faith,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "repair_triggered": repair,
        }
    })

# Extract aggregated stats
agg_match = re.search(r'mean_f1:\s*([\d.]+)\s+std_f1:\s*([\d.]+)\s+mean_contains:\s*([\d.]+)\s+std_contains:\s*([\d.]+)\s+mean_faithfulness:\s*([\d.]+)\s+std_faithfulness:\s*([\d.]+)\s+mean_runtime_seconds:\s*([\d.]+)\s+total_runtime_seconds:\s*([\d.]+)\s+sample_size:\s*(\d+)\s+repair_rate:\s*([\d.]+)', content)

if agg_match:
    r1_aggregated = {
        "vsr_pathrag_r1": {
            "mean_f1": float(agg_match.group(1)),
            "std_f1": float(agg_match.group(2)),
            "mean_contains": float(agg_match.group(3)),
            "std_contains": float(agg_match.group(4)),
            "mean_faithfulness": float(agg_match.group(5)),
            "std_faithfulness": float(agg_match.group(6)),
            "mean_runtime_seconds": float(agg_match.group(7)),
            "total_runtime_seconds": float(agg_match.group(8)),
            "sample_size": int(agg_match.group(9)),
            "repair_rate": float(agg_match.group(10)),
        }
    }
    
    # Also add naive_rag and hyde from baselines
    with open(r'd:\GitHubProjects\PathRAG\results\benchmark\ultradomain_baselines_n30.json', 'r', encoding='utf-8') as f:
        baselines = json.load(f)
    
    r1_aggregated["naive_rag"] = baselines["aggregated"]["naive_rag"]
    r1_aggregated["hyde"] = baselines["aggregated"]["hyde"]
    
    r1_data = {
        "experiment": "full_comparison_r1_original_repair",
        "dataset": "ultradomain",
        "sample_size": 30,
        "aggregated": r1_aggregated,
        "vsr_per_question": questions,
    }
    
    out_path = r'd:\GitHubProjects\PathRAG\results\benchmark\ultradomain_comparison_r1_n30.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(r1_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"R1 data saved to: {out_path}")
    print(f"R1 questions parsed: {len(questions)}")
    print(f"R1 VSR F1: {r1_aggregated['vsr_pathrag_r1']['mean_f1']}")
    print(f"R1 VSR Faith: {r1_aggregated['vsr_pathrag_r1']['mean_faithfulness']}")
    print(f"R1 VSR Repair: {r1_aggregated['vsr_pathrag_r1']['repair_rate']}")
else:
    print("Could not find aggregated stats in R1 log")
    # Try manual extraction
    f1_match = re.search(r'mean_f1:\s*([\d.]+)', content)
    faith_match = re.search(r'mean_faithfulness:\s*([\d.]+)', content)
    repair_match = re.search(r'repair_rate:\s*([\d.]+)', content)
    if f1_match:
        print(f"R1 mean_f1: {f1_match.group(1)}")
    if faith_match:
        print(f"R1 mean_faithfulness: {faith_match.group(1)}")
    if repair_match:
        print(f"R1 repair_rate: {repair_match.group(1)}")
