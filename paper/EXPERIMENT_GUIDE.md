# VSR-PathRAG 论文实验运行指南

## 1. 环境准备

### 1.1 安装依赖
```bash
pip install datasets scipy numpy openai python-dotenv
```

### 1.2 配置 API Key
确保 `.env` 文件中配置了：
```
OPENAI_API_KEY=sk-xxx
LLM_MODEL=gpt-4o-mini
```

### 1.3 确保 PathRAG 图索引已构建
```bash
# 如果还没有构建图索引
python -c "from api.features.rag_manager import get_rag_instance; rag = get_rag_instance(); print('RAG ready')"
```

## 2. 实验运行

### 2.1 历史快速验证（已被 Mix130 / locked 326 取代）
```bash
python experiments/run_all_experiments.py --quick
```
这会运行：
- Naive RAG / HyDE / VSR-PathRAG 三个方法对比
- VSR vs HyDE 和 VSR vs Naive RAG 的 Pairwise 对比
- 7 组 Ablation Study

### 2.2 完整实验（所有数据集, ~2小时）
```bash
python experiments/run_all_experiments.py --full
```

### 2.3 单独运行各组件

#### Benchmark 对比
```bash
# UltraDomain
python experiments/run_standard_benchmark.py --dataset ultradomain --method all --sample_size 30

# HotpotQA (需要安装 datasets 库)
python experiments/run_standard_benchmark.py --dataset hotpotqa --method all --sample_size 100
```

#### Pairwise Comparison
```bash
python experiments/run_pairwise_comparison.py --method_a vsr_pathrag --method_b hyde --dataset ultradomain --n 30
```

#### Ablation Study
```bash
python experiments/run_ablation_study.py --dataset ultradomain --sample_size 30
```

#### 统计显著性检验
```bash
python experiments/run_statistical_tests.py \
  --result_a results/benchmark/ultradomain_vsr_pathrag_full_n30.json \
  --result_b results/benchmark/ultradomain_hyde_full_n30.json
```

## 3. 结果输出

所有结果保存在 `results/` 目录下：
- `results/benchmark/` - Benchmark 对比结果
- `results/pairwise/` - Pairwise 对比结果
- `results/ablation/` - Ablation Study 结果
- `results/statistical_tests/` - 统计检验结果
- `results/paper/` - 完整实验套件结果

## 4. 论文表格填充

运行完实验后，使用以下脚本生成论文表格：

```bash
# 查看所有聚合结果
python -c "
import json
from pathlib import Path

# 加载快速实验结果
with open('results/paper/quick_suite_results.json', 'r') as f:
    data = json.load(f)

# 打印 Benchmark 对比
print('=== Benchmark Results ===')
for key, val in data.items():
    if key.startswith('benchmark_'):
        print(f'{key}:')
        for metric, value in val.items():
            if isinstance(value, float):
                print(f'  {metric}: {value:.4f}')

# 打印 Pairwise 结果
print('\n=== Pairwise Results ===')
for key, val in data.items():
    if key.startswith('pairwise_'):
        print(f'{key}:')
        print(f'  Average win rate A: {val.get(\"average_win_rate_a\", 0):.1f}%')
        for dim, wr in val.get('win_rates', {}).items():
            print(f'  {dim}: A={wr.get(list(wr.keys())[0], 0)} B={wr.get(list(wr.keys())[1], 0)} Tie={wr.get(\"ties\", 0)}')
"
```

## 5. 论文结构

论文文件位于 `paper/` 目录：
- `vsr_pathrag.tex` - LaTeX 论文主体
- `vsr_pathrag_paper_framework.md` - 论文框架（Markdown 版本）

### 论文三大理论贡献：
1. **Theorem 1 (MPS)**: 贪心算法对双目标加权集合覆盖达到 $O(\ln n)$ 近似比
2. **Theorem 2 (CPN)**: 反事实必要性认证的假阳性率 $\leq \epsilon$，假阴性率 $= 0$
3. **Theorem 3 (RCS)**: PAC-Bayesian 泛化界 $\hat{R} + O(\sqrt{(KL + \ln(1/\delta))/m})$

## 6. 投稿建议

### 适合的期刊/会议：
- **IEEE TKDE** (SCI Q1, IF~8.9) - 知识工程方向
- **Information Processing & Management** (SCI Q1, IF~7.5) - 信息检索方向
- **ACL/EMNLP** (CCF-A) - 如果增加 NLP 理论深度
- **AAAI/IJCAI** (CCF-A) - 如果增加推理方向的新颖性

### 关键审稿点预判：
1. **理论深度**: 三个定理提供了近似比、正确性、泛化界 → 需要完整证明
2. **实验规模**: Mix130 严格消融和 locked 326 严格 holdout 已完成；HotpotQA/2Wiki 属于可选扩展，不是当前待跑项
3. **Baseline 强度**: GraphRAG-lite 与 LightRAG current-130 代表性对比已完成；full official GraphRAG/LightRAG rerun 属于可选扩展
4. **可复现性**: 代码已完整 → 需要提供 Docker 镜像
5. **写作质量**: 需要专业英文润色
