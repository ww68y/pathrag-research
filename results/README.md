# 实验结果目录

结果按实验问题分目录保存。论文写作优先使用 `paper_tables/` 和 `reproducibility/` 中的冻结汇总，单次 JSONL/日志用于追溯和复核。

## 论文优先级

1. `paper_tables/`：论文表格和推荐 claim framing
2. `enhanced_ablation_aligned/`：VSR 严格消融与跨数据集验证
3. `retrieval_baselines/`：Naive / HyDE 对比
4. `sage_locked_unseen75/`、`sage_locked_*`：冻结后的 holdout 与稳定性证据
5. `cost_latency/`、`cost_instrumented/`：成本、延迟和质量审计
6. `reproducibility/`：脚本、数据、结果哈希与复现清单

其他目录保留探索、候选策略、历史基线和诊断结果，不应在论文中未经核对直接引用。

## 原根目录结果的分类归档

为避免历史结果继续平铺在 `results/` 根层，旧文件已按文件名和实验用途归入以下目录：

- `historical_baselines/`：Naive、HyDE、Direct Context、Adaptive、GraphRAG、LightRAG 等基线
- `historical_pairwise/`：成对比较与 win-rate 文件
- `historical_datasets/`：Biology、Legal、Domain、UltraDomain 等数据集实验文件
- `historical_interactive/`：Interactive Agent v3-v25 系列
- `historical_sage/`：MSE/SAGE 历史候选与路由实验
- `historical_vsr/`：VSR v1-v77 历史实验和版本化结果
- `historical_ablation/`：早期消融试验
- `historical_reports/`：阶段报告、状态说明、迁移指南和其他汇总

这些 `historical_*` 目录是历史兼容区；论文正文优先引用上面的冻结结果目录。
