# 实验脚本索引

`experiments/` 保留可复现实验脚本，按文件名前缀识别用途。脚本路径暂不整体搬迁，以保持历史命令、论文引用和复现清单兼容。

## 论文主线

- `run_aligned_ablation_rerun.py`：UltraDomain Mix130 严格消融
- `run_enhanced_experiments.py`、`run_full_v2_context_rescue.py`：VSR 主线与 rescue
- `run_retrieval_baseline.py`：Naive / HyDE 检索基线
- `run_graphrag_lite_baseline.py`、`run_lightrag_baseline.py`：图 RAG 对比
- `run_pairwise_comparison.py`、`evaluate_pairwise_winrate.py`：成对比较
- `run_statistical_tests.py`：统计检验
- `build_paper_ready_tables.py`、`build_paper_latex_tables.py`：论文表格生成
- `build_reproducibility_manifest.py`：复现清单生成

## 结果处理

- `build_*`：构造数据、策略或实验输入
- `run_*`：运行实验
- `evaluate_*` / `audit_*`：评分、质量审计和风险检查
- `summarize_*`：从已有结果生成汇总
- `verify_*` / `check_*`：一致性与进度检查
- `analyze_*` / `compare_*`：机制分析和方法比较

## 推荐入口

```powershell
# 快速检查（会运行 API/模型调用，请确认 .env）
python experiments/run_all_experiments.py --quick

# 论文表格与复现清单（离线汇总已有结果）
python experiments/build_paper_ready_tables.py
python experiments/build_paper_latex_tables.py
python experiments/build_reproducibility_manifest.py

# 研究档案索引
python experiments/build_research_archive.py
python experiments/verify_research_archive.py
```

历史输出文本和一次性审计文件位于 `experiments/_artifacts/`，不作为新的实验入口。
