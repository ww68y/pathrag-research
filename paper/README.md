# 论文材料

- `vsr_pathrag.tex`：论文主文档
- `vsr_pathrag_paper_framework.md`：论文结构与论证框架
- `EXPERIMENT_READINESS_FOR_SCI.md`：当前实验就绪度与 claim 边界
- `EXPERIMENT_GUIDE.md`：实验运行说明
- `generated_experiment_tables.tex`：可直接 include 的 LaTeX 表格片段
- `PATHRAG_PAPER_DRAFT_ZH.md`：当前唯一维护的中文论文主稿（含摘要、工作稿目录、正文和结果表）
- `PAPER_FIGURES.md`：论文图稿源、精确绘图数据和 PNG/SVG 预览索引；正文中的图已按章节位置嵌入主稿
- `patent_materials/`：专利参考材料，不属于论文实验主证据
- `build/`：LaTeX 编译生成的辅助文件

论文当前最稳妥的结果来源是 `../results/paper_tables/PAPER_READY_EXPERIMENT_TABLES.md`。其中已明确区分 VSR 内部消融、检索基线、SAGE 冻结候选和探索性分支。

查看论文时，只需打开 `PATHRAG_PAPER_DRAFT_ZH.md`；需要修改图时再打开 `PAPER_FIGURES.md` 和 `figures/` 下的 SVG/PNG。后续论文修改统一在中文主稿中进行，不再新增版本文件。图稿中的数值表是后期重绘时的依据，PNG 用于兼容普通 Markdown 预览器，SVG 用于后期编辑。

中文主稿不单独设置图表目录：图在正文第一次讨论对应方法或结果的位置直接出现，并按出现顺序编号为图 1—图 12；表格统一采用三线表样式。工作稿目录仅用于当前编辑导航，定稿导出时再删除。`PAPER_FIGURES.md` 作为图稿源、精确数据备份和可编辑图片索引。

## 在 VS Code / Cursor 中预览

1. 打开 `PATHRAG_PAPER_DRAFT_ZH.md` 或 `PAPER_FIGURES.md`。
2. 按 `Ctrl+Shift+V` 打开 Markdown 预览；或按 `Ctrl+K`，再按 `V` 在旁边打开预览。
3. 项目已推荐 `bierner.markdown-mermaid` 和 `shd101wyy.markdown-preview-enhanced` 扩展，分别用于 Mermaid 图和公式/增强预览。

如果仍然看到源码，说明当前打开的是编辑器标签页，请在右上角点击“打开侧边预览”图标，或者重新执行第 2 步。刚安装扩展后，请执行一次 `Developer: Reload Window`。正文优先引用 PNG，因此即使 Markdown 预览器不支持 Mermaid，框架图、消融图、三维图等仍应正常显示；`PAPER_FIGURES.md` 中的 Mermaid 代码用于后续结构修改。

推荐使用 Markdown Preview Enhanced 的命令 `Markdown Preview Enhanced: Open Preview`，它对正文中的 `$$...$$` 公式支持更好。
