# datas 目录说明

这个目录只放本地数据和可重建运行产物。

- `ds002680/`：保留的原始 OpenNeuro 数据集，不作为可重建产物清理。
- `artifacts/`：状态转移等后续训练缓存、模型权重、图像表示、扫描结果。
- `experiments_3way/`：RP 三集实验的固定划分、质量报告、冻结权重和汇总。`representations/` 可重建，已 gitignore；划分和 `*.json` / `*.md` 汇总应保留。
- `experiments_3way/transition/`：对齐老师方案的状态转移三集产物，与 RP 冻结结果隔离。

清理时优先删除 `artifacts/` 和可重建的 `experiments_3way/representations/`；不要删除 `ds002680/`，除非明确要重新下载或重新解压数据集。
