# Changelog

## 2026-08-30

- 初始化 EEG 工程：`.claude/`、配置驱动方法变体、阶段 R 优先（先 R 后 M）
- 论文摘录移至 `.claude/reference/papers_rp_eeg.md`
- 约定用旋转机械 raw_datas 做工程验证
- 已写：`config/default.yaml`、`data/`、`representation/phase_space.py` + `recurrence_plot.py`
- **暂停**：R 通路其余代码（models/scripts/跑通）未做；详见 `status.md`

### 正式测试：每类 1000

- `num_windows_per_class=1000`；`main_pipeline.py config/default.yaml`
- best_val_acc≈0.5674；checkpoint：`datas/checkpoints/best_rp_cnn.pt`

### 正式测试：每类 10000

- `num_windows_per_class=10000`；best_val_acc≈0.7859；early stop @16
- checkpoint 覆盖写入 `datas/checkpoints/best_rp_cnn.pt`
