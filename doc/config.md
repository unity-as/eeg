# 个体实验配置项

正式：`config/eeg_ws_r.yaml`、`eeg_ws_m.yaml`  
通路：`eeg_ws_smoke_*.yaml`（`max_samples: 200`，`epochs: 2`）  
正式：`max_samples: 1500`，`epochs: 200`，**`epoch_samples: 512`**（256/400/512/800/1000 扫描后锁定）

**已删除（可推出）：** `subjects`（= `subject`）、`val_ratio` / `test_ratio`（= `1-train_ratio`）、`cache_dir` / `checkpoint_dir` / `save_best_model`（由 subject + representation 生成）、`paper_align`（由 representation + attention 生成）、`import_*`（正式必须按配置从 BIDS 建图）。

换人：`data.subject` 或 `python main_pipeline.py config/eeg_ws_r.yaml sub-003`。  
改窗长：yaml 里 `data.epoch_samples`（点数，@1000 Hz 即毫秒），或  
`python main_pipeline.py config/eeg_ws_r.yaml data.epoch_samples=512`。  
扫多个窗：`python scripts/sweep_epoch.py config/eeg_ws_r.yaml 400,512,800,1000`。

个体缓存/权重按窗写入 `datas/eeg_cache_ws_{rp|mrp}/{subject}/ep{N}`，扫窗时并排对比用；覆盖旧正式即可。

| 段 | 项 | 含义 |
|----|----|------|
| method | representation | `rp` / `mrp` |
| | attention | `none` / `se` / `additive` |
| | rhythm_filter / rhythm_band | 论文1 带通；默认关。开则按 δ/θ/α/β |
| | quality_assess | 论文1 图像质量，默认关 |
| | embed_select | `auto`：AMI 求 τ、FNN 求 m；`fixed`：用下面两行 |
| | embedding_dim / time_delay | m、τ；auto 时只作回退，结果写入 meta |
| | epsilon_mode | `percentile`（分位）/ `std`（论文1 kσ）/ `diameter`（最大直径比例） |
| | recurrence_percentile | percentile 模式的分位 |
| | epsilon_std_k / diameter_frac | std / diameter 模式的系数 |
| | epsilon | 绝对阈值；非 null 时优先 |
| | normalize | `zscore` / `none` |
| | sampling_rate | 节律滤波用，ds002680 为 1000 |
| | mrp_multichannel | 仅 M：`joint` 多导一张；`per_channel` 每导一张 |
| | rp_image_size | 图边长 |
| model | conv_channels | 各卷积层通道，如 `[32,64,128]` |
| | fc_hidden | 分类头隐层，`[]` 表示 GAP 后直接分类 |
| | kernel_size / pool_size | 卷积核 / 池化 |
| data | subject | 哪个人，如 `sub-002` |
| | max_samples | 训练规模：整数=上限；`null`=该人全部 |
| | max_runs | 最多几个 run；`null`=不限 |
| | train_ratio | 训练集占比；其余为合法评估池（早停用整池） |
| | eval_samples | 报告准确率：从合法池**不放回**抽这么多窗；`null`=用尽 |
| | stratify | 是否按类分层切分 |
| | force_rebuild | true 则忽略旧缓存 |
| | epoch_samples | 刺激后点数（@1000 Hz = 毫秒）；改则重建图 |
| | bids_root / class_map | 数据路径、类别 |
| train | augment | 增强总开关 |
| | gaussian_std / channel_dropout | 仅 augment=true 时生效 |
| | epochs / batch_size / optimizer / lr / weight_decay / momentum | 优化 |
| | dropout | 分类头 Dropout |
| | stop_on | `val_loss` 或 `val_acc`（评估集=非训练） |
| | early_stop_patience / seed / device / num_workers | 早停、复现、设备 |
| output | plot_curve / plot_confusion / run_tsne | 曲线、混淆矩阵、特征 t-SNE（报告窗） |
| | tsne_perplexity / tsne_iter | t-SNE 参数；样本少时 perplexity 会自动下调 |

缓存指纹含 subject、max_samples、max_runs、epoch_samples、method。改其中任一项会重建图，不改则跳过。路径已含 `ep{epoch_samples}`，同被试不同窗可并存。
