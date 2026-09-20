# 配置键

当前主线配置是 `config/eeg_ws_transition_3way.yaml`。RP 冻结配置是 `config/eeg_ws_r_3way_frozen.yaml`。协议和命令见 `doc/method.md`，不要在这里重写。

**已删除（可推出）：** `subjects`（= `subject`）、`val_ratio` / `test_ratio`（= `1-train_ratio`）、`cache_dir` / `checkpoint_dir` / `save_best_model`（由 subject + representation 生成）、`paper_align`（由 representation + attention 生成）、`import_*`（正式必须按配置从 BIDS 建图）。

换人：`data.subject` 或 `python main_pipeline.py config/eeg_ws_r.yaml sub-003`。  
改窗长：yaml 里 `data.epoch_samples`（点数，@1000 Hz 即毫秒），或  
`python main_pipeline.py config/eeg_ws_r.yaml data.epoch_samples=512`。  
扫多个窗：`python scripts/sweep_epoch.py config/eeg_ws_r.yaml 400,512,800,1000`。

个体缓存/权重按窗写入 `datas/artifacts/eeg_cache_ws_{rp|mrp}/{subject}/ep{N}` 和 `datas/artifacts/checkpoints_eeg_ws_{rp|mrp}/{subject}/ep{N}`，扫窗时并排对比用；覆盖旧正式即可。

| 段 | 项 | 含义 |
|----|----|------|
| method | representation | `rp` / `mrp` / `transition` |
| | attention | `none` / `se` / `additive` / `self`。`self` 在卷积前对空间位置做自注意力 |
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
| | symbol_bins | 仅 transition：幅值状态/节点数量 |
| | symbol_strategy | 仅 transition：`uniform` 等幅值分区；`quantile` 分位分区 |
| | bin_scope | 仅 transition：`per_epoch` 各试次自己的范围；`train_global` 只用训练集各导最小、最大幅值，且 `normalize` 为 `none` |
| | transition_steps | 仅 transition：状态转移时间步列表，如 `[1,2,3]` |
| | transition_weight | 仅 transition：`count` / `probability` / `row_probability` |
| | transition_multichannel | 仅 transition：`mean` 各导矩阵平均；`stack` 各导各时间步叠通道 |
| | include_self_transition | 仅 transition：是否保留同一状态自转移 |
| model | arch | `cnn`（默认）或 `gcn`。两者互斥，不叠在一起。`gcn` 与注意力 `self` 的代码在 `feat/transition-gcn`，不在本分支 |
| | gcn_hidden / gcn_layers / gcn_pool / gcn_layout / gcn_steps / gcn_attention | 仅 `arch=gcn`。`relational` 把每导的多个时间步当作同一批节点上的不同边；`independent` 每张矩阵一张图。`gcn_steps` 是时间步数。`gcn_attention` 为 true 时在边上做图注意力，默认关 |
| | conv_channels | 各卷积层通道，如 `[32,64,128]`。仅 CNN |
| | fc_hidden | 分类头隐层，`[]` 表示 GAP 后直接分类 |
| | kernel_size / pool_size | 卷积核 / 池化 |
| data | subject | 哪个人，如 `sub-002` |
| | max_samples | 训练规模：整数=上限；`null`=该人全部 |
| | max_samples_per_class | 小样本冒烟用：每类最多取多少窗；`null`=不用按类截断 |
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
| | device | `cuda` 或 `cpu`。`cuda` 是可选项：当前 Python 能用 CUDA 就用 GPU，否则用 CPU，不因此停训练。本机 RTX 5060（sm_120）需 `torch` 的 `cu130` 及以上 wheel；`cu126` 能检测到卡但算不了 |
| | require_gpu | `false`：没有 CUDA 不报错，改走 CPU。`true`：必须有 CUDA，否则停止 |
| | early_stop_patience / seed / num_workers | 早停、复现、DataLoader 进程数 |
| output | plot_curve / plot_confusion / run_tsne | 曲线、混淆矩阵、特征 t-SNE（报告窗） |
| | tsne_perplexity / tsne_iter | t-SNE 参数；样本少时 perplexity 会自动下调 |

缓存指纹含 subject、max_samples、max_runs、epoch_samples、method。改其中任一项会重建图，不改则跳过。路径已含 `ep{epoch_samples}`，同被试不同窗可并存。
