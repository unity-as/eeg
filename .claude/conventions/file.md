# file.md — 项目文件结构

## 分层

```
EEG/
├── .claude/                 # AI 上下文（见 CLAUDE.md）
├── config/
│   ├── eeg_ws_r.yaml        # 个体内正式 R（无参默认）
│   ├── eeg_ws_r_3way.yaml   # 独立三集 RP+SE CNN 实验
│   ├── eeg_ws_r_3way_frozen.yaml  # 冻结参数的正式三集配置
│   ├── eeg_ws_transition.yaml       # 个体内正式 Transition（旧两池）
│   ├── eeg_ws_transition_3way.yaml  # 三集协议对齐老师方案
│   ├── eeg_ws_smoke_transition.yaml # 个体内冒烟 Transition
│   ├── eeg_ws_smoke_transition_3way.yaml # 三集 Transition 冒烟
│   ├── eeg_ws_m.yaml        # 个体内正式 M
│   ├── eeg_ws_smoke_r.yaml  # 个体内冒烟 R
│   ├── eeg_ws_smoke_m.yaml  # 个体内冒烟 M
│   ├── eeg_mch_r.yaml       # 多通道 R，max_epochs=200（旧）
│   ├── eeg_mch_m.yaml       # 多通道 M，max_epochs=200（旧）
│   ├── eeg_full_r.yaml      # 2 被试全 run R（旧跨被试）
│   ├── eeg_full_m.yaml      # 2 被试全 run M（旧跨被试）
│   ├── eeg_smoke_r.yaml     # 旧 EEG 冒烟 R
│   ├── eeg_smoke_m.yaml     # 旧 EEG 冒烟 M
│   ├── eeg_loso2_smoke_r.yaml  # 2-Fold LOSO 冒烟（已结束）
│   └── eeg_loso2_r.yaml     # 2-Fold LOSO 正式（已结束）
├── data/
│   ├── eeg_ds002680.py      # EEG 4 类 epoch / 按被试打包
│   ├── augment.py           # 训练期高斯噪声 + 通道丢失
│   └── __init__.py
├── experiments/             # 独立三集实验的共享逻辑
│   ├── common.py            # 事件正确性、固定三集划分
│   ├── build_dataset.py     # 兼容入口：校正后的数据集构建
│   ├── train.py             # 兼容入口：现有实验训练器
│   └── config_rp_se.yaml
├── representation/
│   ├── phase_space.py
│   ├── embed_params.py      # AMI 求 τ、FNN 求 m
│   ├── rp_core.py           # 距离 / ε / z-score / 缩放
│   ├── recurrence_plot.py   # 经典 RP + build_representation
│   ├── gpu_rp.py            # CUDA 批量 RP 构建
│   ├── state_transition.py  # 幅值符号化 + 多时间步转移矩阵
│   ├── modified_rp.py       # 灰度 MRP；joint 或多导叠图
│   ├── quality.py
│   ├── rhythm.py            # 论文1 带通，接入 step1
│   └── __init__.py
├── models/
│   ├── attention.py         # none / se / additive
│   ├── cnn.py               # conv_channels / fc_hidden 可配
│   └── __init__.py
├── utils/
│   ├── io_utils.py
│   ├── visualizer.py
│   └── __init__.py
├── scripts/
│   ├── step1_build_dataset.py
│   ├── step2_train.py
│   ├── sweep_epoch.py       # 扫 data.epoch_samples
│   ├── sweep_transition_steps.py
│   ├── sweep_symbol_bins.py
│   ├── sweep_transition_weight.py
│   └── experiments/         # 三集数据、sealed-test 训练、调参
│       ├── build_correct_only_dataset.py
│       ├── train_3way.py
│       └── tune.py
├── datas/                   # 运行产物（gitignore 部分）
├── doc/                     # 人读文档，一处只维护一份
│   ├── method.md            # 协议、表示、命令
│   ├── results.md           # 全部数字
│   ├── config.md            # 配置键
│   ├── teacher_improvement_directions.md  # 老师原话，不记结果
│   └── final_results_rp_se_cnn.docx       # 2026-09-18 Word 快照，不再改
├── main_pipeline.py
├── requirements.txt
└── README.md
```

## 依赖方向

`scripts` / `main_pipeline` → `data` + `representation` + `models` + `utils`  
`scripts/experiments` / `experiments` → 同上，另用固定三集划分  
`models` 不依赖 `scripts`；`representation` 不依赖 `models`。

## 维护约定

- 新增/删除/移动源码文件后，**立即更新本文件**中的树状结构。
- 方法变体优先加模块 + 配置项，避免复制整条流水线。
