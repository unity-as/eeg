# file.md — 项目文件结构

## 分层

```
EEG/
├── .claude/                 # AI 上下文（见 CLAUDE.md）
├── config/
│   ├── default.yaml         # 正式默认（R）
│   ├── smoke.yaml           # 冒烟小配置
│   ├── b_r.yaml             # 方案 B + 阶段 R 试跑
│   ├── m_1k.yaml            # 阶段 M：每类 1k（旧协议，仅对照）
│   └── m_1w.yaml            # 阶段 M：每类 1w（旧协议，仅对照）
├── data/
│   ├── dataset.py           # 读 raw、窗采样
│   └── __init__.py
├── representation/
│   ├── phase_space.py
│   ├── recurrence_plot.py   # 经典 RP + build_representation
│   ├── modified_rp.py       # M 占位
│   ├── quality.py
│   ├── rhythm.py
│   └── __init__.py
├── models/
│   ├── attention.py         # none / se；additive 未实现
│   ├── cnn.py
│   └── __init__.py
├── utils/
│   ├── io_utils.py
│   ├── visualizer.py
│   └── __init__.py
├── scripts/
│   ├── step1_build_dataset.py
│   └── step2_train.py
├── datas/                   # 运行产物（gitignore 部分）
├── main_pipeline.py
├── requirements.txt
└── README.md
```

## 依赖方向

`scripts` / `main_pipeline` → `data` + `representation` + `models` + `utils`  
`models` 不依赖 `scripts`；`representation` 不依赖 `models`。

## 维护约定

- 新增/删除/移动源码文件后，**立即更新本文件**中的树状结构。
- 方法变体优先加模块 + 配置项，避免复制整条流水线。
