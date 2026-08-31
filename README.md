# EEG — RP/MRP + CNN（配置切换论文变体）

单一工程，OpenNeuro ds002680 **个体内** Go-nogo 四分类。默认阶段 **R**：经典 RP + SE 注意力。

## 快速开始

```bash
cd D:\project\Python\EEG
pip install -r requirements.txt
# 按本机 CUDA 安装匹配的 torch

# 冒烟
python main_pipeline.py config/eeg_ws_smoke_r.yaml

# 正式默认（个体内 R）
python main_pipeline.py
# 等同
python main_pipeline.py config/eeg_ws_r.yaml
```

分步：

```bash
python scripts/step1_build_dataset.py
python scripts/step2_train.py
```

## 配置要点

见 `config/eeg_ws_r.yaml` 的 `method.*`：`representation`（rp/mrp）、`attention`（none/se/additive）。  
个体实验项见 `doc/config.md`。

## 文档

- `.claude/CLAUDE.md`
- `.claude/docs/architecture.md`
- `.claude/records/status.md`
- `doc/goal.md`
