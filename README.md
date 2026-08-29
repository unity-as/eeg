# EEG — RP/MRP + CNN（配置切换论文变体）

单一工程，对照旋转机械流水线；默认阶段 **R**：经典 RP + SE 注意力，用振动 `raw_datas` 五分类验证。

## 快速开始

```bash
cd D:\project\Python\EEG
pip install -r requirements.txt
# 按本机 CUDA 安装匹配的 torch

# 冒烟（小数据）
python main_pipeline.py config/smoke.yaml

# 正式默认配置
python main_pipeline.py
```

分步：

```bash
python scripts/step1_build_dataset.py
python scripts/step2_train.py
```

## 配置要点

见 `config/default.yaml` 的 `method.*`：`representation`（rp/mrp）、`attention`（none/se/additive）。

## 文档

- `.claude/CLAUDE.md`
- `.claude/docs/architecture.md`
- `.claude/records/status.md`
