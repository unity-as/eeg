# 架构与分阶段方案

## 目标

单一工程复现/组合三篇 RP 系方法；用配置切换。主任务是 ds002680 **个体内** Go-nogo 四分类。

## 流水线

```
BIDS epoch → (可选 z-score) → (可选节律带通) → AMI/FNN 定 m、τ → 相空间 → RP/MRP → (可选质量评估) → CNN(+注意力) → 分类
```

分步脚本 + `main_pipeline.py` 串跑；图表示为 RP/MRP。

## 配置映射

| method 字段 | 含义 | 阶段 |
|-------------|------|------|
| representation: rp | 经典递归图 | R |
| representation: mrp | 改进递归图 | M |
| attention: none/se | 无 / SE 通道注意力 | R |
| attention: additive | 加性注意力 | M |
| rhythm_filter | EEG 节律带通；默认 false（Go-nogo 不要开） | 论文1 可选 |
| quality_assess | 论文1 质量指标 | 后置可开 |
| embed_select | auto=AMI+FNN；fixed=yaml 的 m、τ | 论文1/3 |
| epsilon_mode | percentile / std / diameter | 阈值规则 |

## 验证顺序

1. **R**：rp + se，个体内（`config/eeg_ws_r.yaml`）
2. **M**：mrp + additive（`config/eeg_ws_m.yaml`）
3. 论文向：按配置贴近各篇设置与指标
