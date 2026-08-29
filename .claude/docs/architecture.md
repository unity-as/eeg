# 架构与分阶段方案

## 目标

单一工程复现/组合三篇 RP 系方法；用配置切换；先振动五分类工程验证，再论文对齐。

## 流水线

```
raw .npy → 随机窗 → (可选频带) → 相空间 → RP/MRP → (可选质量评估) → CNN(+注意力) → 分类
```

对照旋转机械：分步脚本 + `main_pipeline.py` 串跑；本项目图表示为 RP 而非可视图。

## 配置映射

| method 字段 | 含义 | 阶段 |
|-------------|------|------|
| representation: rp | 经典递归图 | R |
| representation: mrp | 改进递归图 | M |
| attention: none/se | 无 / SE 通道注意力 | R |
| attention: additive | 加性注意力 | M |
| rhythm_filter | EEG 节律；振动默认 false | 后置 |
| quality_assess | 论文1 质量指标 | 后置可开 |

## 验证顺序

1. **R**：rp + se，振动 raw_datas，五分类跑通
2. **M**：mrp + additive
3. 论文向：按配置贴近各篇设置与指标
