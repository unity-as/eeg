# 当前状态

- **阶段**：方案 B 已落地，并用 R（rp+se）试跑完成
- **协议**：每条记录先按时间 0.7/0.15/0.15 切 train/val/test，段内 `stride=window=256` 不重叠采窗；val early-stop，test 最终报告

## 方案 B + R 结果（`config/b_r.yaml`）

- 样本：train 36868 / val 7898 / test 7898
- **best_val_acc ≈ 0.798**；**test_acc ≈ 0.777**
- 模型：`datas/checkpoints_B_r/best_rp_cnn_B.pt`

## 对照（旧协议有泄漏，仅历史参考）

| 协议 | 设置 | 指标 |
|------|------|------|
| 旧 | R 1w 窗级随机 split | val≈0.79（不可信） |
| 旧 | M 1w | val≈0.88（不可信） |
| **B** | R 时间切分+不重叠 | **test≈0.78**（更干净） |

## 待办

- [ ] 可选：同一协议跑 M（`mrp+additive`）
- [ ] commit 方案 B 改动

## 阻塞

无
