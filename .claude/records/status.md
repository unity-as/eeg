# 当前状态

- **阶段**：R 正式测试（每类 1e4）已跑完
- **工作区**：`D:\project\Python\EEG`

## 正式结果（default，1w/类）

- 样本：5 × 10000 = 50000；train 40000 / test 10000
- 方法：rp + se；窗长 256；RP 64×64；CUDA
- **best_val_acc ≈ 0.7859**（约第 8 epoch）；early stop 于 epoch 16
- 产物：`datas/rp_images/`、`datas/checkpoints/best_rp_cnn.pt`

## 对照

| 每类样本 | best_val_acc |
|----------|--------------|
| 40 (smoke) | ~0.47 |
| 1000 | ~0.57 |
| **10000** | **~0.79** |

## 待办

- [ ] 阶段 M：MRP + additive
- [ ] 论文对齐 / 涨点 / Web / 独立 venv

## 阻塞

无
