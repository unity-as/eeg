# 方法

数字只写在 `doc/results.md`。配置键只写在 `doc/config.md`。老师原话整理在 `doc/teacher_improvement_directions.md`，那里不记实验结果。

## 现在做什么

主线是老师的状态转移网络：EEG 按幅值符号化，做成多时间步有向加权转移矩阵，再交给 CNN。RP/MRP 只作 baseline。GCN、EEGNet、1D CNN 都还没做。

数据协议沿用队友已经冻结的三集，不改他们的划分和 RP 结果。

## 协议

只在这里维护。

- 数据：OpenNeuro ds002680，Go-nogo 四类（cat_go / cat_nogo / rec_go / rec_nogo）
- 范围：个体内。14 人（sub-002 至 sub-015）各自训练，不混人，不报跨人准确率
- 试次：`correct_only`。有 response 时用 correct/incorrect；没有 response 时 target 算错、distractor 算对
- 窗：刺激后 800 ms
- 划分：每人 70/15/15，分层，`split_seed=42`
- 划分文件：`datas/experiments_3way/splits/`，只读复用，不要覆盖
- 训练种子：42、43、44。按 `val_acc` 早停
- test：冻结前不要打开，打开必须同时加 `--evaluate-test --confirm-test`。`bins=6`、按导堆叠这一次已经用已有权重评过，数字在 `doc/results.md`。其他设置不要再开

跨人 LOSO 已放弃，不再作为主线。

## 当前 Transition 表示

配置：`config/eeg_ws_transition_3way.yaml`。冒烟：`config/eeg_ws_smoke_transition_3way.yaml`。

- `symbol_bins=6`，`symbol_strategy=uniform`，`bin_scope=per_epoch`
- `transition_steps=[1, 2, 3]`，`transition_weight=probability`
- `transition_multichannel=mean`，保留自转移
- 输入 `[3, 6, 6]`。CNN 只用两层 `[32, 64]`，三次池化会把 6×6 压成 0
- 注意力 SE。产物在 `datas/experiments_3way/transition/`，与 RP 冻结权重分开

旧两池配置 `config/eeg_ws_transition.yaml` 不是这条协议，不要拿它的结果和三集比。

## RP baseline

冻结参数只在 `config/eeg_ws_r_3way_frozen.yaml`。test 已经评过，这条线不能再调参。

代码里仍保留经典 RP、MRP、无注意力、SE、additive。论文对照见 `.claude/reference/papers_rp_eeg.md`。入口配置是 `config/eeg_ws_r.yaml`、`config/eeg_ws_m.yaml`，不是当前主线。

## 命令

```bash
python scripts/experiments/build_correct_only_dataset.py --config config/eeg_ws_smoke_transition_3way.yaml --force
python scripts/experiments/train_3way.py --config config/eeg_ws_smoke_transition_3way.yaml

python scripts/experiments/build_correct_only_dataset.py --config config/eeg_ws_transition_3way.yaml
python scripts/experiments/train_3way.py --config config/eeg_ws_transition_3way.yaml
```

## 还没做

节点数只在各导平均下扫过。`bins=6`、按导堆叠的 test 已经评过，不要再对其他节点数开 test。GCN 还没做。
