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
- test：调参只看 val。方案在验证集定下之后评一次 test，必须同时加 `--evaluate-test --confirm-test`。当前 `bins=6`、按导堆叠已经评过，数字在 `doc/results.md`，不再拿这次 test 去选别的设置。新方案另开分支，定下后同样评 test

跨人 LOSO 已放弃，不再作为主线。

## 当前 Transition 表示

配置：`config/eeg_ws_transition_3way.yaml`。冒烟：`config/eeg_ws_smoke_transition_3way.yaml`。

- `symbol_bins=6`，`symbol_strategy=uniform`，`bin_scope=per_epoch`
- `transition_steps=[1, 2, 3]`，`transition_weight=probability`
- `transition_multichannel=mean`，保留自转移
- 输入 `[3, 6, 6]`。CNN 只用两层 `[32, 64]`，三次池化会把 6×6 压成 0
- 注意力 SE。产物在 `datas/experiments_3way/transition/`，与 RP 冻结权重分开

旧两池配置 `config/eeg_ws_transition.yaml` 不是这条协议，不要拿它的结果和三集比。

统一节点编号用 `config/eeg_ws_transition_3way_global.yaml`：`bin_scope=train_global`，按导堆叠，`normalize=none`。分档边界只从该人训练集计算。产物不覆盖按试次分档的目录。test 已评，数字在 `doc/results.md`。

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

验收对照的数字只加在 `doc/results.md`，不另开对照文件。`master` 是 CNN 分支，只收这份对照。GCN 实现留在 `feat/transition-gcn`，不合并代码。

统一节点编号已评完，数字在 `doc/results.md`。配置在 `feat/transition-gcn` 的 `config/eeg_ws_transition_3way_global.yaml`。

- CNN：矩阵进卷积网络。注意力是开关。优化参数是 `lr=1e-3`、`batch_size=64`，配置在 GCN 分支的 `config/eeg_ws_transition_3way_global_none_opt.yaml` 和 `config/eeg_ws_transition_3way_global_self_opt.yaml`。14 人上这次扫描没有超过现有参数。数字在 `doc/results.md`。
- GCN：和 CNN 互斥。分支 `feat/transition-gcn`。不加图注意力与加图注意力都已评 test。没有扫学习率。数字在 `doc/results.md`。

手选网络指标加分类器仍是单独一行对照，不接到这两条路上。验证集定下后评一次 test。数字只并进 `doc/results.md`。
