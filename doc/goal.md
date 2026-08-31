# 项目目标（2026-08-31 重定）

## 一句话

在 **单个被试内部**，用三篇论文那套 **RP/MRP + CNN（可切换注意力）**，把 ds002680 的 **Go-nogo 四分类** 做到可复现、且 test 明显高于随机。

## 用什么数据

| 项 | 定值 |
|----|------|
| 数据集 | OpenNeuro **ds002680**（`datas/ds002680/`） |
| 用法 | **一次只用一个人**，由 `data.subject` 指定（默认 `sub-002`；命令行可覆盖） |
| 标签 | 仅 `trial_type=stimulus`：cat_go / cat_nogo / rec_go / rec_nogo（4 类） |
| 样本 | 刺激 onset 起 **512** 点 @ 1000 Hz（窗长扫描后锁定）；每导单独成图后叠成 `[31, 64, 64]` |
| 划分 | **个体内**：`train_ratio` 切出训练；其余为合法池。报告准确率从不放回抽 `eval_samples` 窗（null=用尽），并出混淆矩阵 |
| 不用 | 跨人 LOSO、多人混训、振动五分类当主任务、Bonn/CHB 癫痫库（除非以后单独立项复现论文） |

盘上可以留着 14 个被试；默认不把他们拼成一个跨人问题。若对多人各训各测，只按人列表，不报跨人准确率。

## 通过什么方法

配置切换，不拆成三个仓库。

- **阶段 R（当前主路径）**：经典递归图 + SE 注意力 CNN（`representation=rp`，`attention=se`）
  - **对齐论文1** Shankar 2021 BSPC：EEG→RP→CNN；本配置**未开**其节律滤波与 RP 质量评估
  - **对齐论文2** Hao et al. 2021 AIP：RP + CNN，并用 SE 一类通道注意力
- **阶段 M（对照）**：改进递归图 + 加性注意力（`representation=mrp`，`attention=additive`）
  - **对齐论文3** Huang 2023 BSPC（MRP-Net）：modified RP + additive attention CNN
- 相空间：sub-002 @512 ms 锁定 **m=3、τ=12**（AMI/FNN 中位数，`embed_select=fixed`）；要重估则改 `auto`
- ε 默认仍为距离 10% 分位；可改为论文1 的 kσ 或直径比例
- 各导 **z-score** 后再成图（论文1，保持开）。相对无归一化的工程默认，R 在 τ=12 时约 0.803→0.749；不关 z-score 冲分，后续改 ε / 训练适配新尺度
- 图边长 64
- 训练：Adam，`weight_decay=1e-4`，分类头 Dropout 0.3；增强由 `train.augment` 开关，默认关
- 论文 1 的节律滤波、RP 质量评估：已接入流水线，默认关（Go-nogo 不宜带通到 δ）

不在本阶段把主输入改成 Conv1d；若个体 test 仍贴随机，再另开对比。

## 目标是什么

**要证明的：** 这套二维递归图 + CNN，在 **同一个人、没见过的窗** 上，能否学到 Go-nogo 四分类，而不是只背训练集。

**成功线（个体 test）：** 稳定高于 4 类随机 **0.25**。先不设论文式 90%；通路能复现，正式再看能否到 0.4 / 0.6 档。

**不作为目标：**

- 跨人泛化、零校准
- 对齐论文 Bonn/SWEC 的 93–100%
- 临床癫痫检测产品

对外只写：`subject=…, within-subject, test=…`。
