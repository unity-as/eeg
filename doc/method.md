# 方法

数字只写在 `doc/results.md`。配置键只写在 `doc/config.md`。老师原话整理在 `doc/teacher_improvement_directions.md`。

本分支只做旋转机械故障诊断。脑电四分类不在这里跑，代码和数字在标签 `eeg-cnn`、`eeg-gcn`、`rp-mrp-baseline`。

## 现在做什么

东南大学 DDS：轴承一套模型，齿轮一套模型。幅值符号化 → 多时间步有向加权转移矩阵 → CNN 或 GCN（`model.arch`）。五类 Softmax，交叉熵训练。

## 协议

- 数据：东南大学 Drivetrain Dynamic Simulator。原始文件在 `datas/seu/`
- 两套任务，互不拿去验对方：轴承五类（health / ball / inner / outer / comb），齿轮五类（Health / Chipped / Miss / Root / Surface）
- 工况 `20_0`、`30_2` 分训分测，再报两个准确率的平均
- 通道：行星齿轮箱 2、3、4（x/y/z）。不用转矩和其余振动
- 窗：800 点，步长 800，不重叠。先按时间 70/15/15 切开，再切窗
- 分档：`bin_scope=train_global`，只用该任务该工况的训练集
- 转移：步长 `[1,2,3]`，概率权重，三通道堆叠，输入 `[9,6,6]`
- 训练种子默认 42。按 `val_acc` 早停
- test 默认封闭。评测须同时加 `--evaluate-test --confirm-test`
- 四条对照：CNN 关注意力 / CNN 自注意力 / GCN 关图注意力 / GCN 开图注意力。表示和划分冻第一版。各自扫学习率 `{1e-4, 5e-4, 1e-3}` 和 dropout `{0.1, 0.2, 0.3}`。选参看两工况 val 平均。扫参目录不覆盖第一版权重。

## 命令

```bash
python scripts/experiments/build_seu_dataset.py --config config/seu_smoke_bearing_cnn.yaml --force
python scripts/experiments/train_seu.py --config config/seu_smoke_bearing_cnn.yaml

python scripts/experiments/build_seu_dataset.py --config config/seu_bearing_cnn.yaml
python scripts/experiments/train_seu.py --config config/seu_bearing_cnn.yaml
python scripts/experiments/train_seu.py --config config/seu_bearing_gcn.yaml

python scripts/experiments/build_seu_dataset.py --config config/seu_gear_cnn.yaml
python scripts/experiments/train_seu.py --config config/seu_gear_cnn.yaml
python scripts/experiments/train_seu.py --config config/seu_gear_gcn.yaml

# 已有权重出验证集混淆矩阵和 t-SNE，不打开测试集
python scripts/experiments/plot_seu_val.py

# 四条线扫 lr / dropout，只看 val
python scripts/experiments/sweep_seu.py
```

训练结束后也会把图写到 `doc/figures/seu/`。t-SNE 用分类头之前的向量。
