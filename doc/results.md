# 实验结果

方法见 `doc/method.md`。本分支只记旋转机械。脑电数字在标签 `eeg-cnn` / `eeg-gcn`。

第一版：种子 42，测试集未开。每种工况单独训。平均是两个工况验证准确率的算术平均。

| 任务 | 方法 | 20-0 val | 30-2 val | 平均 |
|---|---|---:|---:|---:|
| 轴承 | CNN | 0.6857 | 0.9776 | 0.8316 |
| 轴承 | GCN | 0.7296 | 0.9898 | 0.8597 |
| 齿轮 | CNN | 0.9194 | 0.8582 | 0.8888 |
| 齿轮 | GCN | 0.9184 | 0.8194 | 0.8689 |

轴承 20-0 明显低于 30-2。测试集还没评。

论文图在 `doc/figures/seu/`（300 dpi），打包 `doc/figures/seu_val_figures.zip`。文件名是 `任务_模型_工况_图种.png`：`cnn` 是不加注意力的 CNN，`gcn` 是不加图注意力的 GCN。混淆矩阵来自验证集预测；t-SNE 是分类头之前的向量。每类验证 196 窗。行是真值，列是预测。

重新出图：`python scripts/experiments/plot_seu_val.py`。不打开测试集。

## 轴承 CNN

20-0 `bearing_cnn_20_0_{confusion,tsne}.png`

|  | health | ball | inner | outer | comb |
|---|---:|---:|---:|---:|---:|
| health | 149 | 44 | 1 | 0 | 2 |
| ball | 63 | 99 | 1 | 2 | 31 |
| inner | 11 | 0 | 185 | 0 | 0 |
| outer | 1 | 6 | 0 | 163 | 26 |
| comb | 15 | 50 | 0 | 55 | 76 |

30-2 `bearing_cnn_30_2_{confusion,tsne}.png`

|  | health | ball | inner | outer | comb |
|---|---:|---:|---:|---:|---:|
| health | 196 | 0 | 0 | 0 | 0 |
| ball | 1 | 192 | 0 | 2 | 1 |
| inner | 0 | 0 | 196 | 0 | 0 |
| outer | 2 | 10 | 0 | 184 | 0 |
| comb | 6 | 0 | 0 | 0 | 190 |

## 轴承 GCN

20-0 `bearing_gcn_20_0_{confusion,tsne}.png`

|  | health | ball | inner | outer | comb |
|---|---:|---:|---:|---:|---:|
| health | 137 | 50 | 3 | 0 | 6 |
| ball | 29 | 122 | 6 | 3 | 36 |
| inner | 5 | 0 | 191 | 0 | 0 |
| outer | 1 | 3 | 0 | 167 | 25 |
| comb | 8 | 32 | 1 | 57 | 98 |

30-2 `bearing_gcn_30_2_{confusion,tsne}.png`

|  | health | ball | inner | outer | comb |
|---|---:|---:|---:|---:|---:|
| health | 194 | 0 | 0 | 0 | 2 |
| ball | 0 | 193 | 0 | 3 | 0 |
| inner | 0 | 0 | 196 | 0 | 0 |
| outer | 0 | 5 | 0 | 191 | 0 |
| comb | 0 | 0 | 0 | 0 | 196 |

## 齿轮 CNN

20-0 `gear_cnn_20_0_{confusion,tsne}.png`

|  | Health | Chipped | Miss | Root | Surface |
|---|---:|---:|---:|---:|---:|
| Health | 196 | 0 | 0 | 0 | 0 |
| Chipped | 0 | 195 | 1 | 0 | 0 |
| Miss | 0 | 7 | 173 | 5 | 11 |
| Root | 1 | 0 | 5 | 189 | 1 |
| Surface | 0 | 0 | 36 | 12 | 148 |

30-2 `gear_cnn_30_2_{confusion,tsne}.png`

|  | Health | Chipped | Miss | Root | Surface |
|---|---:|---:|---:|---:|---:|
| Health | 196 | 0 | 0 | 0 | 0 |
| Chipped | 0 | 168 | 14 | 4 | 10 |
| Miss | 0 | 12 | 138 | 9 | 37 |
| Root | 1 | 1 | 4 | 175 | 15 |
| Surface | 0 | 10 | 17 | 5 | 164 |

## 齿轮 GCN

20-0 `gear_gcn_20_0_{confusion,tsne}.png`

|  | Health | Chipped | Miss | Root | Surface |
|---|---:|---:|---:|---:|---:|
| Health | 196 | 0 | 0 | 0 | 0 |
| Chipped | 0 | 194 | 2 | 0 | 0 |
| Miss | 0 | 5 | 166 | 6 | 19 |
| Root | 0 | 0 | 5 | 184 | 7 |
| Surface | 0 | 0 | 28 | 8 | 160 |

30-2 `gear_gcn_30_2_{confusion,tsne}.png`

|  | Health | Chipped | Miss | Root | Surface |
|---|---:|---:|---:|---:|---:|
| Health | 196 | 0 | 0 | 0 | 0 |
| Chipped | 0 | 155 | 27 | 4 | 10 |
| Miss | 0 | 14 | 139 | 21 | 22 |
| Root | 1 | 1 | 7 | 178 | 9 |
| Surface | 0 | 20 | 28 | 13 | 135 |
