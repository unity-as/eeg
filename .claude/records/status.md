# 当前状态

- **定位**：个体内 Go-nogo 四分类；方法为可配置 RP/MRP+CNN（见 `doc/goal.md`）
- **数据**：`datas/ds002680/`（14 被试；当前批次 `sub-002`）
- **窗长**：已锁定 **`epoch_samples=512`**
- **相空间**：`embed_select=fixed`，m=3，**τ=12**；**z-score 保持开**（论文1，不关）；M 多导 `joint`；节律默认关
- 旋转机械残余已删；无参入口 `config/eeg_ws_r.yaml`

## 2-Fold LOSO（已结束，不作主线）

- 正式 R 均值 0.209；报告：`doc/individual_focus_report.md`

## 当前：个体内 sub-002，512 ms，1500 窗

工程默认（τ=1，无 z-score，M 每导一张）：R **0.803** / M **0.734**（对照，不改回）。

论文向（z-score；M=joint）：

| | τ=12 | τ=1 |
|--|--|--|
| R | **0.749** | 0.718 |
| M | **0.552** | 0.475 |

开 z-score 后 R 低于工程默认约 5 点；不关。尺度变了，后面在 **z-score 仍开** 的前提下改 ε / 训练。M 掉分主因 joint，同样只记录。  
数字：`doc/within_subject_sub002_results.md`。

## 待办

- [ ] 在 z-score + τ=12 下改其它参数（先 ε：`std` / 分位；再 `lr`）
- [ ] 缓存/权重路径带上 τ，避免对照覆盖
- [ ] 需要时换 `sub-003`
- [ ] commit
