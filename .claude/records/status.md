# 当前状态

分支：`feat/transition-gcn`。主干仍是 `master`。旧 RP/MRP 线：标签 `rp-mrp-baseline`。

主线、协议、命令：`doc/method.md`。数字：`doc/results.md`。

## 待办

- [x] 冒烟
- [x] 正式 14 人 val，test 封闭
- [x] 同一划分上扫节点数和各导平均/堆叠，只看 val
- [x] `bins=6`、按导堆叠的 test 已评。见 `doc/results.md`
- [x] 新方案在验证集定下后评 test。见 `doc/method.md`
- [x] `master` 上完成必选：统一节点编号。见 `doc/results.md`
- [x] CNN 自注意力开关已评 test。见 `doc/results.md`
- [x] GCN 不加注意力，三种子 val 已齐，不重跑。test 未开。见 `doc/results.md`
- [x] GCN 不加注意力和加图注意力都已评 test。见 `doc/results.md`
- [ ] 回到 CNN。先优化参数，再跑加注意力和不加注意力。现有参数的「不加」val 已记，见 `doc/results.md`，不算优化后的那一轮
- [ ] 把 GCN 文档合进 `master`。综合结果，review 代码，实验数据和结论只记在 `doc/results.md`
- [ ] 手选指标仍是对照行
- [x] commit