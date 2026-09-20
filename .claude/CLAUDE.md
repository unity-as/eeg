# EEG — 状态转移网络 + CNN

## 项目背景

OpenNeuro **ds002680** 个体内 Go-nogo 四分类。当前主线是老师的状态转移矩阵 + CNN；RP/MRP 是 baseline。人读文档只维护 `doc/method.md`、`doc/results.md`、`doc/config.md`，不要把同一件事再写进新文件。

## 目录导航（.claude/）

| 目录 | 用途 |
|------|------|
| `conventions/` | 本项目规则与文件结构事实 |
| `docs/` | 设计决定、架构、方案结论 |
| `records/` | `status.md` 进度；`history.md` changelog |
| `reference/` | 论文摘录等外部资料 |
| `plans/` | AI 计划输出 |
| `skills/` | 项目技能（按需添加） |

## 方法

见 `doc/method.md`。查进度看 `.claude/records/status.md`，不要在这里抄数字。

## 分支

`master` 就是当前这套框架：状态转移为主线，RP/MRP 为 baseline。不要在 `master` 上换成另一套方案。

多方案各开分支。旧的 RP/MRP 论文对齐线在标签 `rp-mrp-baseline`。

## 红线

- 未经用户允许：勿改 `.claude/conventions/`、`.claude/skills/`
- 未明确维护者的文件：改前征求同意
- 含糊需求：先追问对齐，再动手（用户说「直接做 / 别问了 / 按你判断执行」除外）

## 事件 → 检查文件

| 事件 | 检查 |
|------|------|
| 新建/删除/移动项目文件 | `.claude/conventions/file.md`（并更新结构） |
| 查论文要点 | `.claude/reference/papers_rp_eeg.md` |
| 查进度 | `.claude/records/status.md` |
| 完成一阶段 | 更新 `status.md` + 追加 `history.md` |
