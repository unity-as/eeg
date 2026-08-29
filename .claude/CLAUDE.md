# EEG — 递归图 + CNN 时序分类（论文 1/2/3 可配置组合）

## 项目背景

基于三篇 RP/MRP + CNN（可加注意力）的 EEG 癫痫相关论文，在**单一工程**内用配置切换方法变体。当前优先用旋转机械振动 `raw_datas` 做五分类工程验证，再分阶段补论文向对齐。代码流水线对照 `D:\project\Python\rotating_machinery_diagnosis`；`.claude/` 职责对照 `infantry_main/.claude`。

## 目录导航（.claude/）

| 目录 | 用途 |
|------|------|
| `conventions/` | 本项目规则与文件结构事实 |
| `docs/` | 设计决定、架构、方案结论 |
| `records/` | `status.md` 进度；`history.md` changelog |
| `reference/` | 论文摘录等外部资料 |
| `plans/` | AI 计划输出 |
| `skills/` | 项目技能（按需添加） |

## 方法配置（摘要）

详见 `config/default.yaml` 与 `.claude/docs/architecture.md`。

- **阶段 R（当前）**：`representation=rp`，`attention=se`（或 `none`），`rhythm_filter=false`
- **阶段 M（后续）**：`representation=mrp`，`attention=additive`
- 论文 1 选参/质量评估、节律滤波：模块可插拔，默认关

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
