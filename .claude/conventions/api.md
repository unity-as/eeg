# api.md — 模块接口约定

## 配置

- 使用 OmegaConf 加载 `config/default.yaml`
- 方法开关以 `method.*` 为准；未实现的组合应明确报错，禁止静默回退到错误路径

## representation

- `build_phase_space(signal, m, tau) -> ndarray [T', m]`
- `build_recurrence_plot(signal, m, tau, epsilon, ...) -> ndarray [H, W] float32 in [0,1]`
- `build_modified_rp(...)`：阶段 M 实现前调用即 `NotImplementedError`

## models

- `build_model(num_classes, cfg) -> nn.Module`
- 输入：`[B, 1, H, W]` RP 图；输出 logits `[B, C]`

## data

- 类别关键字匹配文件名（小写包含）：与旋转机械一致 `normal|inner|outer|combine|roll`
- `collect_windows_per_class(...)` 返回 `{class: [1d arrays]}`
