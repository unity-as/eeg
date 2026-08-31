# api.md — 模块接口约定

## 配置

- 使用 OmegaConf 加载 yaml（无参默认 `config/eeg_ws_r.yaml`）
- 方法开关以 `method.*` 为准；未实现的组合应明确报错，禁止静默回退到错误路径

## representation

- `build_phase_space(signal, m, tau) -> ndarray [T', m]`（多导 `[T', C*m]`）
- `build_recurrence_plot(...)`：Heaviside RP
- `build_modified_rp(...)`：灰度 MRP；多导 `joint` 或 `per_channel`
- `resolve_embed_params`：AMI + FNN

## models

- `build_model(num_classes, cfg) -> nn.Module`
- 输入：`[B, C, H, W]` RP 图（多通道 EEG 时 C=导联数）；输出 logits `[B, num_classes]`

## data

- `data.eeg_ds002680`：BIDS epoch，4 类 `cat_go | cat_nogo | rec_go | rec_nogo`
- 按被试打包 `X_<subject>.npy` / `y_<subject>.npy`
