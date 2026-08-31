"""改进递归图（论文3 / 阶段 M）。

相对经典 RP 的改动：不用 Heaviside 二值化，保留灰度
R_ij = exp(-d_ij / ε)。

多通道：
- joint：各导在同一时刻组成状态向量再嵌入（论文3 多导）
- per_channel：每导一张图再叠通道（与阶段 R 同形）
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .phase_space import build_phase_space
from .rp_core import pairwise_distances, resolve_epsilon, resize_square


def _soft_mrp(traj: np.ndarray, method_cfg, image_size: Optional[int]) -> np.ndarray:
    dist = pairwise_distances(traj)
    eps_abs = method_cfg.get("epsilon", None)
    if eps_abs in (None, "null"):
        eps_abs = None
    else:
        eps_abs = float(eps_abs)
    eps = resolve_epsilon(
        dist,
        epsilon=eps_abs,
        mode=str(method_cfg.get("epsilon_mode", "percentile")),
        percentile=float(method_cfg.get("recurrence_percentile", 0.1)),
        std_k=float(method_cfg.get("epsilon_std_k", 0.25)),
        diameter_frac=float(method_cfg.get("diameter_frac", 0.1)),
    )
    mrp = np.exp(-dist / eps).astype(np.float32)
    return resize_square(mrp, image_size)


def build_modified_rp(signal, method_cfg) -> np.ndarray:
    m = int(method_cfg.embedding_dim)
    tau = int(method_cfg.time_delay)
    image_size = int(method_cfg.rp_image_size)
    arr = np.asarray(signal, dtype=np.float64)
    arr = np.squeeze(arr)
    how = str(method_cfg.get("mrp_multichannel", "joint")).lower()
    if arr.ndim == 1:
        traj = build_phase_space(arr, m=m, tau=tau)
        return _soft_mrp(traj, method_cfg, image_size)
    if arr.ndim != 2:
        raise ValueError(f"MRP 仅支持 1D 或 2D，收到 shape={arr.shape}")
    if how in ("per_channel", "stack", "channel"):
        maps = [
            _soft_mrp(build_phase_space(arr[c], m=m, tau=tau), method_cfg, image_size)
            for c in range(arr.shape[0])
        ]
        return np.stack(maps, axis=0).astype(np.float32)
    traj = build_phase_space(arr, m=m, tau=tau)
    return _soft_mrp(traj, method_cfg, image_size)
