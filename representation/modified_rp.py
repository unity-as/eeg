"""改进递归图（论文3 / 阶段 M）。

工程实现说明（摘录未给全公式时的可运行近似）：
- 经典 RP：二值 Θ(ε - d_ij)
- 本 MRP：软递归图 R_ij = exp(-d_ij / ε)，保留灰度动力学信息，再缩放到固定边长
- 多通道：对各通道分别算 MRP 后取平均（振动当前为单通道）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .phase_space import build_phase_space


def _pairwise_distances(traj: np.ndarray) -> np.ndarray:
    sq = np.sum(traj * traj, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (traj @ traj.T), 0.0)
    return np.sqrt(d2)


def _soft_mrp_from_1d(
    signal: np.ndarray,
    m: int,
    tau: int,
    epsilon: Optional[float],
    recurrence_percentile: float,
    image_size: Optional[int],
) -> np.ndarray:
    traj = build_phase_space(signal, m=m, tau=tau)
    dist = _pairwise_distances(traj)
    if epsilon is None:
        iu = np.triu_indices_from(dist, k=1)
        eps = float(np.quantile(dist[iu], recurrence_percentile))
    else:
        eps = float(epsilon)
    eps = max(eps, 1e-8)
    mrp = np.exp(-dist / eps).astype(np.float32)
    if image_size is not None and (mrp.shape[0] != image_size or mrp.shape[1] != image_size):
        img = Image.fromarray((np.clip(mrp, 0, 1) * 255.0).astype(np.uint8), mode="L")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        mrp = np.asarray(img, dtype=np.float32) / 255.0
    return mrp


def build_modified_rp(signal, method_cfg) -> np.ndarray:
    m = int(method_cfg.embedding_dim)
    tau = int(method_cfg.time_delay)
    eps = method_cfg.get("epsilon", None)
    if eps in (None, "null"):
        eps = None
    else:
        eps = float(eps)
    pct = float(method_cfg.recurrence_percentile)
    image_size = int(method_cfg.rp_image_size)

    arr = np.asarray(signal, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return _soft_mrp_from_1d(arr, m, tau, eps, pct, image_size)
    if arr.ndim == 2:
        # [C, T] 多通道：各通道 MRP 平均（要求 C 为通道维且 C>1）
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        maps = [
            _soft_mrp_from_1d(arr[c], m, tau, eps, pct, image_size)
            for c in range(arr.shape[0])
        ]
        return np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    raise ValueError(f"MRP 仅支持 1D 或 2D 信号，收到 shape={np.asarray(signal).shape}")
