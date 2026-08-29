"""经典递归图（论文 1/2 共用表示；阶段 R）。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .phase_space import build_phase_space


def _pairwise_distances(traj: np.ndarray) -> np.ndarray:
    # ||a-b||^2 = |a|^2 + |b|^2 - 2 a·b
    sq = np.sum(traj * traj, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (traj @ traj.T), 0.0)
    return np.sqrt(d2)


def build_recurrence_plot(
    signal: np.ndarray,
    m: int = 3,
    tau: int = 1,
    epsilon: Optional[float] = None,
    recurrence_percentile: float = 0.1,
    image_size: Optional[int] = 64,
) -> np.ndarray:
    traj = build_phase_space(signal, m=m, tau=tau)
    dist = _pairwise_distances(traj)
    if epsilon is None:
        # 上三角（不含对角）分位数作阈值
        iu = np.triu_indices_from(dist, k=1)
        eps = float(np.quantile(dist[iu], recurrence_percentile))
    else:
        eps = float(epsilon)
    rp = (dist <= eps).astype(np.float32)
    if image_size is not None and (rp.shape[0] != image_size or rp.shape[1] != image_size):
        img = Image.fromarray((rp * 255.0).astype(np.uint8), mode="L")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        rp = np.asarray(img, dtype=np.float32) / 255.0
    return rp


def build_representation(signal: np.ndarray, method_cfg) -> Tuple[np.ndarray, dict]:
    """按配置构建二维表示；mrp 阶段未实现则报错。"""
    kind = str(method_cfg.representation).lower()
    meta = {}

    if kind == "rp":
        eps = method_cfg.get("epsilon", None)
        rp = build_recurrence_plot(
            signal,
            m=int(method_cfg.embedding_dim),
            tau=int(method_cfg.time_delay),
            epsilon=None if eps in (None, "null") else float(eps),
            recurrence_percentile=float(method_cfg.recurrence_percentile),
            image_size=int(method_cfg.rp_image_size),
        )
    elif kind == "mrp":
        from .modified_rp import build_modified_rp

        rp = build_modified_rp(signal, method_cfg)
    else:
        raise ValueError(f"未知 representation: {kind}")

    if bool(method_cfg.get("quality_assess", False)):
        from .quality import assess_rp_quality

        meta["quality"] = assess_rp_quality(rp)
    return rp, meta
