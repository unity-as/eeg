"""经典递归图（论文 1/2 共用表示；阶段 R）。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .phase_space import build_phase_space
from .rp_core import pairwise_distances, resolve_epsilon, resize_square, zscore_channels
from .rhythm import apply_rhythm_filter


def build_recurrence_plot(
    signal: np.ndarray,
    m: int = 3,
    tau: int = 1,
    epsilon: Optional[float] = None,
    recurrence_percentile: float = 0.1,
    image_size: Optional[int] = 64,
    epsilon_mode: str = "percentile",
    std_k: float = 0.25,
    diameter_frac: float = 0.1,
) -> np.ndarray:
    traj = build_phase_space(signal, m=m, tau=tau)
    dist = pairwise_distances(traj)
    eps = resolve_epsilon(
        dist,
        epsilon=epsilon,
        mode=epsilon_mode,
        percentile=recurrence_percentile,
        std_k=std_k,
        diameter_frac=diameter_frac,
    )
    rp = (dist <= eps).astype(np.float32)
    return resize_square(rp, image_size)


def _preprocess(arr: np.ndarray, method_cfg) -> np.ndarray:
    fs = method_cfg.get("sampling_rate", None)
    if fs is None:
        fs = 1000.0
    arr = apply_rhythm_filter(
        arr,
        enabled=bool(method_cfg.get("rhythm_filter", False)),
        fs=float(fs),
        band=method_cfg.get("rhythm_band", "delta"),
    )
    if str(method_cfg.get("normalize", "zscore")).lower() in ("zscore", "z", "std"):
        arr = zscore_channels(arr)
    return arr


def _build_1d(signal: np.ndarray, method_cfg) -> np.ndarray:
    kind = str(method_cfg.representation).lower()
    if kind == "rp":
        eps = method_cfg.get("epsilon", None)
        return build_recurrence_plot(
            signal,
            m=int(method_cfg.embedding_dim),
            tau=int(method_cfg.time_delay),
            epsilon=None if eps in (None, "null") else float(eps),
            recurrence_percentile=float(method_cfg.get("recurrence_percentile", 0.1)),
            image_size=int(method_cfg.rp_image_size),
            epsilon_mode=str(method_cfg.get("epsilon_mode", "percentile")),
            std_k=float(method_cfg.get("epsilon_std_k", 0.25)),
            diameter_frac=float(method_cfg.get("diameter_frac", 0.1)),
        )
    if kind == "mrp":
        from .modified_rp import build_modified_rp

        return build_modified_rp(signal, method_cfg)
    raise ValueError(f"未知 representation: {kind}")


def build_representation(signal: np.ndarray, method_cfg) -> Tuple[np.ndarray, dict]:
    """1D → [H,W]；多通道 RP → [C,H,W]；多通道 MRP-joint → [H,W]。"""
    arr = np.asarray(signal, dtype=np.float64)
    arr = np.squeeze(arr)
    arr = _preprocess(arr, method_cfg)
    meta = {}
    kind = str(method_cfg.representation).lower()
    if arr.ndim == 1:
        rp = _build_1d(arr, method_cfg)
    elif arr.ndim == 2:
        if kind == "mrp":
            rp = _build_1d(arr, method_cfg)
        else:
            maps = [_build_1d(arr[c], method_cfg) for c in range(arr.shape[0])]
            rp = np.stack(maps, axis=0).astype(np.float32)
    else:
        raise ValueError(f"表示输入须为 1D 或 [C,T]，收到 {arr.shape}")

    if bool(method_cfg.get("quality_assess", False)):
        from .quality import assess_rp_quality

        meta["quality"] = assess_rp_quality(rp if rp.ndim == 2 else np.mean(rp, axis=0))
    return rp, meta
