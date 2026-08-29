"""相空间重构（Takens）。"""
from __future__ import annotations

import numpy as np


def build_phase_space(signal: np.ndarray, m: int, tau: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = x.shape[0]
    length = n - (m - 1) * tau
    if length <= 1:
        raise ValueError(f"信号过短无法嵌入: n={n}, m={m}, tau={tau}")
    traj = np.empty((length, m), dtype=np.float64)
    for i in range(m):
        traj[:, i] = x[i * tau : i * tau + length]
    return traj
