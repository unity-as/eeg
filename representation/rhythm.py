"""节律/频带滤波（论文1）；振动任务默认关闭。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# 名义采样率仅在显式开启 rhythm_filter 且提供 bands 时使用
DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def apply_rhythm_filter(
    signal: np.ndarray,
    enabled: bool,
    fs: Optional[float] = None,
    band: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    if not enabled:
        return np.asarray(signal, dtype=np.float32)
    if fs is None or band is None:
        raise ValueError("rhythm_filter=true 时需要提供 fs 与 band=(low, high)")
    # 简单 FFT 带通，避免强依赖 scipy（有则更好，这里纯 numpy）
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    spec = np.fft.rfft(x)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    spec[~mask] = 0
    y = np.fft.irfft(spec, n=n)
    return y.astype(np.float32)
