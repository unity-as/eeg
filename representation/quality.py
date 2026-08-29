"""论文1 风格的 RP 质量评估（可选）。"""
from __future__ import annotations

import numpy as np


def assess_rp_quality(rp: np.ndarray) -> dict:
    x = np.asarray(rp, dtype=np.float64)
    flat = x.ravel()
    # 二值/灰度熵
    hist, _ = np.histogram(flat, bins=32, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    entropy = float(-np.sum(hist * np.log(hist)))
    rms = float(np.sqrt(np.mean(flat ** 2)))
    std = float(np.std(flat))
    mean = float(np.mean(flat))
    skew = float(np.mean(((flat - mean) / (std + 1e-12)) ** 3))
    return {
        "entropy": entropy,
        "rms": rms,
        "std": std,
        "skewness": skew,
    }
