"""节律带通（论文1：δ/θ/α/β）。默认关；Go-nogo 主路径不要开。"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _band_hz(band: Union[str, Tuple[float, float], None]) -> Tuple[float, float]:
    if band is None:
        return DEFAULT_BANDS["delta"]
    if isinstance(band, str):
        key = band.lower()
        if key not in DEFAULT_BANDS:
            raise ValueError(f"未知 rhythm_band: {band}，可选 {list(DEFAULT_BANDS)}")
        return DEFAULT_BANDS[key]
    low, high = float(band[0]), float(band[1])
    return low, high


def apply_rhythm_filter(
    signal: np.ndarray,
    enabled: bool,
    fs: Optional[float] = None,
    band: Union[str, Tuple[float, float], None] = None,
) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64)
    if not enabled:
        return arr
    if fs is None:
        raise ValueError("rhythm_filter=true 时需要 sampling_rate / fs")
    low, high = _band_hz(band)
    from mne.filter import filter_data

    squeezed = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeezed = True
    y = filter_data(
        arr.astype(np.float64),
        sfreq=float(fs),
        l_freq=low,
        h_freq=high,
        verbose=False,
    )
    if squeezed:
        y = y[0]
    return y.astype(np.float64)
