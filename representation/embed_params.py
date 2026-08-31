"""论文1/3：互信息求 τ，假近邻求 m。"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .phase_space import build_phase_space
from .rp_core import zscore_1d


def average_mutual_information(x: np.ndarray, max_lag: int, n_bins: int = 16) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    max_lag = int(max(1, max_lag))
    out = np.empty(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        a, b = x[:-lag], x[lag:]
        h2, _, _ = np.histogram2d(a, b, bins=n_bins)
        pxy = h2 / max(h2.sum(), 1.0)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        nz = pxy > 0
        den = px * py
        out[lag - 1] = float(np.sum(pxy[nz] * np.log(pxy[nz] / np.maximum(den[nz], 1e-12))))
    return out


def first_ami_minimum(ami: np.ndarray) -> int:
    """第一个局部最小对应的滞后（从 1 计）。没有谷则取全局最小。"""
    a = np.asarray(ami, dtype=np.float64)
    if a.size == 0:
        return 1
    for i in range(1, a.size - 1):
        if a[i] <= a[i - 1] and a[i] <= a[i + 1]:
            return i + 1
    return int(np.argmin(a)) + 1


def false_nearest_neighbor_ratio(
    x: np.ndarray,
    tau: int,
    m: int,
    rtol: float = 15.0,
    atol: Optional[float] = None,
) -> float:
    """Kennel 假近邻比例。m 维最近邻在 m+1 维变得很远则计假。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.shape[0]
    tau = max(int(tau), 1)
    m = max(int(m), 1)
    if n - m * tau < 8:
        return 1.0
    ym = build_phase_space(x, m=m, tau=tau)
    ym1 = build_phase_space(x, m=m + 1, tau=tau)
    n_use = min(len(ym), len(ym1))
    ym, ym1 = ym[:n_use], ym1[:n_use]
    if n_use < 4:
        return 1.0
    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree")
    nn.fit(ym)
    dist, idx = nn.kneighbors(ym, n_neighbors=2)
    d_m = dist[:, 1]
    j = idx[:, 1]
    d_extra = np.abs(ym1[:, -1] - ym1[j, -1])
    ok = d_m > 1e-12
    ra = np.zeros(n_use, dtype=np.float64)
    ra[ok] = d_extra[ok] / d_m[ok]
    false = ra > float(rtol)
    if atol is not None:
        d_m1 = np.sqrt(d_m * d_m + d_extra * d_extra)
        false = false | (d_m1 > float(atol))
    return float(np.mean(false))


def estimate_m_tau(
    x: np.ndarray,
    max_tau: int = 40,
    max_m: int = 8,
    fnn_thresh: float = 0.1,
) -> Tuple[int, int]:
    x = zscore_1d(np.asarray(x, dtype=np.float64).ravel())
    n = x.shape[0]
    if n < 16:
        return 2, 1
    max_tau = int(max(1, min(max_tau, n // 8)))
    ami = average_mutual_information(x, max_tau)
    tau = max(1, min(first_ami_minimum(ami), max_tau))
    chosen_m = max_m
    for m in range(1, max_m + 1):
        if n - m * tau < 16:
            chosen_m = max(2, m - 1)
            break
        ratio = false_nearest_neighbor_ratio(x, tau, m)
        if ratio < fnn_thresh:
            chosen_m = max(2, m)
            break
    return int(chosen_m), int(tau)


def _channel_traces(sig: np.ndarray, max_channels: int) -> List[np.ndarray]:
    a = np.asarray(sig, dtype=np.float64)
    a = np.squeeze(a)
    if a.ndim == 1:
        return [a]
    c = a.shape[0]
    if c <= max_channels:
        idx = range(c)
    else:
        idx = np.linspace(0, c - 1, max_channels).astype(int)
    return [a[i] for i in idx]


def estimate_from_signals(
    signals: Sequence[np.ndarray],
    n_epochs: int = 16,
    max_channels: int = 8,
    max_tau: int = 40,
    max_m: int = 8,
) -> Tuple[int, int, dict]:
    """从若干 epoch、若干导联取中位数 m、τ（确定性等距抽样）。"""
    n = len(signals)
    if n == 0:
        return 3, 1, {"reason": "empty", "n_traces": 0}
    n_use = min(int(n_epochs), n)
    eidx = np.linspace(0, n - 1, n_use).astype(int)
    ms, taus = [], []
    for ei in eidx:
        for tr in _channel_traces(signals[int(ei)], max_channels):
            m, tau = estimate_m_tau(tr, max_tau=max_tau, max_m=max_m)
            ms.append(m)
            taus.append(tau)
    m = int(np.median(ms)) if ms else 3
    tau = int(np.median(taus)) if taus else 1
    m = max(2, m)
    tau = max(1, tau)
    info = {
        "n_epochs": int(n_use),
        "n_traces": int(len(ms)),
        "m_median": m,
        "tau_median": tau,
        "m_samples": [int(v) for v in ms],
        "tau_samples": [int(v) for v in taus],
    }
    return m, tau, info


def resolve_embed_params(signals: Sequence[np.ndarray], method) -> Tuple[int, int, dict]:
    kind = str(method.get("embed_select", "auto")).lower()
    fallback_m = int(method.get("embedding_dim", 3) or 3)
    fallback_tau = int(method.get("time_delay", 1) or 1)
    if kind in ("fixed", "manual", "off"):
        return fallback_m, fallback_tau, {"mode": "fixed"}
    m, tau, info = estimate_from_signals(signals)
    info["mode"] = "auto"
    info["yaml_fallback"] = {"embedding_dim": fallback_m, "time_delay": fallback_tau}
    return m, tau, info
