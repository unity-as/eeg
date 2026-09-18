"""State-transition matrix representation.

This module discretizes a time series into amplitude states, then counts
directed transitions across one or more time steps.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _as_steps(raw) -> list[int]:
    if raw is None:
        return [1]
    if isinstance(raw, (int, np.integer)):
        steps = [int(raw)]
    else:
        steps = [int(s) for s in list(raw)]
    steps = sorted(set(s for s in steps if s > 0))
    if not steps:
        raise ValueError("transition_steps must contain at least one positive integer")
    return steps


def symbolize_uniform(x: np.ndarray, bins: int) -> np.ndarray:
    """Map one signal to integer states 0..bins-1 using per-epoch amplitude bins."""
    x = np.asarray(x, dtype=np.float64).ravel()
    bins = int(bins)
    if bins < 2:
        raise ValueError("symbol_bins must be >= 2")
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.int64)
    scaled = (x - lo) / (hi - lo)
    states = np.floor(scaled * bins).astype(np.int64)
    return np.clip(states, 0, bins - 1)


def symbolize_quantile(x: np.ndarray, bins: int) -> np.ndarray:
    """Map one signal to states using per-epoch quantile edges."""
    x = np.asarray(x, dtype=np.float64).ravel()
    bins = int(bins)
    if bins < 2:
        raise ValueError("symbol_bins must be >= 2")
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    if edges.size and float(np.max(edges) - np.min(edges)) < 1e-12:
        return np.zeros_like(x, dtype=np.int64)
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def symbolize(x: np.ndarray, bins: int, strategy: str = "uniform") -> np.ndarray:
    strategy = str(strategy).lower()
    if strategy in ("uniform", "equal", "amplitude"):
        return symbolize_uniform(x, bins)
    if strategy in ("quantile", "percentile"):
        return symbolize_quantile(x, bins)
    raise ValueError(f"unknown symbol_strategy: {strategy}")


def transition_matrix(
    states: np.ndarray,
    bins: int,
    step: int,
    weight: str = "probability",
    include_self: bool = True,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64).ravel()
    bins = int(bins)
    step = int(step)
    mat = np.zeros((bins, bins), dtype=np.float32)
    if step <= 0:
        raise ValueError("transition step must be positive")
    if states.size <= step:
        return mat
    src = states[:-step]
    dst = states[step:]
    valid = (src >= 0) & (src < bins) & (dst >= 0) & (dst < bins)
    if not include_self:
        valid = valid & (src != dst)
    if np.any(valid):
        np.add.at(mat, (src[valid], dst[valid]), 1.0)
    weight = str(weight).lower()
    if weight in ("count", "counts", "raw"):
        return mat
    if weight in ("probability", "prob", "global"):
        total = float(np.sum(mat))
        return mat / total if total > 0 else mat
    if weight in ("row_probability", "row_prob", "row"):
        rows = np.sum(mat, axis=1, keepdims=True)
        return np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0)
    raise ValueError(f"unknown transition_weight: {weight}")


def _build_channel(signal: np.ndarray, method_cfg) -> np.ndarray:
    bins = int(method_cfg.get("symbol_bins", 6))
    strategy = str(method_cfg.get("symbol_strategy", "uniform"))
    steps = _as_steps(method_cfg.get("transition_steps", [1, 2, 3]))
    weight = str(method_cfg.get("transition_weight", "probability"))
    include_self = bool(method_cfg.get("include_self_transition", True))
    states = symbolize(signal, bins=bins, strategy=strategy)
    maps = [
        transition_matrix(states, bins=bins, step=s, weight=weight, include_self=include_self)
        for s in steps
    ]
    return np.stack(maps, axis=0).astype(np.float32)


def build_state_transition(signal: np.ndarray, method_cfg) -> np.ndarray:
    """Return [steps, bins, bins] or [channels*steps, bins, bins]."""
    arr = np.asarray(signal, dtype=np.float64)
    arr = np.squeeze(arr)
    how = str(method_cfg.get("transition_multichannel", "mean")).lower()
    if arr.ndim == 1:
        return _build_channel(arr, method_cfg)
    if arr.ndim != 2:
        raise ValueError(f"transition input must be 1D or [C,T], got shape={arr.shape}")
    maps = [_build_channel(arr[c], method_cfg) for c in range(arr.shape[0])]
    stacked = np.stack(maps, axis=0).astype(np.float32)  # [C,S,B,B]
    if how in ("mean", "avg", "average"):
        return np.mean(stacked, axis=0).astype(np.float32)
    if how in ("stack", "per_channel", "channel"):
        c, s, b, _ = stacked.shape
        return stacked.reshape(c * s, b, b).astype(np.float32)
    raise ValueError(f"unknown transition_multichannel: {how}")
