from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def gpu_available() -> bool:
    return bool(torch.cuda.is_available())


def _phase_space(x: torch.Tensor, m: int, tau: int) -> torch.Tensor:
    """x: [B, C, T] -> [B, C, L, m]. Ties broken by original time order."""
    batch, channels, n_times = x.shape
    length = n_times - (m - 1) * tau
    if length <= 1:
        raise ValueError(f"signal is too short for embedding: T={n_times}, m={m}, tau={tau}")
    base = torch.arange(length, device=x.device)
    offsets = torch.arange(m, device=x.device) * tau
    idx = base[:, None] + offsets[None, :]  # [L, m]
    traj = x[:, :, idx]  # [B, C, L, m]
    return traj.reshape(batch, channels, length, m)


def _epsilon_from_distances(
    dist: torch.Tensor,
    *,
    epsilon: float | None,
    mode: str,
    percentile: float,
    std_k: float,
    diameter_frac: float,
) -> torch.Tensor:
    """Return thresholds [B, C, 1, 1]."""
    if epsilon is not None:
        return torch.full(
            (dist.shape[0], dist.shape[1], 1, 1),
            max(float(epsilon), 1e-8),
            device=dist.device,
            dtype=dist.dtype,
        )

    length = dist.shape[-1]
    mask = torch.triu(
        torch.ones((length, length), device=dist.device, dtype=torch.bool),
        diagonal=1,
    )
    batch, channels, _, _ = dist.shape
    off = dist.reshape(batch, channels, -1)[:, :, mask.reshape(-1)]  # [B, C, N]
    mode = str(mode).lower()
    if mode in {"std", "sigma", "paper1"}:
        eps = torch.std(off, dim=-1, correction=0) * float(std_k)
    elif mode in {"diameter", "max", "marwan"}:
        eps = torch.amax(off, dim=-1) * float(diameter_frac)
    else:
        eps = torch.quantile(off, float(percentile), dim=-1)
    return torch.clamp(eps, min=1e-8).view(dist.shape[0], dist.shape[1], 1, 1)


def build_rp_batch_gpu(
    epochs: Sequence[np.ndarray] | np.ndarray,
    method_cfg,
    *,
    device: str = "cuda",
    max_batch_size: int = 8,
) -> np.ndarray:
    """Build RP tensors for a batch of [C, T] epochs on CUDA.

    Returns uint8 [B, C, image_size, image_size] images.  The implementation
    mirrors the CPU pipeline for fixed m/tau, per-channel z-score, percentile
    epsilon, and 64x64 bilinear resizing.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")
    if str(method_cfg.get("representation", "rp")).lower() != "rp":
        raise ValueError("GPU builder currently supports representation=rp only")
    if bool(method_cfg.get("rhythm_filter", False)):
        raise ValueError("GPU builder does not implement rhythm_filter=true")
    if bool(method_cfg.get("quality_assess", False)):
        raise ValueError("GPU builder does not implement quality_assess=true")

    arr = np.asarray(epochs, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"epochs must be [B, C, T], got {arr.shape}")
    m = int(method_cfg.get("embedding_dim", 3))
    tau = max(int(method_cfg.get("time_delay", 1)), 1)
    image_size = int(method_cfg.get("rp_image_size", 64))
    percentile = float(method_cfg.get("recurrence_percentile", 0.1))
    epsilon_raw = method_cfg.get("epsilon", None)
    epsilon = None if epsilon_raw in (None, "null") else float(epsilon_raw)
    epsilon_mode = str(method_cfg.get("epsilon_mode", "percentile"))
    std_k = float(method_cfg.get("epsilon_std_k", 0.25))
    diameter_frac = float(method_cfg.get("diameter_frac", 0.1))
    normalize = str(method_cfg.get("normalize", "zscore")).lower()
    if normalize not in {"zscore", "z", "std", "none", "off"}:
        raise ValueError(f"unsupported normalize mode: {normalize}")

    outputs = []
    batch_size = max(1, int(max_batch_size))
    dev = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            stop = min(start + batch_size, len(arr))
            x = torch.as_tensor(arr[start:stop], dtype=torch.float32, device=dev)
            if normalize in {"zscore", "z", "std"}:
                mean = x.mean(dim=2, keepdim=True)
                std = x.std(dim=2, unbiased=False, keepdim=True)
                x = (x - mean) / torch.clamp(std, min=1e-12)

            traj = _phase_space(x, m=m, tau=tau)
            batch, channels, length, _ = traj.shape
            flat = traj.reshape(batch * channels, length, -1)
            dist = torch.cdist(flat, flat, p=2).reshape(batch, channels, length, length)
            eps = _epsilon_from_distances(
                dist,
                epsilon=epsilon,
                mode=epsilon_mode,
                percentile=percentile,
                std_k=std_k,
                diameter_frac=diameter_frac,
            )
            rp = (dist <= eps).to(torch.float32)
            batch, channels, length, _ = rp.shape
            rp = F.interpolate(
                rp.reshape(batch * channels, 1, length, length),
                size=(image_size, image_size),
                mode="area",
            ).reshape(batch, channels, image_size, image_size)
            image = torch.clamp(torch.round(rp * 255.0), 0, 255).to(torch.uint8)
            outputs.append(image.cpu())
    return torch.cat(outputs, dim=0).numpy()

