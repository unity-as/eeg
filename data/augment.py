"""训练期增强：高斯噪声 + 通道丢失（作用于 [B,C,H,W]）。"""
from __future__ import annotations

import torch


def augment_rp_batch(
    x: torch.Tensor,
    gaussian_std: float = 0.05,
    channel_dropout: float = 0.2,
) -> torch.Tensor:
    """
    x: [B, C, H, W]，值约在 [0,1]。
    channel_dropout: 每个通道被置零的概率（电极级）。
    """
    if x.ndim != 4:
        raise ValueError(f"期望 [B,C,H,W]，收到 {tuple(x.shape)}")
    y = x
    if channel_dropout > 0 and x.shape[1] > 1:
        keep = (torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) >= channel_dropout).to(x.dtype)
        # 避免整样本通道全丢
        all_drop = keep.view(x.size(0), -1).sum(dim=1) == 0
        if all_drop.any():
            keep[all_drop, 0] = 1
        y = y * keep
    if gaussian_std > 0:
        y = y + torch.randn_like(y) * gaussian_std
        y = y.clamp(0.0, 1.0)
    return y
