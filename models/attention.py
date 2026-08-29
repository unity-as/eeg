"""注意力模块：none / se（阶段 R）；additive（阶段 M 占位）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class IdentityAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SEAttention(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


def build_attention(kind: str, channels: int) -> nn.Module:
    k = str(kind).lower()
    if k in ("none", "identity", "off"):
        return IdentityAttention()
    if k == "se":
        return SEAttention(channels)
    if k == "additive":
        raise NotImplementedError(
            "method.attention=additive 属于阶段 M，尚未实现。请使用 none 或 se。"
        )
    raise ValueError(f"未知 attention: {kind}")
