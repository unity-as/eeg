"""注意力模块：none / se（阶段 R）；additive（阶段 M）。"""
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


class AdditiveAttention(nn.Module):
    """Bahdanau 式加性空间注意力（用于 CNN 特征图）。

    score = v^T tanh(W x)，再对空间维 softmax，得到注意力图。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.W = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.v(torch.tanh(self.W(x)))
        b, _, h, w = e.shape
        alpha = torch.softmax(e.view(b, -1), dim=1).view(b, 1, h, w)
        return x * alpha


def build_attention(kind: str, channels: int) -> nn.Module:
    k = str(kind).lower()
    if k in ("none", "identity", "off"):
        return IdentityAttention()
    if k == "se":
        return SEAttention(channels)
    if k == "additive":
        return AdditiveAttention(channels)
    raise ValueError(f"未知 attention: {kind}")
