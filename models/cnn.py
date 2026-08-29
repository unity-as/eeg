"""RP 图像 CNN 分类器。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import build_attention


class RPCNN(nn.Module):
    def __init__(self, num_classes: int, attention: str = "se", dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.attn = build_attention(attention, channels=128)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attn(x)
        return self.head(x)


def build_model(num_classes: int, cfg) -> nn.Module:
    method = cfg.method
    train = cfg.train
    return RPCNN(
        num_classes=num_classes,
        attention=str(method.attention),
        dropout=float(train.dropout),
    )
