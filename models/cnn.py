"""RP 图像 CNN；卷积通道与全连接隐层由 config.model 指定。"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .attention import build_attention


class RPCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        attention: str = "se",
        dropout: float = 0.3,
        in_channels: int = 1,
        conv_channels: Sequence[int] = (32, 64, 128),
        fc_hidden: Sequence[int] = (),
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        convs = []
        ch = in_channels
        pad = kernel_size // 2
        for c in conv_channels:
            c = int(c)
            convs.extend(
                [
                    nn.Conv2d(ch, c, kernel_size=kernel_size, padding=pad),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(pool_size),
                ]
            )
            ch = c
        if not convs:
            raise ValueError("model.conv_channels 不能为空")
        self.features = nn.Sequential(*convs)
        self.attn = build_attention(attention, channels=ch)
        head: list[nn.Module] = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        prev = ch
        for h in fc_hidden:
            h = int(h)
            head.extend([nn.Dropout(dropout), nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        head.extend([nn.Dropout(dropout), nn.Linear(prev, num_classes)])
        self.head = nn.Sequential(*head)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """注意力后的全局向量，供 t-SNE。"""
        x = self.features(x)
        x = self.attn(x)
        return nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attn(x)
        return self.head(x)


def build_model(num_classes: int, cfg, in_channels: int = 1) -> nn.Module:
    method = cfg.method
    train = cfg.train
    model_cfg = cfg.get("model") if cfg.get("model") is not None else {}
    conv = model_cfg.get("conv_channels", [32, 64, 128])
    fc = model_cfg.get("fc_hidden", [])
    return RPCNN(
        num_classes=num_classes,
        attention=str(method.attention),
        dropout=float(train.dropout),
        in_channels=int(in_channels),
        conv_channels=list(conv),
        fc_hidden=list(fc) if fc is not None else [],
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        pool_size=int(model_cfg.get("pool_size", 2)),
    )
