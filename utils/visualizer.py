"""简单训练曲线绘制。"""
from __future__ import annotations

import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_train_curves(
    train_loss: List[float],
    val_acc: List[float],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(train_loss, color="C0", label="train_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(val_acc, color="C1", label="val_acc")
    ax2.set_ylabel("acc", color="C1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
