"""step1：时间切分 + 不重叠窗 → 表示图像缓存（train/val/test 分开存）。"""
from __future__ import annotations

import os
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import collect_split_windows, split_windows_to_arrays
from representation.recurrence_plot import build_representation
from representation.rhythm import apply_rhythm_filter
from utils.io_utils import ensure_dir, save_json


def _encode(signals, labels, method):
    images = []
    for sig in tqdm(signals, desc="build images"):
        sig = apply_rhythm_filter(sig, enabled=False)
        rp, _meta = build_representation(sig, method)
        images.append(rp.astype(np.float32))
    if not images:
        return np.zeros((0, 1, int(method.rp_image_size), int(method.rp_image_size)), dtype=np.float32), labels
    x = np.stack(images, axis=0)[:, None, :, :]
    return x, labels.astype(np.int64)


def main(cfg_path: str = "config/default.yaml") -> None:
    cfg = OmegaConf.load(cfg_path)
    method = cfg.method
    data_cfg = cfg.data

    if bool(method.rhythm_filter):
        raise NotImplementedError(
            "当前振动验证默认 rhythm_filter=false；开启前需配置 fs/band。"
        )

    window_size = int(data_cfg.window_size)
    stride = int(data_cfg.get("stride", window_size))
    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    val_ratio = float(data_cfg.get("val_ratio", 0.15))
    max_w = data_cfg.get("max_windows_per_class", None)
    if max_w is not None:
        max_w = int(max_w)

    print(f"protocol=B  window={window_size} stride={stride} split={train_ratio}/{val_ratio}/rest")
    split_windows = collect_split_windows(
        raw_data_dir=str(data_cfg.raw_data_dir),
        class_map=dict(data_cfg.class_map),
        window_size=window_size,
        stride=stride,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_windows_per_class=max_w,
    )

    cache_dir = ensure_dir(str(data_cfg.cache_dir))
    counts = {}
    for split in ("train", "val", "test"):
        signals, labels = split_windows_to_arrays(split_windows, dict(data_cfg.class_map), split)
        x, y = _encode(signals, labels, method)
        np.save(os.path.join(cache_dir, f"X_{split}.npy"), x)
        np.save(os.path.join(cache_dir, f"y_{split}.npy"), y)
        counts[split] = int(y.shape[0])
        print(f"saved {split}: X{x.shape} y{y.shape}")

    if counts["train"] == 0:
        raise RuntimeError("训练集为空，请检查 raw_data_dir / 切分比例 / 窗长")

    save_json(
        {
            "protocol": "B_temporal_then_nonoverlap",
            "counts": counts,
            "class_map": dict(data_cfg.class_map),
            "method": OmegaConf.to_container(method, resolve=True),
            "window_size": window_size,
            "stride": stride,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "max_windows_per_class": max_w,
        },
        os.path.join(cache_dir, "meta.json"),
    )
    print(f"已保存: {cache_dir}  counts={counts}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    main(cfg)
