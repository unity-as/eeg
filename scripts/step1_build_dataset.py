"""step1：窗采样 → RP 图像缓存。"""
from __future__ import annotations

import os
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import collect_windows_per_class, windows_to_arrays
from representation.recurrence_plot import build_representation
from representation.rhythm import apply_rhythm_filter
from utils.io_utils import ensure_dir, save_json


def main(cfg_path: str = "config/default.yaml") -> None:
    cfg = OmegaConf.load(cfg_path)
    method = cfg.method
    data_cfg = cfg.data

    if bool(method.rhythm_filter):
        raise NotImplementedError(
            "当前振动验证默认 rhythm_filter=false；开启前需配置 fs/band。"
        )

    windows = collect_windows_per_class(
        raw_data_dir=str(data_cfg.raw_data_dir),
        class_map=dict(data_cfg.class_map),
        window_size=int(data_cfg.window_size),
        num_windows_per_class=int(data_cfg.num_windows_per_class),
        random_state=int(data_cfg.random_state),
    )
    signals, labels = windows_to_arrays(windows, dict(data_cfg.class_map))
    if len(signals) == 0:
        raise RuntimeError("未采到任何窗口，请检查 raw_data_dir 与类别关键字。")

    cache_dir = ensure_dir(str(data_cfg.cache_dir))
    images = []
    for sig in tqdm(signals, desc="build RP"):
        sig = apply_rhythm_filter(sig, enabled=False)
        rp, _meta = build_representation(sig, method)
        images.append(rp.astype(np.float32))

    x = np.stack(images, axis=0)[:, None, :, :]  # [N,1,H,W]
    y = labels.astype(np.int64)
    np.save(os.path.join(cache_dir, "X.npy"), x)
    np.save(os.path.join(cache_dir, "y.npy"), y)
    save_json(
        {
            "num_samples": int(x.shape[0]),
            "image_shape": list(x.shape[1:]),
            "class_map": dict(data_cfg.class_map),
            "method": OmegaConf.to_container(method, resolve=True),
            "window_size": int(data_cfg.window_size),
            "num_windows_per_class": int(data_cfg.num_windows_per_class),
        },
        os.path.join(cache_dir, "meta.json"),
    )
    print(f"已保存: {cache_dir}  X{x.shape} y{y.shape}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    main(cfg)
