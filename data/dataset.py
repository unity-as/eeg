"""从 raw .npy 按类别关键字采样固定长度窗口（对照旋转机械 dataset）。"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np


CLASS_KEYS = ("normal", "inner", "outer", "combine", "roll")


def collect_windows_per_class(
    raw_data_dir: str,
    class_map: Dict[str, int],
    window_size: int,
    num_windows_per_class: int,
    random_state: int,
) -> Dict[str, List[np.ndarray]]:
    random.seed(random_state)
    np.random.seed(random_state)

    class_files = {cls: [] for cls in class_map}
    for fname in os.listdir(raw_data_dir):
        if not fname.endswith(".npy"):
            continue
        lower = fname.lower()
        for cls in class_map:
            if cls in lower:
                class_files[cls].append(os.path.join(raw_data_dir, fname))
                break

    windows_per_class: Dict[str, List[np.ndarray]] = {}
    for cls, files in class_files.items():
        if not files:
            print(f"警告：类别 '{cls}' 无文件")
            windows_per_class[cls] = []
            continue

        valid = []
        for fp in files:
            try:
                mmap = np.load(fp, mmap_mode="r")
                if mmap.shape[0] >= window_size:
                    valid.append(fp)
            except Exception as e:
                print(f"警告：跳过 {fp}: {e}")

        if not valid:
            windows_per_class[cls] = []
            continue

        segs = []
        for _ in range(num_windows_per_class):
            fp = valid[random.randint(0, len(valid) - 1)]
            sig = np.load(fp, mmap_mode="r")
            start = random.randint(0, len(sig) - window_size)
            segs.append(np.asarray(sig[start : start + window_size], dtype=np.float32).copy())
        windows_per_class[cls] = segs
    return windows_per_class


def windows_to_arrays(
    windows_per_class: Dict[str, List[np.ndarray]],
    class_map: Dict[str, int],
) -> Tuple[List[np.ndarray], np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for cls, segs in windows_per_class.items():
        label = int(class_map[cls])
        for seg in segs:
            xs.append(seg)
            ys.append(label)
    return xs, np.asarray(ys, dtype=np.int64)
