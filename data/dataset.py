"""时间段先切分，再各段内不重叠采窗（方案 B，防窗级泄漏）。"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np


def _class_files(raw_data_dir: str, class_map: Dict[str, int]) -> Dict[str, List[str]]:
    class_files = {cls: [] for cls in class_map}
    for fname in os.listdir(raw_data_dir):
        if not fname.endswith(".npy"):
            continue
        lower = fname.lower()
        for cls in class_map:
            if cls in lower:
                class_files[cls].append(os.path.join(raw_data_dir, fname))
                break
    return class_files


def temporal_ranges(
    n: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Tuple[int, int]]:
    """连续区间：[0,t) train，[t,t+v) val，[t+v,n) test。"""
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("需要 train_ratio>0, val_ratio>=0, 且 train+val < 1")
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return {
        "train": (0, t),
        "val": (t, t + v),
        "test": (t + v, n),
    }


def nonoverlap_windows(
    signal: np.ndarray,
    start: int,
    end: int,
    window_size: int,
    stride: int,
) -> List[np.ndarray]:
    if stride <= 0:
        raise ValueError("stride 必须 > 0")
    if stride < window_size:
        raise ValueError("方案 B 要求 stride >= window_size，禁止重叠窗")
    sig = np.asarray(signal)
    segs: List[np.ndarray] = []
    pos = start
    while pos + window_size <= end:
        w = np.asarray(sig[pos : pos + window_size], dtype=np.float32).squeeze().copy()
        if w.ndim != 1 or w.shape[0] != window_size:
            raise ValueError(f"窗形状异常: {w.shape}")
        segs.append(w)
        pos += stride
    return segs


def collect_split_windows(
    raw_data_dir: str,
    class_map: Dict[str, int],
    window_size: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    max_windows_per_class: int | None = None,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    返回 {split: {class: [windows...]}}
    split in train/val/test。
    """
    class_files = _class_files(raw_data_dir, class_map)
    out = {
        "train": {cls: [] for cls in class_map},
        "val": {cls: [] for cls in class_map},
        "test": {cls: [] for cls in class_map},
    }

    for cls, files in class_files.items():
        if not files:
            print(f"警告：类别 '{cls}' 无文件")
            continue
        for fp in files:
            try:
                sig = np.load(fp, mmap_mode="r")
            except Exception as e:
                print(f"警告：跳过 {fp}: {e}")
                continue
            n = int(sig.shape[0])
            if n < window_size:
                print(f"警告：{fp} 过短，跳过")
                continue
            ranges = temporal_ranges(n, train_ratio, val_ratio)
            for split, (a, b) in ranges.items():
                segs = nonoverlap_windows(sig, a, b, window_size, stride)
                if max_windows_per_class is not None:
                    remain = max_windows_per_class - len(out[split][cls])
                    if remain <= 0:
                        continue
                    segs = segs[:remain]
                out[split][cls].extend(segs)

        for split in ("train", "val", "test"):
            print(f"  {cls}/{split}: {len(out[split][cls])} windows")

    return out


def split_windows_to_arrays(
    split_windows: Dict[str, Dict[str, List[np.ndarray]]],
    class_map: Dict[str, int],
    split: str,
) -> Tuple[List[np.ndarray], np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for cls, segs in split_windows[split].items():
        label = int(class_map[cls])
        for seg in segs:
            xs.append(seg)
            ys.append(label)
    return xs, np.asarray(ys, dtype=np.int64)
