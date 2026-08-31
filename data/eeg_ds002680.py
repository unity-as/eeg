"""ds002680 Go-nogo：stimulus 事件 → 4 类 epoch（方案 A）。"""
from __future__ import annotations

import os
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd

# categorization/recognition × go/nogo
CLASS_MAP = {
    "cat_go": 0,
    "cat_nogo": 1,
    "rec_go": 2,
    "rec_nogo": 3,
}

VALUE_TO_CLASS = {
    "animal_target": "cat_go",
    "animal_distractor": "cat_nogo",
    "easy_target": "rec_go",
    "difficult_target": "rec_go",
    "nonanimal_target": "rec_go",
    "easy_distractor": "rec_nogo",
    "difficult_distractor": "rec_nogo",
    "nonanimal_distractor": "rec_nogo",
}


def _find_set_files(bids_root: str, subjects: Sequence[str]) -> List[str]:
    files: List[str] = []
    for sub in subjects:
        pattern = os.path.join(bids_root, sub, "ses-*", "eeg", f"{sub}_*_eeg.set")
        files.extend(sorted(glob(pattern)))
    return files


def _load_events(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    stim = df[df["trial_type"] == "stimulus"].copy()
    stim = stim[stim["value"].isin(VALUE_TO_CLASS.keys())]
    return stim.reset_index(drop=True)


def _epoch_multichannel(
    data: np.ndarray,
    sfreq: float,
    onset_sec: float,
    epoch_samples: int,
) -> Optional[np.ndarray]:
    """data: [n_channels, n_times] → [n_channels, epoch_samples]。"""
    start = int(round(onset_sec * sfreq))
    end = start + epoch_samples
    if start < 0 or end > data.shape[1]:
        return None
    return np.asarray(data[:, start:end], dtype=np.float32)


def collect_epochs_by_subject(
    bids_root: str,
    subjects: Sequence[str],
    epoch_samples: int = 256,
    max_epochs_per_subject: Optional[int] = None,
    max_runs_per_subject: Optional[int] = None,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    返回 {subject: {class_name: [arrays of shape (C, T)]}}
    """
    out: Dict[str, Dict[str, List[np.ndarray]]] = {
        sub: {name: [] for name in CLASS_MAP} for sub in subjects
    }
    for sub in subjects:
        set_files = _find_set_files(bids_root, [sub])
        if max_runs_per_subject is not None:
            set_files = set_files[: max_runs_per_subject]
        print(f"{sub}: {len(set_files)} runs")
        for set_path in set_files:
            events_path = set_path.replace("_eeg.set", "_events.tsv")
            if not os.path.exists(events_path):
                print(f"  skip (no events): {set_path}")
                continue
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
            data = raw.get_data()
            sfreq = float(raw.info["sfreq"])
            events = _load_events(events_path)
            for _, row in events.iterrows():
                cname = VALUE_TO_CLASS[str(row["value"])]
                if max_epochs_per_subject is not None:
                    total = sum(len(v) for v in out[sub].values())
                    if total >= max_epochs_per_subject:
                        break
                ep = _epoch_multichannel(data, sfreq, float(row["onset"]), epoch_samples)
                if ep is None:
                    continue
                out[sub][cname].append(ep)
            if max_epochs_per_subject is not None:
                total = sum(len(v) for v in out[sub].values())
                if total >= max_epochs_per_subject:
                    break
        counts = {k: len(v) for k, v in out[sub].items()}
        print(f"  epochs {counts} total={sum(counts.values())}")
    return out


def pack_subject(
    by_subject: Dict[str, Dict[str, List[np.ndarray]]],
    subject: str,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """单个被试 → (signals, labels)。"""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for cname, segs in by_subject[subject].items():
        lab = CLASS_MAP[cname]
        for s in segs:
            xs.append(s)
            ys.append(lab)
    return xs, np.asarray(ys, dtype=np.int64)


def subject_split_to_arrays(
    by_subject: Dict[str, Dict[str, List[np.ndarray]]],
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> Dict[str, Tuple[List[np.ndarray], np.ndarray]]:
    """按被试划分；返回 train/val/test → (signals, labels)。三集被试必须互斥。"""
    train_set, val_set, test_set = set(train_subjects), set(val_subjects), set(test_subjects)
    overlap = {
        "train∩val": train_set & val_set,
        "train∩test": train_set & test_set,
        "val∩test": val_set & test_set,
    }
    leaked = {k: v for k, v in overlap.items() if v}
    if leaked:
        raise ValueError(f"train/val/test 被试重叠（会污染评估）: {leaked}")

    def pack(subs: Sequence[str]):
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for sub in subs:
            sx, sy = pack_subject(by_subject, sub)
            xs.extend(sx)
            ys.append(sy)
        y = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
        return xs, y

    return {
        "train": pack(train_subjects),
        "val": pack(val_subjects),
        "test": pack(test_subjects),
    }
