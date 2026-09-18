from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.eeg_ds002680 import CLASS_MAP, VALUE_TO_CLASS


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def available_subjects(root: str | Path) -> List[str]:
    root = resolve_path(root)
    return sorted(p.name for p in root.glob("sub-*") if p.is_dir())


def run_files_for_subject(root: str | Path, subject: str) -> List[Path]:
    root = resolve_path(root)
    return sorted(root.glob(f"{subject}/ses-*/eeg/{subject}_*_eeg.set"))


def load_events(tsv_path: str | Path) -> pd.DataFrame:
    """Load an event table without filtering or changing event order."""
    return pd.read_csv(tsv_path, sep="\t")


def annotate_event_correctness(events: pd.DataFrame) -> pd.DataFrame:
    """Attach the dataset's explicit correct/incorrect label to stimulus rows.

    ds002680 stores a response event with value ``correct`` or ``incorrect``
    after a trial when such a response event exists.  For trials without a
    response event, a target trial is incorrect and a distractor trial is
    correct.  The stimulus ``response_time`` column alone is not sufficient:
    some timely target responses are explicitly marked incorrect, and some
    correct no-go trials use ``response_time == 1000``.
    """

    out = events.copy()
    out["correct"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    pending_idx = None

    def finish_pending() -> None:
        nonlocal pending_idx
        if pending_idx is None:
            return
        value = str(out.at[pending_idx, "value"])
        out.at[pending_idx, "correct"] = bool(not value.endswith("_target"))
        pending_idx = None

    for idx, row in out.iterrows():
        trial_type = str(row.get("trial_type", ""))
        if trial_type == "stimulus" and str(row.get("value", "")) in VALUE_TO_CLASS:
            finish_pending()
            pending_idx = idx
        elif trial_type == "response" and pending_idx is not None:
            value = str(row.get("value", "")).strip().lower()
            if value in {"correct", "incorrect"}:
                out.at[pending_idx, "correct"] = bool(value == "correct")
                pending_idx = None

    finish_pending()
    return out


def load_stimulus_events(tsv_path: str | Path) -> pd.DataFrame:
    """Return valid stimulus rows with event_index and explicit correctness."""

    df = annotate_event_correctness(load_events(tsv_path))
    stim = df[df["trial_type"] == "stimulus"].copy()
    stim = stim[stim["value"].isin(VALUE_TO_CLASS.keys())].copy()
    stim["event_index"] = stim.index.astype(int)
    return stim.reset_index(drop=True)


def derive_correct(stim: pd.DataFrame) -> np.ndarray:
    """Return correctness produced by :func:`load_stimulus_events`."""

    if "correct" not in stim.columns:
        raise ValueError(
            "correct column is missing; load stimuli through load_stimulus_events()"
        )
    if stim["correct"].isna().any():
        missing = int(stim["correct"].isna().sum())
        raise ValueError(f"correctness is missing for {missing} stimulus rows")
    return stim["correct"].astype(bool).to_numpy()


def epoch_from_raw(
    data: np.ndarray,
    sfreq: float,
    onset_sec: float,
    epoch_samples: int,
) -> np.ndarray | None:
    start = int(round(float(onset_sec) * float(sfreq)))
    end = start + int(epoch_samples)
    if start < 0 or end > data.shape[1]:
        return None
    return np.asarray(data[:, start:end], dtype=np.float32)


def class_count_dict(labels: np.ndarray) -> dict[str, int]:
    names = [k for k, _ in sorted(CLASS_MAP.items(), key=lambda kv: int(kv[1]))]
    values, counts = np.unique(labels, return_counts=True)
    return {names[int(v)]: int(c) for v, c in zip(values, counts)}


def make_three_way_split(y: np.ndarray, cfg, sample_ids: list[str] | None = None) -> dict:
    """Create a deterministic stratified train/val/test split.

    ``data.split_seed`` controls the split and is intentionally independent of
    ``train.seed`` so repeated model runs with different training seeds keep the
    exact same trial partition.
    """

    seed = int(cfg.data.get("split_seed", cfg.train.get("seed", 42)))
    train_ratio = float(cfg.data.train_ratio)
    val_ratio = float(cfg.data.val_ratio)
    test_ratio = float(cfg.data.test_ratio)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    idx = np.arange(len(y), dtype=np.int64)
    idx_train, idx_hold = train_test_split(
        idx,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        shuffle=True,
        stratify=y if bool(cfg.data.get("stratify", True)) else None,
    )
    val_share = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test = train_test_split(
        idx_hold,
        train_size=val_share,
        random_state=seed + 1,
        shuffle=True,
        stratify=y[idx_hold] if bool(cfg.data.get("stratify", True)) else None,
    )

    indices = {
        "train": idx_train.tolist(),
        "val": idx_val.tolist(),
        "test": idx_test.tolist(),
    }
    split = {
        "version": 1,
        "split_seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "indices": indices,
        "sizes": {k: int(len(v)) for k, v in indices.items()},
        "class_counts": {k: class_count_dict(y[np.asarray(v, dtype=np.int64)]) for k, v in indices.items()},
    }
    if sample_ids is not None:
        if len(sample_ids) != len(y):
            raise ValueError("sample_ids length must match y length")
        split["sample_ids"] = {
            k: [sample_ids[i] for i in v] for k, v in indices.items()
        }
    return split
