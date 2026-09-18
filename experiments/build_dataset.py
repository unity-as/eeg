from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import mne
import numpy as np
from omegaconf import OmegaConf

from experiments.common import (
    available_subjects,
    class_count_dict,
    derive_correct,
    epoch_from_raw,
    load_stimulus_events,
    make_three_way_split,
    resolve_path,
    run_files_for_subject,
)
from data.eeg_ds002680 import CLASS_MAP, VALUE_TO_CLASS
from representation.recurrence_plot import build_representation
from utils.io_utils import ensure_dir, save_json


def _sample_id(subject: str, run_path: Path, event_index: int, onset: float, value: str) -> str:
    return (
        f"{subject}|{run_path.parent.parent.name}|{run_path.name}|"
        f"{int(event_index)}|{float(onset):.6f}|{value}"
    )


def build_subject(subject: str, cfg, force: bool = False) -> None:
    data_cfg = cfg.data
    out_dir = resolve_path(data_cfg.output_root) / subject
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        print(f"{subject}: cache exists, skip: {out_dir}")
        return

    run_files = run_files_for_subject(data_cfg.bids_root, subject)
    max_runs = data_cfg.get("max_runs", None)
    if max_runs is not None:
        run_files = run_files[: int(max_runs)]
    max_samples = data_cfg.get("max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)

    xs: List[np.ndarray] = []
    ys: List[int] = []
    sample_ids: List[str] = []
    event_rows = []
    skipped_wrong = 0
    skipped_bad_epoch = 0

    print(f"{subject}: {len(run_files)} runs")
    for run_path in run_files:
        events_path = Path(str(run_path).replace("_eeg.set", "_events.tsv"))
        if not events_path.exists():
            print(f"  skip no events: {run_path.name}")
            continue
        raw = mne.io.read_raw_eeglab(str(run_path), preload=True, verbose=False)
        data = raw.get_data()
        sfreq = float(raw.info["sfreq"])
        events = load_stimulus_events(events_path)
        correctness = derive_correct(events)

        for event_i, row in events.iterrows():
            if max_samples is not None and len(xs) >= max_samples:
                break
            is_correct = bool(correctness[int(event_i)])
            if bool(data_cfg.get("correct_only", False)) and not is_correct:
                skipped_wrong += 1
                continue
            ep = epoch_from_raw(
                data,
                sfreq,
                float(row["onset"]),
                int(data_cfg.epoch_samples),
            )
            if ep is None:
                skipped_bad_epoch += 1
                continue
            image, _ = build_representation(ep, cfg.method)
            image = np.asarray(image, dtype=np.float32)
            if image.ndim == 2:
                image = image[None, :, :]
            if image.ndim != 3:
                raise ValueError(f"representation must be [C,H,W], got {image.shape}")
            label = int(CLASS_MAP[VALUE_TO_CLASS[str(row["value"])]])
            sid = _sample_id(
                subject,
                run_path,
                int(row["event_index"]),
                float(row["onset"]),
                str(row["value"]),
            )
            xs.append(image)
            ys.append(label)
            sample_ids.append(sid)
            event_rows.append(
                {
                    "sample_id": sid,
                    "run": run_path.name,
                    "session": run_path.parent.parent.name,
                    "event_index": int(row["event_index"]),
                    "onset": float(row["onset"]),
                    "value": str(row["value"]),
                    "label": label,
                    "correct": is_correct,
                    "response_time": row.get("response_time", None),
                }
            )
        if max_samples is not None and len(xs) >= max_samples:
            break

    if not xs:
        raise RuntimeError(f"{subject}: no samples after filtering")

    X = np.stack(xs, axis=0).astype(np.float32)
    kind = str(cfg.method.get("representation", "rp")).lower()
    if kind == "transition":
        if not np.isfinite(X).all():
            raise RuntimeError(f"{subject}: transition cache contains NaN or inf")
    else:
        # Preserve legacy cache behavior: RPs are already roughly [0, 1].
        X = np.clip(np.rint(X * 255.0), 0, 255).astype(np.uint8)
    y = np.asarray(ys, dtype=np.int64)
    split = make_three_way_split(y, cfg, sample_ids=sample_ids)
    out_dir = Path(ensure_dir(str(out_dir)))
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    save_json(split, str(out_dir / "split_indices.json"))
    with open(out_dir / "events.jsonl", "w", encoding="utf-8") as f:
        for row in event_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "subject": subject,
        "source": "ds002680",
        "n_samples": int(len(y)),
        "correct_only": bool(data_cfg.get("correct_only", False)),
        "skipped_wrong": int(skipped_wrong),
        "skipped_bad_epoch": int(skipped_bad_epoch),
        "max_samples": None if max_samples is None else int(max_samples),
        "max_runs": None if max_runs is None else int(max_runs),
        "epoch_samples": int(data_cfg.epoch_samples),
        "class_counts": class_count_dict(y),
        "input_shape": list(X.shape[1:]),
        "method": OmegaConf.to_container(cfg.method, resolve=True),
        "split": split["sizes"],
        "split_class_counts": split["class_counts"],
        "split_seed": int(split["split_seed"]),
    }
    save_json(meta, str(out_dir / "meta.json"))
    print(f"{subject}: saved X={X.shape} y={y.shape} -> {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/config_rp_se.yaml")
    parser.add_argument("--subjects", default=None, help="comma separated; default from config")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(resolve_path(args.config))
    if args.all:
        subjects = available_subjects(cfg.data.bids_root)
    elif args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = list(cfg.data.get("subjects", available_subjects(cfg.data.bids_root)))
    for subject in subjects:
        build_subject(subject, cfg, force=args.force)


if __name__ == "__main__":
    main()
