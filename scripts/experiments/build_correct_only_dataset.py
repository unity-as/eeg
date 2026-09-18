from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The CUDA environment owns torch; the EEG environment owns MNE/pandas.
# Appending preserves the CUDA torch while making its dependencies importable.
_DEP_SITE = Path(
    os.environ.get("EEG_DEP_SITE_PACKAGES", r"D:\MINICODA\envs\eeg\Lib\site-packages")
)
if _DEP_SITE.exists() and str(_DEP_SITE) not in sys.path:
    sys.path.append(str(_DEP_SITE))

import mne
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from data.eeg_ds002680 import CLASS_MAP, VALUE_TO_CLASS
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
from representation.gpu_rp import build_rp_batch_gpu
from representation.recurrence_plot import build_representation
from utils.io_utils import ensure_dir, load_json, save_json


CLASS_NAMES = [k for k, _ in sorted(CLASS_MAP.items(), key=lambda kv: int(kv[1]))]


def _representation_kind(cfg) -> str:
    return str(cfg.method.get("representation", "rp")).lower()


def _encode_epochs(epochs, cfg, use_gpu: bool, rp_device: str, rp_batch_size: int) -> np.ndarray:
    """Return [N,C,H,W]. RP stays uint8; transition stays float32 probabilities."""
    kind = _representation_kind(cfg)
    if kind == "rp" and use_gpu:
        return build_rp_batch_gpu(
            epochs,
            cfg.method,
            device=rp_device,
            max_batch_size=rp_batch_size,
        )
    built = []
    for epoch in epochs:
        image, _ = build_representation(epoch, cfg.method)
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image = image[None, :, :]
        if image.ndim != 3:
            raise ValueError(f"representation must be [C,H,W], got {image.shape}")
        if kind == "transition":
            built.append(image.astype(np.float32))
        else:
            built.append(np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8))
    return np.stack(built, axis=0)


def _config_fingerprint(cfg) -> str:
    text = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _session_name(run_path: Path) -> str:
    return run_path.parent.parent.name


def _sample_id(subject: str, run_path: Path, event_index: int, onset: float, value: str) -> str:
    return (
        f"{subject}|{_session_name(run_path)}|{run_path.name}|"
        f"{int(event_index)}|{float(onset):.6f}|{value}"
    )


def _empty_class_counts() -> dict:
    return {name: {"total": 0, "correct": 0, "wrong": 0} for name in CLASS_NAMES}


def quality_report_subject(subject: str, cfg) -> dict:
    """Scan only event tables and summarize correctness before filtering."""

    data_cfg = cfg.data
    run_files = run_files_for_subject(data_cfg.bids_root, subject)
    max_runs = data_cfg.get("max_runs", None)
    if max_runs is not None:
        run_files = run_files[: int(max_runs)]

    per_class = _empty_class_counts()
    per_run = []
    total = correct = 0
    for run_path in run_files:
        events_path = Path(str(run_path).replace("_eeg.set", "_events.tsv"))
        if not events_path.exists():
            per_run.append({"run": run_path.name, "status": "missing_events"})
            continue
        events = load_stimulus_events(events_path)
        correctness = derive_correct(events)
        run_total = int(len(events))
        run_correct = int(correctness.sum())
        total += run_total
        correct += run_correct
        for event_i, row in events.iterrows():
            class_name = VALUE_TO_CLASS[str(row["value"])]
            is_correct = bool(correctness[int(event_i)])
            per_class[class_name]["total"] += 1
            per_class[class_name]["correct" if is_correct else "wrong"] += 1
        per_run.append(
            {
                "run": run_path.name,
                "session": _session_name(run_path),
                "total": run_total,
                "correct": run_correct,
                "wrong": run_total - run_correct,
            }
        )

    return {
        "subject": subject,
        "n_runs": int(len(run_files)),
        "total_stimuli": int(total),
        "correct": int(correct),
        "wrong": int(total - correct),
        "correct_rule": (
            "Use response event value=correct/incorrect when present; "
            "otherwise target is incorrect and distractor is correct."
        ),
        "per_class": per_class,
        "per_run": per_run,
    }


def _split_fingerprint(split: dict) -> str:
    ids = split.get("sample_ids")
    if ids is None:
        raise ValueError("split file has no sample_ids")
    payload = json.dumps(ids, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_or_create_split(subject: str, y: np.ndarray, sample_ids: list[str], cfg, overwrite: bool) -> tuple[dict, Path]:
    split_path = resolve_path(cfg.data.splits_dir) / f"split_{subject}.json"
    if split_path.exists() and not overwrite:
        split = load_json(str(split_path))
        existing_ids = split.get("sample_ids")
        if existing_ids is None:
            raise RuntimeError(
                f"{split_path} lacks sample_ids; rerun with --overwrite-split only "
                "if replacing the frozen split is intentional"
            )
        for split_name in ("train", "val", "test"):
            if existing_ids.get(split_name) != [sample_ids[i] for i in split["indices"][split_name]]:
                raise RuntimeError(
                    f"{split_path} does not match the current sample order. "
                    "Refusing to change a frozen split; use --overwrite-split explicitly."
                )
        return split, split_path

    if split_path.exists() and overwrite:
        print(f"{subject}: overwriting frozen split {split_path}")
    split = make_three_way_split(y, cfg, sample_ids=sample_ids)
    split["subject"] = subject
    split["sample_id_fingerprint"] = _split_fingerprint(split)
    save_json(split, str(split_path))
    return split, split_path


def _serialize_response_time(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def build_subject(subject: str, cfg, force: bool = False, overwrite_split: bool = False) -> dict:
    data_cfg = cfg.data
    out_dir = resolve_path(data_cfg.output_root) / subject
    split_path = resolve_path(data_cfg.splits_dir) / f"split_{subject}.json"
    quality = quality_report_subject(subject, cfg)
    quality_dir = Path(ensure_dir(str(resolve_path(data_cfg.quality_dir))))
    save_json(quality, str(quality_dir / f"quality_{subject}.json"))

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        meta = load_json(str(meta_path))
        print(f"{subject}: cache exists, skip: {out_dir}")
        return meta

    run_files = run_files_for_subject(data_cfg.bids_root, subject)
    max_runs = data_cfg.get("max_runs", None)
    if max_runs is not None:
        run_files = run_files[: int(max_runs)]
    max_samples = data_cfg.get("max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)

    xs = []
    ys = []
    sample_ids = []
    event_rows = []
    pending_epochs = []
    pending_labels = []
    pending_rows = []
    skipped_wrong = 0
    skipped_bad_epoch = 0

    kind = _representation_kind(cfg)
    rp_device = str(data_cfg.get("rp_device", "cuda"))
    rp_batch_size = int(data_cfg.get("rp_batch_size", 8))
    use_gpu = kind == "rp" and rp_device.startswith("cuda")
    if use_gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "data.rp_device=cuda but CUDA is unavailable. Run with the CUDA PyTorch Python."
            )
        print(f"{subject}: RP device={rp_device} batch_size={rp_batch_size}")
    else:
        print(f"{subject}: representation={kind} device=cpu batch_size={rp_batch_size}")

    def flush_pending() -> None:
        if not pending_epochs:
            return
        images = _encode_epochs(
            pending_epochs, cfg, use_gpu, rp_device, rp_batch_size
        )
        out_dtype = np.float32 if kind == "transition" else np.uint8
        for image, label, row in zip(images, pending_labels, pending_rows):
            xs.append(np.asarray(image, dtype=out_dtype))
            ys.append(int(label))
            sample_ids.append(str(row["sample_id"]))
            event_rows.append(row)
        pending_epochs.clear()
        pending_labels.clear()
        pending_rows.clear()

    print(f"{subject}: building from {len(run_files)} runs")
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
            if max_samples is not None and len(xs) + len(pending_epochs) >= max_samples:
                break
            is_correct = bool(correctness[int(event_i)])
            if bool(data_cfg.get("correct_only", False)) and not is_correct:
                skipped_wrong += 1
                continue
            epoch = epoch_from_raw(
                data,
                sfreq,
                float(row["onset"]),
                int(data_cfg.epoch_samples),
            )
            if epoch is None:
                skipped_bad_epoch += 1
                continue

            class_name = VALUE_TO_CLASS[str(row["value"])]
            label = int(CLASS_MAP[class_name])
            sid = _sample_id(
                subject,
                run_path,
                int(row["event_index"]),
                float(row["onset"]),
                str(row["value"]),
            )
            pending_epochs.append(epoch)
            pending_labels.append(label)
            pending_rows.append(
                {
                    "sample_id": sid,
                    "subject": subject,
                    "session": _session_name(run_path),
                    "run": run_path.name,
                    "event_index": int(row["event_index"]),
                    "onset": float(row["onset"]),
                    "value": str(row["value"]),
                    "class_name": class_name,
                    "label": label,
                    "correct": is_correct,
                    "response_time": _serialize_response_time(row.get("response_time", None)),
                }
            )
            if len(pending_epochs) >= rp_batch_size:
                flush_pending()
        flush_pending()
        if max_samples is not None and len(xs) >= max_samples:
            break

    flush_pending()
    if not xs:
        raise RuntimeError(f"{subject}: no samples after filtering")

    X = np.stack(xs, axis=0)
    if kind == "transition":
        X = X.astype(np.float32)
        if not np.isfinite(X).all():
            raise RuntimeError(f"{subject}: transition cache contains NaN or inf")
    else:
        X = X.astype(np.uint8)
    y = np.asarray(ys, dtype=np.int64)
    split, split_path = _load_or_create_split(
        subject, y, sample_ids, cfg, overwrite=overwrite_split
    )

    out_dir = Path(ensure_dir(str(out_dir)))
    arrays = {}
    for split_name in ("train", "val", "test"):
        idx = np.asarray(split["indices"][split_name], dtype=np.int64)
        np.save(out_dir / f"X_{split_name}.npy", X[idx])
        np.save(out_dir / f"y_{split_name}.npy", y[idx])
        arrays[split_name] = {
            "n": int(len(idx)),
            "X_shape": list(X[idx].shape),
            "class_counts": class_count_dict(y[idx]),
        }

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
        "representation": kind,
        "dtype": str(X.dtype),
        "rp_device": rp_device if kind == "rp" else "cpu",
        "rp_batch_size": int(rp_batch_size),
        "class_counts": class_count_dict(y),
        "input_shape": list(X.shape[1:]),
        "method": OmegaConf.to_container(cfg.method, resolve=True),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "split_file": str(split_path),
        "split_sizes": split["sizes"],
        "split_class_counts": split["class_counts"],
        "split_seed": int(split["split_seed"]),
        "split_sample_id_fingerprint": split["sample_id_fingerprint"],
        "arrays": arrays,
        "quality": quality,
        "config_fingerprint": _config_fingerprint(cfg),
    }
    save_json(meta, str(out_dir / "meta.json"))
    print(f"{subject}: saved {X.shape} -> {out_dir}")
    return meta


def write_quality_summary(subjects: list[str], cfg) -> dict:
    quality_dir = Path(ensure_dir(str(resolve_path(cfg.data.quality_dir))))
    reports = {}
    totals = {"total_stimuli": 0, "correct": 0, "wrong": 0}
    per_class = _empty_class_counts()
    for subject in subjects:
        report = quality_report_subject(subject, cfg)
        reports[subject] = report
        for key in totals:
            totals[key] += int(report[key])
        for name in CLASS_NAMES:
            for key in ("total", "correct", "wrong"):
                per_class[name][key] += int(report["per_class"][name][key])
        save_json(report, str(quality_dir / f"quality_{subject}.json"))
    summary = {
        "subjects": subjects,
        "totals": totals,
        "per_class": per_class,
        "by_subject": reports,
        "correct_rule": (
            "Use response event value=correct/incorrect when present; "
            "otherwise target is incorrect and distractor is correct."
        ),
    }
    save_json(summary, str(quality_dir / "quality_report.json"))
    return summary


def parse_subjects(cfg, args) -> list[str]:
    if args.subjects:
        return [s.strip() for s in args.subjects.split(",") if s.strip()]
    if args.all:
        return available_subjects(cfg.data.bids_root)
    return list(cfg.data.get("subjects", available_subjects(cfg.data.bids_root)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build quality report and frozen 3-way dataset (RP or transition)."
    )
    parser.add_argument("--config", default="config/eeg_ws_r_3way.yaml")
    parser.add_argument("--subjects", default=None, help="comma separated")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="write event quality reports only")
    parser.add_argument("--force", action="store_true", help="rebuild dataset cache")
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="replace an existing frozen split file; do not use unless intentional",
    )
    parser.add_argument("--epoch-samples", type=int, default=None)
    parser.add_argument("--recurrence-percentile", type=float, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--quality-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(resolve_path(args.config))
    overrides = {}
    if args.epoch_samples is not None:
        overrides["data.epoch_samples"] = args.epoch_samples
    if args.recurrence_percentile is not None:
        overrides["method.recurrence_percentile"] = args.recurrence_percentile
    if args.output_root is not None:
        overrides["data.output_root"] = args.output_root
    if args.splits_dir is not None:
        overrides["data.splits_dir"] = args.splits_dir
    if args.quality_dir is not None:
        overrides["data.quality_dir"] = args.quality_dir
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))

    subjects = parse_subjects(cfg, args)
    summary = write_quality_summary(subjects, cfg)
    print(
        "quality: total={total_stimuli} correct={correct} wrong={wrong}".format(
            **summary["totals"]
        )
    )
    if args.report_only:
        return
    for subject in subjects:
        build_subject(subject, cfg, force=args.force, overwrite_split=args.overwrite_split)


if __name__ == "__main__":
    main()
