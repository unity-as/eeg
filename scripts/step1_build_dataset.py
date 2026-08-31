"""step1：构建表示图像缓存（ds002680 EEG 4 类）。"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from representation.embed_params import resolve_embed_params
from representation.recurrence_plot import build_representation
from utils.io_utils import ensure_dir, load_json, load_run_cfg, save_json


def _encode(signals, labels, method):
    images = []
    for sig in tqdm(signals, desc="build images"):
        rp, _meta = build_representation(sig, method)
        rp = np.asarray(rp, dtype=np.float32)
        if rp.ndim == 2:
            rp = rp[None, :, :]
        elif rp.ndim != 3:
            raise ValueError(f"表示输出须为 [H,W] 或 [C,H,W]，收到 {rp.shape}")
        images.append(rp)
    if not images:
        h = int(method.rp_image_size)
        return np.zeros((0, 1, h, h), dtype=np.float32), labels
    x = np.stack(images, axis=0)
    return x, labels.astype(np.int64)


def _save_splits(cache_dir, splits, method, meta_extra):
    cache_dir = ensure_dir(cache_dir)
    train_sig = splits.get("train", ((), None))[0]
    method, embed_info = _resolve_method(list(train_sig), method)
    counts = {}
    for split, (signals, labels) in splits.items():
        x, y = _encode(signals, labels, method)
        np.save(os.path.join(cache_dir, f"X_{split}.npy"), x)
        np.save(os.path.join(cache_dir, f"y_{split}.npy"), y)
        counts[split] = int(y.shape[0])
        print(f"saved {split}: X{x.shape} y{y.shape}")
    if counts.get("train", 0) == 0:
        raise RuntimeError("训练集为空")
    meta = {
        "counts": counts,
        "method": OmegaConf.to_container(method, resolve=True),
        "embed_resolved": embed_info,
    }
    meta.update(meta_extra)
    save_json(meta, os.path.join(cache_dir, "meta.json"))
    print(f"已保存: {cache_dir}  counts={counts}")


def _resolve_method(signals, method):
    method = OmegaConf.create(OmegaConf.to_container(method, resolve=True))
    m, tau, info = resolve_embed_params(signals, method)
    OmegaConf.update(method, "embedding_dim", int(m), merge=True)
    OmegaConf.update(method, "time_delay", int(tau), merge=True)
    mode = info.get("mode", str(method.get("embed_select", "auto")))
    print(
        f"相空间 {mode}  m={int(m)}  tau={int(tau)}"
        + (f"  n_traces={info.get('n_traces')}" if info.get("n_traces") is not None else "")
    )
    return method, info


def _save_subjects(cache_dir, by_sub, subjects, method, meta_extra):
    from data.eeg_ds002680 import pack_subject

    cache_dir = ensure_dir(cache_dir)
    packed = {}
    all_sig = []
    for sub in subjects:
        signals, labels = pack_subject(by_sub, sub)
        packed[sub] = (signals, labels)
        all_sig.extend(signals)
    method, embed_info = _resolve_method(all_sig, method)
    counts = {}
    for sub in subjects:
        signals, labels = packed[sub]
        x, y = _encode(signals, labels, method)
        np.save(os.path.join(cache_dir, f"X_{sub}.npy"), x)
        np.save(os.path.join(cache_dir, f"y_{sub}.npy"), y)
        counts[sub] = int(y.shape[0])
        print(f"saved {sub}: X{x.shape} y{y.shape}")
    if sum(counts.values()) == 0:
        raise RuntimeError("按被试缓存为空")
    meta = {
        "counts": counts,
        "method": OmegaConf.to_container(method, resolve=True),
        "embed_resolved": embed_info,
    }
    meta.update(meta_extra)
    save_json(meta, os.path.join(cache_dir, "meta.json"))
    print(f"已保存: {cache_dir}  counts={counts}")


def _import_split_as_subjects(cfg):
    """把旧的 X_train/X_val 缓存映射成 X_<subject>.npy，避免重复算 RP。"""
    from data.eeg_ds002680 import CLASS_MAP

    data_cfg = cfg.data
    src = str(data_cfg.import_split_cache)
    mapping = dict(data_cfg.import_split_map)
    dest = ensure_dir(str(data_cfg.cache_dir))
    counts = {}
    for split, sub in mapping.items():
        xp, yp = os.path.join(src, f"X_{split}.npy"), os.path.join(src, f"y_{split}.npy")
        if not (os.path.exists(xp) and os.path.exists(yp)):
            raise FileNotFoundError(f"复用缓存缺少 {split}: {xp}")
        dest_x = os.path.join(dest, f"X_{sub}.npy")
        dest_y = os.path.join(dest, f"y_{sub}.npy")
        shutil.copy2(xp, dest_x)
        shutil.copy2(yp, dest_y)
        counts[str(sub)] = int(np.load(dest_y).shape[0])
        print(f"import {split} → {sub}: {counts[str(sub)]}")
    meta = {
        "counts": counts,
        "method": OmegaConf.to_container(cfg.method, resolve=True),
        "protocol": str(data_cfg.get("protocol", "loso2")),
        "source": "ds002680",
        "class_map": CLASS_MAP,
        "subjects": list(data_cfg.subjects),
        "imported_from": src,
        "import_split_map": {str(k): str(v) for k, v in mapping.items()},
        "epoch_samples": int(data_cfg.epoch_samples),
    }
    save_json(meta, os.path.join(dest, "meta.json"))
    print(f"已保存: {dest}  counts={counts}")


def _sample_cap(data_cfg):
    raw = data_cfg.get("max_samples", None)
    if raw is None:
        raw = data_cfg.get("max_epochs_per_subject", None)
    return None if raw is None else int(raw)


def _fingerprint(cfg) -> dict:
    method = OmegaConf.to_container(cfg.method, resolve=True)
    method.pop("paper_align", None)
    d = cfg.data
    cap = _sample_cap(d)
    if cap is not None:
        cap = int(cap)
    runs = d.get("max_runs", None)
    if runs is not None:
        runs = int(runs)
    return {
        "subject": str(d.subject),
        "max_samples": cap,
        "max_runs": runs,
        "epoch_samples": int(d.epoch_samples),
        "method": method,
    }


def _within_cache_ok(cfg) -> bool:
    cache_dir = str(cfg.data.cache_dir)
    sub = str(cfg.data.subject)
    meta_p = os.path.join(cache_dir, "meta.json")
    xp = os.path.join(cache_dir, f"X_{sub}.npy")
    yp = os.path.join(cache_dir, f"y_{sub}.npy")
    if not (os.path.exists(meta_p) and os.path.exists(xp) and os.path.exists(yp)):
        return False
    meta = load_json(meta_p)
    return meta.get("fingerprint") == _fingerprint(cfg)


def build_ds002680(cfg):
    from data.eeg_ds002680 import CLASS_MAP, collect_epochs_by_subject, subject_split_to_arrays

    data_cfg = cfg.data
    subjects = list(data_cfg.subjects)
    protocol = str(data_cfg.get("protocol", "subject_split")).lower()
    if data_cfg.get("subject") is not None:
        print(
            f"data.subject={data_cfg.subject}  protocol={protocol}  "
            f"max_samples={_sample_cap(data_cfg)}  epoch_samples={int(data_cfg.epoch_samples)}"
        )

    if protocol == "within_subject":
        if not bool(data_cfg.get("force_rebuild", False)) and _within_cache_ok(cfg):
            print(f"缓存与配置一致，跳过重建: {data_cfg.cache_dir}")
            return
        by_sub = collect_epochs_by_subject(
            bids_root=str(data_cfg.bids_root),
            subjects=subjects,
            epoch_samples=int(data_cfg.epoch_samples),
            max_epochs_per_subject=_sample_cap(data_cfg),
            max_runs_per_subject=data_cfg.get("max_runs", None),
        )
        _save_subjects(
            str(data_cfg.cache_dir),
            by_sub,
            subjects,
            cfg.method,
            {
                "source": "ds002680",
                "class_map": CLASS_MAP,
                "subjects": subjects,
                "epoch_samples": int(data_cfg.epoch_samples),
                "protocol": protocol,
                "fingerprint": _fingerprint(cfg),
            },
        )
        return

    if protocol == "loso2" and data_cfg.get("import_split_cache"):
        _import_split_as_subjects(cfg)
        return

    by_sub = collect_epochs_by_subject(
        bids_root=str(data_cfg.bids_root),
        subjects=subjects,
        epoch_samples=int(data_cfg.epoch_samples),
        max_epochs_per_subject=_sample_cap(data_cfg),
        max_runs_per_subject=data_cfg.get("max_runs", None),
    )
    common_meta = {
        "source": "ds002680",
        "class_map": CLASS_MAP,
        "subjects": subjects,
        "epoch_samples": int(data_cfg.epoch_samples),
    }
    if protocol == "loso2":
        _save_subjects(
            str(data_cfg.cache_dir),
            by_sub,
            subjects,
            cfg.method,
            {**common_meta, "protocol": protocol},
        )
        return

    splits = subject_split_to_arrays(
        by_sub,
        train_subjects=list(data_cfg.train_subjects),
        val_subjects=list(data_cfg.val_subjects),
        test_subjects=list(data_cfg.test_subjects),
    )
    _save_splits(
        str(data_cfg.cache_dir),
        splits,
        cfg.method,
        {
            **common_meta,
            "protocol": "subject_split_epochs",
            "train_subjects": list(data_cfg.train_subjects),
            "val_subjects": list(data_cfg.val_subjects),
            "test_subjects": list(data_cfg.test_subjects),
        },
    )


def main(cfg_path: str = "config/eeg_ws_r.yaml") -> None:
    cfg = load_run_cfg(cfg_path)
    source = str(cfg.data.get("source", "ds002680")).lower()
    if source in ("ds002680", "eeg", "openneuro"):
        build_ds002680(cfg)
    else:
        raise ValueError(f"未知 data.source: {source}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/eeg_ws_r.yaml"
    main(cfg)
