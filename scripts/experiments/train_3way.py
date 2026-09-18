from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEP_SITE = Path(
    os.environ.get("EEG_DEP_SITE_PACKAGES", r"D:\MINICODA\envs\eeg\Lib\site-packages")
)
if _DEP_SITE.exists() and str(_DEP_SITE) not in sys.path:
    sys.path.append(str(_DEP_SITE))

from data.eeg_ds002680 import CLASS_MAP
from experiments.common import available_subjects, class_count_dict, resolve_path
from models.cnn import build_model
from utils.io_utils import ensure_dir, load_json, save_json

CLASS_NAMES = [k for k, _ in sorted(CLASS_MAP.items(), key=lambda kv: int(kv[1]))]


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(cfg) -> torch.device:
    requested = str(cfg.train.get("device", "cuda"))
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if bool(cfg.train.get("require_gpu", False)):
            raise RuntimeError("CUDA is required but unavailable in this Python environment")
    return torch.device("cpu")


def load_array_pair(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x_path = data_dir / f"X_{split}.npy"
    y_path = data_dir / f"y_{split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"missing {split} arrays: {x_path}, {y_path}")
    x = np.load(x_path)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    return x, y


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
    )


def evaluate(model, data_loader, device, criterion=None, predictions: bool = False):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    true_all = []
    pred_all = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            if criterion is not None:
                loss_sum += float(criterion(logits, yb).item()) * int(yb.numel())
            if predictions:
                true_all.extend(yb.cpu().numpy().tolist())
                pred_all.extend(pred.cpu().numpy().tolist())

    loss = float(loss_sum / total) if total else 0.0
    acc = float(correct / total) if total else 0.0
    if predictions:
        return loss, acc, np.asarray(true_all, dtype=np.int64), np.asarray(pred_all, dtype=np.int64)
    return loss, acc


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    return {
        f"{prefix}_acc": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        f"{prefix}_balanced_acc": float(
            balanced_accuracy_score(y_true, y_pred) if len(y_true) else 0.0
        ),
        f"{prefix}_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0
        ),
        f"{prefix}_confusion_matrix": cm.astype(int).tolist(),
        f"{prefix}_class_counts": class_count_dict(y_true),
    }


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    np.savetxt(
        path,
        np.column_stack([np.arange(len(y_true)), y_true, y_pred]),
        delimiter=",",
        header="sample_index,true_label,pred_label",
        comments="",
        fmt="%d",
    )


def train_one(subject: str, seed: int, cfg, evaluate_test: bool, confirm_test: bool) -> dict:
    if evaluate_test and not confirm_test:
        raise RuntimeError("test evaluation requires --confirm-test after parameters are frozen")

    data_dir = resolve_path(cfg.data.output_root) / subject
    meta = load_json(str(data_dir / "meta.json"))
    X_train, y_train = load_array_pair(data_dir, "train")
    X_val, y_val = load_array_pair(data_dir, "val")
    X_test = y_test = None
    if evaluate_test:
        X_test, y_test = load_array_pair(data_dir, "test")

    set_seed(seed, deterministic=bool(cfg.train.get("deterministic", True)))
    device = get_device(cfg)
    workers = int(cfg.train.get("num_workers", 0))
    batch_size = int(cfg.train.batch_size)
    train_loader = make_loader(X_train, y_train, batch_size, workers, True)
    val_loader = make_loader(X_val, y_val, batch_size, workers, False)

    criterion = nn.CrossEntropyLoss()
    model = build_model(len(CLASS_NAMES), cfg, in_channels=int(X_train.shape[1])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    run_dir = Path(ensure_dir(str(resolve_path(cfg.train.output_dir) / subject / f"seed_{seed}")))
    ckpt_path = run_dir / str(cfg.train.get("checkpoint_name", "best.pt"))
    stop_on = str(cfg.train.get("stop_on", "val_acc"))
    patience = int(cfg.train.get("early_stop_patience", 30))
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = {}
    bad_epochs = 0
    train_losses = []
    val_accs = []

    print(
        f"{subject} seed={seed}: train={len(y_train)} val={len(y_val)} "
        f"test={'sealed' if not evaluate_test else len(y_test)} shape={X_train.shape[1:]} device={device}"
    )
    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss, val_acc, val_true, val_pred = evaluate(
            model, val_loader, device, criterion, predictions=True
        )
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        print(
            f"epoch {epoch:03d} loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if stop_on == "val_loss":
            improved = val_loss < best_val_loss
        else:
            improved = val_acc > best_val_acc

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_metrics = metrics_from_predictions(val_true, val_pred, "val")
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch}")
                break

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _, _, val_true, val_pred = evaluate(model, val_loader, device, criterion, predictions=True)
    save_predictions(run_dir / "val_predictions.csv", val_true, val_pred)

    result = {
        "subject": subject,
        "seed": int(seed),
        "protocol": "within_subject_3way",
        "method": "{}_{}_cnn".format(
            str(cfg.method.get("representation", "rp")).lower(),
            str(cfg.method.get("attention", "none")).lower(),
        ),
        "correct_only": bool(cfg.data.get("correct_only", False)),
        "epoch_samples": int(cfg.data.epoch_samples),
        "split_seed": int(meta["split_seed"]),
        "split_sample_id_fingerprint": str(meta["split_sample_id_fingerprint"]),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(train_losses)),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_evaluated": bool(evaluate_test),
        "input_shape": list(X_train.shape[1:]) if X_train.ndim else [],
        "class_names": CLASS_NAMES,
        "train_class_counts": class_count_dict(y_train),
        "val_class_counts": class_count_dict(y_val),
        **best_metrics,
    }

    if evaluate_test:
        test_loss, test_acc, test_true, test_pred = evaluate(
            model, make_loader(X_test, y_test, batch_size, workers, False), device, criterion, predictions=True
        )
        save_predictions(run_dir / "test_predictions.csv", test_true, test_pred)
        result.update(
            {
                "test_loss": float(test_loss),
                "test_acc_once": float(test_acc),
                "test_evaluated_after_freeze": True,
                **metrics_from_predictions(test_true, test_pred, "test"),
            }
        )

    save_json(result, str(run_dir / "metrics.json"))
    print(
        f"{subject} seed={seed}: best_val_acc={best_val_acc:.4f} "
        f"best_epoch={best_epoch} test_evaluated={evaluate_test}"
    )
    return result


def parse_subjects(cfg, args) -> list[str]:
    if args.subjects:
        return [s.strip() for s in args.subjects.split(",") if s.strip()]
    if args.all:
        return available_subjects(cfg.data.bids_root)
    return list(cfg.data.get("subjects", available_subjects(cfg.data.bids_root)))


def parse_seeds(cfg, args) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    seeds = cfg.train.get("seeds", None)
    if seeds is not None:
        return [int(s) for s in seeds]
    return [int(cfg.train.get("seed", 42))]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train frozen-split 3-way CNN (RP or transition) with test sealed by default."
    )
    parser.add_argument("--config", default="config/eeg_ws_r_3way.yaml")
    parser.add_argument("--subjects", default=None, help="comma separated")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", default=None, help="comma separated training seeds")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="required with --evaluate-test after representation/model/training parameters are frozen",
    )
    parser.add_argument("--epoch-samples", type=int, default=None)
    parser.add_argument("--recurrence-percentile", type=float, default=None)
    parser.add_argument("--output-root", default=None)
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
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))

    subjects = parse_subjects(cfg, args)
    seeds = parse_seeds(cfg, args)
    for subject in subjects:
        for seed in seeds:
            train_one(
                subject,
                seed,
                cfg,
                evaluate_test=bool(args.evaluate_test),
                confirm_test=bool(args.confirm_test),
            )


if __name__ == "__main__":
    main()
