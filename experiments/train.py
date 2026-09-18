from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import available_subjects, resolve_path
from data.eeg_ds002680 import CLASS_MAP
from models.cnn import build_model
from utils.io_utils import ensure_dir, load_json, save_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg) -> torch.device:
    requested = str(cfg.train.get("device", "cuda"))
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def loader(x, y, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
    )


def evaluate(model, data_loader, device, criterion=None, predictions=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    true_all, pred_all = [], []
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
    acc = float(correct / total) if total else 0.0
    loss = float(loss_sum / total) if total else 0.0
    if predictions:
        return loss, acc, np.asarray(true_all, dtype=np.int64), np.asarray(pred_all, dtype=np.int64)
    return loss, acc

def train_subject(subject: str, cfg) -> None:
    data_dir = resolve_path(cfg.data.output_root) / subject
    X = np.load(data_dir / "X.npy")
    if X.dtype == np.uint8:
        X = X.astype(np.float32) / 255.0
    else:
        X = X.astype(np.float32)
    y = np.load(data_dir / "y.npy")
    split = load_json(str(data_dir / "split_indices.json"))
    train_idx = np.asarray(split["indices"]["train"], dtype=np.int64)
    val_idx = np.asarray(split["indices"]["val"], dtype=np.int64)
    test_idx = np.asarray(split["indices"]["test"], dtype=np.int64)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    seed = int(cfg.train.seed)
    set_seed(seed)
    device = get_device(cfg)
    out_dir = Path(ensure_dir(str(resolve_path(cfg.train.output_dir) / subject)))
    ckpt_path = out_dir / str(cfg.train.get("checkpoint_name", "best.pt"))
    workers = int(cfg.train.get("num_workers", 0))
    batch_size = int(cfg.train.batch_size)
    train_loader = loader(X_train, y_train, batch_size, workers, True)
    val_loader = loader(X_val, y_val, batch_size, workers, False)
    test_loader = loader(X_test, y_test, batch_size, workers, False)

    criterion = nn.CrossEntropyLoss()
    model = build_model(len(CLASS_MAP), cfg, in_channels=int(X_train.shape[1])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    stop_on = str(cfg.train.get("stop_on", "val_loss"))
    patience = int(cfg.train.early_stop_patience)
    best_val_acc, best_val_loss, best_epoch, bad = -1.0, float("inf"), 0, 0
    losses, val_accs = [], []

    print(
        f"{subject}: train={len(y_train)} val={len(y_val)} test={len(y_test)} "
        f"shape={X_train.shape[1:]} device={device} stop_on={stop_on}"
    )
    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        losses.append(train_loss)
        val_accs.append(val_acc)
        print(
            f"epoch {epoch:03d} loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        improved = val_acc > best_val_acc if stop_on == "val_acc" else val_loss < best_val_loss
        if improved:
            best_val_acc, best_val_loss, best_epoch, bad = val_acc, val_loss, epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}")
                break

    from utils.visualizer import plot_confusion_matrix, plot_train_curves

    plot_train_curves(losses, val_accs, str(out_dir / "train_curve.png"))
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, device, criterion, predictions=True
    )
    names = [k for k, _ in sorted(CLASS_MAP.items(), key=lambda kv: int(kv[1]))]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    plot_confusion_matrix(
        cm,
        names,
        str(out_dir / "test_confusion_matrix.png"),
        title=f"{subject} independent test",
    )
    np.savetxt(
        out_dir / "test_predictions.csv",
        np.column_stack([np.arange(len(y_true)), y_true, y_pred]),
        delimiter=",",
        header="sample_index,true_label,pred_label",
        comments="",
        fmt="%d",
    )
    metrics = {
        "subject": subject,
        "protocol": "within_subject_3way",
        "correct_only": bool(cfg.data.get("correct_only", False)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "input_shape": list(X_train.shape[1:]),
        "class_names": names,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(losses)),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    save_json(metrics, str(out_dir / "metrics.json"))
    print(
        f"{subject}: best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f} "
        f"balanced_acc={metrics['test_balanced_acc']:.4f} "
        f"macro_f1={metrics['test_macro_f1']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/config_rp_se.yaml")
    parser.add_argument("--subjects", default=None, help="comma separated")
    parser.add_argument("--all", action="store_true")
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
        train_subject(subject, cfg)


if __name__ == "__main__":
    main()
