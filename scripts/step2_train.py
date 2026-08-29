"""step2：训练 / 评估 RP-CNN。"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.cnn import build_model
from utils.io_utils import ensure_dir, save_json
from utils.visualizer import plot_train_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def main(cfg_path: str = "config/default.yaml") -> None:
    cfg = OmegaConf.load(cfg_path)
    data_cfg = cfg.data
    train_cfg = cfg.train
    set_seed(int(train_cfg.seed))

    cache_dir = str(data_cfg.cache_dir)
    x_path = os.path.join(cache_dir, "X.npy")
    y_path = os.path.join(cache_dir, "y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError("缺少缓存，请先运行 scripts/step1_build_dataset.py")

    X = np.load(x_path)
    y = np.load(y_path)
    num_classes = len(dict(data_cfg.class_map))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(data_cfg.test_ratio),
        random_state=int(data_cfg.random_state),
        stratify=y,
    )

    device = torch.device(
        str(train_cfg.device) if torch.cuda.is_available() and str(train_cfg.device).startswith("cuda") else "cpu"
    )
    print(f"device={device}  train={len(y_train)} test={len(y_test)}")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        ),
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(train_cfg.num_workers),
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test),
        ),
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(train_cfg.num_workers),
    )

    model = build_model(num_classes, cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = ensure_dir(str(train_cfg.checkpoint_dir))
    best_path = os.path.join(ckpt_dir, str(train_cfg.save_best_model))
    best_acc = -1.0
    patience = int(train_cfg.early_stop_patience)
    bad = 0
    hist_loss, hist_acc = [], []

    for epoch in range(1, int(train_cfg.epochs) + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        accs = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                accs.append(accuracy(model(xb), yb))
        train_loss = float(np.mean(losses)) if losses else 0.0
        val_acc = float(np.mean(accs)) if accs else 0.0
        hist_loss.append(train_loss)
        hist_acc.append(val_acc)
        print(f"epoch {epoch:03d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": best_acc,
                    "epoch": epoch,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}")
                break

    if bool(cfg.output.plot_curve):
        plot_train_curves(hist_loss, hist_acc, os.path.join(ckpt_dir, "train_curve.png"))

    save_json(
        {"best_val_acc": best_acc, "epochs_ran": len(hist_loss), "device": str(device)},
        os.path.join(ckpt_dir, "metrics.json"),
    )
    print(f"best_val_acc={best_acc:.4f}  saved={best_path}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    main(cfg)
