"""step2：train 训练，val early-stop，test 最终报告。"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

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


def eval_loader(model, loader, device) -> float:
    model.eval()
    accs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            accs.append(accuracy(model(xb), yb))
    return float(np.mean(accs)) if accs else 0.0


def main(cfg_path: str = "config/default.yaml") -> None:
    cfg = OmegaConf.load(cfg_path)
    data_cfg = cfg.data
    train_cfg = cfg.train
    set_seed(int(train_cfg.seed))

    cache_dir = str(data_cfg.cache_dir)
    paths = {
        s: (
            os.path.join(cache_dir, f"X_{s}.npy"),
            os.path.join(cache_dir, f"y_{s}.npy"),
        )
        for s in ("train", "val", "test")
    }
    for s, (xp, yp) in paths.items():
        if not (os.path.exists(xp) and os.path.exists(yp)):
            raise FileNotFoundError(
                f"缺少 {s} 缓存，请先用方案 B 运行 scripts/step1_build_dataset.py"
            )

    X_train, y_train = np.load(paths["train"][0]), np.load(paths["train"][1])
    X_val, y_val = np.load(paths["val"][0]), np.load(paths["val"][1])
    X_test, y_test = np.load(paths["test"][0]), np.load(paths["test"][1])
    num_classes = len(dict(data_cfg.class_map))

    device = torch.device(
        str(train_cfg.device)
        if torch.cuda.is_available() and str(train_cfg.device).startswith("cuda")
        else "cpu"
    )
    print(
        f"device={device}  train={len(y_train)} val={len(y_val)} test={len(y_test)}"
    )

    bs = int(train_cfg.batch_size)
    nw = int(train_cfg.num_workers)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
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
    best_val = -1.0
    patience = int(train_cfg.early_stop_patience)
    bad = 0
    hist_loss, hist_val = [], []

    for epoch in range(1, int(train_cfg.epochs) + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_acc = eval_loader(model, val_loader, device)
        hist_loss.append(train_loss)
        hist_val.append(val_acc)
        print(f"epoch {epoch:03d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": best_val,
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

    # 加载最佳后在 test 上报告
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_acc = eval_loader(model, test_loader, device)

    if bool(cfg.output.plot_curve):
        plot_train_curves(hist_loss, hist_val, os.path.join(ckpt_dir, "train_curve.png"))

    metrics = {
        "protocol": "B_temporal_then_nonoverlap",
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "epochs_ran": len(hist_loss),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "device": str(device),
    }
    save_json(metrics, os.path.join(ckpt_dir, "metrics.json"))
    print(
        f"best_val_acc={best_val:.4f}  test_acc={test_acc:.4f}  saved={best_path}"
    )


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    main(cfg)
