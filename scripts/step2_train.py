"""step2：固定划分训练，或 loso2（内部 8:2 早停 + 换人测试）。"""
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.augment import augment_rp_batch
from models.cnn import build_model
from utils.io_utils import ensure_dir, load_run_cfg, save_json
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from utils.visualizer import plot_confusion_matrix, plot_train_curves, plot_tsne


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_loader(model, loader, device, criterion=None):
    """准确率为全部样本微平均，不是 batch acc 再平均。"""
    model.eval()
    n, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            n += int(yb.numel())
            if criterion is not None:
                loss_sum += float(criterion(logits, yb).item()) * int(yb.numel())
    acc = float(correct / n) if n else 0.0
    if criterion is None:
        return acc
    return (float(loss_sum / n) if n else 0.0), acc


def _device(train_cfg) -> torch.device:
    return torch.device(
        str(train_cfg.device)
        if torch.cuda.is_available() and str(train_cfg.device).startswith("cuda")
        else "cpu"
    )


def _aug_params(train_cfg):
    use_aug = bool(train_cfg.get("augment", False))
    gauss_std = float(train_cfg.get("gaussian_std", 0.05) or 0.0) if use_aug else 0.0
    ch_drop = float(train_cfg.get("channel_dropout", 0.2) or 0.0) if use_aug else 0.0
    return use_aug, gauss_std, ch_drop


def _loader(x, y, bs, nw, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
    )


def train_one_split(
    cfg,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    device,
    ckpt_path,
    curve_path=None,
    stop_on="val_acc",
):
    """训一折。stop_on: val_acc（越大越好）或 val_loss（越小越好）。"""
    train_cfg = cfg.train
    num_classes = len(dict(cfg.data.class_map))
    bs, nw = int(train_cfg.batch_size), int(train_cfg.num_workers)
    use_aug, gauss_std, ch_drop = _aug_params(train_cfg)
    print(
        f"in_channels={int(X_train.shape[1])}  image={tuple(X_train.shape[2:])}  "
        f"n={len(y_train)}/{len(y_val)}/{len(y_test)}  "
        f"augment={use_aug}  dropout={float(train_cfg.dropout)}  "
        f"weight_decay={float(train_cfg.weight_decay)}"
    )

    train_loader = _loader(X_train, y_train, bs, nw, True)
    val_loader = _loader(X_val, y_val, bs, nw, False)
    test_loader = _loader(X_test, y_test, bs, nw, False)

    model = build_model(num_classes, cfg, in_channels=int(X_train.shape[1])).to(device)
    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    if opt_name == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg.lr),
            weight_decay=float(train_cfg.weight_decay),
        )
    elif opt_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(train_cfg.lr),
            weight_decay=float(train_cfg.weight_decay),
            momentum=float(train_cfg.get("momentum", 0.9)),
        )
    else:
        raise ValueError(f"未知 train.optimizer: {opt_name}")
    criterion = nn.CrossEntropyLoss()
    stop_on = str(train_cfg.get("stop_on", stop_on))
    patience = int(train_cfg.early_stop_patience)
    bad = 0
    hist_loss, hist_val = [], []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, int(train_cfg.epochs) + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if gauss_std > 0 or ch_drop > 0:
                xb = augment_rp_batch(xb, gaussian_std=gauss_std, channel_dropout=ch_drop)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss, val_acc = eval_loader(model, val_loader, device, criterion)
        hist_loss.append(train_loss)
        hist_val.append(val_acc)
        print(
            f"epoch {epoch:03d}  loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        improved = (
            val_acc > best_val_acc if stop_on == "val_acc" else val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}")
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_acc = eval_loader(model, test_loader, device)
    if curve_path and bool(cfg.output.plot_curve):
        plot_train_curves(hist_loss, hist_val, curve_path)
    return {
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_acc": float(test_acc),
        "epochs_ran": len(hist_loss),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }


def run_fixed_split(cfg) -> None:
    train_cfg = cfg.train
    set_seed(int(train_cfg.seed))
    cache_dir = str(cfg.data.cache_dir)
    paths = {
        s: (os.path.join(cache_dir, f"X_{s}.npy"), os.path.join(cache_dir, f"y_{s}.npy"))
        for s in ("train", "val", "test")
    }
    for s, (xp, yp) in paths.items():
        if not (os.path.exists(xp) and os.path.exists(yp)):
            raise FileNotFoundError(f"缺少 {s} 缓存，请先运行 scripts/step1_build_dataset.py")

    X_train, y_train = np.load(paths["train"][0]), np.load(paths["train"][1])
    X_val, y_val = np.load(paths["val"][0]), np.load(paths["val"][1])
    X_test, y_test = np.load(paths["test"][0]), np.load(paths["test"][1])
    device = _device(train_cfg)
    print(f"device={device}  train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    ckpt_dir = ensure_dir(str(train_cfg.checkpoint_dir))
    best_path = os.path.join(ckpt_dir, str(train_cfg.save_best_model))
    metrics = train_one_split(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        device,
        best_path,
        os.path.join(ckpt_dir, "train_curve.png"),
        stop_on="val_acc",
    )
    metrics.update({"protocol": "fixed_split", "device": str(device)})
    save_json(metrics, os.path.join(ckpt_dir, "metrics.json"))
    print(
        f"best_val_acc={metrics['best_val_acc']:.4f}  "
        f"test_acc={metrics['test_acc']:.4f}  saved={best_path}"
    )


def take_unique_windows(X, y, n, seed: int, stratify: bool = True):
    """从合法池中不放回抽取 n 个窗；n 为 None 或 >= 池大小则全用。"""
    if n is None or int(n) >= len(y):
        return X, y
    n = int(n)
    if n <= 0:
        raise ValueError("data.eval_samples 必须为正或 null")
    idx = np.arange(len(y))
    if stratify:
        try:
            idx, _ = train_test_split(
                idx, train_size=n, random_state=seed, shuffle=True, stratify=y
            )
        except ValueError:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(y), size=n, replace=False)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def collect_embeddings(model, loader, device):
    feats, ys = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            feats.append(model.embed(xb.to(device)).cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def collect_predictions(model, loader, device):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            ys.append(yb.numpy())
            ps.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def split_train_rest(X, y, seed: int, train_ratio=0.7, stratify=True):
    """训练集 vs 其余全部（其余用于早停和最终准确率）。"""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("data.train_ratio 必须在 (0, 1) 内")
    kwargs = dict(test_size=1.0 - train_ratio, random_state=seed, shuffle=True)
    if stratify:
        kwargs["stratify"] = y
    return train_test_split(X, y, **kwargs)


def run_within_subject(cfg) -> None:
    train_cfg = cfg.train
    data_cfg = cfg.data
    if data_cfg.get("subject") is not None:
        subject = str(data_cfg.subject)
    else:
        subjects = [str(s) for s in list(data_cfg.subjects)]
        if len(subjects) != 1:
            raise ValueError(f"within_subject 请指定 data.subject，或 subjects 长度为 1，收到 {subjects}")
        subject = subjects[0]
    seed = int(train_cfg.seed)
    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    stratify = bool(data_cfg.get("stratify", True))
    cache_dir = str(data_cfg.cache_dir)
    xp, yp = os.path.join(cache_dir, f"X_{subject}.npy"), os.path.join(
        cache_dir, f"y_{subject}.npy"
    )
    if not (os.path.exists(xp) and os.path.exists(yp)):
        raise FileNotFoundError(f"缺少 {subject} 缓存，请先运行 step1（protocol=within_subject）")

    set_seed(seed)
    X, y = np.load(xp), np.load(yp)
    X_tr, X_ev, y_tr, y_ev = split_train_rest(
        X, y, seed, train_ratio, stratify=stratify
    )
    eval_n = data_cfg.get("eval_samples", None)
    X_rep, y_rep = take_unique_windows(X_ev, y_ev, eval_n, seed, stratify=stratify)
    device = _device(train_cfg)
    ckpt_dir = ensure_dir(str(train_cfg.checkpoint_dir))
    print(
        f"protocol=within_subject  subject={subject}  device={device}  "
        f"n_train={len(y_tr)} n_pool={len(y_ev)} n_report={len(y_rep)}  "
        f"train_ratio={train_ratio}  align={cfg.method.get('paper_align', '')}"
    )
    ckpt_path = os.path.join(ckpt_dir, str(train_cfg.save_best_model))
    metrics = train_one_split(
        cfg,
        X_tr,
        y_tr,
        X_ev,
        y_ev,
        X_rep,
        y_rep,
        device,
        ckpt_path,
        os.path.join(ckpt_dir, "train_curve.png"),
        stop_on=str(train_cfg.get("stop_on", "val_loss")),
    )
    class_map = dict(data_cfg.class_map)
    names = [k for k, _ in sorted(class_map.items(), key=lambda kv: int(kv[1]))]
    labels = list(range(len(names)))
    model = build_model(len(class_map), cfg, in_channels=int(X_tr.shape[1])).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    y_true, y_pred = collect_predictions(
        model, _loader(X_rep, y_rep, int(train_cfg.batch_size), 0, False), device
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion_matrix (rows=true, cols=pred) " + " ".join(names))
    print(cm)
    if bool(cfg.output.get("plot_confusion", True)):
        plot_confusion_matrix(
            cm,
            names,
            os.path.join(ckpt_dir, "confusion_matrix.png"),
            title=f"{subject} report n={len(y_rep)}",
        )
    tsne_path = None
    if bool(cfg.output.get("run_tsne", False)) and len(y_rep) >= 4:
        feats, y_emb = collect_embeddings(
            model, _loader(X_rep, y_rep, int(train_cfg.batch_size), 0, False), device
        )
        perp = float(cfg.output.get("tsne_perplexity", 30))
        perp = max(5.0, min(perp, (len(y_emb) - 1) / 3.0))
        n_iter = int(cfg.output.get("tsne_iter", 1000))
        kw = dict(
            n_components=2,
            perplexity=perp,
            init="pca",
            random_state=seed,
            learning_rate="auto",
        )
        try:
            reducer = TSNE(**kw, max_iter=n_iter)
        except TypeError:
            reducer = TSNE(**kw, n_iter=n_iter)
        xy = reducer.fit_transform(feats)
        tsne_path = os.path.join(ckpt_dir, "tsne.png")
        plot_tsne(xy, y_emb, names, tsne_path, title=f"{subject} t-SNE n={len(y_emb)}")
        print(f"saved t-SNE {tsne_path}")
    metrics.update(
        {
            "protocol": "within_subject",
            "subject": subject,
            "device": str(device),
            "train_ratio": train_ratio,
            "n_eval_pool": int(len(y_ev)),
            "n_report": int(len(y_rep)),
            "eval_samples": None if eval_n is None else int(eval_n),
            "class_names": names,
            "confusion_matrix": cm.astype(int).tolist(),
            "tsne_path": tsne_path,
            "paper_align": str(cfg.method.get("paper_align", "")),
            "representation": str(cfg.method.representation),
            "attention": str(cfg.method.attention),
        }
    )
    save_json(metrics, os.path.join(ckpt_dir, "metrics.json"))
    print(
        f"subject={subject}  best_val_acc={metrics['best_val_acc']:.4f}  "
        f"report_acc={metrics['test_acc']:.4f}  n_report={len(y_rep)}"
    )


def run_loso2(cfg) -> None:
    train_cfg = cfg.train
    data_cfg = cfg.data
    subjects = [str(s) for s in list(data_cfg.subjects)]
    if len(subjects) != 2:
        raise ValueError(f"loso2 需要恰好 2 个被试，收到 {subjects}")
    val_ratio = float(data_cfg.get("internal_val_ratio", 0.2))
    seed = int(train_cfg.seed)
    cache_dir = str(data_cfg.cache_dir)
    device = _device(train_cfg)
    ckpt_dir = ensure_dir(str(train_cfg.checkpoint_dir))
    print(f"protocol=loso2  device={device}  subjects={subjects}  val_ratio={val_ratio}")

    folds = {}
    for fold_i, train_id in enumerate(subjects):
        test_id = [s for s in subjects if s != train_id][0]
        xp_tr, yp_tr = os.path.join(cache_dir, f"X_{train_id}.npy"), os.path.join(
            cache_dir, f"y_{train_id}.npy"
        )
        xp_te, yp_te = os.path.join(cache_dir, f"X_{test_id}.npy"), os.path.join(
            cache_dir, f"y_{test_id}.npy"
        )
        for p in (xp_tr, yp_tr, xp_te, yp_te):
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少被试缓存 {p}，请先运行 step1（protocol=loso2）")

        X_full, y_full = np.load(xp_tr), np.load(yp_tr)
        X_test, y_test = np.load(xp_te), np.load(yp_te)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_full,
            y_full,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )
        print(
            f"\n=== Fold {fold_i + 1}: train={train_id} → test={test_id}  "
            f"{len(y_tr)}/{len(y_va)}/{len(y_test)} ==="
        )
        set_seed(seed + fold_i)
        fold = train_one_split(
            cfg,
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_test,
            y_test,
            device,
            os.path.join(ckpt_dir, f"best_fold_{test_id}.pt"),
            os.path.join(ckpt_dir, f"train_curve_{test_id}.png"),
            stop_on="val_loss",
        )
        fold["train_subject"] = train_id
        fold["test_subject"] = test_id
        folds[test_id] = fold
        print(f"Acc_{test_id}={fold['test_acc']:.4f}  best_epoch={fold['best_epoch']}")

    accs = [folds[s]["test_acc"] for s in subjects]
    mean_acc = float(np.mean(accs))
    metrics = {
        "protocol": "loso2",
        "folds": folds,
        "test_acc_mean": mean_acc,
        "device": str(device),
        "subjects": subjects,
        "internal_val_ratio": val_ratio,
        "seed": seed,
    }
    save_json(metrics, os.path.join(ckpt_dir, "metrics.json"))
    print(f"\n=== Final Cross-Subject Accuracy (Mean) === {mean_acc:.4f}")
    for s in subjects:
        print(f"  Acc_{s}={folds[s]['test_acc']:.4f}")


def main(cfg_path: str = "config/eeg_ws_r.yaml") -> None:
    cfg = load_run_cfg(cfg_path)
    protocol = str(cfg.data.get("protocol", "subject_split")).lower()
    if protocol == "loso2":
        run_loso2(cfg)
    elif protocol == "within_subject":
        run_within_subject(cfg)
    else:
        run_fixed_split(cfg)


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/eeg_ws_r.yaml"
    main(cfg)
