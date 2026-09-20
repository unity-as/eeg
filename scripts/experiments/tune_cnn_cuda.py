"""Val-only CNN hyperparameter probe on CUDA. Does not open test."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.train_3way import train_one


def main() -> None:
    base = OmegaConf.load(ROOT / "config/eeg_ws_transition_3way_global_none.yaml")
    base.train.require_gpu = True
    base.train.device = "cuda"
    base.train.seeds = [42]
    base.data.subjects = ["sub-002", "sub-008", "sub-015"]
    candidates = [
        {"lr": 0.0003, "dropout": 0.2, "batch_size": 32},
        {"lr": 0.0005, "dropout": 0.2, "batch_size": 32},
        {"lr": 0.001, "dropout": 0.2, "batch_size": 32},
        {"lr": 0.0005, "dropout": 0.3, "batch_size": 32},
        {"lr": 0.001, "dropout": 0.2, "batch_size": 64},
    ]
    rows = []
    for i, cand in enumerate(candidates):
        cfg = OmegaConf.merge(base, OmegaConf.create({"train": cand}))
        cfg.train.output_dir = f"./datas/experiments_3way/transition/tune_cnn_cuda/c{i}"
        cfg.train.checkpoint_name = "best_tune.pt"
        vals = []
        for subject in cfg.data.subjects:
            result = train_one(subject, 42, cfg, evaluate_test=False, confirm_test=False)
            vals.append(float(result["best_val_acc"]))
        mean_val = sum(vals) / len(vals)
        rows.append({**cand, "mean_val": mean_val, "vals": vals})
        print(f"CAND {i} {cand} mean_val={mean_val:.4f} vals={vals}", flush=True)
    best = max(rows, key=lambda r: r["mean_val"])
    out = ROOT / "datas/experiments_3way/transition/tune_cnn_cuda/best.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(best), out)
    print(f"BEST {best}", flush=True)


if __name__ == "__main__":
    main()
