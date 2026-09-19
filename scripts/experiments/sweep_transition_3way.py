"""三集协议上扫 Transition 表示。只看 val，不打开 test。

bins=6、各导平均复用已有 checkpoint。其余写入独立目录。

用法：
  python scripts/experiments/sweep_transition_3way.py
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from experiments.common import resolve_path
from utils.io_utils import load_json, save_json

BASE = "config/eeg_ws_transition_3way.yaml"
SWEEP_ROOT = "./datas/experiments_3way/transition/sweep"
VARIANTS = (
    {"tag": "b6_mean", "symbol_bins": 6, "transition_multichannel": "mean", "reuse_official": True},
    {"tag": "b5_mean", "symbol_bins": 5, "transition_multichannel": "mean"},
    {"tag": "b8_mean", "symbol_bins": 8, "transition_multichannel": "mean"},
    {"tag": "b10_mean", "symbol_bins": 10, "transition_multichannel": "mean"},
    {"tag": "b12_mean", "symbol_bins": 12, "transition_multichannel": "mean"},
    {"tag": "b6_stack", "symbol_bins": 6, "transition_multichannel": "stack"},
)


def _cfg(variant: dict):
    cfg = OmegaConf.load(resolve_path(BASE))
    if variant.get("reuse_official"):
        return cfg
    tag = variant["tag"]
    return OmegaConf.merge(
        cfg,
        {
            "method": {
                "symbol_bins": int(variant["symbol_bins"]),
                "transition_multichannel": variant["transition_multichannel"],
            },
            "data": {
                "output_root": f"{SWEEP_ROOT}/representations/{tag}",
                "quality_dir": f"{SWEEP_ROOT}/quality",
            },
            "train": {"output_dir": f"{SWEEP_ROOT}/checkpoints/{tag}"},
        },
    )


def _run(script: str, cfg_path: Path) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "experiments" / script), "--config", str(cfg_path)]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _metrics_path(cfg, subject: str, seed: int) -> Path:
    return resolve_path(cfg.train.output_dir) / subject / f"seed_{seed}" / "metrics.json"


def _row(tag: str, variant: dict, metrics: list[dict]) -> dict:
    by_subject = {}
    for item in metrics:
        by_subject.setdefault(item["subject"], []).append(float(item["best_val_acc"]))
    subject_means = [sum(v) / len(v) for v in by_subject.values()]
    n = len(subject_means)
    mean = sum(subject_means) / n
    var = sum((x - mean) ** 2 for x in subject_means) / n
    std = math.sqrt(var)
    ci = 1.96 * std / math.sqrt(n) if n else 0.0
    return {
        "tag": tag,
        "symbol_bins": int(variant["symbol_bins"]),
        "transition_multichannel": variant["transition_multichannel"],
        "n_subjects": n,
        "n_runs": len(metrics),
        "test_evaluated": any(bool(item.get("test_evaluated")) for item in metrics),
        "mean_val_acc": mean,
        "std_val_acc": std,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
    }


def main() -> None:
    base = OmegaConf.load(resolve_path(BASE))
    subjects = list(base.data.subjects)
    seeds = [int(s) for s in base.train.seeds]
    cfg_dir = resolve_path(f"{SWEEP_ROOT}/configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        cfg = _cfg(variant)
        tag = variant["tag"]
        cfg_path = cfg_dir / f"{tag}.yaml"
        OmegaConf.save(cfg, cfg_path)
        missing_cache = any(
            not (resolve_path(cfg.data.output_root) / subject / "meta.json").exists()
            for subject in subjects
        )
        if missing_cache:
            _run("build_correct_only_dataset.py", cfg_path)
        missing_train = any(
            not _metrics_path(cfg, subject, seed).exists()
            for subject in subjects
            for seed in seeds
        )
        if missing_train:
            _run("train_3way.py", cfg_path)
        collected = []
        for subject in subjects:
            for seed in seeds:
                path = _metrics_path(cfg, subject, seed)
                item = load_json(str(path))
                if item.get("test_evaluated"):
                    raise RuntimeError(f"{path} opened test; sweep must stay sealed")
                collected.append(item)
        rows.append(_row(tag, variant, collected))
        print(
            f"{tag}: val={rows[-1]['mean_val_acc']:.4f} "
            f"n={rows[-1]['n_runs']} test_evaluated={rows[-1]['test_evaluated']}"
        )

    out = resolve_path(f"{SWEEP_ROOT}/sweep_summary.json")
    save_json({"config": BASE, "seeds": seeds, "rows": rows}, str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
