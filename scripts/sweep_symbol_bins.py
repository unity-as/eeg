"""扫 symbol_bins。bins=6 复用正式 [1,2,3] 结果。

用法：
  python scripts/sweep_symbol_bins.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.io_utils import load_json, save_json

BINS = (5, 6, 8, 10, 12)


def _dirs(subject: str, epoch: int, bins: int) -> tuple[str, str]:
    tag = f"b{bins}"
    cache = f"./datas/artifacts/eeg_cache_ws_transition/{subject}/ep{epoch}/{tag}"
    ckpt = f"./datas/artifacts/checkpoints_eeg_ws_transition/{subject}/ep{epoch}/{tag}"
    return cache, ckpt


def _row(bins: int, metrics: dict, ckpt: str) -> dict:
    return {
        "symbol_bins": bins,
        "report_acc": metrics.get("test_acc"),
        "best_epoch": metrics.get("best_epoch"),
        "epochs_ran": metrics.get("epochs_ran"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "checkpoint_dir": ckpt,
    }


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/eeg_ws_transition.yaml"
    subject, epoch = "sub-002", 512
    rows = []
    py = sys.executable
    for bins in BINS:
        cache, ckpt = _dirs(subject, epoch, bins)
        mp = os.path.join(ROOT, ckpt, "metrics.json")
        if bins == 6 and not os.path.isfile(mp):
            official = os.path.join(
                ROOT,
                "datas",
                "artifacts",
                "checkpoints_eeg_ws_transition",
                subject,
                f"ep{epoch}",
                "metrics.json",
            )
            if os.path.isfile(official):
                ckpt = f"./datas/artifacts/checkpoints_eeg_ws_transition/{subject}/ep{epoch}"
                mp = official
        if os.path.isfile(mp):
            print(f"\n===== 已有 bins={bins}，跳过 =====")
        else:
            print(f"\n===== sweep symbol_bins={bins} =====")
            cmd = [
                py,
                os.path.join(ROOT, "main_pipeline.py"),
                cfg_path,
                f"method.symbol_bins={bins}",
                f"data.cache_dir={cache}",
                f"train.checkpoint_dir={ckpt}",
            ]
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"bins={bins} 失败，停止。")
                sys.exit(result.returncode)
        rows.append(_row(bins, load_json(mp), ckpt))

    out = os.path.join(ROOT, "datas", "artifacts", "sweeps", "symbol_bins.json")
    save_json({"config": cfg_path, "rows": rows}, out)
    print("\nbins  report  best_epoch")
    for r in rows:
        acc = r["report_acc"]
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
        print(f"{r['symbol_bins']:>4}  {acc_s}  {r['best_epoch']}")
    print(f"汇总: {out}")


if __name__ == "__main__":
    main()
