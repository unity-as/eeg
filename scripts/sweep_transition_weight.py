"""扫 transition_weight。probability 复用正式结果。

用法：
  python scripts/sweep_transition_weight.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.io_utils import load_json, save_json

WEIGHTS = ("probability", "row_probability", "count")


def _dirs(subject: str, epoch: int, weight: str) -> tuple[str, str]:
    cache = f"./datas/artifacts/eeg_cache_ws_transition/{subject}/ep{epoch}/w_{weight}"
    ckpt = f"./datas/artifacts/checkpoints_eeg_ws_transition/{subject}/ep{epoch}/w_{weight}"
    return cache, ckpt


def _row(weight: str, metrics: dict, ckpt: str) -> dict:
    return {
        "transition_weight": weight,
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
    for weight in WEIGHTS:
        cache, ckpt = _dirs(subject, epoch, weight)
        mp = os.path.join(ROOT, ckpt, "metrics.json")
        if weight == "probability" and not os.path.isfile(mp):
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
            print(f"\n===== 已有 weight={weight}，跳过 =====")
        else:
            print(f"\n===== sweep transition_weight={weight} =====")
            cmd = [
                py,
                os.path.join(ROOT, "main_pipeline.py"),
                cfg_path,
                f"method.transition_weight={weight}",
                f"data.cache_dir={cache}",
                f"train.checkpoint_dir={ckpt}",
            ]
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"weight={weight} 失败，停止。")
                sys.exit(result.returncode)
        rows.append(_row(weight, load_json(mp), ckpt))

    out = os.path.join(ROOT, "datas", "artifacts", "sweeps", "transition_weight.json")
    save_json({"config": cfg_path, "rows": rows}, out)
    print("\nweight            report  best_epoch")
    for r in rows:
        acc = r["report_acc"]
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
        print(f"{r['transition_weight']:<16} {acc_s}  {r['best_epoch']}")
    print(f"汇总: {out}")


if __name__ == "__main__":
    main()
