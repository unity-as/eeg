"""扫 transition_steps。各组合写入独立目录，避免盖掉正式 [1,2,3]。

用法：
  python scripts/sweep_transition_steps.py
  python scripts/sweep_transition_steps.py config/eeg_ws_transition.yaml
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.io_utils import load_json, save_json

DEFAULT_STEPS = ([1], [1, 2], [1, 2, 3])


def _tag(steps) -> str:
    return "s" + "".join(str(int(s)) for s in steps)


def _dirs(subject: str, epoch: int, steps) -> tuple[str, str]:
    tag = _tag(steps)
    cache = f"./datas/artifacts/eeg_cache_ws_transition/{subject}/ep{epoch}/{tag}"
    ckpt = f"./datas/artifacts/checkpoints_eeg_ws_transition/{subject}/ep{epoch}/{tag}"
    return cache, ckpt


def _row(steps, metrics: dict, ckpt: str) -> dict:
    return {
        "transition_steps": list(steps),
        "report_acc": metrics.get("test_acc"),
        "best_val_acc": metrics.get("best_val_acc"),
        "best_epoch": metrics.get("best_epoch"),
        "epochs_ran": metrics.get("epochs_ran"),
        "n_report": metrics.get("n_report"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "checkpoint_dir": ckpt,
    }


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/eeg_ws_transition.yaml"
    subject, epoch = "sub-002", 512
    rows = []
    py = sys.executable
    for steps in DEFAULT_STEPS:
        cache, ckpt = _dirs(subject, epoch, steps)
        mp = os.path.join(ROOT, ckpt, "metrics.json")
        if steps == [1, 2, 3] and not os.path.isfile(mp):
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
        steps_arg = "method.transition_steps=[" + ",".join(str(s) for s in steps) + "]"
        if os.path.isfile(mp):
            print(f"\n===== 已有 {_tag(steps)}，跳过训练 =====")
        else:
            print(f"\n===== sweep {_tag(steps)} {steps} =====")
            cmd = [
                py,
                os.path.join(ROOT, "main_pipeline.py"),
                cfg_path,
                steps_arg,
                f"data.cache_dir={cache}",
                f"train.checkpoint_dir={ckpt}",
            ]
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"{_tag(steps)} 失败，停止。")
                sys.exit(result.returncode)
        if not os.path.isfile(mp):
            raise FileNotFoundError(mp)
        rows.append(_row(steps, load_json(mp), ckpt))

    out = os.path.join(ROOT, "datas", "artifacts", "sweeps", "transition_steps.json")
    save_json({"config": cfg_path, "rows": rows}, out)
    print("\nsteps        report  best_epoch  epochs")
    for r in rows:
        acc = r["report_acc"]
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
        print(f"{r['transition_steps']!s:<12} {acc_s}  {r['best_epoch']}  {r['epochs_ran']}")
    print(f"汇总: {out}")


if __name__ == "__main__":
    main()
