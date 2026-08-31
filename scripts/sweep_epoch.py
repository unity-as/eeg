"""扫 data.epoch_samples。各窗写入 ep{N} 以便对比；覆盖旧正式结果即可。

用法：
  python scripts/sweep_epoch.py config/eeg_ws_r.yaml
  python scripts/sweep_epoch.py config/eeg_ws_r.yaml 256,400,512,800,1000
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.io_utils import load_json, load_run_cfg, save_json

DEFAULT_WINDOWS = (256, 400, 512, 800, 1000)


def _collapse(cm):
    """4 类顺序 cat_go, cat_nogo, rec_go, rec_nogo → cat/rec 与 go/nogo 准确率。"""
    n = sum(sum(row) for row in cm)
    if n == 0:
        return None, None
    cat_ok = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    rec_ok = cm[2][2] + cm[2][3] + cm[3][2] + cm[3][3]
    go_ok = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
    nogo_ok = cm[1][1] + cm[1][3] + cm[3][1] + cm[3][3]
    return (cat_ok + rec_ok) / n, (go_ok + nogo_ok) / n


def _row_from_metrics(ep: int, metrics: dict, ckpt: str) -> dict:
    cm = metrics.get("confusion_matrix") or []
    cat_rec, go_nogo = _collapse(cm) if len(cm) == 4 else (None, None)
    return {
        "epoch_samples": ep,
        "report_acc": metrics.get("test_acc"),
        "best_val_acc": metrics.get("best_val_acc"),
        "best_epoch": metrics.get("best_epoch"),
        "epochs_ran": metrics.get("epochs_ran"),
        "n_report": metrics.get("n_report"),
        "cat_vs_rec": cat_rec,
        "go_vs_nogo": go_nogo,
        "checkpoint_dir": ckpt,
    }


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/eeg_ws_r.yaml"
    if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
        windows = tuple(int(x) for x in sys.argv[2].split(",") if x.strip())
    else:
        windows = DEFAULT_WINDOWS

    rows = []
    py = sys.executable
    for ep in windows:
        print(f"\n===== sweep epoch_samples={ep} =====")
        cmd = [py, os.path.join(ROOT, "main_pipeline.py"), cfg_path, f"data.epoch_samples={ep}"]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"ep={ep} 失败，停止。")
            sys.exit(result.returncode)
        run_cfg = load_run_cfg(cfg_path, ["sweep", cfg_path, f"data.epoch_samples={ep}"])
        ckpt = str(run_cfg.train.checkpoint_dir)
        mp = os.path.join(ROOT, ckpt, "metrics.json")
        if not os.path.isfile(mp):
            mp = os.path.join(ckpt, "metrics.json")
        rows.append(_row_from_metrics(ep, load_json(mp), ckpt))

    out = os.path.join(ROOT, "datas", "sweeps", "epoch_ws_r.json")
    save_json({"config": cfg_path, "rows": rows}, out)
    print("\n窗长  report  cat/rec  go/nogo  早停")
    for r in rows:
        cr = f"{r['cat_vs_rec']:.3f}" if r["cat_vs_rec"] is not None else "-"
        gn = f"{r['go_vs_nogo']:.3f}" if r["go_vs_nogo"] is not None else "-"
        acc = f"{r['report_acc']:.3f}" if r["report_acc"] is not None else "-"
        print(f"{r['epoch_samples']:>4}  {acc}   {cr}    {gn}    {r['epochs_ran']}")
    print(f"汇总: {out}")


if __name__ == "__main__":
    main()
