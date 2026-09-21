"""Four SEU methods sweep lr and dropout on val only. Test stays sealed.

Methods: CNN none, CNN self, GCN, GCN graph attention.
Representation and splits stay at the first-run settings.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_runtime import method_name, resolve_path
from scripts.experiments.train_seu import train_one
from utils.io_utils import ensure_dir, load_json, save_json

TASKS = ("bearing", "gear")
CONDITIONS = ("20_0", "30_2")
LRS = (1e-4, 5e-4, 1e-3)
DROPOUTS = (0.1, 0.2, 0.3)
METHODS = (
    {"tag": "cnn_none", "base": "cnn", "attention": "none", "gcn_attention": False},
    {"tag": "cnn_self", "base": "cnn", "attention": "self", "gcn_attention": False},
    {"tag": "gcn", "base": "gcn", "attention": "none", "gcn_attention": False},
    {"tag": "gcn_attn", "base": "gcn", "attention": "none", "gcn_attention": True},
)
SEED = 42
SWEEP_ROOT = "./datas/experiments_seu/sweep"


def base_config(task: str, base: str) -> str:
    return f"config/seu_{task}_{base}.yaml"


def run_name(lr: float, dropout: float) -> str:
    return f"lr{lr:g}_drop{dropout:g}"


def apply_method(cfg, spec: dict, lr: float, dropout: float, out_dir: str) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.method.attention = spec["attention"]
    cfg.model.gcn_attention = bool(spec["gcn_attention"])
    cfg.train.lr = float(lr)
    cfg.train.dropout = float(dropout)
    cfg.train.output_dir = out_dir
    cfg.train.checkpoint_name = f"best_{spec['tag']}.pt"
    cfg.train.evaluate_test = False
    cfg.train.seeds = [SEED]
    cfg.train.seed = SEED


def summarize(trials: list[dict]) -> dict:
    best = {}
    for spec in METHODS:
        for task in TASKS:
            tagged = [
                t
                for t in trials
                if t.get("tag") == spec["tag"] and t.get("task") == task and t.get("ok")
            ]
            by_run: dict[str, list[dict]] = {}
            for t in tagged:
                by_run.setdefault(t["run"], []).append(t)
            ranked = []
            for run, rows in by_run.items():
                if len(rows) < len(CONDITIONS):
                    continue
                mean_acc = float(sum(r["val_acc"] for r in rows) / len(rows))
                ranked.append(
                    {
                        "run": run,
                        "lr": rows[0]["lr"],
                        "dropout": rows[0]["dropout"],
                        "mean_val_acc": mean_acc,
                        "by_condition": {r["condition"]: r["val_acc"] for r in rows},
                    }
                )
            ranked.sort(key=lambda x: x["mean_val_acc"], reverse=True)
            best[f"{task}_{spec['tag']}"] = ranked[0] if ranked else None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="SEU val sweep; does not open test.")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--methods", default=",".join(s["tag"] for s in METHODS))
    args = parser.parse_args()
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    method_tags = {t.strip() for t in args.methods.split(",") if t.strip()}

    sweep_root = Path(ensure_dir(str(resolve_path(SWEEP_ROOT))))
    summary_path = sweep_root / "summary.json"
    trials = []
    if summary_path.exists():
        prev = load_json(str(summary_path))
        trials = list(prev.get("trials", []))
    done = {(t.get("tag"), t.get("task"), t.get("run"), t.get("condition")) for t in trials}

    for spec in METHODS:
        if spec["tag"] not in method_tags:
            continue
        for task in tasks:
            cfg_path = resolve_path(base_config(task, spec["base"]))
            for lr in LRS:
                for dropout in DROPOUTS:
                    run = run_name(lr, dropout)
                    out_dir = str(sweep_root / spec["tag"] / task / run)
                    for condition in CONDITIONS:
                        key = (spec["tag"], task, run, condition)
                        if key in done:
                            print(f"skip {key}", flush=True)
                            continue
                        cfg = OmegaConf.load(cfg_path)
                        apply_method(cfg, spec, lr, dropout, out_dir)
                        print(
                            f"train {spec['tag']} {task} {condition} {run} "
                            f"method={method_name(cfg)}",
                            flush=True,
                        )
                        result = train_one(
                            task,
                            condition,
                            SEED,
                            cfg,
                            evaluate_test=False,
                            confirm_test=False,
                            write_paper_figures=False,
                        )
                        row = {
                            "ok": True,
                            "tag": spec["tag"],
                            "task": task,
                            "condition": condition,
                            "run": run,
                            "lr": float(lr),
                            "dropout": float(dropout),
                            "method": method_name(cfg),
                            "val_acc": float(result["best_val_acc"]),
                            "best_epoch": int(result["best_epoch"]),
                            "output_dir": out_dir,
                        }
                        trials.append(row)
                        done.add(key)
                        save_json(
                            {"trials": copy.deepcopy(trials), "best": summarize(trials)},
                            str(summary_path),
                        )
                        print(
                            f"done {spec['tag']} {task} {condition} {run} "
                            f"val={row['val_acc']:.4f}",
                            flush=True,
                        )

    payload = {"trials": trials, "best": summarize(trials)}
    save_json(payload, str(summary_path))
    print("best", payload["best"], flush=True)


if __name__ == "__main__":
    main()
