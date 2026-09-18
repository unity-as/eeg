from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_correct_only_dataset import build_subject as build_dataset_subject
from build_correct_only_dataset import parse_subjects, write_quality_summary
from experiments.common import resolve_path
from train_3way import parse_seeds, train_one
from utils.io_utils import ensure_dir, save_json


def clone_cfg(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def aggregate_results(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = ["best_val_acc", "val_balanced_acc", "val_macro_f1", "best_epoch", "epochs_ran"]
    out = {"n_runs": len(records)}
    for key in keys:
        values = np.asarray([float(r[key]) for r in records if key in r], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean()) if values.size else 0.0
        out[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return out


def sort_key(result: dict) -> tuple:
    return (
        -float(result.get("val_balanced_acc_mean", 0.0)),
        -float(result.get("val_macro_f1_mean", 0.0)),
        -float(result.get("best_val_acc_mean", 0.0)),
        float(result.get("val_balanced_acc_std", 0.0)),
    )


def write_summary(path: Path, stage: str, results: list[dict]) -> None:
    ordered = sorted(results, key=sort_key)
    save_json(
        {
            "stage": stage,
            "selection_order": [
                "validation balanced accuracy mean",
                "validation macro F1 mean",
                "validation accuracy mean",
                "validation balanced accuracy std (lower is better)",
            ],
            "test_used": False,
            "results": ordered,
        },
        str(path),
    )
    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "config",
                "n_runs",
                "best_val_acc_mean",
                "best_val_acc_std",
                "val_balanced_acc_mean",
                "val_balanced_acc_std",
                "val_macro_f1_mean",
                "val_macro_f1_std",
                "best_epoch_mean",
                "epochs_ran_mean",
            ],
        )
        writer.writeheader()
        for result in ordered:
            row = {k: result.get(k, "") for k in writer.fieldnames}
            row["config"] = str(result.get("config", ""))
            writer.writerow(row)


def run_representation(args, cfg) -> None:
    subjects = parse_subjects(cfg, args)
    seeds = parse_seeds(cfg, args)
    base_root = resolve_path(cfg.data.output_root)
    out_root = Path(ensure_dir(str(base_root / "tuning_representation")))
    write_quality_summary(subjects, cfg)

    results = []
    for epoch_samples in [int(v) for v in cfg.tuning.representation.time_windows]:
        for percentile in [float(v) for v in cfg.tuning.representation.recurrence_percentiles]:
            tag = f"ep{epoch_samples}_p{percentile:.2f}"
            local = clone_cfg(cfg)
            local.data.epoch_samples = epoch_samples
            local.method.recurrence_percentile = percentile
            local.data.output_root = str(base_root / "representations" / tag)
            local.train.output_dir = str(out_root / tag / "checkpoints")
            records = []
            print(f"=== representation {tag}: subjects={subjects} seeds={seeds} ===")
            for subject in subjects:
                build_dataset_subject(subject, local)
                for seed in seeds:
                    records.append(train_one(subject, seed, local, evaluate_test=False, confirm_test=False))
            result = {
                "tag": tag,
                "config": {
                    "epoch_samples": epoch_samples,
                    "recurrence_percentile": percentile,
                },
                **aggregate_results(records),
                "runs": records,
            }
            results.append(result)
    write_summary(out_root / "summary.json", "representation", results)


def run_model(args, cfg) -> None:
    subjects = parse_subjects(cfg, args)
    seeds = parse_seeds(cfg, args)
    base_root = resolve_path(cfg.data.output_root)
    if args.data_root:
        base_root = resolve_path(args.data_root)
        cfg.data.output_root = str(base_root)
    out_root = Path(ensure_dir(str(base_root / "tuning_model")))

    combinations = []
    for channels in cfg.tuning.model.conv_channels:
        for dropout in cfg.tuning.model.dropout:
            for lr in cfg.tuning.model.lr:
                for batch_size in cfg.tuning.model.batch_size:
                    combinations.append(
                        {
                            "conv_channels": [int(v) for v in channels],
                            "dropout": float(dropout),
                            "lr": float(lr),
                            "batch_size": int(batch_size),
                        }
                    )
    if args.max_configs is not None:
        combinations = combinations[: int(args.max_configs)]

    results = []
    for combo in combinations:
        tag = (
            f"ch{'-'.join(str(v) for v in combo['conv_channels'])}"
            f"_do{combo['dropout']:.1f}_lr{combo['lr']:.0e}_bs{combo['batch_size']}"
        )
        local = clone_cfg(cfg)
        local.model.conv_channels = combo["conv_channels"]
        local.train.dropout = combo["dropout"]
        local.train.lr = combo["lr"]
        local.train.batch_size = combo["batch_size"]
        local.train.output_dir = str(out_root / tag / "checkpoints")
        records = []
        print(f"=== model {tag}: subjects={subjects} seeds={seeds} ===")
        for subject in subjects:
            for seed in seeds:
                records.append(train_one(subject, seed, local, evaluate_test=False, confirm_test=False))
        result = {"tag": tag, "config": combo, **aggregate_results(records), "runs": records}
        results.append(result)
    write_summary(out_root / "summary.json", "model", results)


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-only tuning for RP+SE CNN.")
    parser.add_argument("--config", default="config/eeg_ws_r_3way.yaml")
    parser.add_argument("--stage", choices=["representation", "model"], required=True)
    parser.add_argument("--subjects", default=None, help="comma separated")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", default=None, help="comma separated training seeds")
    parser.add_argument("--data-root", default=None, help="frozen data root for model-stage tuning")
    parser.add_argument("--max-configs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(resolve_path(args.config))
    if args.stage == "representation":
        run_representation(args, cfg)
    else:
        run_model(args, cfg)


if __name__ == "__main__":
    main()
