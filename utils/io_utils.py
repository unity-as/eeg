from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_PAPER_ALIGN = {
    ("rp", "se"): "论文1 Shankar 2021 RP+CNN（节律/质量是否开见 method）；论文2 Hao 2021 RP+CNN+SE",
    ("rp", "none"): "论文1 Shankar 2021 RP+CNN（无注意力）",
    ("mrp", "additive"): "论文3 Huang 2023 MRP-Net（MRP + 加性注意力 CNN）",
}


def _parse_cli_overrides(argv: List[str]):
    """argv[2:]：无等号视为 data.subject；key=value 写入 OmegaConf（裸 epoch_samples 视为 data.）。"""
    subject = None
    dots: List[str] = []
    for arg in argv[2:]:
        s = str(arg)
        if s.startswith("-"):
            continue
        if "=" in s:
            key, _, val = s.partition("=")
            if "." not in key:
                key = f"data.{key}"
            dots.append(f"{key}={val}")
        elif subject is None:
            subject = s
    return subject, dots


def load_run_cfg(cfg_path: str, argv: Optional[List[str]] = None):
    """读 yaml；命令行可覆盖 subject / data.epoch_samples=512；补齐可由其它项推出的路径。"""
    from omegaconf import OmegaConf

    argv = list(sys.argv if argv is None else argv)
    cfg = OmegaConf.load(cfg_path)
    subject, dots = _parse_cli_overrides(argv)
    if subject is not None:
        OmegaConf.update(cfg, "data.subject", subject, merge=True)
    if dots:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dots))
    if cfg.get("data") is not None and cfg.data.get("subject") is not None:
        sub = str(cfg.data.subject)
        OmegaConf.update(cfg, "data.subjects", [sub], merge=True)
        rep = str(cfg.method.representation) if cfg.get("method") is not None else "rp"
        att = str(cfg.method.attention) if cfg.get("method") is not None else "none"
        ep = int(cfg.data.epoch_samples) if cfg.data.get("epoch_samples") is not None else 256
        if not cfg.data.get("cache_dir"):
            OmegaConf.update(
                cfg, "data.cache_dir", f"./datas/eeg_cache_ws_{rep}/{sub}/ep{ep}"
            )
        if cfg.get("train") is not None and not cfg.train.get("checkpoint_dir"):
            OmegaConf.update(
                cfg,
                "train.checkpoint_dir",
                f"./datas/checkpoints_eeg_ws_{rep}/{sub}/ep{ep}",
            )
        if cfg.get("train") is not None and not cfg.train.get("save_best_model"):
            OmegaConf.update(cfg, "train.save_best_model", f"best_{rep}_{sub}.pt")
        if cfg.get("method") is not None and not cfg.method.get("paper_align"):
            OmegaConf.update(
                cfg,
                "method.paper_align",
                _PAPER_ALIGN.get((rep, att), f"{rep}+{att}"),
            )
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
