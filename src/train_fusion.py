"""End-to-end trainer for Stage 3 fusion variants (and the F0 sanity
unimodal baselines).

Joint (not sequential) training: one optimizer over all fusion +
head parameters, loss = loss_expr + loss_va + loss_au. Encoders are
frozen by construction (we train on cached features, the encoders are
not part of the graph).

Usage:
    python -m src.train_fusion --config configs/stage3_f4_xattn.yaml
    python -m src.train_fusion --config configs/stage3_f0_visual_only.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.datasets.affwild2_mtl import (
    compute_au_class_weights,
    compute_expr_class_weights,
)
from src.datasets.bimodal import BimodalFrameDataset, BimodalWindowDataset
from src.fusion.base import FusionConfig, count_parameters
from src.fusion.f4_xattn import CrossModalAttention
from src.fusion.f5_lmf import LMFusion
from src.fusion.f6c_iaca import IACAGate
from src.fusion.f6d_mbt import MBTFusion
from src.fusion.variants import (
    AudioOnly,
    EarlyConcat,
    LearnedBlend,
    TaskGate,
    VisualOnly,
)
from src.heads.mtl_head import loss_va, make_loss_aus


FUSION_REGISTRY = {
    "visual_only": (VisualOnly, False),
    "audio_only": (AudioOnly, False),
    "f1_concat": (EarlyConcat, False),
    "f2_blend": (LearnedBlend, False),
    "f3_gate": (TaskGate, False),
    "f4_xattn": (CrossModalAttention, True),   # True = wants temporal window
    "f5_lmf": (LMFusion, False),
    "f6c_iaca": (IACAGate, True),
    "f6d_mbt": (MBTFusion, True),
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(variant: str, cfg: FusionConfig) -> nn.Module:
    if variant not in FUSION_REGISTRY:
        raise KeyError(
            f"Unknown fusion variant '{variant}'. "
            f"Available: {list(FUSION_REGISTRY)}"
        )
    ctor, _wants_window = FUSION_REGISTRY[variant]
    return ctor(cfg)


def wants_window(variant: str) -> bool:
    return FUSION_REGISTRY[variant][1]


def _build_datasets(cfg, variant: str):
    if wants_window(variant):
        ds_cls = BimodalWindowDataset
        extra = {"window": int(cfg.fusion.get("window", 5))}
    else:
        ds_cls = BimodalFrameDataset
        extra = {}

    train_ds = ds_cls(
        annotations_file=cfg.data.train_annotations,
        visual_cache_dir=cfg.visual.features_cache,
        audio_cache_dir=cfg.audio.aligned_dir,
        **extra,
    )
    val_ds = ds_cls(
        annotations_file=cfg.data.val_annotations,
        visual_cache_dir=cfg.visual.features_cache,
        audio_cache_dir=cfg.audio.aligned_dir,
        **extra,
    )
    return train_ds, val_ds


def _joint_loss(
    pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    expr_class_weights: torch.Tensor,
    au_loss_fn,
    loss_weights: Dict[str, float],
) -> torch.Tensor:
    expr_logits, va_pred, au_pred = pred

    device = expr_logits.device
    m_expr = batch["m_expr"].to(device)
    m_va = batch["m_va"].to(device)
    m_au = batch["m_au"].to(device)

    loss_expr = torch.zeros((), device=device)
    if m_expr.sum() > 0:
        sel = m_expr > 0
        loss_expr = nn.functional.cross_entropy(
            expr_logits[sel],
            batch["y_expr"].to(device)[sel],
            weight=expr_class_weights,
        )

    loss_va_val = torch.zeros((), device=device)
    if m_va.sum() > 0:
        sel = m_va > 0
        loss_va_val = loss_va(
            batch["y_va"].to(device)[sel], va_pred[sel]
        )

    loss_au_val = torch.zeros((), device=device)
    if m_au.sum() > 0:
        sel = m_au > 0
        loss_au_val = au_loss_fn(au_pred[sel], batch["y_aus"].to(device)[sel])

    return (
        loss_weights.get("expr", 1.0) * loss_expr
        + loss_weights.get("va", 1.0) * loss_va_val
        + loss_weights.get("au", 1.0) * loss_au_val
    )


def train_from_config(cfg) -> Path:
    _set_seed(int(cfg.seed))
    device = cfg.train.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU.")
        device = "cpu"

    results_dir = Path(cfg.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, results_dir / "config.yaml")

    variant = cfg.fusion.variant
    train_ds, val_ds = _build_datasets(cfg, variant)

    fus_cfg = FusionConfig(
        v_dim=int(cfg.visual.feature_dim),
        scores_dim=int(cfg.visual.scores_dim),
        a_dim=int(cfg.audio.feature_dim),
        num_expr=int(cfg.data.num_expr),
        num_aus=int(cfg.data.num_aus),
        hidden=int(cfg.fusion.get("hidden", 384)),
        dropout=float(cfg.fusion.get("dropout", 0.3)),
        au_hidden=int(cfg.head.au_hidden),
    )
    model = build_model(variant, fus_cfg).to(device)
    print(f"[fusion={variant}] trainable params = {count_parameters(model):,}")

    # Class weights from the train split (reuse Stage 1 utilities).
    anno_train = train_ds.anno
    class_w_dict = compute_expr_class_weights(
        anno_train.y_expr, anno_train.mask_expr, fus_cfg.num_expr
    )
    expr_class_weights = torch.tensor(
        [class_w_dict[i] for i in range(fus_cfg.num_expr)], dtype=torch.float32
    ).to(device)
    au_weights_np = compute_au_class_weights(anno_train.y_aus, anno_train.mask_au)
    au_loss_fn = make_loss_aus(au_weights_np)

    loss_weights = OmegaConf.to_container(
        cfg.train.get("loss_weights", {"expr": 1.0, "va": 1.0, "au": 1.0})
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )

    best_val, best_state = float("inf"), None
    log_rows: List[Dict] = []
    for epoch in range(1, int(cfg.train.epochs) + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            pred = model(
                batch["v_feat"].to(device),
                batch["v_scores"].to(device),
                batch["a_feat"].to(device),
            )
            loss = _joint_loss(
                pred, batch, expr_class_weights, au_loss_fn, loss_weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = batch["y_expr"].size(0)
            tot += loss.item() * bs
            n += bs
        train_loss = tot / max(1, n)

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch["v_feat"].to(device),
                    batch["v_scores"].to(device),
                    batch["a_feat"].to(device),
                )
                loss = _joint_loss(
                    pred, batch, expr_class_weights, au_loss_fn, loss_weights
                )
                bs = batch["y_expr"].size(0)
                vtot += loss.item() * bs
                vn += bs
        val_loss = vtot / max(1, vn)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "improved": int(improved),
                "seconds": time.time() - t0,
            }
        )
        print(
            f"[{variant}] ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}"
            f"  best={best_val:.4f}{'  *' if improved else ''}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    ck_path = results_dir / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "variant": variant,
            "fus_cfg": fus_cfg.__dict__,
            "expr_class_weights": class_w_dict,
            "au_class_weights": au_weights_np,
            "trainable_params": count_parameters(model),
        },
        ck_path,
    )

    log_path = results_dir / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"[fusion={variant}] checkpoint -> {ck_path}")
    return ck_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Stage 3 fusion variant.")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config key (repeatable). Dotted keys address nested "
        "fields, e.g. --set seed=1 --set output.results_dir=results/foo. The "
        "fully-resolved config is still saved to results/<run>/config.yaml.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    train_from_config(cfg)


if __name__ == "__main__":
    main()
