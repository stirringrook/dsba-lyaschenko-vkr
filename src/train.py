"""Sequential per-head training for the Stage 1 visual-only baseline.

Trains the three heads from :mod:`src.heads.mtl_head` one after another
(EXPR -> VA -> AU), each with its own Adam(lr=1e-3) optimizer and its
own best-val-loss checkpoint, exactly as in cells 47 / 50 / 53 of
``mtl.ipynb``. Writes ``results/<run_name>/{best.pt, log.csv, config.yaml}``.

Usage:
    python -m src.train --config configs/stage1_visual.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.affwild2_mtl import (
    MTLAnnotations,
    compute_au_class_weights,
    compute_expr_class_weights,
    read_mtl_annotations,
)
from src.heads.mtl_head import (
    AUHead,
    ExprHead,
    MTLHead,
    MTLHeadConfig,
    VAHead,
    loss_va,
    make_loss_aus,
)
from src.utils.io import load_features_dir
from src.utils.metrics import metric_for_Exp


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def _assemble_X(anno: MTLAnnotations, filename2features) -> np.ndarray:
    """Concatenate [features, scores] per row. Cell 32 ``FEATURES_SCORES``."""
    parts = []
    for image_path in anno.df["image_path"].tolist():
        feat, sco = filename2features[image_path]
        parts.append(np.concatenate([feat, sco]))
    return np.asarray(parts, dtype=np.float32)


def load_split(
    annotations_file: str | Path,
    features_dir: str | Path,
) -> Tuple[np.ndarray, MTLAnnotations]:
    """Load per-frame features aligned with an annotation split.

    Returns:
        ``(X, annotations)`` where ``X`` is the concatenated feature/score
        matrix of shape ``(N, D)`` and ``annotations`` has the labels,
        masks, and ``videoname_frames`` for later smoothing.
    """
    filename2features, _ = load_features_dir(features_dir)
    anno = read_mtl_annotations(annotations_file, features_index=filename2features)
    X = _assemble_X(anno, filename2features)
    return X, anno


# ---------------------------------------------------------------------------
# Training loop for a single head
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_head(
    head: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    loss_fn,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    l2: float,
    device: str,
    log_rows: List[Dict],
    head_name: str,
    score_fn=None,
) -> nn.Module:
    """Train ``head`` with Adam + early-stop on val. Returns best state.

    By default, best-checkpoint selection minimises the same ``loss_fn``
    on the validation split (matches the ``SaveBestModel`` pattern from
    cells 47 / 50 / 53 of ``mtl.ipynb``). Pass ``score_fn(head, X_val,
    y_val) -> float`` (higher = better) to override --- used for the
    EXPR head's ``head.expr_select_by: f1_macro`` option, which selects
    on the actual scoring metric instead of class-weighted CE.
    """
    head = head.to(device)
    optim = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=l2)
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    higher_is_better = score_fn is not None
    best_score = float("-inf") if higher_is_better else float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    for epoch in range(1, epochs + 1):
        head.train()
        t0 = time.time()
        train_loss_sum = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = head(xb)
            loss = loss_fn(out, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_sum += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = train_loss_sum / max(1, n_seen)

        head.eval()
        with torch.no_grad():
            val_out = head(X_val.to(device))
            val_loss = loss_fn(val_out, y_val.to(device)).item()
            if score_fn is not None:
                score = float(score_fn(head, X_val.to(device), y_val.to(device)))
            else:
                score = val_loss

        if higher_is_better:
            improved = score > best_score
        else:
            improved = score < best_score
        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        log_rows.append(
            {
                "head": head_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_score": score,
                "best_score": best_score,
                "improved": int(improved),
                "seconds": time.time() - t0,
            }
        )
        suffix = "" if score_fn is None else f"  score={score:.4f}"
        print(
            f"[{head_name}] epoch {epoch:3d}  train={train_loss:.4f}  "
            f"val={val_loss:.4f}{suffix}  best={best_score:.4f}"
            f"{'  *' if improved else ''}"
        )

    head.load_state_dict(best_state)
    return head


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def train_from_config(cfg) -> Path:
    _set_seed(int(cfg.seed))

    device = cfg.train.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    results_dir = Path(cfg.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, results_dir / "config.yaml")

    print("[stage1] loading features and annotations ...")
    X_train_np, anno_train = load_split(cfg.data.train_annotations, cfg.features_cache)
    X_val_np, anno_val = load_split(cfg.data.val_annotations, cfg.features_cache)

    feature_dim = int(cfg.backbone.feature_dim)
    scores_dim = int(cfg.backbone.scores_dim)
    assert X_train_np.shape[1] == feature_dim + scores_dim, (
        f"Config says input_dim={feature_dim + scores_dim} but got "
        f"X_train shape {X_train_np.shape}."
    )

    head_cfg = MTLHeadConfig(
        feature_dim=feature_dim,
        scores_dim=scores_dim,
        num_expr=int(cfg.data.num_expr),
        num_aus=int(cfg.data.num_aus),
        au_hidden=int(cfg.head.au_hidden),
    )
    model = MTLHead(head_cfg)

    X_train_t = torch.from_numpy(X_train_np)
    X_val_t = torch.from_numpy(X_val_np)

    log_rows: List[Dict] = []

    # --------- EXPR ----------
    expr_train_mask = anno_train.mask_expr == 1
    expr_val_mask = anno_val.mask_expr == 1
    class_w_dict = compute_expr_class_weights(
        anno_train.y_expr, anno_train.mask_expr, head_cfg.num_expr
    )
    class_w = torch.tensor(
        [class_w_dict[i] for i in range(head_cfg.num_expr)], dtype=torch.float32
    ).to(device)

    def _expr_loss(logits, target):
        return nn.functional.cross_entropy(logits, target.long(), weight=class_w)

    expr_select_by = str(cfg.head.get("expr_select_by", "val_loss")).lower()
    if expr_select_by == "f1_macro":
        def _expr_score(head, X, y):
            head.eval()
            with torch.no_grad():
                logits = head(X)
                pred = logits.argmax(dim=-1).cpu().numpy()
                target = y.cpu().numpy()
            f1, _, _ = metric_for_Exp(target, pred, class_num=head_cfg.num_expr)
            return f1
        expr_score_fn = _expr_score
        print("[expr] selecting best.pt by macro-F1 (head.expr_select_by=f1_macro)")
    elif expr_select_by == "val_loss":
        expr_score_fn = None
    else:
        raise ValueError(
            f"head.expr_select_by must be 'val_loss' or 'f1_macro', got {expr_select_by!r}"
        )

    _train_head(
        model.expr,
        X_train_t[expr_train_mask],
        torch.from_numpy(anno_train.y_expr[expr_train_mask]),
        X_val_t[expr_val_mask],
        torch.from_numpy(anno_val.y_expr[expr_val_mask]),
        loss_fn=_expr_loss,
        epochs=int(cfg.train.epochs_expr),
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.train.learning_rate),
        l2=float(cfg.head.l2_reg),
        device=device,
        log_rows=log_rows,
        head_name="expr",
        score_fn=expr_score_fn,
    )

    # --------- VA ----------
    va_train_mask = anno_train.mask_va == 1
    va_val_mask = anno_val.mask_va == 1
    _train_head(
        model.va,
        X_train_t[va_train_mask],
        torch.from_numpy(anno_train.y_va[va_train_mask]),
        X_val_t[va_val_mask],
        torch.from_numpy(anno_val.y_va[va_val_mask]),
        loss_fn=loss_va,
        epochs=int(cfg.train.epochs_va),
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.train.learning_rate),
        l2=float(cfg.head.l2_reg),
        device=device,
        log_rows=log_rows,
        head_name="va",
    )

    # --------- AU ----------
    au_train_mask = anno_train.mask_au == 1
    au_val_mask = anno_val.mask_au == 1
    au_weights_np = compute_au_class_weights(anno_train.y_aus, anno_train.mask_au)
    au_loss = make_loss_aus(au_weights_np)

    def _au_loss(pred, target):
        return au_loss(pred, target)

    _train_head(
        model.au,
        X_train_t[au_train_mask],
        torch.from_numpy(anno_train.y_aus[au_train_mask]).float(),
        X_val_t[au_val_mask],
        torch.from_numpy(anno_val.y_aus[au_val_mask]).float(),
        loss_fn=_au_loss,
        epochs=int(cfg.train.epochs_au),
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.train.learning_rate),
        l2=float(cfg.head.l2_reg),
        device=device,
        log_rows=log_rows,
        head_name="au",
    )

    # Save bundled checkpoint.
    ckpt_path = results_dir / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "head_cfg": head_cfg.__dict__,
            "au_class_weights": au_weights_np,
            "expr_class_weights": class_w_dict,
        },
        ckpt_path,
    )

    log_path = results_dir / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"[stage1] saved checkpoint to {ckpt_path}")
    print(f"[stage1] saved training log to {log_path}")
    return ckpt_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 MTL head training.")
    p.add_argument("--config", required=True, help="Path to a YAML config.")
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
