"""Evaluation for the Stage 1 visual-only MTL baseline.

Computes ``ccc_V, ccc_A, CCC_VA, F1_EXPR (macro), F1_AU (macro @ t=0.5),
F1_AU* (best-threshold), P_MTL`` on the validation split and writes a
Markdown table to ``<results_dir>/metrics.md``. Optionally applies the
per-video Gaussian smoothing from :mod:`src.smoothing` to EXPR and VA
(AU smoothing hurts F1; confirmed cell 66).

Usage:
    python -m src.eval --config configs/stage1_visual.yaml \
        --checkpoint results/stage1_visual_mbf/best.pt
    python -m src.eval --config configs/stage1_visual.yaml \
        --checkpoint results/stage1_visual_mbf/best.pt \
        --smooth --sigma 100000 --delta 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf

from src.heads.mtl_head import MTLHead, MTLHeadConfig
from src.smoothing import gaussian_smooth_per_video
from src.train import load_split
from src.utils.metrics import (
    f1_macro_au,
    f1_score_max,
    metric_for_Exp,
    metric_for_VA,
    p_mtl,
)


def _predict_all(
    model: MTLHead, X_val: np.ndarray, device: str, batch: int = 4096
):
    model.eval()
    xt = torch.from_numpy(X_val.astype(np.float32))
    preds_expr, preds_va, preds_au = [], [], []
    with torch.no_grad():
        for i in range(0, xt.size(0), batch):
            xb = xt[i : i + batch].to(device)
            e, v, a = model(xb)
            # The notebook uses softmax-probabilities for EXPR argmax; we do the same.
            preds_expr.append(torch.softmax(e, dim=-1).cpu().numpy())
            preds_va.append(v.cpu().numpy())
            preds_au.append(a.cpu().numpy())
    return (
        np.concatenate(preds_expr, axis=0),
        np.concatenate(preds_va, axis=0),
        np.concatenate(preds_au, axis=0),
    )


def evaluate_mtl(
    checkpoint: str | Path,
    config_path: str | Path,
    *,
    smooth: bool = False,
    sigma: float = 100000.0,
    delta: int = 50,
    output_dir: str | Path | None = None,
) -> Dict[str, float]:
    """Run evaluation on the validation split. Returns a metric dict."""
    cfg = OmegaConf.load(config_path)
    device = cfg.train.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    X_val, anno_val = load_split(cfg.data.val_annotations, cfg.features_cache)

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head_cfg_dict = ck["head_cfg"]
    head_cfg = MTLHeadConfig(**head_cfg_dict)
    model = MTLHead(head_cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    pred_expr_prob, pred_va, pred_au = _predict_all(model, X_val, device)

    # Optional per-video smoothing for EXPR + VA (NOT AU).
    if smooth:
        pred_expr_prob = gaussian_smooth_per_video(
            pred_expr_prob, anno_val.videoname_frames, sigma=sigma, delta=delta
        )
        pred_va = gaussian_smooth_per_video(
            pred_va, anno_val.videoname_frames, sigma=sigma, delta=delta
        )

    # ---- VA ----
    va_m = anno_val.mask_va == 1
    gt_V = anno_val.y_va[va_m, 0]
    gt_A = anno_val.y_va[va_m, 1]
    pV = pred_va[va_m, 0]
    pA = pred_va[va_m, 1]
    ccc_V, ccc_A, ccc_VA = metric_for_VA(gt_V, gt_A, pV, pA)

    # ---- EXPR ----
    ex_m = anno_val.mask_expr == 1
    y_pred_expr = pred_expr_prob.argmax(axis=1)
    f1_expr, acc_expr, _ = metric_for_Exp(
        anno_val.y_expr[ex_m], y_pred_expr[ex_m], class_num=head_cfg.num_expr
    )

    # ---- AU ----
    au_m = anno_val.mask_au == 1
    y_aus = anno_val.y_aus[au_m]
    p_aus = pred_au[au_m]
    f1_au_05 = f1_macro_au(y_aus, p_aus, threshold=0.5)
    f1_au_best, t_au_best = f1_score_max(y_aus, p_aus, thresh=np.arange(0.1, 1.0, 0.1))

    # ---- P_MTL ----
    p_total = p_mtl(ccc_VA, f1_expr, f1_au_05)
    p_total_best = p_mtl(ccc_VA, f1_expr, f1_au_best)

    metrics = {
        "ccc_V": ccc_V,
        "ccc_A": ccc_A,
        "CCC_VA": ccc_VA,
        "F1_EXPR_macro": f1_expr,
        "ACC_EXPR": acc_expr,
        "F1_AU@0.5": f1_au_05,
        "F1_AU_best": f1_au_best,
        "t_AU_best": t_au_best,
        "P_MTL@0.5": p_total,
        "P_MTL_best": p_total_best,
        "smoothing": float(smooth),
        "sigma": float(sigma) if smooth else float("nan"),
        "delta": float(delta) if smooth else float("nan"),
    }

    if output_dir is None:
        output_dir = Path(cfg.output.results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_smoothed" if smooth else ""
    md_path = output_dir / f"metrics{suffix}.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Stage 1 metrics (smoothing={'on' if smooth else 'off'})\n\n")
        if smooth:
            fh.write(f"sigma={sigma}, delta={delta}\n\n")
        fh.write("| metric | value |\n| --- | --- |\n")
        for k, v in metrics.items():
            fh.write(f"| {k} | {v:.4f} |\n")
        fh.write("\n## Paper A reproduction targets (cell 60, 'aligned')\n\n")
        fh.write("| backbone | ccc_V | ccc_A | F1_EXPR | P_MTL |\n")
        fh.write("| --- | --- | --- | --- | --- |\n")
        fh.write("| enet_b0_8_va_mtl | 0.4433 | 0.3422 | 0.5040 | 1.2896 |\n")
        fh.write("| mbf_va_mtl       | 0.4503 | 0.2870 | 0.4891 | 1.2264 |\n")

    print(f"[stage1] metrics written to {md_path}")
    for k, v in metrics.items():
        print(f"  {k:>14s} = {v:.4f}")
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 evaluation.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--smooth", action="store_true", help="Apply per-video Gaussian smoothing to EXPR and VA.")
    p.add_argument("--sigma", type=float, default=100000.0)
    p.add_argument("--delta", type=int, default=50)
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    evaluate_mtl(
        checkpoint=args.checkpoint,
        config_path=args.config,
        smooth=args.smooth,
        sigma=args.sigma,
        delta=args.delta,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
