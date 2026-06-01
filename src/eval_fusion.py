"""Evaluator for Stage 3 fusion variants + F0 post-hoc late blend.

For every trainable variant (F1-F5 and the unimodal F0 baselines), this
module computes ``ccc_V, ccc_A, CCC_VA, F1_EXPR, F1_AU@0.5, F1_AU*,
P_MTL`` on the validation split and writes a Markdown table. F0 is
handled as a special post-hoc mode: it loads two separately-trained
unimodal checkpoints and grid-searches ``w`` per task.

Usage:
    python -m src.eval_fusion --config configs/stage3_f4_xattn.yaml \
        --checkpoint results/stage3_f4_xattn/best.pt

    # F0 late blend (requires two trained unimodal heads):
    python -m src.eval_fusion --mode f0_grid \
        --visual-checkpoint  results/stage3_visual_only/best.pt \
        --audio-checkpoint   results/stage3_audio_only/best.pt \
        --config configs/stage3_f0_grid.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.datasets.bimodal import BimodalFrameDataset, BimodalWindowDataset
from src.fusion.base import FusionConfig
from src.fusion.f0_grid import grid_search_blend
from src.fusion.f6a_dirichlet import grid_search_dirichlet
from src.train_fusion import FUSION_REGISTRY, build_model, wants_window
from src.utils.metrics import (
    f1_macro_au,
    f1_score_max,
    metric_for_Exp,
    metric_for_VA,
    p_mtl,
)


def _collect_predictions(
    model, loader, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expr_logits, va_pred, au_pred = [], [], []
    y_expr, y_va, y_aus, m_expr, m_va, m_au = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            e, v, a = model(
                batch["v_feat"].to(device),
                batch["v_scores"].to(device),
                batch["a_feat"].to(device),
            )
            expr_logits.append(e.cpu().numpy())
            va_pred.append(v.cpu().numpy())
            au_pred.append(a.cpu().numpy())
            y_expr.append(batch["y_expr"].numpy())
            y_va.append(batch["y_va"].numpy())
            y_aus.append(batch["y_aus"].numpy())
            m_expr.append(batch["m_expr"].numpy())
            m_va.append(batch["m_va"].numpy())
            m_au.append(batch["m_au"].numpy())
    return tuple(np.concatenate(x, axis=0) for x in (
        expr_logits, va_pred, au_pred,
        y_expr, y_va, y_aus,
        m_expr, m_va, m_au,
    ))


def _metrics_from_preds(
    expr_logits: np.ndarray,
    va_pred: np.ndarray,
    au_pred: np.ndarray,
    y_expr: np.ndarray,
    y_va: np.ndarray,
    y_aus: np.ndarray,
    m_expr: np.ndarray,
    m_va: np.ndarray,
    m_au: np.ndarray,
) -> Dict[str, float]:
    ex = m_expr == 1
    pred_expr = expr_logits.argmax(axis=-1)
    f1_expr, acc_expr, _ = metric_for_Exp(y_expr[ex], pred_expr[ex])

    vm = m_va == 1
    ccc_V, ccc_A, ccc_VA = metric_for_VA(
        y_va[vm, 0], y_va[vm, 1], va_pred[vm, 0], va_pred[vm, 1]
    )

    am = m_au == 1
    if am.sum() == 0:
        # Dataset has no AU labels (e.g. CREMA-D, RAVDESS). Skip AU metrics.
        f1_au_05 = 0.0
        f1_au_best, t_au_best = 0.0, 0.5
    else:
        f1_au_05 = f1_macro_au(y_aus[am], au_pred[am], threshold=0.5)
        f1_au_best, t_au_best = f1_score_max(
            y_aus[am], au_pred[am], thresh=np.arange(0.1, 1.0, 0.1)
        )

    return {
        "ccc_V": ccc_V,
        "ccc_A": ccc_A,
        "CCC_VA": ccc_VA,
        "F1_EXPR_macro": f1_expr,
        "ACC_EXPR": acc_expr,
        "F1_AU@0.5": f1_au_05,
        "F1_AU_best": f1_au_best,
        "t_AU_best": t_au_best,
        "P_MTL@0.5": p_mtl(ccc_VA, f1_expr, f1_au_05),
        "P_MTL_best": p_mtl(ccc_VA, f1_expr, f1_au_best),
    }


def _build_val_loader(cfg, variant: str):
    ds_cls = BimodalWindowDataset if wants_window(variant) else BimodalFrameDataset
    extra = {"window": int(cfg.fusion.get("window", 5))} if wants_window(variant) else {}
    ds = ds_cls(
        annotations_file=cfg.data.val_annotations,
        visual_cache_dir=cfg.visual.features_cache,
        audio_cache_dir=cfg.audio.aligned_dir,
        **extra,
    )
    return DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=False)


def evaluate_variant(
    checkpoint: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, float]:
    cfg = OmegaConf.load(config_path)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    variant = ck["variant"]
    fus_cfg = FusionConfig(**ck["fus_cfg"])
    model = build_model(variant, fus_cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    loader = _build_val_loader(cfg, variant)
    preds = _collect_predictions(model, loader, device)
    metrics = _metrics_from_preds(*preds)
    metrics["variant"] = variant
    metrics["trainable_params"] = int(ck.get("trainable_params", 0))

    if output_dir is None:
        output_dir = Path(cfg.output.results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_md(output_dir / "metrics.md", variant, metrics)
    return metrics


def evaluate_f0_grid(
    visual_checkpoint: str | Path,
    audio_checkpoint: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, float]:
    cfg = OmegaConf.load(config_path)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"

    def _load(ckp):
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        fus_cfg = FusionConfig(**ck["fus_cfg"])
        m = build_model(ck["variant"], fus_cfg)
        m.load_state_dict(ck["state_dict"])
        return m.to(device), ck["variant"]

    m_v, _ = _load(visual_checkpoint)
    m_a, _ = _load(audio_checkpoint)
    loader = _build_val_loader(cfg, "visual_only")  # frame-level dataset shared

    preds_v = _collect_predictions(m_v, loader, device)
    preds_a = _collect_predictions(m_a, loader, device)

    (v_expr, v_va, v_au, y_expr, y_va, y_aus, m_expr, m_va, m_au) = preds_v
    (a_expr, a_va, a_au, *_ ) = preds_a

    res = grid_search_blend(
        v_expr, v_va, v_au,
        a_expr, a_va, a_au,
        y_expr, y_va, y_aus,
        m_expr, m_va, m_au,
    )

    metrics = dict(res.metrics)
    metrics["w_expr"] = res.w_expr
    metrics["w_va"] = res.w_va
    metrics["w_au"] = res.w_au
    metrics["variant"] = "f0_grid"
    if output_dir is None:
        output_dir = Path(cfg.output.results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_md(output_dir / "metrics_f0.md", "f0_grid", metrics)
    return metrics


def evaluate_f6a_dirichlet(
    checkpoints: Dict[str, str | Path],
    config_path: str | Path,
    output_dir: str | Path | None = None,
    step: float = 0.2,
) -> Dict[str, float]:
    """F6a post-hoc Dirichlet ensemble over a dict ``{name: checkpoint}``.

    Loads every checkpoint, runs one validation forward pass per model,
    and grid-searches a per-task simplex blend via
    :func:`src.fusion.f6a_dirichlet.grid_search_dirichlet`. All checkpoints
    must have been trained against the same val split (the function uses
    the first checkpoint's frame-level val loader).
    """
    cfg = OmegaConf.load(config_path)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"

    expr_logits, vas, aus = [], [], []
    y_state = None
    names = list(checkpoints.keys())
    for name in names:
        ckp = checkpoints[name]
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        fus_cfg = FusionConfig(**ck["fus_cfg"])
        model = build_model(ck["variant"], fus_cfg)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        loader = _build_val_loader(cfg, ck["variant"])
        preds = _collect_predictions(model, loader, device)
        expr_logits.append(preds[0])
        vas.append(preds[1])
        aus.append(preds[2])
        if y_state is None:
            y_state = preds[3:]
        del model

    y_expr, y_va, y_aus, m_expr, m_va, m_au = y_state
    res = grid_search_dirichlet(
        expr_logits, vas, aus,
        y_expr, y_va, y_aus,
        m_expr, m_va, m_au,
        variant_names=names,
        step=step,
    )
    metrics = dict(res.metrics)
    metrics["variant"] = "f6a_dirichlet"
    metrics["w_expr"] = res.w_expr
    metrics["w_va"] = res.w_va
    metrics["w_au"] = res.w_au
    metrics["t_au"] = res.t_au
    metrics["variants"] = res.variants

    if output_dir is None:
        output_dir = Path(cfg.output.results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_md(output_dir / "metrics_f6a.md", "f6a_dirichlet", metrics)
    return metrics


def _write_md(path: Path, variant: str, metrics: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Stage 3 metrics - variant={variant}\n\n")
        fh.write("| metric | value |\n| --- | --- |\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                fh.write(f"| {k} | {float(v):.4f} |\n")
            else:
                fh.write(f"| {k} | {v} |\n")
    print(f"[stage3] wrote {path}")
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k:>18s} = {float(v):.4f}")
        else:
            print(f"  {k:>18s} = {v}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 3 evaluator.")
    p.add_argument(
        "--mode",
        default="variant",
        choices=["variant", "f0_grid"],
        help="'variant' for F1-F5 / unimodals; 'f0_grid' for post-hoc late blend.",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", help="Required for --mode variant.")
    p.add_argument("--visual-checkpoint", help="Required for --mode f0_grid.")
    p.add_argument("--audio-checkpoint", help="Required for --mode f0_grid.")
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.mode == "variant":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --mode variant")
        evaluate_variant(
            checkpoint=args.checkpoint,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    else:
        if not (args.visual_checkpoint and args.audio_checkpoint):
            raise SystemExit(
                "--visual-checkpoint and --audio-checkpoint are required for "
                "--mode f0_grid"
            )
        evaluate_f0_grid(
            visual_checkpoint=args.visual_checkpoint,
            audio_checkpoint=args.audio_checkpoint,
            config_path=args.config,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
