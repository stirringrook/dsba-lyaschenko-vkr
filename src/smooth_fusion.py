"""Per-video Gaussian smoothing grid for Stage 3 fusion checkpoints.

Loads a single trained fusion variant, runs the validation forward pass
once, caches predictions, then sweeps the Paper-A grid
(sigma, delta) over EXPR softmax probabilities and VA regressions
(AU is never smoothed, per Paper-A cell 66 and the docstring of
:mod:`src.smoothing`).

Reports raw vs.\\ best-smoothed metrics on both ``P_MTL@0.5`` and
``P_MTL_best`` (the latter re-tunes the AU threshold per Paper-A
cell 66's scheme).

Usage::

    python -m src.smooth_fusion \\
        --checkpoint results/stage3_f4_xattn/best.pt \\
        --config configs/stage3_f4_xattn.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.special import softmax
from torch.utils.data import DataLoader

from src.datasets.bimodal import BimodalFrameDataset, BimodalWindowDataset
from src.fusion.base import FusionConfig
from src.smoothing import gaussian_smooth_per_video
from src.train_fusion import build_model, wants_window
from src.utils.metrics import (
    f1_macro_au,
    f1_score_max,
    metric_for_Exp,
    metric_for_VA,
    p_mtl,
)


_DEFAULT_SIGMAS = (0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1e3, 1e4, 1e5)
_DEFAULT_DELTAS = (1, 5, 10, 50, 100)


def _build_val_dataset(cfg, variant: str):
    if wants_window(variant):
        ds_cls = BimodalWindowDataset
        extra = {"window": int(cfg.fusion.get("window", 5))}
    else:
        ds_cls = BimodalFrameDataset
        extra = {}
    return ds_cls(
        annotations_file=cfg.data.val_annotations,
        visual_cache_dir=cfg.visual.features_cache,
        audio_cache_dir=cfg.audio.aligned_dir,
        **extra,
    )


def _forward_once(checkpoint: Path, config_path: Path):
    cfg = OmegaConf.load(config_path)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    variant = ck["variant"]
    fus_cfg = FusionConfig(**ck["fus_cfg"])
    model = build_model(variant, fus_cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    dataset = _build_val_dataset(cfg, variant)
    loader = DataLoader(dataset, batch_size=int(cfg.train.batch_size), shuffle=False)

    expr_logits, va_pred, au_pred = [], [], []
    y_expr, y_va, y_aus, m_expr, m_va, m_au = [], [], [], [], [], []
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

    cat = lambda xs: np.concatenate(xs, axis=0)
    return {
        "variant": variant,
        "videoname_frames": dataset.videoname_frames,
        "expr_prob": softmax(cat(expr_logits), axis=-1),
        "va": cat(va_pred),
        "au": cat(au_pred),
        "y_expr": cat(y_expr),
        "y_va": cat(y_va),
        "y_aus": cat(y_aus),
        "m_expr": cat(m_expr),
        "m_va": cat(m_va),
        "m_au": cat(m_au),
    }


def _metrics(state, expr_prob, va, *, au_threshold: float | None = None) -> Dict[str, float]:
    ex = state["m_expr"] == 1
    pred_expr = expr_prob.argmax(axis=-1)
    f1_expr, acc_expr, _ = metric_for_Exp(state["y_expr"][ex], pred_expr[ex])

    vm = state["m_va"] == 1
    ccc_V, ccc_A, ccc_VA = metric_for_VA(
        state["y_va"][vm, 0], state["y_va"][vm, 1], va[vm, 0], va[vm, 1]
    )

    am = state["m_au"] == 1
    if am.sum() == 0:
        f1_au_05 = 0.0
        f1_au_best, t_au_best = 0.0, 0.5
    else:
        f1_au_05 = f1_macro_au(state["y_aus"][am], state["au"][am], threshold=0.5)
        if au_threshold is None:
            f1_au_best, t_au_best = f1_score_max(
                state["y_aus"][am], state["au"][am], thresh=np.arange(0.1, 1.0, 0.1)
            )
        else:
            f1_au_best = f1_macro_au(state["y_aus"][am], state["au"][am], threshold=au_threshold)
            t_au_best = float(au_threshold)

    return {
        "ccc_V": float(ccc_V),
        "ccc_A": float(ccc_A),
        "CCC_VA": float(ccc_VA),
        "F1_EXPR_macro": float(f1_expr),
        "ACC_EXPR": float(acc_expr),
        "F1_AU@0.5": float(f1_au_05),
        "F1_AU_best": float(f1_au_best),
        "t_AU_best": float(t_au_best),
        "P_MTL@0.5": float(p_mtl(ccc_VA, f1_expr, f1_au_05)),
        "P_MTL_best": float(p_mtl(ccc_VA, f1_expr, f1_au_best)),
    }


def smoothing_grid(
    checkpoint: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    sigmas: Tuple[float, ...] = _DEFAULT_SIGMAS,
    deltas: Tuple[int, ...] = _DEFAULT_DELTAS,
) -> Dict:
    state = _forward_once(Path(checkpoint), Path(config_path))
    variant = state["variant"]

    raw = _metrics(state, state["expr_prob"], state["va"])
    raw_au_threshold = raw["t_AU_best"]

    rows: List[Dict] = [{"sigma": None, "delta": None, **raw}]
    best_p, best_row = raw["P_MTL@0.5"], rows[0]

    vn_frames = state["videoname_frames"]
    for sigma in sigmas:
        for delta in deltas:
            sm_expr = gaussian_smooth_per_video(
                state["expr_prob"], vn_frames, sigma=float(sigma), delta=int(delta)
            )
            sm_va = gaussian_smooth_per_video(
                state["va"], vn_frames, sigma=float(sigma), delta=int(delta)
            )
            m = _metrics(state, sm_expr, sm_va, au_threshold=raw_au_threshold)
            row = {"sigma": float(sigma), "delta": int(delta), **m}
            rows.append(row)
            if m["P_MTL@0.5"] > best_p:
                best_p, best_row = m["P_MTL@0.5"], row
            print(
                f"[{variant}] sigma={sigma:>8} delta={delta:>3}  "
                f"P_MTL@0.5={m['P_MTL@0.5']:.4f}  CCC_VA={m['CCC_VA']:.4f}  "
                f"F1_EXPR={m['F1_EXPR_macro']:.4f}"
            )

    if output_dir is None:
        cfg = OmegaConf.load(config_path)
        output_dir = Path(cfg.output.results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "variant": variant,
        "raw": raw,
        "best": best_row,
        "delta_over_raw": best_row["P_MTL@0.5"] - raw["P_MTL@0.5"],
        "grid_sigmas": list(map(float, sigmas)),
        "grid_deltas": list(map(int, deltas)),
        "rows": rows,
    }
    (output_dir / "smoothing.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_md(output_dir / "smoothing.md", summary)
    return summary


def _write_md(path: Path, summary: Dict) -> None:
    raw, best = summary["raw"], summary["best"]
    lines = [
        f"# Stage 3 smoothing grid - variant={summary['variant']}\n",
        "## Raw vs. best-smoothed",
        "| metric | raw | best smoothed | delta |",
        "| --- | --- | --- | --- |",
    ]
    for k in (
        "ccc_V", "ccc_A", "CCC_VA", "F1_EXPR_macro",
        "F1_AU@0.5", "F1_AU_best", "P_MTL@0.5", "P_MTL_best",
    ):
        d = best[k] - raw[k]
        lines.append(f"| {k} | {raw[k]:.4f} | {best[k]:.4f} | {d:+.4f} |")
    lines.append("")
    lines.append(f"Best key: sigma={best['sigma']}  delta={best['delta']}")
    lines.append(f"AU threshold held at raw best t={raw['t_AU_best']:.2f} "
                 "across the grid (smoothing is never applied to AU).")
    lines.append("")
    lines.append("## Full grid (sorted by P_MTL@0.5)")
    lines.append("| sigma | delta | CCC_VA | F1_EXPR | F1_AU@0.5 | P_MTL@0.5 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in sorted(summary["rows"], key=lambda x: -x["P_MTL@0.5"])[:15]:
        s = "raw" if r["sigma"] is None else f"{r['sigma']:g}"
        d = "raw" if r["delta"] is None else f"{r['delta']}"
        lines.append(
            f"| {s} | {d} | {r['CCC_VA']:.4f} | {r['F1_EXPR_macro']:.4f} | "
            f"{r['F1_AU@0.5']:.4f} | {r['P_MTL@0.5']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[smooth_fusion] wrote {path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    smoothing_grid(
        checkpoint=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
