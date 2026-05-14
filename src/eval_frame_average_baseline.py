"""Frame-average baseline using EmotiEffLib pretrained 10-D scores.

Supervisor's prior: "averaging per-frame predictions from my pretrained
models is competitive". This script materialises that baseline so that
every learned model in the report has an apples-to-apples zero-train
comparison row.

The 10-D `scores` saved by ``src.features.extract_visual`` is the
EmotiEffLib head's raw output: indices 0..7 are AffectNet-8 EXPR
logits, index 8 is valence, index 9 is arousal. AU is not predicted
and is reported as n/a.

Two aggregation modes:
  * ``frame``: per-frame predictions, evaluated against the per-frame
    Aff-Wild2 MTL annotations. Anchors the lower bound the trained
    heads have to beat at the same granularity.
  * ``video``: average the 10-D scores (softmax for EXPR, mean for VA)
    across all frames in a video, then broadcast back to every annotated
    frame. This matches the per-video frame-averaging the supervisor
    referenced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.special import softmax

from src.train import load_split
from src.utils.metrics import f1_macro_au, metric_for_Exp, metric_for_VA, p_mtl


def _per_frame_scores(X: np.ndarray) -> np.ndarray:
    return X[:, -10:].astype(np.float32)


def _video_average(scores: np.ndarray, videoname_frames):
    """Replace each row's 10-D vector with the mean over its video.

    EXPR uses mean of softmax (Paper-A-style averaging of probabilities).
    VA uses mean of raw values.
    """
    out = scores.copy()
    expr_prob = softmax(scores[:, :8], axis=1)
    by_video: dict[str, list[int]] = {}
    for i, (vid, _) in enumerate(videoname_frames):
        by_video.setdefault(vid, []).append(i)
    for idx in by_video.values():
        idx_arr = np.asarray(idx, dtype=np.int64)
        mean_prob = expr_prob[idx_arr].mean(axis=0)
        out[idx_arr, :8] = np.log(mean_prob + 1e-12)  # logit-space stand-in for argmax
        out[idx_arr, 8] = scores[idx_arr, 8].mean()
        out[idx_arr, 9] = scores[idx_arr, 9].mean()
    return out


def _metrics(anno, scores: np.ndarray, num_expr: int = 8) -> dict:
    va_m = anno.mask_va == 1
    ccc_V, ccc_A, ccc_VA = metric_for_VA(
        anno.y_va[va_m, 0], anno.y_va[va_m, 1],
        scores[va_m, 8], scores[va_m, 9],
    )
    ex_m = anno.mask_expr == 1
    y_pred_expr = scores[:, :8].argmax(axis=1)
    f1_expr, _, _ = metric_for_Exp(
        anno.y_expr[ex_m], y_pred_expr[ex_m], class_num=num_expr,
    )
    return {
        "ccc_V": float(ccc_V),
        "ccc_A": float(ccc_A),
        "ccc_VA": float(ccc_VA),
        "f1_expr": float(f1_expr),
        "f1_au": None,           # 10-D vector has no AU head
        "p_mtl_va_expr": float(ccc_VA + f1_expr),  # P_MTL' (3-task analogue)
    }


def evaluate(config_path: str | Path) -> dict:
    cfg = OmegaConf.load(config_path)
    X_val, anno_val = load_split(cfg.data.val_annotations, cfg.features_cache)
    raw_scores = _per_frame_scores(X_val)
    avg_scores = _video_average(raw_scores, anno_val.videoname_frames)
    return {
        "config": str(config_path),
        "frame": _metrics(anno_val, raw_scores),
        "video": _metrics(anno_val, avg_scores),
        "n_frames": int(len(anno_val)),
        "n_videos": len({v for v, _ in anno_val.videoname_frames}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Stage-1 config (e.g. configs/aw2_stage1_enet.yaml)")
    ap.add_argument("--out", default=None,
                    help="JSON path for the metrics dict")
    args = ap.parse_args()
    result = evaluate(args.config)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
