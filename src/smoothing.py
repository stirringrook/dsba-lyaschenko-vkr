"""Per-video Gaussian smoothing of frame-level predictions.

Port of cells 62-66 of ``EmotiEffLib-main/training_and_examples/ABAW/ABAW7/mtl.ipynb``.
Smoothing is applied to EXPR probabilities and VA regressions; it is
NOT applied to AU probabilities because cell 66 confirmed it hurts AU
F1 in the notebook's grid search.

Grid-search defaults:
    sigma in {0.1, 1, 10, 50, 100, 500, 1000, 10000, 100000}
    delta in {1, 5, 10, 50, 100}
With sigma=100000 the Gaussian becomes effectively a box filter of
width ``2*delta + 1``. Paper A's reported number uses this extreme.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def gaussian_smooth_per_video(
    scores: np.ndarray,
    videoname_frames: Sequence[Tuple[str, int]],
    sigma: float,
    delta: int,
) -> np.ndarray:
    """Per-video Gaussian-weighted smoothing over a +/-delta window.

    For each row ``i``, replaces ``scores[i]`` with the Gaussian-weighted
    average of ``scores[j]`` where ``j`` is in ``[i - delta, i + delta]``
    **AND** ``videoname_frames[j][0] == videoname_frames[i][0]``. The
    weight is ``exp(-(frame_j - frame_i)^2 / (2 * sigma^2))``.

    The notebook walks the row order blindly and relies on same-video
    contiguity inside that order. We do the same and therefore assume
    ``videoname_frames`` was produced by the dataset parser (rows for a
    given video appear consecutively in annotation-file order).

    Args:
        scores: Shape ``(N, K)`` array of frame-level predictions.
        videoname_frames: List of ``(videoname, frame_index)`` with
            ``len == N``, aligned row-for-row with ``scores``.
        sigma: Standard deviation of the Gaussian kernel in frame-index units.
        delta: Half-window size (in rows, not frame indices).

    Returns:
        Smoothed array of shape ``(N, K)``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n != len(videoname_frames):
        raise ValueError(
            f"Length mismatch: scores={n} rows, videoname_frames={len(videoname_frames)}"
        )
    if delta < 0:
        raise ValueError("delta must be >= 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    out = np.zeros_like(scores)
    two_sigma_sq = 2.0 * sigma * sigma

    for i in range(n):
        video_i, frame_i = videoname_frames[i]
        lo = max(0, i - delta)
        hi = min(n, i + delta + 1)
        w_sum = 0.0
        acc = np.zeros(scores.shape[1], dtype=np.float64)
        for j in range(lo, hi):
            video_j, frame_j = videoname_frames[j]
            if video_j != video_i:
                continue
            k = np.exp(-((frame_j - frame_i) ** 2) / two_sigma_sq)
            acc += k * scores[j]
            w_sum += k
        out[i] = acc / w_sum  # w_sum >= exp(0) = 1 because j=i is always in-window
    return out.astype(scores.dtype, copy=False)


def grid_search_smoothing(
    y_true: np.ndarray,
    scores: np.ndarray,
    videoname_frames: Sequence[Tuple[str, int]],
    metric_fn,
    sigmas: Sequence[float] = (0.1, 1, 10, 50, 100, 500, 1000, 10000, 100000),
    deltas: Sequence[int] = (1, 5, 10, 50, 100),
):
    """Grid-search ``(sigma, delta)`` against an arbitrary metric.

    Args:
        y_true: Ground-truth array passed unchanged to ``metric_fn``.
        scores: Frame-level predictions to smooth.
        videoname_frames: Same alignment as ``gaussian_smooth_per_video``.
        metric_fn: ``f(y_true, y_smoothed) -> float``; higher is better.
        sigmas / deltas: Values to grid over (notebook defaults).

    Returns:
        ``(best_metric, best_sigma, best_delta, best_smoothed)``.
    """
    best = (-np.inf, None, None, None)
    for sigma in sigmas:
        for delta in deltas:
            smoothed = gaussian_smooth_per_video(
                scores, videoname_frames, sigma=sigma, delta=delta
            )
            m = float(metric_fn(y_true, smoothed))
            if m > best[0]:
                best = (m, float(sigma), int(delta), smoothed)
    return best
