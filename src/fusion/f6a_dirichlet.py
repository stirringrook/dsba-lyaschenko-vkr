"""F6a - Dirichlet-weighted post-hoc ensemble over F1-F5 + visual_only.

No trainable parameters. Loads N trained variants' validation predictions
once, then searches a per-task simplex weight vector ``w in Delta^N`` that
maximises macro-F1 (EXPR), CCC-VA, and threshold-tuned F1_AU
respectively. Writes the best metric block plus the chosen weights.

References: Dresvyanskiy et al. (CVPRW 2024) "Hierarchical Network for
Facial Emotion Recognition" (the SUN team's ABAW-7 entry, which uses a
Dirichlet-weighted fusion of multiple branches).

This module is invoked from :mod:`src.eval_fusion` via ``--mode f6a``;
see also ``configs/stage3_f6a_dirichlet.yaml`` for the variant list.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.utils.metrics import CCC_score, f1_score_max, metric_for_Exp


@dataclass
class DirichletResult:
    variants: List[str]
    w_expr: List[float]
    w_va: List[float]
    w_au: List[float]
    t_au: float
    metrics: Dict[str, float]


def _softmax_safe(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def _simplex_grid(n_components: int, step: float = 0.2) -> List[Tuple[float, ...]]:
    """All weight vectors of length ``n_components`` summing to 1 on a step-grid.

    For ``n=5, step=0.2`` this yields C(n+k-1, k-1) = 126 points where
    ``k = 1/step + 1 = 6``; cheap enough to grid-search per task.
    """
    levels = int(round(1.0 / step)) + 1
    grid = []
    for combo in product(range(levels), repeat=n_components):
        if sum(combo) != levels - 1:
            continue
        grid.append(tuple(c * step for c in combo))
    return grid


def grid_search_dirichlet(
    expr_logits: List[np.ndarray],
    vas: List[np.ndarray],
    aus: List[np.ndarray],
    y_expr: np.ndarray,
    y_va: np.ndarray,
    y_aus: np.ndarray,
    m_expr: np.ndarray,
    m_va: np.ndarray,
    m_au: np.ndarray,
    *,
    variant_names: Sequence[str],
    step: float = 0.2,
    au_thresh_grid: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> DirichletResult:
    n = len(expr_logits)
    if not (n == len(vas) == len(aus) == len(variant_names)):
        raise ValueError("expr_logits / vas / aus / variant_names must align")
    grid = _simplex_grid(n, step=step)

    probs = [_softmax_safe(L) for L in expr_logits]

    # ---- EXPR ----
    em = m_expr == 1
    best_f1_expr, best_w_expr = -1.0, [1.0 / n] * n
    for w in grid:
        blended = sum(wi * pi for wi, pi in zip(w, probs))
        y_hat = blended.argmax(axis=-1)
        f1, _, _ = metric_for_Exp(y_expr[em], y_hat[em])
        if f1 > best_f1_expr:
            best_f1_expr, best_w_expr = float(f1), list(w)

    # ---- VA ----
    vm = m_va == 1
    best_ccc_avg, best_w_va = -np.inf, [1.0 / n] * n
    best_cV = best_cA = 0.0
    for w in grid:
        blended = sum(wi * vi for wi, vi in zip(w, vas))
        cV = CCC_score(y_va[vm, 0], blended[vm, 0])
        cA = CCC_score(y_va[vm, 1], blended[vm, 1])
        avg = 0.5 * (cV + cA)
        if avg > best_ccc_avg:
            best_ccc_avg, best_w_va = float(avg), list(w)
            best_cV, best_cA = float(cV), float(cA)

    # ---- AU ----
    am = m_au == 1
    best_f1_au, best_w_au, best_t_au = 0.0, [1.0 / n] * n, 0.5
    if am.sum() > 0:
        for w in grid:
            blended = sum(wi * ai for wi, ai in zip(w, aus))
            f1_t, t_t = f1_score_max(
                y_aus[am], blended[am], thresh=au_thresh_grid
            )
            if f1_t > best_f1_au:
                best_f1_au, best_w_au, best_t_au = float(f1_t), list(w), float(t_t)

    p_mtl = 2 * best_ccc_avg + best_f1_expr + best_f1_au

    return DirichletResult(
        variants=list(variant_names),
        w_expr=best_w_expr,
        w_va=best_w_va,
        w_au=best_w_au,
        t_au=best_t_au,
        metrics={
            "ccc_V": best_cV,
            "ccc_A": best_cA,
            "CCC_VA": best_ccc_avg,
            "F1_EXPR_macro": best_f1_expr,
            "F1_AU_best": best_f1_au,
            "t_AU_best": best_t_au,
            "P_MTL": p_mtl,
        },
    )
