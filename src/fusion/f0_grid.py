"""F0 - Grid-searched late blend (HSEmotion ABAW-8 baseline).

F0 has no trainable parameters of its own. It reuses the unimodal
predictions from the already-trained :class:`VisualOnly` and
:class:`AudioOnly` heads and, at evaluation time, grid-searches a
per-task scalar blend weight ``w in [0, 1]`` (step 0.1) plus, for AUs,
a decision threshold ``t in {0.2, ..., 0.7}``. The resulting blend and
threshold are reported alongside the fused metrics.

Matches the strategy in cells 76-85 of ``bah.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from src.utils.metrics import CCC_score, f1_score_max, metric_for_Exp


@dataclass
class GridSearchResult:
    w_expr: float
    w_va: float
    w_au: float
    t_au: float
    metrics: Dict[str, float]


def _softmax_safe(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax."""
    x = logits - logits.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def grid_search_blend(
    v_expr_logits: np.ndarray,
    v_va: np.ndarray,
    v_au: np.ndarray,
    a_expr_logits: np.ndarray,
    a_va: np.ndarray,
    a_au: np.ndarray,
    y_expr: np.ndarray,
    y_va: np.ndarray,
    y_aus: np.ndarray,
    m_expr: np.ndarray,
    m_va: np.ndarray,
    m_au: np.ndarray,
    *,
    w_grid: Sequence[float] = tuple(round(x, 2) for x in np.arange(0.0, 1.001, 0.1)),
    au_thresh_grid: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> GridSearchResult:
    """Blend unimodal predictions per task and grid-search ``w``.

    Returns the best ``(w_expr, w_va, w_au, t_au)`` along with the
    resulting ccc_V/A, F1_EXPR, F1_AU, and P_MTL metrics.
    """
    v_prob = _softmax_safe(v_expr_logits)
    a_prob = _softmax_safe(a_expr_logits)

    # ---- EXPR ----
    em = m_expr == 1
    best_f1_expr, best_w_expr = -1.0, 0.0
    for w in w_grid:
        probs = w * v_prob + (1 - w) * a_prob
        y_hat = probs.argmax(axis=-1)
        f1, _, _ = metric_for_Exp(y_expr[em], y_hat[em])
        if f1 > best_f1_expr:
            best_f1_expr, best_w_expr = f1, float(w)

    # ---- VA (CCC on each dim; blend with a shared w) ----
    vm = m_va == 1
    best_ccc_avg, best_w_va = -np.inf, 0.0
    best_cV = best_cA = 0.0
    for w in w_grid:
        blended = w * v_va + (1 - w) * a_va
        cV = CCC_score(y_va[vm, 0], blended[vm, 0])
        cA = CCC_score(y_va[vm, 1], blended[vm, 1])
        avg = 0.5 * (cV + cA)
        if avg > best_ccc_avg:
            best_ccc_avg, best_w_va = avg, float(w)
            best_cV, best_cA = cV, cA

    # ---- AU (joint sweep over w and threshold) ----
    am = m_au == 1
    best_f1_au, best_w_au, best_t_au = 0.0, 0.0, 0.5
    if am.sum() > 0:
        for w in w_grid:
            blended = w * v_au + (1 - w) * a_au
            f1_t, t_t = f1_score_max(
                y_aus[am], blended[am], thresh=au_thresh_grid
            )
            if f1_t > best_f1_au:
                best_f1_au, best_w_au, best_t_au = f1_t, float(w), float(t_t)

    p_mtl = 2 * best_ccc_avg + best_f1_expr + best_f1_au
    # Paper A's P_MTL = CCC_V + CCC_A + F1_EXPR + F1_AU. 2 * avg == CCC_V + CCC_A,
    # hence the multiplier above.

    return GridSearchResult(
        w_expr=best_w_expr,
        w_va=best_w_va,
        w_au=best_w_au,
        t_au=best_t_au,
        metrics={
            "ccc_V": float(best_cV),
            "ccc_A": float(best_cA),
            "CCC_VA": float(best_ccc_avg),
            "F1_EXPR_macro": float(best_f1_expr),
            "F1_AU_best": float(best_f1_au),
            "t_AU_best": float(best_t_au),
            "P_MTL": float(p_mtl),
        },
    )
