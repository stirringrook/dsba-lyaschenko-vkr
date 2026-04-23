"""Evaluation metrics for ABAW MTL.

Ports the three helpers from cells 3 and 34 of
``EmotiEffLib-main/training_and_examples/ABAW/ABAW7/mtl.ipynb``
(``CCC_score``, ``metric_for_VA``, ``metric_for_Exp``) to NumPy +
scikit-learn. Adds ``f1_score_max`` for per-AU threshold tuning
(cell 34 ``print_au``) and ``p_mtl`` for the competition metric.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def CCC_score(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient.

    Matches cell 3 of ``mtl.ipynb`` byte-for-byte so the reproduction
    numbers are directly comparable.

    Args:
        x: 1-D array of ground-truth values.
        y: 1-D array of predicted values with the same shape as ``x``.

    Returns:
        The concordance correlation coefficient in [-1, 1]. When either
        input is constant the function returns 0.0 (matches the limit of
        the CCC formula as the variance goes to 0).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    vx = x - x.mean()
    vy = y - y.mean()
    denom_rho = np.sqrt((vx * vx).sum()) * np.sqrt((vy * vy).sum())
    if denom_rho == 0.0:
        return 0.0
    rho = (vx * vy).sum() / denom_rho
    x_s = x.std()
    y_s = y.std()
    denom = x_s ** 2 + y_s ** 2 + (x.mean() - y.mean()) ** 2
    if denom == 0.0:
        return 0.0
    return float(2.0 * rho * x_s * y_s / denom)


def metric_for_VA(
    gt_V: np.ndarray,
    gt_A: np.ndarray,
    pred_V: np.ndarray,
    pred_A: np.ndarray,
) -> Tuple[float, float, float]:
    """Per-dimension and averaged CCC for valence/arousal.

    Returns:
        ``(ccc_V, ccc_A, 0.5 * (ccc_V + ccc_A))``.
    """
    ccc_V = CCC_score(gt_V, pred_V)
    ccc_A = CCC_score(gt_A, pred_A)
    return ccc_V, ccc_A, 0.5 * (ccc_V + ccc_A)


def metric_for_Exp(
    gt: np.ndarray,
    pred: np.ndarray,
    class_num: int = 8,
) -> Tuple[float, float, np.ndarray]:
    """Macro F1 / accuracy / per-class F1 for expression classification.

    Args:
        gt: 1-D array of integer labels in ``[0, class_num)``.
        pred: 1-D array of predicted integer labels.
        class_num: Number of expression classes.

    Returns:
        ``(F1_macro, accuracy, F1_per_class)`` where ``F1_per_class`` is a
        NumPy array of length ``class_num``.
    """
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    acc = accuracy_score(gt, pred)
    per_class = np.zeros(class_num, dtype=np.float64)
    for i in range(class_num):
        per_class[i] = f1_score((gt == i).astype(int), (pred == i).astype(int))
    return float(per_class.mean()), float(acc), per_class


def f1_score_max(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresh: Iterable[float] = np.arange(0.1, 1.0, 0.1),
) -> Tuple[float, float]:
    """Grid-searched macro F1 for multi-label AU predictions.

    For each threshold ``t`` in ``thresh`` computes the mean over AUs of
    ``f1_score(y_true, y_score >= t)`` and returns the best ``(f1, t)``.
    Matches the logic in cell 34 of the notebook.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresh:
        y_hat = (y_score >= t).astype(int)
        per_au = [
            f1_score(y_true[:, i], y_hat[:, i]) for i in range(y_true.shape[1])
        ]
        mean_f1 = float(np.mean(per_au))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_t = float(t)
    return best_f1, best_t


def f1_macro_au(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """Mean per-AU F1 at a fixed threshold. Paper A reports F1_AU at t=0.5."""
    y_true = np.asarray(y_true).astype(int)
    y_hat = (np.asarray(y_score) >= threshold).astype(int)
    per_au = [f1_score(y_true[:, i], y_hat[:, i]) for i in range(y_true.shape[1])]
    return float(np.mean(per_au))


def p_mtl(ccc_VA: float, f1_expr: float, f1_au: float) -> float:
    """Competition metric: ``P_MTL = CCC_VA + F1_EXPR + F1_AU``."""
    return float(ccc_VA + f1_expr + f1_au)
