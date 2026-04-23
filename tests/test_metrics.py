"""Unit tests for src/utils/metrics.py (no dataset required)."""

import numpy as np
import pytest

from src.utils.metrics import (
    CCC_score,
    f1_macro_au,
    f1_score_max,
    metric_for_Exp,
    metric_for_VA,
    p_mtl,
)


def test_ccc_identical_is_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    assert CCC_score(x, x) == pytest.approx(1.0, abs=1e-9)


def test_ccc_constant_is_zero():
    x = np.zeros(100)
    y = np.ones(100)
    assert CCC_score(x, y) == 0.0
    assert CCC_score(x, x) == 0.0  # both constant -> degenerate


def test_ccc_matches_reference_implementation():
    """Check against the algebraically-equivalent form (ddof=0 everywhere).

    The notebook's secondary ``CCC_numpy`` mixes ``np.cov`` (ddof=1) with
    ``np.var`` (ddof=0), which is a small inconsistency. The primary
    ``CCC_score`` (cell 3) is ddof=0-throughout, so we cross-check against
    that form of the formula.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    y = 0.8 * x + 0.2 * rng.standard_normal(1000)

    s_xy = ((x - x.mean()) * (y - y.mean())).mean()  # ddof=0 covariance
    x_m, y_m = x.mean(), y.mean()
    s_x_sq, s_y_sq = x.var(), y.var()
    ref = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)

    assert CCC_score(x, y) == pytest.approx(ref, rel=1e-6)


def test_ccc_negative_correlation():
    x = np.linspace(-1, 1, 200)
    y = -x
    assert CCC_score(x, y) == pytest.approx(-1.0, abs=1e-9)


def test_metric_for_va_averages_components():
    rng = np.random.default_rng(1)
    gt_V = rng.standard_normal(300)
    gt_A = rng.standard_normal(300)
    pred_V = gt_V + 0.1 * rng.standard_normal(300)
    pred_A = gt_A + 0.3 * rng.standard_normal(300)

    ccc_V, ccc_A, avg = metric_for_VA(gt_V, gt_A, pred_V, pred_A)
    assert avg == pytest.approx(0.5 * (ccc_V + ccc_A))
    assert 0.8 < ccc_V < 1.0
    assert 0.5 < ccc_A < 1.0


def test_metric_for_exp_perfect_prediction():
    gt = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=int)
    pred = gt.copy()
    f1_macro, acc, per_class = metric_for_Exp(gt, pred, class_num=8)
    assert acc == 1.0
    assert f1_macro == 1.0
    assert per_class.shape == (8,)


def test_f1_score_max_picks_best_threshold():
    rng = np.random.default_rng(7)
    n, au = 200, 4
    y_true = (rng.random((n, au)) > 0.7).astype(int)
    y_score = y_true * 0.8 + 0.1 * rng.random((n, au))

    best_f1, best_t = f1_score_max(y_true, y_score, thresh=np.arange(0.1, 1.0, 0.1))
    assert 0.0 <= best_f1 <= 1.0
    assert 0.1 <= best_t <= 0.9


def test_f1_macro_au_shape():
    rng = np.random.default_rng(3)
    y_true = (rng.random((50, 12)) > 0.5).astype(int)
    y_score = rng.random((50, 12))
    out = f1_macro_au(y_true, y_score, threshold=0.5)
    assert 0.0 <= out <= 1.0


def test_p_mtl_is_sum():
    assert p_mtl(0.5, 0.3, 0.4) == pytest.approx(1.2)
