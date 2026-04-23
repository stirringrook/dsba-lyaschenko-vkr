"""Unit tests for src/smoothing.py (no dataset required)."""

import numpy as np
import pytest

from src.smoothing import gaussian_smooth_per_video


def _vnf(videos_and_counts):
    """Helper to build a ``videoname_frames`` list in annotation order."""
    out = []
    for name, n in videos_and_counts:
        for i in range(n):
            out.append((name, i))
    return out


def test_delta_zero_is_identity():
    scores = np.arange(20, dtype=np.float64).reshape(10, 2)
    vnf = _vnf([("v1", 10)])
    out = gaussian_smooth_per_video(scores, vnf, sigma=1.0, delta=0)
    np.testing.assert_allclose(out, scores)


def test_does_not_mix_videos():
    """Row ``i`` must not draw weight from rows in a different video."""
    scores = np.array([[10.0], [10.0], [-10.0], [-10.0]])
    vnf = [("v1", 0), ("v1", 1), ("v2", 0), ("v2", 1)]
    out = gaussian_smooth_per_video(scores, vnf, sigma=1e9, delta=50)
    # Both videos are internally constant, so smoothing should be a no-op.
    np.testing.assert_allclose(out, scores)


def test_large_sigma_approaches_box_filter():
    """With huge sigma the Gaussian weights collapse to 1 -> plain box mean."""
    scores = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    vnf = _vnf([("v1", 5)])
    out = gaussian_smooth_per_video(scores, vnf, sigma=1e9, delta=1)
    # Row 0: mean(1,2) = 1.5; row 1: mean(1,2,3) = 2; ...; row 4: mean(4,5) = 4.5.
    expected = np.array([[1.5], [2.0], [3.0], [4.0], [4.5]])
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_small_sigma_peaks_at_center():
    """With sigma -> 0 only row i itself contributes (weight=1, neighbours ~0)."""
    scores = np.array([[0.0], [1.0], [100.0], [1.0], [0.0]])
    vnf = _vnf([("v1", 5)])
    out = gaussian_smooth_per_video(scores, vnf, sigma=0.1, delta=2)
    np.testing.assert_allclose(out, scores, rtol=1e-6, atol=1e-6)


def test_shape_mismatch_raises():
    scores = np.zeros((5, 3))
    vnf = [("v1", 0), ("v1", 1)]
    with pytest.raises(ValueError):
        gaussian_smooth_per_video(scores, vnf, sigma=1.0, delta=1)


def test_multiclass_smoothing_preserves_shape():
    rng = np.random.default_rng(0)
    scores = rng.random((30, 8))
    vnf = _vnf([("a", 10), ("b", 20)])
    out = gaussian_smooth_per_video(scores, vnf, sigma=10.0, delta=5)
    assert out.shape == scores.shape
    assert np.all(out >= 0)
