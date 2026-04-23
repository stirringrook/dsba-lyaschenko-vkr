"""Tests for the ffmpeg fallback helpers in :mod:`extract_faces_from_video`.

The heavy path (cv2 + MediaPipe + a real ffmpeg subprocess) is exercised
by the notebook on real data. These tests cover just the pure helpers
whose logic is easy to get wrong.
"""

from __future__ import annotations

import pytest

from src.features.extract_faces_from_video import (
    _pad_bbox_px,
    _parse_rational,
    _sample_indices,
)


def test_parse_rational_handles_common_forms():
    assert _parse_rational("30/1") == 30.0
    assert _parse_rational("30000/1001") == pytest.approx(29.970, rel=1e-4)
    assert _parse_rational("25") == 25.0
    assert _parse_rational("0/0") == 0.0
    assert _parse_rational("") == 0.0
    assert _parse_rational("bogus") == 0.0


def test_sample_indices_respects_target_fps():
    # 30 fps source, 5 fps target -> every 6th frame.
    got = _sample_indices(n_frames=30, source_fps=30.0, target_fps=5.0)
    assert got == [0, 6, 12, 18, 24]


def test_sample_indices_empty_on_zero_frames():
    assert _sample_indices(0, 30.0, 5.0) == []


def test_pad_bbox_px_clamps_to_image():
    # A bbox near the bottom-right corner, padded generously, must still
    # land inside the image and have positive area.
    x0, y0, x1, y1 = _pad_bbox_px(
        x=90, y=90, w=20, h=20, expand=0.5, img_w=100, img_h=100,
    )
    assert 0 <= x0 < x1 <= 100
    assert 0 <= y0 < y1 <= 100


def test_pad_bbox_px_expands_correctly_in_interior():
    # 40x40 bbox at (30,30), 0.5 expand -> grow by 10 px in each direction.
    x0, y0, x1, y1 = _pad_bbox_px(
        x=30, y=30, w=40, h=40, expand=0.5, img_w=200, img_h=200,
    )
    assert (x0, y0, x1, y1) == (20, 20, 80, 80)
