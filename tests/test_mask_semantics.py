"""Unit tests for src/datasets/affwild2_mtl.py.

Uses a synthetic 4-row annotation file covering the four canonical
missing-value patterns described in cell 32 of the notebook:
    1. fully labelled frame
    2. VA-only frame
    3. EXPR-only frame
    4. AU-only frame
Plus one "nothing labelled" row that must be dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.affwild2_mtl import (
    compute_au_class_weights,
    compute_expr_class_weights,
    read_mtl_annotations,
)


def _write_annotation_file(tmp_path: Path) -> Path:
    au_zero = ",".join(["0"] * 12)
    au_minus = ",".join(["-1"] * 12)
    au_mixed_1 = ",".join(["1"] + ["0"] * 11)
    lines = [
        "image,valence,arousal,expression," + ",".join(f"AU{i}" for i in range(1, 13)),
        # fully-labelled
        f"vid1/0001.jpg,0.5,0.2,3,{au_mixed_1}",
        # VA-only (expr = -1, AU negative)
        f"vid1/0002.jpg,-0.1,0.4,-1,{au_minus}",
        # EXPR-only (VA = -5, AU negative)
        f"vid2/0100.jpg,-5,-5,2,{au_minus}",
        # AU-only (VA = -5, EXPR = -1)
        f"vid2/0101.jpg,-5,-5,-1,{au_mixed_1}",
        # nothing labelled - must be dropped
        f"vid3/0001.jpg,-5,-5,-1,{au_minus}",
    ]
    path = tmp_path / "annotations.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_four_pattern_parse(tmp_path: Path):
    path = _write_annotation_file(tmp_path)
    anno = read_mtl_annotations(path)

    # The "nothing labelled" row is dropped.
    assert len(anno) == 4
    assert anno.num_missed == 0

    df = anno.df.set_index("image_path")
    # Row 1: all masks on.
    r1 = df.loc["vid1/0001.jpg"]
    assert (r1["mask_va"], r1["mask_expr"], r1["mask_au"]) == (1.0, 1.0, 1.0)
    assert r1["valence"] == pytest.approx(0.5)
    assert r1["expr"] == 3
    assert r1["au1"] == 1

    # Row 2: VA only; EXPR and AU masks off, values zeroed.
    r2 = df.loc["vid1/0002.jpg"]
    assert (r2["mask_va"], r2["mask_expr"], r2["mask_au"]) == (1.0, 0.0, 0.0)
    assert r2["expr"] == 0
    assert r2["au1"] == 0

    # Row 3: EXPR only.
    r3 = df.loc["vid2/0100.jpg"]
    assert (r3["mask_va"], r3["mask_expr"], r3["mask_au"]) == (0.0, 1.0, 0.0)
    assert r3["valence"] == 0.0 and r3["arousal"] == 0.0

    # Row 4: AU only.
    r4 = df.loc["vid2/0101.jpg"]
    assert (r4["mask_va"], r4["mask_expr"], r4["mask_au"]) == (0.0, 0.0, 1.0)
    assert r4["expr"] == 0


def test_features_index_filter_counts_missed(tmp_path: Path):
    path = _write_annotation_file(tmp_path)
    # Only row 1 has a cached feature.
    anno = read_mtl_annotations(path, features_index={"vid1/0001.jpg"})
    assert len(anno) == 1
    # Rows 2, 3, 4 were valid but not cached -> counted as missed.
    assert anno.num_missed == 3


def test_videoname_frames_alignment(tmp_path: Path):
    path = _write_annotation_file(tmp_path)
    anno = read_mtl_annotations(path)
    assert anno.videoname_frames == [
        ("vid1", 1),
        ("vid1", 2),
        ("vid2", 100),
        ("vid2", 101),
    ]


def test_expr_class_weights_min_is_one(tmp_path: Path):
    # 10 frames: class 0 appears 6x, class 1 appears 4x.
    y_expr = np.array([0] * 6 + [1] * 4 + [3])
    mask = np.concatenate([np.ones(10), np.zeros(1)])  # last row masked out
    cw = compute_expr_class_weights(y_expr, mask, num_classes=8)
    # Minority class gets the higher weight; min weight normalised to 1.
    assert min(cw.values()) == pytest.approx(1.0)
    assert cw[1] > cw[0]


def test_au_class_weights_shape_and_balance():
    rng = np.random.default_rng(0)
    y_aus = (rng.random((200, 12)) > 0.7).astype(int)
    mask = np.ones(200, dtype=np.float32)
    w = compute_au_class_weights(y_aus, mask)
    assert w.shape == (12, 2)
    # The rarer class in each column should get a weight > 1.
    for i in range(12):
        neg = (y_aus[:, i] == 0).sum()
        pos = (y_aus[:, i] == 1).sum()
        rarer_idx = 1 if pos < neg else 0
        assert w[i, rarer_idx] >= 1.0
