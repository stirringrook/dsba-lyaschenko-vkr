"""Unit tests for the VA-mapping half of :mod:`src.datasets.label_mapping`."""

from __future__ import annotations

import pytest

from src.datasets.label_mapping import (
    CREMAD_INTENSITY_SCALE,
    EXPR_NAME_TO_AFFWILD2_IDX,
    RAVDESS_INTENSITY_SCALE,
    VA_ANCHOR,
    crema_d_code_to_emotion,
    emotion_to_va,
    ravdess_code_to_emotion,
)


def test_anchors_inside_unit_square():
    for emo, (v, a) in VA_ANCHOR.items():
        assert -1.0 <= v <= 1.0, f"{emo} valence out of range"
        assert -1.0 <= a <= 1.0, f"{emo} arousal out of range"


def test_emotion_to_va_scales_arousal_magnitude_only():
    # Happiness anchor is (+0.8, +0.5). At half intensity, arousal -> 0.25.
    v, a = emotion_to_va("Happiness", intensity_scale=0.5)
    assert v == pytest.approx(0.8, abs=1e-6)
    assert a == pytest.approx(0.25, abs=1e-6)


def test_emotion_to_va_clips_to_unit_interval():
    # An intensity scale of 10x on Fear (anchor arousal +0.8) must clip to +1.
    v, a = emotion_to_va("Fear", intensity_scale=10.0)
    assert a == 1.0
    # Valence is never scaled, so it stays at the anchor.
    assert v == pytest.approx(-0.6, abs=1e-6)


def test_neutral_anchor_is_zero():
    v, a = emotion_to_va("Neutral", intensity_scale=1.0)
    assert v == 0.0 and a == 0.0


def test_emotion_to_va_raises_on_unknown():
    with pytest.raises(KeyError):
        emotion_to_va("ExtremelyUnlikely", intensity_scale=1.0)


def test_crema_d_code_round_trip():
    # Every code must map to an emotion that has a VA anchor.
    codes = ["NEU", "ANG", "DIS", "FEA", "HAP", "SAD"]
    for c in codes:
        emo = crema_d_code_to_emotion(c)
        assert emo in VA_ANCHOR
        assert emo in EXPR_NAME_TO_AFFWILD2_IDX


def test_ravdess_code_round_trip():
    for c in range(1, 9):
        emo = ravdess_code_to_emotion(c)
        assert emo in VA_ANCHOR
        assert emo in EXPR_NAME_TO_AFFWILD2_IDX


def test_crema_d_intensity_scale_is_monotonic():
    # LO < MD < HI < XX in the magnitude multiplier.
    prev = -1.0
    for code in ("LO", "MD", "HI", "XX"):
        assert CREMAD_INTENSITY_SCALE[code] > prev
        prev = CREMAD_INTENSITY_SCALE[code]


def test_ravdess_intensity_scale_present_for_both_levels():
    assert 1 in RAVDESS_INTENSITY_SCALE
    assert 2 in RAVDESS_INTENSITY_SCALE
    assert RAVDESS_INTENSITY_SCALE[2] > RAVDESS_INTENSITY_SCALE[1]
