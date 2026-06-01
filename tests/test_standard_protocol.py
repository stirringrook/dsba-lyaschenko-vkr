"""Tests for the standard-protocol re-evaluator.

Covers the parts that don't need a trained checkpoint:
1. Native-label recovery from CREMA-D and RAVDESS filenames.
2. Restricted argmax over the dataset's native subset.
3. Per-clip mean-softmax aggregation.
4. The strict ravdess8 reading (Calm collapses to Neutral; Calm clips
   are counted as misses).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.label_mapping import (
    AFFWILD2_IDX_FOR_CREMAD,
    AFFWILD2_IDX_FOR_RAVDESS_8,
    LABEL_SPACE_CREMAD6,
    LABEL_SPACE_RAVDESS7,
    LABEL_SPACE_RAVDESS8,
    NATIVE_CREMAD,
    NATIVE_RAVDESS_7,
    NATIVE_RAVDESS_8,
    aggregate_softmax_per_clip,
    detect_dataset_from_videoname,
    native_emotion_from_cremad_videoname,
    native_emotion_from_ravdess_videoname,
    native_labels_for_clips,
    restricted_argmax,
)


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

class TestNativeLabelRecovery:
    def test_cremad_basic(self):
        assert native_emotion_from_cremad_videoname("1001_DFA_ANG_XX") == "ANG"
        assert native_emotion_from_cremad_videoname("1091_TIE_HAP_LO") == "HAP"
        assert native_emotion_from_cremad_videoname("1014_IEO_NEU_XX") == "NEU"

    def test_cremad_rejects_bad_format(self):
        with pytest.raises(ValueError):
            native_emotion_from_cremad_videoname("not-a-cremad-name")
        with pytest.raises(ValueError):
            native_emotion_from_cremad_videoname("1001_DFA_XYZ_XX")

    def test_ravdess_basic(self):
        # 03 = happiness, 06 = fear, 02 = calm, 01 = neutral, 08 = surprise.
        assert native_emotion_from_ravdess_videoname("01-01-03-02-01-02-12") == "Happy"
        assert native_emotion_from_ravdess_videoname("01-01-06-01-01-01-04") == "Fearful"
        assert native_emotion_from_ravdess_videoname("01-01-02-01-01-01-04") == "Calm"
        assert native_emotion_from_ravdess_videoname("01-01-01-01-01-01-04") == "Neutral"
        assert native_emotion_from_ravdess_videoname("01-01-08-02-01-01-04") == "Surprised"

    def test_ravdess_rejects_bad_format(self):
        with pytest.raises(ValueError):
            native_emotion_from_ravdess_videoname("1001_DFA_ANG_XX")
        with pytest.raises(ValueError):
            native_emotion_from_ravdess_videoname("01-01-99-01-01-01-04")

    def test_dataset_autodetect(self):
        assert detect_dataset_from_videoname("1001_DFA_ANG_XX") == "cremad"
        assert detect_dataset_from_videoname("01-01-03-02-01-02-12") == "ravdess"
        with pytest.raises(ValueError):
            detect_dataset_from_videoname("ambiguous_filename")


# ---------------------------------------------------------------------------
# Restricted argmax
# ---------------------------------------------------------------------------

class TestRestrictedArgmax:
    def _one_hot_8(self, idx: int) -> np.ndarray:
        v = np.zeros(8, dtype=np.float64)
        v[idx] = 1.0
        return v

    def test_cremad_bijective(self):
        # AffWild2 idx 0..5 are valid; 6 and 7 (Surprise, Other) are not.
        for native_code in NATIVE_CREMAD:
            idx = AFFWILD2_IDX_FOR_CREMAD[native_code]
            probs = self._one_hot_8(idx)[None, :]
            picked = restricted_argmax(probs, LABEL_SPACE_CREMAD6)
            assert NATIVE_CREMAD[int(picked[0])] == native_code

    def test_cremad_ignores_invalid_classes(self):
        # Surprise (idx 6) is not in CREMA-D's native set; the model must
        # fall back to the second-best class within the native subset.
        probs = np.zeros((1, 8), dtype=np.float64)
        probs[0, 6] = 0.7    # Surprise
        probs[0, 4] = 0.3    # Happiness
        picked = restricted_argmax(probs, LABEL_SPACE_CREMAD6)
        assert NATIVE_CREMAD[int(picked[0])] == "HAP"

    def test_ravdess7_bijective(self):
        for native_code in NATIVE_RAVDESS_7:
            from src.datasets.label_mapping import AFFWILD2_IDX_FOR_RAVDESS_7
            idx = AFFWILD2_IDX_FOR_RAVDESS_7[native_code]
            probs = self._one_hot_8(idx)[None, :]
            picked = restricted_argmax(probs, LABEL_SPACE_RAVDESS7)
            assert NATIVE_RAVDESS_7[int(picked[0])] == native_code

    def test_ravdess8_strict_calm_decoded_as_neutral(self):
        # The strict ravdess8 reading decodes any class-0 model output as
        # 'Neutral'; Calm is unreachable.
        probs = self._one_hot_8(0)[None, :]
        picked = restricted_argmax(probs, LABEL_SPACE_RAVDESS8)
        assert NATIVE_RAVDESS_8[int(picked[0])] == "Neutral"

    def test_ravdess8_other_classes_round_trip(self):
        # All non-collapsed classes must round-trip cleanly.
        for native_code in NATIVE_RAVDESS_8:
            if native_code in ("Calm", "Neutral"):
                continue
            idx = AFFWILD2_IDX_FOR_RAVDESS_8[native_code]
            probs = self._one_hot_8(idx)[None, :]
            picked = restricted_argmax(probs, LABEL_SPACE_RAVDESS8)
            assert NATIVE_RAVDESS_8[int(picked[0])] == native_code


# ---------------------------------------------------------------------------
# Clip aggregation
# ---------------------------------------------------------------------------

class TestClipAggregation:
    def test_three_clips_two_frames_each(self):
        # 6 frames over 3 clips, all logits identity-like.
        videonames = ["c1", "c1", "c2", "c2", "c3", "c3"]
        logits = np.zeros((6, 8))
        # c1: both frames vote Anger (idx 1)
        logits[0, 1] = 5.0; logits[1, 1] = 5.0
        # c2: both frames vote Disgust (idx 2)
        logits[2, 2] = 5.0; logits[3, 2] = 5.0
        # c3: split vote (idx 4 then idx 5) — averaged softmax should be ~equal.
        logits[4, 4] = 5.0
        logits[5, 5] = 5.0

        probs, order = aggregate_softmax_per_clip(logits, videonames)
        assert order == ("c1", "c2", "c3")
        assert probs.shape == (3, 8)
        assert int(np.argmax(probs[0])) == 1
        assert int(np.argmax(probs[1])) == 2
        # c3's argmax is one of {4, 5} since they're tied; both are acceptable.
        assert int(np.argmax(probs[2])) in (4, 5)
        # Probabilities sum to 1 per clip.
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-9)

    def test_rejects_misaligned_videonames(self):
        with pytest.raises(ValueError):
            aggregate_softmax_per_clip(np.zeros((3, 8)), ["a", "b"])

    def test_rejects_wrong_logit_shape(self):
        with pytest.raises(ValueError):
            aggregate_softmax_per_clip(np.zeros((2, 7)), ["a", "b"])


class TestNativeLabelsForClips:
    def test_cremad_round_trip(self):
        clips = ("1001_DFA_ANG_XX", "1014_IEO_NEU_XX", "1091_TIE_HAP_LO")
        idxs = native_labels_for_clips(clips, LABEL_SPACE_CREMAD6)
        decoded = [NATIVE_CREMAD[i] for i in idxs]
        assert decoded == ["ANG", "NEU", "HAP"]

    def test_ravdess7_merges_calm(self):
        clips = (
            "01-01-01-01-01-01-04",  # Neutral
            "01-01-02-01-01-01-04",  # Calm
            "01-01-03-02-01-02-12",  # Happy
        )
        idxs = native_labels_for_clips(clips, LABEL_SPACE_RAVDESS7)
        decoded = [NATIVE_RAVDESS_7[i] for i in idxs]
        assert decoded == ["Neutral_or_Calm", "Neutral_or_Calm", "Happy"]

    def test_ravdess8_keeps_calm_separate(self):
        clips = (
            "01-01-01-01-01-01-04",  # Neutral
            "01-01-02-01-01-01-04",  # Calm
        )
        idxs = native_labels_for_clips(clips, LABEL_SPACE_RAVDESS8)
        decoded = [NATIVE_RAVDESS_8[i] for i in idxs]
        assert decoded == ["Neutral", "Calm"]
