"""Unit tests for :mod:`src.features.align_audio_to_video` (no torchaudio / ffmpeg)."""

import numpy as np
import pytest

from src.features.align_audio_to_video import align_audio_to_video


def test_linear_interp_matches_numpy_ground_truth():
    T_audio, D = 50, 4
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((T_audio, D)).astype(np.float32)
    T_video = 120

    aligned, has_audio = align_audio_to_video(audio, T_video)
    assert aligned.shape == (T_video, D)
    assert has_audio

    # Endpoints must match the audio endpoints exactly (linear interp at 0 and 1).
    np.testing.assert_allclose(aligned[0], audio[0], rtol=1e-6)
    np.testing.assert_allclose(aligned[-1], audio[-1], rtol=1e-6)

    # Check one interior point against a manual linear-interp computation.
    mid_new = 60
    x_new = mid_new / (T_video - 1)
    idx_in = x_new * (T_audio - 1)
    lo = int(np.floor(idx_in))
    frac = idx_in - lo
    expected = (1 - frac) * audio[lo] + frac * audio[lo + 1]
    np.testing.assert_allclose(aligned[mid_new], expected, rtol=1e-5)


def test_empty_audio_returns_zeros_and_false():
    aligned, has_audio = align_audio_to_video(
        np.zeros((0, 0), dtype=np.float32), num_video_frames=30, fallback_dim=768
    )
    assert aligned.shape == (30, 768)
    assert has_audio is False
    assert not aligned.any()


def test_empty_without_fallback_raises():
    with pytest.raises(ValueError):
        align_audio_to_video(np.zeros((0, 0)), num_video_frames=30)


def test_single_audio_frame_broadcasts():
    audio = np.arange(10, dtype=np.float32)[None, :]  # (1, 10)
    aligned, has_audio = align_audio_to_video(audio, num_video_frames=7)
    assert aligned.shape == (7, 10)
    assert has_audio
    for i in range(7):
        np.testing.assert_allclose(aligned[i], audio[0])


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        align_audio_to_video(np.zeros(10), num_video_frames=5)


def test_downsample_preserves_endpoints():
    T_audio, D = 200, 3
    audio = np.linspace(0, 1, T_audio)[:, None] * np.arange(1, D + 1)
    aligned, _ = align_audio_to_video(audio, num_video_frames=20)
    assert aligned.shape == (20, D)
    np.testing.assert_allclose(aligned[0], audio[0])
    np.testing.assert_allclose(aligned[-1], audio[-1])
