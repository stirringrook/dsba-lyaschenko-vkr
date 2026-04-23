"""Unit tests for :mod:`src.datasets.interim_prep`.

The tests build a tiny synthetic directory layout of zero-byte files whose
names follow the CREMA-D / RAVDESS conventions, then verify that the
parsers, actor-disjoint split, and annotation writer all produce the
expected output.

Downstream, the written annotation files are parsed with the same
AffWild2 reader used by the Aff-Wild2 code path
(:func:`src.datasets.affwild2_mtl.read_mtl_annotations`) --- this is the
acceptance test: the interim data is indistinguishable from a genuine
AffWild2 annotation file as far as the existing loader is concerned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.affwild2_mtl import read_mtl_annotations
from src.datasets.interim_prep import (
    actor_split,
    parse_crema_d_clips,
    parse_ravdess_clips,
    split_summary,
    write_annotations,
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def synthetic_crema_d(tmp_path: Path) -> Path:
    """Synthetic CREMA-D-like directory with 12 flv files over 3 actors.

    Filenames cover four emotions at two intensities each, so the class
    balance is non-trivial but the tests remain deterministic.
    """
    video_dir = tmp_path / "VideoFlash"
    video_dir.mkdir()
    names = [
        # actor 1001: one of each emotion at XX intensity
        "1001_DFA_ANG_XX.flv", "1001_DFA_DIS_XX.flv",
        "1001_DFA_HAP_XX.flv", "1001_DFA_SAD_XX.flv",
        # actor 1002: mixed intensities
        "1002_DFA_ANG_LO.flv", "1002_DFA_HAP_MD.flv",
        "1002_DFA_NEU_XX.flv", "1002_DFA_FEA_HI.flv",
        # actor 1003: all happy (trivial)
        "1003_DFA_HAP_XX.flv", "1003_DFA_HAP_LO.flv",
        "1003_DFA_HAP_MD.flv", "1003_DFA_HAP_HI.flv",
    ]
    for name in names:
        (video_dir / name).touch()
    # A deliberately malformed file that the parser must skip.
    (video_dir / "bogus.flv").touch()
    return video_dir


@pytest.fixture
def synthetic_ravdess(tmp_path: Path) -> Path:
    """Synthetic RAVDESS-like directory with 8 speech clips over 2 actors."""
    base = tmp_path / "RAVDESS"
    base.mkdir()
    actor_a = base / "Actor_01"
    actor_b = base / "Actor_02"
    actor_a.mkdir()
    actor_b.mkdir()
    # Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
    for stem in [
        "01-01-01-01-01-01-01", "01-01-06-02-01-01-01",
        "01-01-03-01-01-01-01", "01-01-05-02-01-01-01",
    ]:
        (actor_a / f"{stem}.mp4").touch()
    for stem in [
        "01-01-02-01-01-01-02", "01-01-04-02-01-01-02",
        "01-01-07-02-01-01-02", "01-01-08-01-01-01-02",
    ]:
        (actor_b / f"{stem}.mp4").touch()
    # A song clip we must skip by default.
    (actor_a / "01-02-01-01-01-01-01.mp4").touch()
    # Audio-only clip we must skip always.
    (actor_a / "03-01-01-01-01-01-01.mp4").touch()
    return base


# ----------------------------------------------------------------------
# CREMA-D parser
# ----------------------------------------------------------------------

def test_parse_crema_d_skips_malformed(synthetic_crema_d):
    clips = parse_crema_d_clips(synthetic_crema_d)
    assert len(clips) == 12
    # 'bogus.flv' must not appear.
    names = {c.videoname for c in clips}
    assert "bogus" not in names


def test_parse_crema_d_intensity_scale_matches_code(synthetic_crema_d):
    clips = parse_crema_d_clips(synthetic_crema_d)
    by_name = {c.videoname: c for c in clips}
    assert by_name["1002_DFA_ANG_LO"].intensity_scale == pytest.approx(0.25)
    assert by_name["1002_DFA_HAP_MD"].intensity_scale == pytest.approx(0.50)
    assert by_name["1002_DFA_FEA_HI"].intensity_scale == pytest.approx(0.75)
    assert by_name["1001_DFA_ANG_XX"].intensity_scale == pytest.approx(1.00)


# ----------------------------------------------------------------------
# RAVDESS parser
# ----------------------------------------------------------------------

def test_parse_ravdess_excludes_song_and_audio_only_by_default(synthetic_ravdess):
    clips = parse_ravdess_clips(synthetic_ravdess)
    assert len(clips) == 8
    for c in clips:
        assert c.source == "RAVDESS"
        # No song, no audio-only in the parsed list.
        assert not c.videoname.startswith("01-02-")
        assert not c.videoname.startswith("03-")


def test_parse_ravdess_neutral_is_forced_to_normal_intensity(synthetic_ravdess):
    clips = parse_ravdess_clips(synthetic_ravdess)
    by_name = {c.videoname: c for c in clips}
    # Neutral has only intensity=01; the mapping must not break on it.
    assert by_name["01-01-01-01-01-01-01"].emotion == "Neutral"
    assert by_name["01-01-01-01-01-01-01"].intensity_scale == pytest.approx(0.5)


# ----------------------------------------------------------------------
# Actor-disjoint split
# ----------------------------------------------------------------------

def test_actor_split_is_disjoint(synthetic_crema_d):
    clips = parse_crema_d_clips(synthetic_crema_d)
    train, val, test = actor_split(clips, val_frac=0.34, test_frac=0.34, seed=0)
    def actors(xs): return {f"{c.source}/{c.actor_id}" for c in xs}
    assert actors(train).isdisjoint(actors(val))
    assert actors(train).isdisjoint(actors(test))
    assert actors(val).isdisjoint(actors(test))
    assert actors(train) | actors(val) | actors(test) == actors(clips)


def test_split_summary_counts_match_totals(synthetic_crema_d):
    clips = parse_crema_d_clips(synthetic_crema_d)
    train, val, test = actor_split(clips, val_frac=0.34, test_frac=0.34, seed=0)
    summary = split_summary(train, val, test)
    total_by_summary = sum(
        sum(d.values()) for d in summary.values()
    )
    assert total_by_summary == len(clips)


# ----------------------------------------------------------------------
# Annotation writer round-trip
# ----------------------------------------------------------------------

def test_write_annotations_roundtrip_through_affwild2_parser(
    synthetic_crema_d, tmp_path,
):
    clips = parse_crema_d_clips(synthetic_crema_d)
    # Give each clip exactly 3 frames; the writer should emit 3 rows per clip
    # except for 'Neutral', which maps to index 0 but must also get 3 rows.
    out_file = tmp_path / "annotations" / "interim.txt"
    total = write_annotations(
        clips,
        output_file=out_file,
        frame_count_for=lambda clip: 3,
    )
    assert total == 3 * len(clips)

    # And the same file must parse cleanly with the existing AffWild2 reader.
    anno = read_mtl_annotations(
        out_file,
        features_index={
            f"{clip.videoname}/{fi:05d}.jpg"
            for clip in clips
            for fi in range(3)
        },
    )
    assert len(anno) == total
    # Every row had a valid VA and a valid EXPR for the interim pipeline.
    assert (anno.mask_va == 1).all()
    assert (anno.mask_expr == 1).all()
    # AU columns are -1 -> mask always off.
    assert (anno.mask_au == 0).all()


def test_write_annotations_empty_clip_list_writes_header_only(tmp_path):
    out_file = tmp_path / "empty.txt"
    total = write_annotations([], out_file, frame_count_for=lambda c: 3)
    assert total == 0
    text = out_file.read_text(encoding="utf-8")
    # A single header line with no trailing data.
    assert text.count("\n") == 1
    assert text.startswith("image,valence,arousal,expression,AU1,")
