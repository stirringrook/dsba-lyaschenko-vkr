"""CREMA-D and RAVDESS -> AffWild2-shaped annotation writer.

The existing Stage-1 / Stage-3 code path reads a single annotation file
(`training_set_annotations.txt` or its validation counterpart) whose format
is ``image_path, valence, arousal, expression_id, AU1 ... AU12`` with
``image_path = "<videoname>/<frame>.jpg"``. See
``src/datasets/affwild2_mtl.py::read_mtl_annotations`` for the exact parser.

This module parses the CREMA-D and RAVDESS filename conventions, applies
the Russell-circumplex VA mapping from
:mod:`src.datasets.emotion_va_mapping`, and writes annotation files that
the existing parser can consume unchanged. It ALSO produces actor-disjoint
train / val / test splits (70/15/15 on CREMA-D; RAVDESS is used in full as
a cross-dataset held-out set).

The module deliberately leaves out the AU columns by filling them with
``-1`` on every row (= 'AU mask off' in the existing parser). This means
the AU loss contributes nothing for interim runs --- exactly what we want.

Typical use::

    from src.datasets.interim_prep import (
        parse_crema_d_clips, write_annotations, actor_split,
    )

    clips = parse_crema_d_clips("data/crema_d/AudioWAV", "data/crema_d/VideoFlash")
    train, val, test = actor_split(clips, seed=42)
    write_annotations(train, "data/crema_d/annotations/train.txt", frame_count_for=fc)
    write_annotations(val,   "data/crema_d/annotations/val.txt",   frame_count_for=fc)
    write_annotations(test,  "data/crema_d/annotations/test.txt",  frame_count_for=fc)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.datasets.emotion_va_mapping import (
    CREMAD_INTENSITY_SCALE,
    EXPR_NAME_TO_AFFWILD2_IDX,
    RAVDESS_INTENSITY_SCALE,
    crema_d_code_to_emotion,
    emotion_to_va,
    ravdess_code_to_emotion,
)

NUM_AUS_DEFAULT = 12
AU_PADDING = [-1] * NUM_AUS_DEFAULT  # keeps the AU mask always off for interim data.


@dataclass(frozen=True)
class ClipRecord:
    """One parsed audiovisual clip ready for annotation writing.

    Attributes:
        videoname: Used both as the annotation's videoname prefix and as the
            filename stem of the cached visual/audio ``.npz``. Must be unique
            within a dataset.
        actor_id: Used for actor-disjoint splitting.
        emotion: Name from :data:`VA_ANCHOR` (``"Anger"``, ``"Happiness"``, etc.).
        intensity_scale: Multiplier applied to the arousal magnitude.
        source: ``"CREMA-D"`` or ``"RAVDESS"``; useful for audit logs.
        source_video_path: Absolute path to the source video file.
        source_audio_path: Optional absolute path to a pre-extracted WAV.
    """

    videoname: str
    actor_id: str
    emotion: str
    intensity_scale: float
    source: str
    source_video_path: Path
    source_audio_path: Optional[Path] = None


# ----------------------------------------------------------------------
# CREMA-D
# ----------------------------------------------------------------------

def parse_crema_d_clips(
    video_dir: str | Path,
    audio_dir: Optional[str | Path] = None,
    video_ext: str = ".flv",
) -> List[ClipRecord]:
    """Walk ``video_dir`` and produce a :class:`ClipRecord` per clip.

    CREMA-D filenames follow ``<ActorID>_<SentenceCode>_<Emotion>_<Intensity>``,
    e.g. ``1001_DFA_ANG_XX.flv``. ``Emotion`` is one of ``ANG, DIS, FEA,
    HAP, NEU, SAD``; ``Intensity`` is ``LO/MD/HI/XX``. The repository also
    ships an ``.mp4`` variant in some mirrors --- pass ``video_ext=".mp4"``
    for those.

    Args:
        video_dir: Directory containing the CREMA-D video files.
        audio_dir: Optional directory holding the matching ``.wav`` files.
            When provided, the :attr:`ClipRecord.source_audio_path` is set;
            otherwise it is left at ``None`` and the audio pipeline extracts
            WAVs from the video track.
        video_ext: File extension of the videos to include.

    Returns:
        Every successfully parsed clip as a :class:`ClipRecord`.
    """
    video_dir = Path(video_dir)
    audio_dir = Path(audio_dir) if audio_dir is not None else None
    if not video_dir.is_dir():
        raise FileNotFoundError(f"CREMA-D video dir not found: {video_dir}")

    out: List[ClipRecord] = []
    for p in sorted(video_dir.glob(f"*{video_ext}")):
        stem = p.stem  # e.g. '1001_DFA_ANG_XX'
        parts = stem.split("_")
        if len(parts) != 4:
            # CREMA-D sometimes ships auxiliary files; skip anything that
            # does not obey the four-field convention.
            continue
        actor_id, sentence_code, emo_code, intensity_code = parts
        try:
            emotion = crema_d_code_to_emotion(emo_code)
        except KeyError:
            continue
        scale = CREMAD_INTENSITY_SCALE.get(intensity_code, 1.0)
        wav_path = None
        if audio_dir is not None:
            candidate = audio_dir / f"{stem}.wav"
            if candidate.exists():
                wav_path = candidate
        out.append(
            ClipRecord(
                videoname=stem,
                actor_id=actor_id,
                emotion=emotion,
                intensity_scale=scale,
                source="CREMA-D",
                source_video_path=p,
                source_audio_path=wav_path,
            )
        )
    return out


# ----------------------------------------------------------------------
# RAVDESS
# ----------------------------------------------------------------------

def parse_ravdess_clips(
    video_dir: str | Path,
    include_song: bool = False,
    video_ext: str = ".mp4",
) -> List[ClipRecord]:
    """Walk ``video_dir`` and produce a :class:`ClipRecord` per clip.

    RAVDESS filenames follow
    ``<Modality>-<VocalChannel>-<Emotion>-<Intensity>-<Statement>-<Repetition>-<Actor>``.
    Modality: 01=AV, 02=V-only, 03=A-only. VocalChannel: 01=speech, 02=song.
    Emotion: 01-08. Intensity: 01=normal, 02=strong (no 'strong' for
    neutral). See Livingstone & Russo 2018.

    Args:
        video_dir: Directory holding the ``Video_Speech_Actor_*/*.mp4`` (and
            optionally song) files. The function recurses into subfolders.
        include_song: Include song clips (VocalChannel==02) in addition to
            speech (VocalChannel==01). Defaults to ``False`` because the
            thesis targets speech fusion.
        video_ext: File extension of the videos to include.

    Returns:
        Every successfully parsed clip as a :class:`ClipRecord`.
    """
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"RAVDESS video dir not found: {video_dir}")

    out: List[ClipRecord] = []
    for p in sorted(video_dir.rglob(f"*{video_ext}")):
        stem = p.stem  # e.g. '01-01-06-02-01-02-12'
        parts = stem.split("-")
        if len(parts) != 7:
            continue
        try:
            modality = int(parts[0])
            vocal = int(parts[1])
            emotion_code = int(parts[2])
            intensity_code = int(parts[3])
            actor_code = int(parts[6])
        except ValueError:
            continue
        if modality not in (1, 2):
            # 03 = audio-only files have no video frames for the visual branch.
            continue
        if vocal == 2 and not include_song:
            continue
        try:
            emotion = ravdess_code_to_emotion(emotion_code)
        except KeyError:
            continue
        # Neutral has only one intensity level; force it to 'normal'.
        if emotion == "Neutral":
            intensity_code = 1
        scale = RAVDESS_INTENSITY_SCALE.get(intensity_code, 1.0)
        out.append(
            ClipRecord(
                videoname=stem,
                actor_id=f"Actor_{actor_code:02d}",
                emotion=emotion,
                intensity_scale=scale,
                source="RAVDESS",
                source_video_path=p,
                source_audio_path=None,
            )
        )
    return out


# ----------------------------------------------------------------------
# Actor-disjoint splitting
# ----------------------------------------------------------------------

def actor_split(
    clips: Sequence[ClipRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ClipRecord], List[ClipRecord], List[ClipRecord]]:
    """Split ``clips`` into train / val / test by actor.

    No actor appears in more than one split, following the protocol
    recommended by CREMA-D's baseline paper (Cao 2014).

    Args:
        clips: Parsed clip records (mixed-dataset input is allowed; the
            actor_id is globally unique when ``source`` is included).
        val_frac: Fraction of ACTORS (not clips) assigned to validation.
        test_frac: Fraction of ACTORS assigned to test.
        seed: RNG seed for reproducibility.

    Returns:
        ``(train, val, test)`` lists of :class:`ClipRecord`.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    actors = sorted({f"{c.source}/{c.actor_id}" for c in clips})
    rng = random.Random(seed)
    rng.shuffle(actors)
    n = len(actors)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    val_set = set(actors[:n_val])
    test_set = set(actors[n_val : n_val + n_test])
    # rest go to train
    train, val, test = [], [], []
    for c in clips:
        key = f"{c.source}/{c.actor_id}"
        if key in val_set:
            val.append(c)
        elif key in test_set:
            test.append(c)
        else:
            train.append(c)
    return train, val, test


# ----------------------------------------------------------------------
# Annotation file writer
# ----------------------------------------------------------------------

def write_annotations(
    clips: Iterable[ClipRecord],
    output_file: str | Path,
    frame_count_for: Callable[[ClipRecord], int],
    num_aus: int = NUM_AUS_DEFAULT,
) -> int:
    """Write an AffWild2-shaped annotation file for the given clips.

    For each clip, emits ``T`` rows (one per frame) where ``T`` is the
    integer returned by ``frame_count_for(clip)``. Each row shares the
    clip's synthetic VA and expression label; AU columns are filled with
    ``-1`` so that the AU mask is always off, which disables the AU loss
    for interim runs.

    The first row of the file is a header identical to AffWild2's:
    ``image,valence,arousal,expression,AU1,...,AU12``. (The existing
    parser skips the first line unconditionally.)

    Args:
        clips: Clip records to write.
        output_file: Destination .txt file.
        frame_count_for: Callable returning the number of extracted face
            frames for each clip. This is the same ``T`` used when writing
            the visual .npz cache.
        num_aus: Number of AU columns. AffWild2 uses 12.

    Returns:
        Total number of rows written (sum of ``frame_count_for`` over
        clips).
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    au_cols = ",".join(str(x) for x in AU_PADDING[:num_aus])
    header = "image,valence,arousal,expression," + ",".join(
        f"AU{i}" for i in range(1, num_aus + 1)
    )
    with output_file.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + "\n")
        for clip in clips:
            if clip.emotion not in EXPR_NAME_TO_AFFWILD2_IDX:
                continue
            expr_idx = EXPR_NAME_TO_AFFWILD2_IDX[clip.emotion]
            v, a = emotion_to_va(clip.emotion, clip.intensity_scale)
            n_frames = int(frame_count_for(clip))
            if n_frames <= 0:
                continue
            for fi in range(n_frames):
                image_path = f"{clip.videoname}/{fi:05d}.jpg"
                fh.write(
                    f"{image_path},{v:.4f},{a:.4f},{expr_idx},{au_cols}\n"
                )
                total_rows += 1
    return total_rows


def split_summary(
    train: Sequence[ClipRecord],
    val: Sequence[ClipRecord],
    test: Sequence[ClipRecord],
) -> Dict[str, Dict[str, int]]:
    """Class-balance summary for a three-way split.

    Returns:
        A nested dictionary ``{split -> {emotion -> clip_count}}`` useful
        for printing in a notebook or for logging.
    """
    out: Dict[str, Dict[str, int]] = {}
    for name, clips in (("train", train), ("val", val), ("test", test)):
        counts: Dict[str, int] = {}
        for c in clips:
            counts[c.emotion] = counts.get(c.emotion, 0) + 1
        out[name] = dict(sorted(counts.items()))
    return out
