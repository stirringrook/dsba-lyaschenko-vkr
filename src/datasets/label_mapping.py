"""Label mapping for the CREMA-D / RAVDESS interim pipeline.

Two complementary halves of the same concern, kept in one module:

1. **Categorical -> synthetic (V, A)** (the *write* side). CREMA-D and
   RAVDESS ship clip-level categorical labels only; to exercise the VA
   head of the three-head MTL architecture we assign each basic emotion a
   fixed Russell-circumplex anchor and scale arousal by clip intensity.
   Used by :mod:`src.datasets.interim_prep` when building annotation files.

2. **Native-label recovery** (the *read* side). Standard-protocol CREMA-D /
   RAVDESS evaluation needs the native class back from the clip filename,
   without the AffWild2 8-class collapse the training pipeline applies.
   Used by :mod:`src.eval_standard_protocol`.

NOTE on the two RAVDESS tables: :func:`ravdess_code_to_emotion` (write
side) maps codes onto the **AffWild2 EXPR vocabulary** ("Happiness",
"Sadness", "Anger", "Fear", "Surprise") so the VA anchors line up, while
:data:`NATIVE_RAVDESS_8` / :func:`native_emotion_from_ravdess_videoname`
(read side) use the **RAVDESS codebook vocabulary** ("Happy", "Sad",
"Angry", "Fearful", "Surprised"). These are deliberately different
spellings of the same eight classes -- do NOT "deduplicate" them.

The VA mapping is a PROXY --- not a replacement for Aff-Wild2's
human-annotated VA --- so callers should treat the resulting CCC numbers
as indicative rather than directly comparable to the Aff-Wild2 leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


# ===========================================================================
# 1. Categorical emotion -> synthetic (V, A) on the Russell circumplex
# ===========================================================================

# Full eight-class AffWild2 EXPR label set (cell 32 of mtl.ipynb).
AFFWILD2_EXPR_LABELS = (
    "Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness",
    "Surprise", "Other",
)

# Russell-circumplex anchors for every expression used by CREMA-D and
# RAVDESS. Values are on the AffWild2 VA scale, ``[-1, 1]``.
VA_ANCHOR: Dict[str, Tuple[float, float]] = {
    "Neutral":   (0.00,  0.00),
    "Anger":     (-0.70, +0.70),
    "Disgust":   (-0.70, +0.30),
    "Fear":      (-0.60, +0.80),
    "Happiness": (+0.80, +0.50),
    "Sadness":   (-0.60, -0.40),
    "Surprise":  (+0.20, +0.80),
    "Calm":      (+0.20, -0.40),  # RAVDESS-only label
    "Other":     (0.00,  0.00),
}

# CREMA-D intensity codes (LO/MD/HI/XX) -> arousal magnitude multiplier.
# XX means 'unspecified' and is treated as full intensity, matching the
# convention used by the dataset's own baseline paper (Cao 2014).
CREMAD_INTENSITY_SCALE: Dict[str, float] = {
    "LO": 0.25,
    "MD": 0.50,
    "HI": 0.75,
    "XX": 1.00,
}

# RAVDESS intensity codes (filename field 4) -> magnitude multiplier.
RAVDESS_INTENSITY_SCALE: Dict[int, float] = {
    1: 0.50,  # "normal"
    2: 1.00,  # "strong"
}

# Canonical AffWild2 class-index mapping. ``-1`` means ``drop``: CREMA-D and
# RAVDESS do not include the ``Other`` class, and CREMA-D additionally has
# no ``Surprise``; we never emit those indices for unsupported sources.
EXPR_NAME_TO_AFFWILD2_IDX: Dict[str, int] = {
    "Neutral":   0,
    "Anger":     1,
    "Disgust":   2,
    "Fear":      3,
    "Happiness": 4,
    "Sadness":   5,
    "Surprise":  6,
    "Other":     7,
    "Calm":      0,  # RAVDESS 'calm' collapses to Neutral at AffWild2 scale.
}


def emotion_to_va(emotion: str, intensity_scale: float = 1.0) -> Tuple[float, float]:
    """Return the synthetic (valence, arousal) for ``emotion`` at a scale.

    Args:
        emotion: One of the keys of :data:`VA_ANCHOR`. Case-sensitive; pass
            names exactly (``"Happiness"``, not ``"happy"``).
        intensity_scale: Multiplier on the anchor's arousal magnitude. Valence
            is left unscaled because Russell's 2-D layout does not move a
            label's valence direction when its arousal intensifies.

    Returns:
        ``(valence, arousal)`` both in ``[-1, 1]``. Values are clipped.
    """
    if emotion not in VA_ANCHOR:
        raise KeyError(
            f"Unknown emotion '{emotion}'. Known: {sorted(VA_ANCHOR)}"
        )
    v, a = VA_ANCHOR[emotion]
    a = max(-1.0, min(1.0, a * float(intensity_scale)))
    return float(v), float(a)


def crema_d_code_to_emotion(code: str) -> str:
    """CREMA-D uses three-letter emotion codes in filenames.

    Filename example: ``1001_DFA_ANG_XX.flv`` -> emotion code ``"ANG"``.
    """
    mapping = {
        "NEU": "Neutral",
        "ANG": "Anger",
        "DIS": "Disgust",
        "FEA": "Fear",
        "HAP": "Happiness",
        "SAD": "Sadness",
    }
    if code not in mapping:
        raise KeyError(f"Unknown CREMA-D emotion code '{code}'.")
    return mapping[code]


def ravdess_code_to_emotion(code: int) -> str:
    """RAVDESS uses a 2-digit emotion code in field 3 of its filenames.

    Filename example: ``03-01-06-02-01-02-12.mp4``; the third field (``06``)
    is the emotion code. Returns names on the **AffWild2 EXPR vocabulary**
    (see the module note); the standard-protocol read side uses a separate
    RAVDESS-codebook table.
    """
    mapping = {
        1: "Neutral",
        2: "Calm",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Fear",
        7: "Disgust",
        8: "Surprise",
    }
    if code not in mapping:
        raise KeyError(f"Unknown RAVDESS emotion code {code}.")
    return mapping[code]


# ===========================================================================
# 2. Native CREMA-D / RAVDESS label recovery (standard-protocol read side)
# ===========================================================================
#
# The interim pipeline maps every dataset's categorical emotion onto the
# 8-class Aff-Wild2 EXPR taxonomy above. That mapping is convenient for
# training but it discards information that *standard-protocol* evaluation
# needs to put back:
#
# * CREMA-D's six native classes (NEU, ANG, DIS, FEA, HAP, SAD) are a proper
#   subset of the eight Aff-Wild2 classes, so the mapping is bijective.
# * RAVDESS's eight native classes include ``Calm`` (code ``02``), which
#   collapses to ``Neutral`` (idx 0) under training. Fair comparison with the
#   published RAVDESS literature has to either (i) merge Calm with Neutral in
#   the ground truth (``ravdess7``) or (ii) accept that all Calm clips count
#   as misses (``ravdess8``).
#
# The helpers below recover the native label from the clip name only; they
# never look at the AffWild2-mapped annotation file.

# CREMA-D native order (matches Cao 2014 baseline tables).
NATIVE_CREMAD: Tuple[str, ...] = ("NEU", "ANG", "DIS", "FEA", "HAP", "SAD")

# RAVDESS native order (matches Livingstone & Russo 2018 codebook).
NATIVE_RAVDESS_8: Tuple[str, ...] = (
    "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised",
)

# RAVDESS 7-class order (Calm merged into Neutral).
NATIVE_RAVDESS_7: Tuple[str, ...] = (
    "Neutral_or_Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised",
)

# AffWild2 8-class index per native CREMA-D class. Bijective.
AFFWILD2_IDX_FOR_CREMAD: Dict[str, int] = {
    "NEU": 0, "ANG": 1, "DIS": 2, "FEA": 3, "HAP": 4, "SAD": 5,
}

# AffWild2 idx per native RAVDESS class (8-class). NOT bijective: Calm and
# Neutral both map to idx 0.
AFFWILD2_IDX_FOR_RAVDESS_8: Dict[str, int] = {
    "Neutral":   0,
    "Calm":      0,  # collapsed under the training pipeline.
    "Happy":     4,
    "Sad":       5,
    "Angry":     1,
    "Fearful":   3,
    "Disgust":   2,
    "Surprised": 6,
}

# AffWild2 idx per RAVDESS 7-class label (Calm merged into Neutral). Bijective.
AFFWILD2_IDX_FOR_RAVDESS_7: Dict[str, int] = {
    "Neutral_or_Calm": 0,
    "Happy":           4,
    "Sad":             5,
    "Angry":           1,
    "Fearful":         3,
    "Disgust":         2,
    "Surprised":       6,
}


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

def native_emotion_from_cremad_videoname(videoname: str) -> str:
    """Recover the native CREMA-D emotion code from a clip filename stem.

    Expected stem format: ``<ActorID>_<SentenceCode>_<EmotionCode>_<Intensity>``
    (e.g. ``1001_DFA_ANG_XX``).

    Args:
        videoname: The .npz / annotation stem.

    Returns:
        One of :data:`NATIVE_CREMAD`.

    Raises:
        ValueError: If the stem does not match the four-field convention or
        the emotion code is not one of CREMA-D's six.
    """
    parts = videoname.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"CREMA-D videoname '{videoname}' is not 4-field "
            "<ActorID>_<SentCode>_<EmoCode>_<Intensity>."
        )
    code = parts[2]
    if code not in NATIVE_CREMAD:
        raise ValueError(
            f"CREMA-D emotion code '{code}' (from '{videoname}') is not in "
            f"{NATIVE_CREMAD}."
        )
    return code


def native_emotion_from_ravdess_videoname(videoname: str) -> str:
    """Recover the native RAVDESS emotion class from a clip filename stem.

    Expected stem format: 7 dash-separated integers
    ``<Mod>-<Vocal>-<Emo>-<Inten>-<Stmt>-<Rep>-<Actor>``.
    The third field is the emotion code 01..08.

    Returns:
        One of :data:`NATIVE_RAVDESS_8`.
    """
    parts = videoname.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"RAVDESS videoname '{videoname}' is not 7-field "
            "<Mod>-<Voc>-<Emo>-<Int>-<Stmt>-<Rep>-<Actor>."
        )
    try:
        code = int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"RAVDESS videoname '{videoname}' has non-integer emotion field "
            f"'{parts[2]}'."
        ) from exc
    table = {
        1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
        5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised",
    }
    if code not in table:
        raise ValueError(
            f"RAVDESS emotion code {code} (from '{videoname}') is not in 1..8."
        )
    return table[code]


def detect_dataset_from_videoname(videoname: str) -> str:
    """Heuristic dataset detector based on the filename convention.

    Returns ``'cremad'`` for the underscore-4-field CREMA-D names and
    ``'ravdess'`` for the dash-7-field RAVDESS names. Raises
    :class:`ValueError` otherwise.
    """
    if videoname.count("_") == 3 and videoname.count("-") == 0:
        return "cremad"
    if videoname.count("-") == 6 and videoname.count("_") == 0:
        return "ravdess"
    raise ValueError(
        f"Cannot detect dataset from videoname '{videoname}'. Expected "
        "CREMA-D '<A>_<S>_<E>_<I>' or RAVDESS '<M>-<V>-<E>-<I>-<S>-<R>-<A>'."
    )


# ---------------------------------------------------------------------------
# Clip-level argmax under a restricted label space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelSpace:
    """The native label space + the AffWild2 index each label maps to."""

    name: str                                  # 'cremad6', 'ravdess7', 'ravdess8'
    classes: Tuple[str, ...]                   # native class names in canonical order
    affwild2_idx: Tuple[int, ...]              # parallel AffWild2 idx per class


LABEL_SPACE_CREMAD6 = LabelSpace(
    name="cremad6",
    classes=NATIVE_CREMAD,
    affwild2_idx=tuple(AFFWILD2_IDX_FOR_CREMAD[c] for c in NATIVE_CREMAD),
)

LABEL_SPACE_RAVDESS7 = LabelSpace(
    name="ravdess7",
    classes=NATIVE_RAVDESS_7,
    affwild2_idx=tuple(AFFWILD2_IDX_FOR_RAVDESS_7[c] for c in NATIVE_RAVDESS_7),
)

LABEL_SPACE_RAVDESS8 = LabelSpace(
    name="ravdess8",
    classes=NATIVE_RAVDESS_8,
    affwild2_idx=tuple(AFFWILD2_IDX_FOR_RAVDESS_8[c] for c in NATIVE_RAVDESS_8),
)


def restricted_argmax(probs: np.ndarray, space: LabelSpace) -> np.ndarray:
    """Argmax over the native classes only.

    For CREMA-D and RAVDESS-7 the AffWild2 indices in ``space`` are unique,
    so we directly slice the 8-D softmax to the native subset and argmax.

    For RAVDESS-8 the mapping is many-to-one (``Calm`` and ``Neutral`` both
    point at idx 0). Two design choices appear in the literature:
    we follow the strict reading where any class-0 model output is decoded
    as ``Neutral`` --- this is the worst case for the model and removes any
    ambiguity about how Calm clips score. The corollary is that the model
    can never get a Calm clip right under ``ravdess8``; readers must
    consult ``ravdess7`` for the apples-to-apples number.

    Args:
        probs: ``(N, 8)`` array of softmax probabilities over the AffWild2
            taxonomy. ``N`` is typically the number of clips after
            aggregation, but the function works at any rank.
        space: The native label space to project onto.

    Returns:
        ``(N,)`` array of native-class integer indices into ``space.classes``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[-1] != 8:
        raise ValueError(
            f"Expected 8-D AffWild2 softmax along the last axis, got "
            f"{probs.shape}."
        )

    if space.name == "ravdess8":
        # Strict ravdess8 reading: pick the native class whose AffWild2 idx
        # has the highest probability AND has a UNIQUE inverse. For idx 0
        # we always decode 'Neutral' rather than 'Calm'.
        idx_to_class = {}
        for ci, idx in enumerate(space.affwild2_idx):
            idx_to_class.setdefault(idx, ci)  # first wins; in canonical order Neutral is first.
        # Allowed AffWild2 indices (deduplicated, ordered).
        allowed = sorted(set(space.affwild2_idx))
        sub = probs[..., allowed]
        sub_argmax = np.argmax(sub, axis=-1)
        winning_affwild_idx = np.array([allowed[i] for i in sub_argmax.flatten()])
        out = np.array([idx_to_class[i] for i in winning_affwild_idx])
        return out.reshape(sub_argmax.shape)

    # cremad6 / ravdess7: bijective subset.
    sub = probs[..., list(space.affwild2_idx)]
    return np.argmax(sub, axis=-1)


# ---------------------------------------------------------------------------
# Per-clip aggregation
# ---------------------------------------------------------------------------

def aggregate_softmax_per_clip(
    expr_logits: np.ndarray,
    videonames: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Aggregate per-frame logits into one mean-softmax vector per clip.

    Args:
        expr_logits: ``(N_frames, 8)`` raw expression logits.
        videonames: Length-``N_frames`` parallel sequence of clip stems.

    Returns:
        ``(probs_per_clip, clip_order)`` where ``probs_per_clip`` is
        ``(N_clips, 8)`` averaged softmax probabilities and ``clip_order``
        lists the corresponding videonames.
    """
    expr_logits = np.asarray(expr_logits, dtype=np.float64)
    if expr_logits.ndim != 2 or expr_logits.shape[1] != 8:
        raise ValueError(
            f"expr_logits must be (N, 8); got {expr_logits.shape}."
        )
    if len(videonames) != expr_logits.shape[0]:
        raise ValueError(
            f"videonames length {len(videonames)} != expr_logits frames "
            f"{expr_logits.shape[0]}."
        )

    # Stable softmax per frame.
    z = expr_logits - expr_logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    probs = ez / ez.sum(axis=1, keepdims=True)

    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for i, vn in enumerate(videonames):
        if vn not in sums:
            sums[vn] = probs[i].copy()
            counts[vn] = 1
        else:
            sums[vn] += probs[i]
            counts[vn] += 1

    clip_order = tuple(sums.keys())
    out = np.stack(
        [sums[vn] / counts[vn] for vn in clip_order], axis=0
    )
    return out, clip_order


def native_labels_for_clips(
    clip_order: Sequence[str],
    space: LabelSpace,
) -> np.ndarray:
    """Recover the native-class index per clip from its videoname.

    For ``cremad6`` and ``ravdess8`` the per-class lookup is direct.
    For ``ravdess7`` we recover the 8-class native label first and then
    merge ``Neutral`` and ``Calm`` into ``Neutral_or_Calm``.
    """
    if space.name == "cremad6":
        out = []
        for vn in clip_order:
            code = native_emotion_from_cremad_videoname(vn)
            out.append(NATIVE_CREMAD.index(code))
        return np.asarray(out, dtype=np.int64)

    if space.name == "ravdess8":
        out = []
        for vn in clip_order:
            code = native_emotion_from_ravdess_videoname(vn)
            out.append(NATIVE_RAVDESS_8.index(code))
        return np.asarray(out, dtype=np.int64)

    if space.name == "ravdess7":
        out = []
        for vn in clip_order:
            code = native_emotion_from_ravdess_videoname(vn)
            if code in ("Neutral", "Calm"):
                out.append(NATIVE_RAVDESS_7.index("Neutral_or_Calm"))
            else:
                out.append(NATIVE_RAVDESS_7.index(code))
        return np.asarray(out, dtype=np.int64)

    raise ValueError(f"Unknown LabelSpace name '{space.name}'.")
