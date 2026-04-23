"""Russell-circumplex mapping from categorical emotions to synthetic (V, A).

CREMA-D and RAVDESS ship clip-level categorical emotion labels only. For the
interim evaluation on these datasets we need a continuous valence/arousal
signal to exercise the VA head of the existing three-head MTL architecture.
The mapping below assigns each basic emotion a fixed anchor
``(V, A) in [-1, 1]^2`` taken from Russell's circumplex and standard
dimensional-emotion norms (Warriner et al. 2013, Russell 1980). The arousal
magnitude is then scaled by the clip's intensity level so that more intense
readings sit further from the origin.

The mapping is deliberately conservative: anchors match the qualitative
placement in Russell's diagram and the intensity scaling is linear. This
is a PROXY --- it is not a replacement for Aff-Wild2's human-annotated VA,
and callers should treat the resulting CCC numbers as indicative rather
than directly comparable to the Aff-Wild2 leaderboard.
"""

from __future__ import annotations

from typing import Dict, Tuple

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
    is the emotion code.
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
