"""Small I/O helpers for the per-video feature cache."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def save_video_features(
    output_path: str | Path,
    features: np.ndarray,
    scores: np.ndarray,
    image_names: Iterable[str],
) -> None:
    """Save per-frame features/scores + ordered image names to ``.npz``.

    The three arrays must share the same leading dimension ``N``
    (number of frames in the video). ``image_names`` is stored as a
    unicode object array to preserve frame-to-filename alignment.
    """
    features = np.asarray(features, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    names = np.asarray(list(image_names), dtype=object)
    assert features.shape[0] == scores.shape[0] == names.shape[0], (
        f"Mismatched lengths: features={features.shape}, scores={scores.shape}, "
        f"names={names.shape}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, features=features, scores=scores, image_names=names)


def load_video_features(path: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a single per-video ``.npz`` saved by :func:`save_video_features`."""
    with np.load(path, allow_pickle=True) as data:
        features = np.array(data["features"])
        scores = np.array(data["scores"])
        image_names = [str(x) for x in data["image_names"]]
    return features, scores, image_names


def load_features_dir(
    cache_dir: str | Path,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    """Load every ``<videoname>.npz`` in ``cache_dir``.

    Returns:
        ``(filename2features, video_sizes)`` where
          * ``filename2features[image_path] = (feature_vec, score_vec)``
            keyed by ``"<videoname>/<frame>.jpg"``, matching the notebook's
            ``filename2featuresAll`` layout.
          * ``video_sizes[videoname] = num_frames`` for smoothing sanity checks.
    """
    cache_dir = Path(cache_dir)
    filename2features: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    video_sizes: Dict[str, int] = {}
    for npz in sorted(cache_dir.glob("*.npz")):
        videoname = npz.stem
        features, scores, image_names = load_video_features(npz)
        video_sizes[videoname] = len(image_names)
        for i, name in enumerate(image_names):
            filename2features[name] = (features[i], scores[i])
    return filename2features, video_sizes
