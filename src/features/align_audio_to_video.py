"""Linear-interpolate audio embeddings onto the visual frame timeline.

Port of cell 28 of ``bah.ipynb``. Given

* ``audio_features`` of shape ``(T_audio, D_a)`` at the encoder's native
  rate (~50 Hz for wav2vec 2.0 / HuBERT at 16 kHz input), and
* a visual frame count ``T_video``,

this module produces ``audio_features_sampled`` of shape
``(T_video, D_a)`` by 1-D linear interpolation along the time axis.
The two caches become perfectly co-indexed by videoname + frame index.

The function also handles the pathological edge cases we expect on
AffWild2:

* ``T_audio == 0`` (silent / missing audio)  -> zeros of shape ``(T_video, D_a)``
  with ``D_a`` inferred from a fallback argument; rows marked via a mask.
* ``T_audio == 1``  -> broadcast the single embedding over the timeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


def align_audio_to_video(
    audio_features: np.ndarray,
    num_video_frames: int,
    fallback_dim: Optional[int] = None,
) -> Tuple[np.ndarray, bool]:
    """Resample ``(T_audio, D_a)`` onto a ``num_video_frames``-long timeline.

    Args:
        audio_features: Shape ``(T_audio, D_a)``. May be empty.
        num_video_frames: Target number of visual frames ``T_video``.
        fallback_dim: Dimension ``D_a`` to use when ``audio_features`` is
            empty. Required in that case; otherwise ignored.

    Returns:
        ``(aligned, has_audio)`` with ``aligned`` shape
        ``(num_video_frames, D_a)`` and ``has_audio`` False only when the
        input was empty.
    """
    if num_video_frames <= 0:
        raise ValueError(f"num_video_frames must be > 0, got {num_video_frames}")
    audio_features = np.asarray(audio_features)
    if audio_features.size == 0 or audio_features.shape[0] == 0:
        if fallback_dim is None:
            raise ValueError(
                "audio_features is empty and fallback_dim was not provided."
            )
        return np.zeros((num_video_frames, fallback_dim), dtype=np.float32), False

    if audio_features.ndim != 2:
        raise ValueError(
            f"Expected (T, D) audio_features, got shape {audio_features.shape}"
        )

    T_audio = audio_features.shape[0]
    if T_audio == 1:
        aligned = np.repeat(audio_features, num_video_frames, axis=0)
        return aligned.astype(np.float32, copy=False), True

    x = np.linspace(0.0, 1.0, T_audio)
    new_x = np.linspace(0.0, 1.0, num_video_frames)
    f = interp1d(x, audio_features, axis=0, kind="linear", copy=False, assume_sorted=True)
    return f(new_x).astype(np.float32), True


def align_caches(
    audio_cache_dir: str | Path,
    visual_cache_dir: str | Path,
    output_dir: str | Path,
    fallback_dim: Optional[int] = None,
) -> None:
    """Write aligned audio caches keyed by videoname.

    For every ``<name>.npz`` under ``visual_cache_dir`` (produced by
    :mod:`src.features.extract_visual`), we look up the matching audio
    cache ``<name>.npz`` under ``audio_cache_dir`` and write the aligned
    counterpart to ``output_dir/<name>.npz`` with keys ``features,
    has_audio``. When the audio cache is missing, we zero-fill.
    """
    audio_cache_dir = Path(audio_cache_dir)
    visual_cache_dir = Path(visual_cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_npz_paths = sorted(visual_cache_dir.glob("*.npz"))
    if not visual_npz_paths:
        raise FileNotFoundError(f"No visual caches under {visual_cache_dir}")

    for vpath in visual_npz_paths:
        name = vpath.stem
        with np.load(vpath, allow_pickle=True) as vdata:
            T_video = int(vdata["features"].shape[0])

        apath = audio_cache_dir / f"{name}.npz"
        if apath.exists():
            with np.load(apath) as adata:
                afeat = adata["features"]
        else:
            afeat = np.zeros((0, 0), dtype=np.float32)

        aligned, has_audio = align_audio_to_video(
            afeat, num_video_frames=T_video, fallback_dim=fallback_dim
        )
        np.savez(
            output_dir / f"{name}.npz",
            features=aligned.astype(np.float32),
            has_audio=np.bool_(has_audio),
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Linear-interpolate audio features onto the visual frame timeline."
    )
    p.add_argument("--audio-cache", required=True, help="Dir of per-video audio .npz")
    p.add_argument("--visual-cache", required=True, help="Dir of per-video visual .npz")
    p.add_argument("--out", dest="output_dir", required=True)
    p.add_argument(
        "--fallback-dim",
        type=int,
        default=None,
        help="Audio feature dim used for zero-filling missing clips.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    align_caches(
        audio_cache_dir=args.audio_cache,
        visual_cache_dir=args.visual_cache,
        output_dir=args.output_dir,
        fallback_dim=args.fallback_dim,
    )


if __name__ == "__main__":
    main()
