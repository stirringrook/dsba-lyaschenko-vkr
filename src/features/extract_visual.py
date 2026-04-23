"""Per-video visual feature extraction for AffWild2 cropped_aligned data.

Reads every ``<video>/<frame>.jpg`` under ``cropped_aligned_dir/`` in
natural frame order, runs them through ``EmotiEffLibRecognizer``, and
writes one ``.npz`` per video to ``output_dir/<videoname>.npz`` with
three arrays: ``features`` ``(N, D)``, ``scores`` ``(N, 10)``, and
``image_names`` ``(N,)`` of ``"<videoname>/<frame>.jpg"`` strings.

The extractor is intentionally a thin rewrite of cell 15 of
``EmotiEffLib-main/training_and_examples/ABAW/ABAW7/mtl.ipynb`` using the
library's public API rather than ``torch.load`` on a raw ``.pt`` file
(see :func:`EmotiEffLibRecognizer`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def _natural_frame_sort_key(name: str) -> int:
    """AffWild2 frames are named ``00001.jpg``, ``00002.jpg`` etc."""
    stem = os.path.splitext(name)[0]
    try:
        return int(stem)
    except ValueError:
        return -1  # put malformed names first; they're easy to spot


def _list_video_frames(video_dir: Path) -> List[str]:
    frames = [f for f in os.listdir(video_dir) if f.lower().endswith(".jpg")]
    frames.sort(key=_natural_frame_sort_key)
    return frames


def _extract_one_video(
    recognizer,
    video_dir: Path,
    batch_size: int = 48,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run the backbone over every frame in ``video_dir``.

    Returns:
        ``(features, scores, image_names)`` with
          * ``features`` shape ``(N, D)``, float32;
          * ``scores``   shape ``(N, 10)`` for MTL models (8 EXPR + 2 VA);
          * ``image_names`` a list of ``"<videoname>/<frame>.jpg"`` strings.
    """
    frame_files = _list_video_frames(video_dir)
    if not frame_files:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            [],
        )

    videoname = video_dir.name
    all_features: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    image_names: List[str] = []

    buf_imgs: List[np.ndarray] = []
    buf_names: List[str] = []

    def _flush():
        if not buf_imgs:
            return
        features = recognizer.extract_features(buf_imgs)     # (B, D)
        _, scores = recognizer.classify_emotions(features, logits=True)  # (B, 10)
        all_features.append(np.asarray(features, dtype=np.float32))
        all_scores.append(np.asarray(scores, dtype=np.float32))
        image_names.extend(buf_names)
        buf_imgs.clear()
        buf_names.clear()

    for fname in frame_files:
        img = Image.open(video_dir / fname).convert("RGB")
        buf_imgs.append(np.asarray(img))
        buf_names.append(f"{videoname}/{fname}")
        if len(buf_imgs) >= batch_size:
            _flush()
    _flush()

    if not all_features:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            [],
        )
    features = np.concatenate(all_features, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    return features, scores, image_names


def extract_visual_features(
    model_name: str,
    cropped_aligned_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    engine: str = "torch",
    batch_size: int = 48,
    device: str | None = None,
    overwrite: bool = False,
) -> None:
    """Extract per-video visual features with ``EmotiEffLibRecognizer``.

    Args:
        model_name: One of ``emotiefflib.facial_analysis.get_model_list()``.
            Stage 1 targets ``"mbf_va_mtl"``.
        cropped_aligned_dir: AffWild2 cropped_aligned root (per-video folders).
        output_dir: Destination for ``<videoname>.npz``.
        engine: ``"torch"`` (default) or ``"onnx"``.
        batch_size: Backbone batch size. 48 matches the notebook default.
        device: Override the auto-picked device. When ``None``, chooses
            ``"cuda"`` if available else ``"cpu"``.
        overwrite: If ``False``, skip videos whose ``.npz`` already exists.
    """
    # Deferred import so unit tests that don't touch the library don't need it.
    import torch
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    from src.utils.io import save_video_features

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    recognizer = EmotiEffLibRecognizer(
        engine=engine, model_name=model_name, device=device
    )

    cropped_aligned_dir = Path(cropped_aligned_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(p for p in cropped_aligned_dir.iterdir() if p.is_dir())
    if not video_dirs:
        raise FileNotFoundError(
            f"No per-video directories found under {cropped_aligned_dir}. "
            "Expected AffWild2 cropped_aligned layout."
        )

    for video_dir in tqdm(video_dirs, desc=f"extract[{model_name}]"):
        out_npz = output_dir / f"{video_dir.name}.npz"
        if out_npz.exists() and not overwrite:
            continue
        features, scores, image_names = _extract_one_video(
            recognizer, video_dir, batch_size=batch_size
        )
        if not image_names:
            continue
        save_video_features(out_npz, features, scores, image_names)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-frame visual features with EmotiEffLibRecognizer."
    )
    p.add_argument("--model", default="mbf_va_mtl", help="EmotiEffLib model name.")
    p.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="Path to cropped_aligned directory (per-video folders of JPEGs).",
    )
    p.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Where to write <videoname>.npz caches.",
    )
    p.add_argument("--engine", default="torch", choices=["torch", "onnx"])
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--device", default=None, help="cuda / cpu; auto if unset.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute caches that already exist.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    extract_visual_features(
        model_name=args.model,
        cropped_aligned_dir=args.input_dir,
        output_dir=args.output_dir,
        engine=args.engine,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
