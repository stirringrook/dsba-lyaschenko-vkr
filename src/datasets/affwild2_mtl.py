"""AffWild2 MTL annotation parser and feature-backed Dataset.

Replicates the mask semantics of cell 32 of
``EmotiEffLib-main/training_and_examples/ABAW/ABAW7/mtl.ipynb``:

* Annotation file format (comma-separated, one line per frame, header row):
  ``image_path, valence, arousal, expression_id, AU1, ..., AU12``.
* Missing-value conventions:
    - Valence or arousal == -5  -> VA mask off, values zeroed.
    - Expression id  == -1      -> EXPR mask off, value zeroed.
    - Any AU < 0                -> AU mask off, vector zeroed.
* A row is kept only if at least one of (VA, EXPR, AU) is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NUM_AUS_DEFAULT = 12


@dataclass
class MTLAnnotations:
    """In-memory MTL annotations for a single split."""

    df: pd.DataFrame                          # one row per kept frame
    videoname_frames: List[Tuple[str, int]]   # (videoname, frame_index) per row
    num_missed: int = 0                       # rows dropped (no cached features)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def y_va(self) -> np.ndarray:
        return self.df[["valence", "arousal"]].to_numpy(dtype=np.float32)

    @property
    def y_expr(self) -> np.ndarray:
        return self.df["expr"].to_numpy(dtype=np.int64)

    @property
    def y_aus(self) -> np.ndarray:
        cols = [c for c in self.df.columns if c.startswith("au")]
        return self.df[cols].to_numpy(dtype=np.int64)

    @property
    def mask_va(self) -> np.ndarray:
        return self.df["mask_va"].to_numpy(dtype=np.float32)

    @property
    def mask_expr(self) -> np.ndarray:
        return self.df["mask_expr"].to_numpy(dtype=np.float32)

    @property
    def mask_au(self) -> np.ndarray:
        return self.df["mask_au"].to_numpy(dtype=np.float32)


def read_mtl_annotations(
    annotation_file: str | Path,
    features_index: Optional[Dict[str, object]] = None,
    num_aus: int = NUM_AUS_DEFAULT,
) -> MTLAnnotations:
    """Parse an ABAW-7 MTL annotation file.

    Args:
        annotation_file: Path to ``training_set_annotations.txt`` or the
            validation equivalent.
        features_index: Optional container (dict or set) of image paths
            ``"<videoname>/<frame>.jpg"``. When provided, rows whose
            ``image_path`` is not in this index are dropped (and counted
            in ``num_missed``). This mirrors the ``imagename in
            filename2featuresAll`` check in cell 32.
        num_aus: Number of action-unit columns. AffWild2 uses 12.

    Returns:
        A :class:`MTLAnnotations` holding a parsed DataFrame plus the
        ``(videoname, frame_index)`` list needed for per-video smoothing.
    """
    path = Path(annotation_file)
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    # The notebook skips the header with ``mtl_lines[1:]``.
    rows: List[Dict] = []
    videoname_frames: List[Tuple[str, int]] = []
    num_missed = 0

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        image_path = parts[0]
        valence = float(parts[1])
        arousal = float(parts[2])
        expression = int(parts[3])
        aus = list(map(int, parts[4 : 4 + num_aus]))

        mask_va = valence > -5 and arousal > -5
        if not mask_va:
            valence = 0.0
            arousal = 0.0

        mask_expr = expression > -1
        if not mask_expr:
            expression = 0

        mask_au = min(aus) >= 0
        if not mask_au:
            aus = [0] * num_aus

        if not (mask_va or mask_expr or mask_au):
            continue

        if features_index is not None and image_path not in features_index:
            num_missed += 1
            continue

        videoname, frame_file = image_path.split("/")
        frame_idx = int(Path(frame_file).stem)

        row: Dict = {
            "image_path": image_path,
            "videoname": videoname,
            "frame_index": frame_idx,
            "valence": valence,
            "arousal": arousal,
            "expr": expression,
            "mask_va": float(mask_va),
            "mask_expr": float(mask_expr),
            "mask_au": float(mask_au),
        }
        for i, v in enumerate(aus, start=1):
            row[f"au{i}"] = int(v)
        rows.append(row)
        videoname_frames.append((videoname, frame_idx))

    df = pd.DataFrame(rows)
    return MTLAnnotations(df=df, videoname_frames=videoname_frames, num_missed=num_missed)


def compute_expr_class_weights(
    y_expr: np.ndarray, mask_expr: np.ndarray, num_classes: int = 8
) -> Dict[int, float]:
    """Replicates cell 35 of the notebook.

    ``emo_cw = (1 / counts); emo_cw /= emo_cw.min()``
    """
    mask = mask_expr == 1
    y = y_expr[mask].astype(int)
    _, counts = np.unique(y, return_counts=True)
    # Ensure ``counts`` covers all classes even when some are absent.
    full = np.ones(num_classes, dtype=np.float64)
    present, cnts = np.unique(y, return_counts=True)
    full[present] = cnts
    cw = 1.0 / full
    cw = cw / cw.min()
    return {int(i): float(cw[i]) for i in range(num_classes)}


def compute_au_class_weights(
    y_aus: np.ndarray, mask_au: np.ndarray
) -> np.ndarray:
    """Per-AU (neg, pos) weighting from cell 36.

    For each AU, ``weight_for_0 = total / (2 * neg)`` and
    ``weight_for_1 = total / (2 * pos)``. Returns shape ``(num_aus, 2)``.
    """
    mask = mask_au == 1
    y = y_aus[mask].astype(int)
    num_aus = y.shape[1]
    weights = np.ones((num_aus, 2), dtype=np.float32)
    for i in range(num_aus):
        counts = np.bincount(y[:, i], minlength=2)
        neg, pos = int(counts[0]), int(counts[1])
        total = neg + pos
        if neg > 0:
            weights[i, 0] = total / (2.0 * neg)
        if pos > 0:
            weights[i, 1] = total / (2.0 * pos)
    return weights
