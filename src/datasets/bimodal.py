"""Co-indexed bimodal feature loader for Stage 3 fusion training.

Walks a visual cache (``<videoname>.npz`` with ``features (N, D_v)`` and
``scores (N, 10)``) and an aligned-audio cache (``<videoname>.npz`` with
``features (N, D_a)``, same ``N``), joins them against the AffWild2 MTL
annotations, and yields per-frame records ready for fusion training.

Exposes two dataset classes:

* :class:`BimodalFrameDataset` - one training sample per labelled frame.
  Suitable for F0, F1, F2, F3, F5 (no temporal context).
* :class:`BimodalWindowDataset` - one sample per labelled frame, but
  emits the surrounding ``+/-window`` frames from the same video for the
  Stage 3 cross-modal-attention variant F4. Boundary rows are edge-padded.

Both return dictionaries with keys
``v_feat, v_scores, a_feat, y_expr, y_va, y_aus, m_expr, m_va, m_au,
videoname, frame_index``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.affwild2_mtl import MTLAnnotations, read_mtl_annotations


def _load_per_video(cache_dir: Path, key: str = "features") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for npz in sorted(cache_dir.glob("*.npz")):
        with np.load(npz, allow_pickle=True) as data:
            out[npz.stem] = np.array(data[key])
    return out


def _load_visual_with_scores(cache_dir: Path):
    feats: Dict[str, np.ndarray] = {}
    scores: Dict[str, np.ndarray] = {}
    image_names: Dict[str, List[str]] = {}
    for npz in sorted(cache_dir.glob("*.npz")):
        with np.load(npz, allow_pickle=True) as data:
            feats[npz.stem] = np.array(data["features"])
            scores[npz.stem] = np.array(data["scores"])
            image_names[npz.stem] = [str(s) for s in data["image_names"]]
    return feats, scores, image_names


class BimodalFrameDataset(Dataset):
    """Per-labelled-frame bimodal dataset (no temporal context)."""

    def __init__(
        self,
        annotations_file: str | Path,
        visual_cache_dir: str | Path,
        audio_cache_dir: str | Path,
    ) -> None:
        self.visual_cache_dir = Path(visual_cache_dir)
        self.audio_cache_dir = Path(audio_cache_dir)

        v_feats, v_scores, image_names = _load_visual_with_scores(self.visual_cache_dir)
        a_feats = _load_per_video(self.audio_cache_dir, key="features")

        # Build a frame -> (video, row) index identical to the one the visual
        # Stage 1 pipeline produces.
        self.v_feats = v_feats
        self.v_scores = v_scores
        self.a_feats = a_feats

        # image_name -> (videoname, row_index)
        self._name_to_row: Dict[str, Tuple[str, int]] = {}
        for videoname, names in image_names.items():
            for i, nm in enumerate(names):
                self._name_to_row[nm] = (videoname, i)

        anno = read_mtl_annotations(
            annotations_file, features_index=self._name_to_row
        )
        self.anno: MTLAnnotations = anno
        # Keep the video/frame list for per-video smoothing at eval time.
        self.videoname_frames = anno.videoname_frames

        self._rows: List[Tuple[str, int]] = [
            self._name_to_row[p] for p in anno.df["image_path"].tolist()
        ]

        # Sanity: audio and visual must share T per video.
        for vname in self.v_feats:
            if vname not in self.a_feats:
                continue
            if self.v_feats[vname].shape[0] != self.a_feats[vname].shape[0]:
                raise ValueError(
                    f"Visual/audio length mismatch for {vname}: "
                    f"{self.v_feats[vname].shape[0]} vs {self.a_feats[vname].shape[0]}. "
                    "Did you run align_audio_to_video?"
                )

        # Infer feature dims (for downstream head sizing).
        any_v = next(iter(self.v_feats.values()))
        self.v_dim = int(any_v.shape[1])
        self.scores_dim = int(next(iter(self.v_scores.values())).shape[1])
        if self.a_feats:
            any_a = next(iter(self.a_feats.values()))
            self.a_dim = int(any_a.shape[1])
        else:
            self.a_dim = 0

    def __len__(self) -> int:
        return len(self._rows)

    def _audio(self, videoname: str, row: int) -> np.ndarray:
        arr = self.a_feats.get(videoname)
        if arr is None:
            return np.zeros(self.a_dim, dtype=np.float32)
        return arr[row]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        videoname, row = self._rows[idx]
        v = self.v_feats[videoname][row]
        s = self.v_scores[videoname][row]
        a = self._audio(videoname, row)
        anno_row = self.anno.df.iloc[idx]

        return {
            "v_feat": torch.from_numpy(np.asarray(v, dtype=np.float32)),
            "v_scores": torch.from_numpy(np.asarray(s, dtype=np.float32)),
            "a_feat": torch.from_numpy(np.asarray(a, dtype=np.float32)),
            "y_expr": torch.tensor(int(anno_row["expr"]), dtype=torch.long),
            "y_va": torch.tensor(
                [float(anno_row["valence"]), float(anno_row["arousal"])],
                dtype=torch.float32,
            ),
            "y_aus": torch.tensor(
                [float(anno_row[f"au{i}"]) for i in range(1, 13)],
                dtype=torch.float32,
            ),
            "m_expr": torch.tensor(float(anno_row["mask_expr"]), dtype=torch.float32),
            "m_va": torch.tensor(float(anno_row["mask_va"]), dtype=torch.float32),
            "m_au": torch.tensor(float(anno_row["mask_au"]), dtype=torch.float32),
        }


class BimodalWindowDataset(BimodalFrameDataset):
    """Like :class:`BimodalFrameDataset` but returns +/-window context.

    Output tensors gain a leading time dimension of length ``2*window+1``.
    Out-of-bounds neighbours are edge-padded (replicate the first/last
    in-video row). Labels/masks are for the center frame only.
    """

    def __init__(
        self,
        annotations_file: str | Path,
        visual_cache_dir: str | Path,
        audio_cache_dir: str | Path,
        window: int = 5,
    ) -> None:
        super().__init__(annotations_file, visual_cache_dir, audio_cache_dir)
        if window < 0:
            raise ValueError("window must be >= 0")
        self.window = int(window)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base = super().__getitem__(idx)
        videoname, row = self._rows[idx]
        v_all = self.v_feats[videoname]
        s_all = self.v_scores[videoname]
        a_all = self.a_feats.get(videoname)

        T = v_all.shape[0]
        idxs = np.clip(
            np.arange(row - self.window, row + self.window + 1), 0, T - 1
        )
        v_seq = v_all[idxs]
        s_seq = s_all[idxs]
        a_seq = (
            a_all[idxs] if a_all is not None
            else np.zeros((len(idxs), self.a_dim), dtype=np.float32)
        )

        base["v_feat"] = torch.from_numpy(np.asarray(v_seq, dtype=np.float32))
        base["v_scores"] = torch.from_numpy(np.asarray(s_seq, dtype=np.float32))
        base["a_feat"] = torch.from_numpy(np.asarray(a_seq, dtype=np.float32))
        return base
