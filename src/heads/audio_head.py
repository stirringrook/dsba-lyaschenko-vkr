"""Audio-only MTL head (Stage 2 sanity baseline).

Takes a per-frame audio embedding ``a_t`` of shape ``(N, D_a)`` and
emits the three MTL outputs with the same activation layout as
:mod:`src.heads.mtl_head`:

    EXPR -> Linear(D_a, 8)  (logits, softmax in eval)
    VA   -> Linear(D_a, 2)  + tanh
    AU   -> Linear(D_a, 128) -> ReLU -> Linear(128, 12) + sigmoid

The loss/optimizer plumbing (CE, CCC-based VA, weighted BCE) is shared
with the visual head via the utilities in :mod:`src.heads.mtl_head`, so
Stage 2 reuses ``train_head`` with a different ``input_dim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioHeadConfig:
    audio_dim: int = 1024           # HuBERT Large default; 768 for wav2vec2-base
    num_expr: int = 8
    num_aus: int = 12
    au_hidden: int = 128


class AudioExprHead(nn.Module):
    def __init__(self, audio_dim: int, num_classes: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(audio_dim, num_classes)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.fc(a)


class AudioVAHead(nn.Module):
    def __init__(self, audio_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(audio_dim, 2)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(a))


class AudioAUHead(nn.Module):
    def __init__(self, audio_dim: int, num_aus: int = 12, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(audio_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_aus)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(F.relu(self.fc1(a))))


class AudioMTLHead(nn.Module):
    """Container with three independently-trainable audio-only heads."""

    def __init__(self, cfg: AudioHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.expr = AudioExprHead(cfg.audio_dim, cfg.num_expr)
        self.va = AudioVAHead(cfg.audio_dim)
        self.au = AudioAUHead(cfg.audio_dim, cfg.num_aus, cfg.au_hidden)

    @property
    def input_dim(self) -> int:
        return self.cfg.audio_dim

    def forward(
        self, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.expr(a), self.va(a), self.au(a)
