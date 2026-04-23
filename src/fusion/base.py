"""Common API for Stage 3 fusion variants.

Every fusion module takes a frame-level (or windowed) triple
``(v_feat, v_scores, a_feat)`` and returns three output tensors with
the same shape conventions as :class:`src.heads.mtl_head.MTLHead`:

    expr_logits: (B, 8)
    va_tanh:     (B, 2)   in [-1, 1]
    au_prob:     (B, 12)  in ( 0,  1 )

Fusion modules that take a temporal window (B, T, D) should mean-pool
across ``T`` internally before emitting center-frame predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    v_dim: int = 1280
    scores_dim: int = 10
    a_dim: int = 1024
    num_expr: int = 8
    num_aus: int = 12
    hidden: int = 512
    dropout: float = 0.3
    au_hidden: int = 128


class FusionModule(nn.Module):
    """Base class. Subclasses implement :meth:`forward`."""

    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MLPHeads(nn.Module):
    """Shared three-head block used by most fusion variants."""

    def __init__(self, input_dim: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.expr = nn.Linear(input_dim, cfg.num_expr)
        self.va = nn.Linear(input_dim, 2)
        self.au1 = nn.Linear(input_dim, cfg.au_hidden)
        self.au2 = nn.Linear(cfg.au_hidden, cfg.num_aus)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expr = self.expr(z)
        va = torch.tanh(self.va(z))
        au = torch.sigmoid(self.au2(F.relu(self.au1(z))))
        return expr, va, au


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Helper used by the results table."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
