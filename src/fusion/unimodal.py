"""Unimodal visual / audio fusion "modules".

These wrap the Stage 1 and Stage 2 heads into the FusionModule API so
F0's post-hoc blend can consume them via the same trainer/evaluator.
"""

from __future__ import annotations

from typing import Tuple

import torch

from src.fusion.base import FusionConfig, FusionModule, MLPHeads


class VisualOnly(FusionModule):
    """Uses ``v_feat + v_scores`` only; audio input is ignored."""

    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__(cfg)
        self.heads = MLPHeads(cfg.v_dim + cfg.scores_dim, cfg)

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.cat([v_feat, v_scores], dim=-1)
        return self.heads(z)


class AudioOnly(FusionModule):
    """Uses ``a_feat`` only; visual input is ignored."""

    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__(cfg)
        self.heads = MLPHeads(cfg.a_dim, cfg)

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.heads(a_feat)
