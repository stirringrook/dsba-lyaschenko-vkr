"""F2 - Late fusion with a learned per-task scalar blend.

Two unimodal heads plus four learned blend scalars (one per task
{EXPR, V, A, AU}), each passed through sigmoid. Blends happen at the
logit / tanh / sigmoid layer (AU is already a probability; we blend
pre-sigmoid logits for stability).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.fusion.base import FusionConfig, FusionModule, MLPHeads


class LearnedBlend(FusionModule):
    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__(cfg)
        self.visual = MLPHeads(cfg.v_dim + cfg.scores_dim, cfg)
        self.audio = MLPHeads(cfg.a_dim, cfg)
        # Four learnable logits -> sigmoid for the four ABAW tasks.
        self.alpha_logits = nn.Parameter(torch.zeros(4))  # {expr, V, A, au}

    def _alphas(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logits)

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = torch.cat([v_feat, v_scores], dim=-1)
        v_expr, v_va, v_au = self.visual(v)
        a_expr, a_va, a_au = self.audio(a_feat)
        a = self._alphas()

        expr = a[0] * v_expr + (1 - a[0]) * a_expr
        # VA: blend each dimension independently so alpha_V / alpha_A can split.
        va = torch.stack(
            [
                a[1] * v_va[:, 0] + (1 - a[1]) * a_va[:, 0],
                a[2] * v_va[:, 1] + (1 - a[2]) * a_va[:, 1],
            ],
            dim=1,
        )
        au = a[3] * v_au + (1 - a[3]) * a_au
        return expr, va, au
