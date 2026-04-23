"""F1 - Early concatenation + MLP.

``[v_feat, v_scores, a_feat] -> LN -> Linear(hidden) -> GELU -> Dropout
-> Linear(hidden//2) -> GELU -> Dropout -> three heads``.

Sized to fit the 1.0 M trainable-parameter Stage 3 budget. For the
primary configuration (v_dim=1280, a_dim=1024, scores=10), hidden=384
gives ~1.0 M trainable params including the three heads.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.fusion.base import FusionConfig, FusionModule, MLPHeads


class EarlyConcat(FusionModule):
    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__(cfg)
        in_dim = cfg.v_dim + cfg.scores_dim + cfg.a_dim
        h1 = cfg.hidden
        h2 = max(128, cfg.hidden // 2)
        self.ln = nn.LayerNorm(in_dim)
        self.body = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.heads = MLPHeads(h2, cfg)

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.cat([v_feat, v_scores, a_feat], dim=-1)
        z = self.body(self.ln(z))
        return self.heads(z)
