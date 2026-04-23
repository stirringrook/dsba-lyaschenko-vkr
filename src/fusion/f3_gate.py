"""F3 - Task-specific per-frame modality gating (direction D3).

A small gate MLP reads the concatenated embedding and emits a per-task,
per-frame gate vector ``alpha in [0,1]^4`` covering the four ABAW tasks
{EXPR, V, A, AU}. Final output per task ``t`` is
``alpha_t * visual_logit_t + (1 - alpha_t) * audio_logit_t``.

The learned ``alpha`` tensor is accessible via :meth:`last_alpha` for
the Chapter 5 interpretability figure (mean alpha per task across the
validation split, per-class boxplots, etc.).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.fusion.base import FusionConfig, FusionModule, MLPHeads


class TaskGate(FusionModule):
    def __init__(self, cfg: FusionConfig, gate_hidden: int = 128) -> None:
        super().__init__(cfg)
        self.visual = MLPHeads(cfg.v_dim + cfg.scores_dim, cfg)
        self.audio = MLPHeads(cfg.a_dim, cfg)
        in_dim = cfg.v_dim + cfg.scores_dim + cfg.a_dim
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 4),  # {EXPR, V, A, AU}
        )
        self._last_alpha: torch.Tensor | None = None

    def last_alpha(self) -> torch.Tensor | None:
        """Last forward's alpha tensor (B, 4). Useful for figures."""
        return self._last_alpha

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = torch.cat([v_feat, v_scores], dim=-1)
        v_expr, v_va, v_au = self.visual(v)
        a_expr, a_va, a_au = self.audio(a_feat)

        gate_in = torch.cat([v, a_feat], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))  # (B, 4)
        self._last_alpha = alpha.detach()

        expr = alpha[:, 0:1] * v_expr + (1 - alpha[:, 0:1]) * a_expr
        va = torch.stack(
            [
                alpha[:, 1] * v_va[:, 0] + (1 - alpha[:, 1]) * a_va[:, 0],
                alpha[:, 2] * v_va[:, 1] + (1 - alpha[:, 2]) * a_va[:, 1],
            ],
            dim=1,
        )
        au = alpha[:, 3:4] * v_au + (1 - alpha[:, 3:4]) * a_au
        return expr, va, au
