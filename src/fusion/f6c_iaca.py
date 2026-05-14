"""F6c - Inconsistency-aware cross-attention gate (IACA on top of F4).

Per Praveen et al. (CVPRW 2024) "Inconsistency-Aware Cross-Modal Attention
for Audio-Visual Fusion in Dimensional Emotion Recognition", the
inconsistency gate downweights the cross-modally-fused output when the
two modalities disagree, falling back to the visual unimodal prediction.

Implementation:
* Same F4 cross-attention block as :class:`CrossModalAttention` produces
  the fused stream :math:`z_{\\mathrm{fused}}`.
* A parallel visual-only branch produces a visual fallback logit per
  task (same head architecture as :class:`MLPHeads`).
* A small two-layer MLP gate reads the per-frame inconsistency signal
  ``g_in = [pooled_v, pooled_a, |pooled_v - pooled_a|]`` and emits a
  per-task gate ``g \\in [0, 1]^{4}`` over {EXPR, V, A, AU}.
* Output per task ``t`` is
  ``logit_t = g_t * fused_t + (1 - g_t) * visual_t``.

Param count at the default settings (``hidden=256``, ``num_heads=4``,
``window=5``, gate hidden 128) sits at $\\approx 2$ M, just above F4
(1.7 M) and below the report's 2 M ceiling.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.fusion.base import FusionConfig, FusionModule, MLPHeads
from src.fusion.f4_xattn import _CrossAttnBlock


class IACAGate(FusionModule):
    def __init__(
        self,
        cfg: FusionConfig,
        hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        gate_hidden: int = 128,
    ) -> None:
        super().__init__(cfg)
        self.hidden = hidden

        # Fused stream (F4-equivalent).
        self.proj_v = nn.Linear(cfg.v_dim + cfg.scores_dim, hidden)
        self.proj_a = nn.Linear(cfg.a_dim, hidden)
        self.block_v = _CrossAttnBlock(hidden, num_heads=num_heads, dropout=dropout)
        self.block_a = _CrossAttnBlock(hidden, num_heads=num_heads, dropout=dropout)
        self.fused_heads = MLPHeads(2 * hidden, cfg)

        # Visual-only fallback stream.
        self.visual_heads = MLPHeads(cfg.v_dim + cfg.scores_dim, cfg)

        # Inconsistency gate. Reads pooled-time visual & audio embeddings
        # plus their absolute difference; emits 4 gate scalars covering
        # {EXPR, V, A, AU}.
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Linear(3 * hidden, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 4),
        )
        self._last_gate: torch.Tensor | None = None

    def last_gate(self) -> torch.Tensor | None:
        """Last forward's gate tensor (B, 4). Useful for figures."""
        return self._last_gate

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if v_feat.dim() == 2:
            v_feat = v_feat.unsqueeze(1)
            v_scores = v_scores.unsqueeze(1)
            a_feat = a_feat.unsqueeze(1)

        # Visual-only fallback uses the center frame of the window.
        T = v_feat.shape[1]
        center = T // 2
        v_center = torch.cat([v_feat[:, center], v_scores[:, center]], dim=-1)
        v_expr_uni, v_va_uni, v_au_uni = self.visual_heads(v_center)

        # Fused F4 stream.
        v_tokens = self.proj_v(torch.cat([v_feat, v_scores], dim=-1))
        a_tokens = self.proj_a(a_feat)
        v_hat = self.block_v(v_tokens, a_tokens)
        a_hat = self.block_a(a_tokens, v_tokens)
        v_pool = v_hat.mean(dim=1)
        a_pool = a_hat.mean(dim=1)
        z = torch.cat([v_pool, a_pool], dim=-1)
        f_expr, f_va, f_au = self.fused_heads(z)

        # Gate.
        gate_in = torch.cat([v_pool, a_pool, torch.abs(v_pool - a_pool)], dim=-1)
        g = torch.sigmoid(self.gate(gate_in))  # (B, 4)
        self._last_gate = g.detach()

        expr = g[:, 0:1] * f_expr + (1 - g[:, 0:1]) * v_expr_uni
        va = torch.stack(
            [
                g[:, 1] * f_va[:, 0] + (1 - g[:, 1]) * v_va_uni[:, 0],
                g[:, 2] * f_va[:, 1] + (1 - g[:, 2]) * v_va_uni[:, 1],
            ],
            dim=1,
        )
        au = g[:, 3:4] * f_au + (1 - g[:, 3:4]) * v_au_uni
        return expr, va, au
