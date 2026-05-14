"""F6d - Attention-bottleneck fusion (MBT).

Per Nagrani et al. (NeurIPS 2021) "Attention Bottlenecks for Multimodal
Fusion", a small set of latent ``bottleneck'' tokens mediates inter-modal
information exchange, keeping per-modality streams unconcatenated. This
cuts cross-modal self-attention FLOPs vs.pairwise attention (F4) at the
cost of one indirection.

Architecture:
* Inputs: windowed (B, T, D_v + scores) and (B, T, D_a). Project to
  ``hidden`` to obtain ``v_tokens`` and ``a_tokens`` of shape (B, T, h).
* ``B`` learnable bottleneck tokens ``z`` of shape (B, B_tokens, h).
* One transformer encoder block per modality with cross-attention into
  ``z`` (modality tokens attend to z; z attends to each modality). The
  updated ``z`` carries cross-modal information that is read out by
  mean-pooling the *bottleneck* tokens.
* Heads consume ``[mean(v_hat), mean(a_hat), mean(z_hat)]``.

At ``hidden=256, num_heads=4, B_tokens=4, window=5`` this lands at
$\\approx 1.0$ M trainable params --- below F4 and within the 2 M
ceiling.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.fusion.base import FusionConfig, FusionModule, MLPHeads
from src.fusion.f4_xattn import _CrossAttnBlock


class MBTFusion(FusionModule):
    def __init__(
        self,
        cfg: FusionConfig,
        hidden: int | None = None,
        num_heads: int = 4,
        dropout: float | None = None,
        num_bottleneck_tokens: int = 4,
    ) -> None:
        super().__init__(cfg)
        # Read hidden + dropout from FusionConfig so the YAML's
        # ``fusion.hidden`` actually takes effect (kwargs override).
        hidden = hidden if hidden is not None else cfg.hidden
        dropout = dropout if dropout is not None else cfg.dropout
        self.hidden = hidden
        self.B = num_bottleneck_tokens

        self.proj_v = nn.Linear(cfg.v_dim + cfg.scores_dim, hidden)
        self.proj_a = nn.Linear(cfg.a_dim, hidden)
        self.bottleneck = nn.Parameter(torch.randn(1, num_bottleneck_tokens, hidden) * 0.02)

        # Each modality reads from + writes to the bottleneck.
        # Block_v: Q = [v_tokens; z], KV = z; bottleneck-side update reuses z.
        # We implement two passes: modality -> z (z writes), then z -> modality
        # (modality writes). Simple and matches the MBT "fusion via bottleneck"
        # description in §3.2 of Nagrani et al.
        self.modality_to_bottleneck_v = _CrossAttnBlock(hidden, num_heads=num_heads, dropout=dropout)
        self.modality_to_bottleneck_a = _CrossAttnBlock(hidden, num_heads=num_heads, dropout=dropout)
        self.bottleneck_to_v = _CrossAttnBlock(hidden, num_heads=num_heads, dropout=dropout)
        self.bottleneck_to_a = _CrossAttnBlock(hidden, num_heads=num_heads, dropout=dropout)

        self.heads = MLPHeads(3 * hidden, cfg)

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

        B = v_feat.shape[0]
        v_tokens = self.proj_v(torch.cat([v_feat, v_scores], dim=-1))   # (B, T, h)
        a_tokens = self.proj_a(a_feat)                                   # (B, T, h)
        z = self.bottleneck.expand(B, -1, -1)                            # (B, K, h)

        # Modality writes into z.
        z_after_v = self.modality_to_bottleneck_v(z, v_tokens)
        z_after_a = self.modality_to_bottleneck_a(z, a_tokens)
        z = 0.5 * (z_after_v + z_after_a)

        # z writes back into each modality.
        v_hat = self.bottleneck_to_v(v_tokens, z)
        a_hat = self.bottleneck_to_a(a_tokens, z)

        z_summary = torch.cat(
            [v_hat.mean(dim=1), a_hat.mean(dim=1), z.mean(dim=1)], dim=-1
        )
        return self.heads(z_summary)
