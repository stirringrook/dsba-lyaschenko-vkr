"""F5 - Low-rank multimodal fusion (tensor fusion, LMF).

Liu et al. (ACL 2018) "Efficient Low-rank Multimodal Fusion with
Modality-Specific Factors". An outer product between reduced
visual and audio embeddings would blow up the parameter count; LMF
factors that outer product as a sum over ``R`` rank-1 terms, turning
the fusion into a gather-sum over modality-specific factor matrices.

Concretely, with reduced modality vectors ``u_v in R^d, u_a in R^d``:

    z = sum_{r=1..R} (W_v[r] @ u_v) * (W_a[r] @ u_a)  in R^h

which requires only ``R * d * h`` parameters per modality.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fusion.base import FusionConfig, FusionModule, MLPHeads


class LMFusion(FusionModule):
    def __init__(
        self,
        cfg: FusionConfig,
        reduce_dim: int = 128,
        out_dim: int = 256,
        rank: int = 4,
    ) -> None:
        super().__init__(cfg)
        self.reduce_v = nn.Sequential(
            nn.Linear(cfg.v_dim + cfg.scores_dim, reduce_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.reduce_a = nn.Sequential(
            nn.Linear(cfg.a_dim, reduce_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        # LMF factors; using +1 rows to absorb the bias term (Liu et al. eqn 4).
        self.W_v = nn.Parameter(torch.randn(rank, reduce_dim + 1, out_dim) * 0.1)
        self.W_a = nn.Parameter(torch.randn(rank, reduce_dim + 1, out_dim) * 0.1)
        self.w_fuse = nn.Parameter(torch.ones(1, rank) / rank)
        self.b_fuse = nn.Parameter(torch.zeros(out_dim))
        self.heads = MLPHeads(out_dim, cfg)

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_v = self.reduce_v(torch.cat([v_feat, v_scores], dim=-1))  # (B, d)
        u_a = self.reduce_a(a_feat)                                  # (B, d)
        ones = torch.ones(u_v.size(0), 1, device=u_v.device, dtype=u_v.dtype)
        u_v = torch.cat([ones, u_v], dim=-1)                         # (B, d+1)
        u_a = torch.cat([ones, u_a], dim=-1)                         # (B, d+1)

        # (R, out) per modality after contraction with the reduced vectors.
        fv = torch.einsum("bd,rdh->brh", u_v, self.W_v)
        fa = torch.einsum("bd,rdh->brh", u_a, self.W_a)
        z = (fv * fa).sum(dim=1)                                     # sum over rank
        z = z + self.b_fuse

        # One scalar weight per rank, broadcast back in for the final projection.
        # Note: equivalent to weighted sum over ranks if we carried w_fuse into
        # the einsum; we keep it explicit so R >= 1 is trivially tunable.
        return self.heads(z)
