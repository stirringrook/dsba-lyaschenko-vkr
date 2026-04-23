"""F4 - Bidirectional cross-modal attention (direction D1, primary candidate).

Expects a temporal window (B, T, D) for both modalities (T = 2*window+1,
center frame at index T//2). Projects to a shared hidden dimension
``h`` and runs two single-layer Transformer encoder blocks with
multi-head attention:

    Block A: Q = visual,  K,V = audio   -> v_hat
    Block B: Q = audio,   K,V = visual  -> a_hat

The two updated streams are concatenated along the feature axis and
mean-pooled across time before entering the three heads. ``h=256``,
4 heads, 1 block each gives ~0.6 M trainable params at the default
feature dimensions.

References: Tsai et al. (ACL 2019) "Multimodal Transformer"; Praveen
et al. (CVPRW 2024) "Recursive Joint Cross-Modal Attention for SER".
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.fusion.base import FusionConfig, FusionModule, MLPHeads


class _CrossAttnBlock(nn.Module):
    """One Transformer encoder layer with cross-attention Q vs K/V."""

    def __init__(self, hidden: int, num_heads: int, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden)
        self.ln_kv = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_ffn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * hidden, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_ln = self.ln_q(q)
        kv_ln = self.ln_kv(kv)
        attn_out, _ = self.attn(q_ln, kv_ln, kv_ln, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.ln_ffn(q))
        return q


class CrossModalAttention(FusionModule):
    def __init__(
        self,
        cfg: FusionConfig,
        hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(cfg)
        self.hidden = hidden
        self.proj_v = nn.Linear(cfg.v_dim + cfg.scores_dim, hidden)
        self.proj_a = nn.Linear(cfg.a_dim, hidden)
        self.block_v = _CrossAttnBlock(hidden, num_heads=num_heads, dropout=dropout)
        self.block_a = _CrossAttnBlock(hidden, num_heads=num_heads, dropout=dropout)
        self.heads = MLPHeads(2 * hidden, cfg)

    def forward(
        self,
        v_feat: torch.Tensor,
        v_scores: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Accept both per-frame (B, D) and windowed (B, T, D) tensors.
        if v_feat.dim() == 2:
            v_feat = v_feat.unsqueeze(1)
            v_scores = v_scores.unsqueeze(1)
            a_feat = a_feat.unsqueeze(1)

        v_tokens = self.proj_v(torch.cat([v_feat, v_scores], dim=-1))  # (B, T, H)
        a_tokens = self.proj_a(a_feat)                                  # (B, T, H)

        v_hat = self.block_v(v_tokens, a_tokens)
        a_hat = self.block_a(a_tokens, v_tokens)

        z = torch.cat([v_hat.mean(dim=1), a_hat.mean(dim=1)], dim=-1)   # (B, 2H)
        return self.heads(z)
