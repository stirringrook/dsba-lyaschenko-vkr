"""Small per-frame fusion variants: unimodal controls + F1/F2/F3.

Collects the four lightweight :class:`~src.fusion.base.FusionModule`
subclasses that share the same per-frame ``(v_feat, v_scores, a_feat)``
contract and the shared :class:`~src.fusion.base.MLPHeads` block:

* :class:`VisualOnly` / :class:`AudioOnly` -- the F0 unimodal controls,
  wrapping the Stage 1 / Stage 2 heads into the FusionModule API.
* :class:`EarlyConcat` (F1) -- early concatenation + MLP.
* :class:`LearnedBlend` (F2) -- late fusion with a learned per-task scalar blend.
* :class:`TaskGate` (F3) -- task-specific per-frame modality gating.

The larger, algorithmically distinct variants (F4 xattn, F5 LMF, the F6
family, F0 grid) keep their own modules.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

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


class EarlyConcat(FusionModule):
    """F1 - Early concatenation + MLP.

    ``[v_feat, v_scores, a_feat] -> LN -> Linear(hidden) -> GELU -> Dropout
    -> Linear(hidden//2) -> GELU -> Dropout -> three heads``. Sized to fit
    the 1.0 M trainable-parameter Stage 3 budget; for the primary
    configuration (v_dim=1280, a_dim=1024, scores=10), hidden=384 gives
    ~1.0 M trainable params including the three heads.
    """

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


class LearnedBlend(FusionModule):
    """F2 - Late fusion with a learned per-task scalar blend.

    Two unimodal heads plus four learned blend scalars (one per task
    {EXPR, V, A, AU}), each passed through sigmoid. Blends happen at the
    logit / tanh / sigmoid layer (AU is already a probability; we blend
    pre-sigmoid logits for stability).
    """

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


class TaskGate(FusionModule):
    """F3 - Task-specific per-frame modality gating (direction D3).

    A small gate MLP reads the concatenated embedding and emits a per-task,
    per-frame gate vector ``alpha in [0,1]^4`` covering the four ABAW tasks
    {EXPR, V, A, AU}. Final output per task ``t`` is
    ``alpha_t * visual_logit_t + (1 - alpha_t) * audio_logit_t``.

    The learned ``alpha`` tensor is accessible via :meth:`last_alpha` for
    the Chapter 5 interpretability figure (mean alpha per task across the
    validation split, per-class boxplots, etc.).
    """

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
