"""PyTorch port of the MTL head defined in cell 44 of mtl.ipynb.

Architecture (identical to the notebook, deliberately tiny):
    x in R^D where D = feature_dim + scores_dim  (e.g. 512 + 10 = 522)

    EXPR head:  slice x[:, :feature_dim]            -> Linear(feature_dim, 8)
                -> softmax,        L2 reg = 1/batch_size
    VA head:    slice x[:, feature_dim:]            -> Linear(scores_dim, 2)
                -> tanh,           L2 reg = 1/batch_size
    AU head:    full x                              -> Linear(D, 128) -> ReLU
                -> Linear(128, 12) -> sigmoid

The three heads are trained independently (cells 47 / 50 / 53), each
with its own optimizer and early-stop-on-val-loss. This module therefore
exposes them as three separate nn.Module classes plus an ``MTLHead``
container that bundles them for evaluation.

Losses:
    * ``loss_va``    : 1 - 0.5 * (CCC(V) + CCC(A))     (cell 37)
    * ``loss_expr``  : class-weighted cross-entropy     (cell 47 via class_weight)
    * ``loss_aus``   : per-AU weighted binary cross-entropy (cell 38)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class ExprHead(nn.Module):
    """EXPR head: takes the *features* slice only (cell 44 Slice(0, D_feat))."""

    def __init__(self, feature_dim: int, num_classes: int = 8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat = x[:, : self.feature_dim]
        return self.fc(x_feat)  # logits; cross-entropy does the softmax


class VAHead(nn.Module):
    """VA head: takes the *scores* slice only (cell 44 Slice(D_feat, 10))."""

    def __init__(self, feature_dim: int, scores_dim: int = 10) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.scores_dim = scores_dim
        self.fc = nn.Linear(scores_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sco = x[:, self.feature_dim : self.feature_dim + self.scores_dim]
        return torch.tanh(self.fc(x_sco))


class AUHead(nn.Module):
    """AU head: takes the full concatenated vector, hidden 128 + ReLU."""

    def __init__(self, input_dim: int, num_aus: int = 12, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_aus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))  # probabilities in (0, 1)


@dataclass
class MTLHeadConfig:
    feature_dim: int = 512
    scores_dim: int = 10
    num_expr: int = 8
    num_aus: int = 12
    au_hidden: int = 128


class MTLHead(nn.Module):
    """Three-head container. Stores the heads as attributes so each one
    can be trained independently with its own optimizer (see ``src.train``)."""

    def __init__(self, cfg: MTLHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.expr = ExprHead(cfg.feature_dim, cfg.num_expr)
        self.va = VAHead(cfg.feature_dim, cfg.scores_dim)
        self.au = AUHead(
            cfg.feature_dim + cfg.scores_dim, cfg.num_aus, cfg.au_hidden
        )

    @property
    def input_dim(self) -> int:
        return self.cfg.feature_dim + self.cfg.scores_dim

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.expr(x), self.va(x), self.au(x)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def ccc_loss_component(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Differentiable CCC on a 1-D tensor pair (matches the Keras ``CCC`` in cell 37)."""
    x = y_true.float()
    y = y_pred.float()
    x_m = x.mean()
    y_m = y.mean()
    vx = x - x_m
    vy = y - y_m
    s_xy = (vx * vy).mean()
    x_v = x.var(unbiased=False)
    y_v = y.var(unbiased=False)
    denom = x_v + y_v + (x_m - y_m) ** 2
    return 2.0 * s_xy / (denom + 1e-8)


def loss_va(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """``1 - 0.5 * (CCC(V) + CCC(A))``; matches cell 37."""
    v_ccc = ccc_loss_component(y_true[:, 0], y_pred[:, 0])
    a_ccc = ccc_loss_component(y_true[:, 1], y_pred[:, 1])
    return 1.0 - 0.5 * (v_ccc + a_ccc)


def loss_expr(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-weighted sparse cross-entropy.

    Equivalent to Keras ``sparse_categorical_crossentropy`` +
    ``class_weight=emo_class_weights`` in cell 47.
    """
    return F.cross_entropy(logits, target.long(), weight=class_weights)


def make_loss_aus(class_weights: np.ndarray):
    """Return a closure computing per-AU weighted BCE (cell 38, ``True`` branch).

    For each sample/AU pair with ground truth ``y`` and probability ``p``,
    the per-element weight is ``w[:, 0]**(1-y) * w[:, 1]**y`` and the loss
    is their mean times the plain BCE.
    """
    w = torch.as_tensor(class_weights, dtype=torch.float32)  # (A, 2)

    def _loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        eps = 1e-7
        ce = -(
            y_true * torch.log(y_pred.clamp(eps, 1.0 - eps))
            + (1.0 - y_true) * torch.log((1.0 - y_pred).clamp(eps, 1.0 - eps))
        )
        weights = w.to(y_pred.device)  # (A, 2)
        w_per = (weights[:, 0] ** (1.0 - y_true)) * (weights[:, 1] ** y_true)
        return (w_per * ce).mean()

    return _loss
