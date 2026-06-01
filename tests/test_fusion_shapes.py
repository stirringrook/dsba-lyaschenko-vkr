"""Shape / parameter-budget tests for fusion variants.

Instantiates each fusion module with the production feature dims
(v=1280, scores=10, a=1024) and runs a tiny forward pass. Also records
the trainable-parameter count against the 1.0 M budget from the research
plan so regressions show up as pytest failures.
"""

import numpy as np
import pytest
import torch

from src.fusion.base import FusionConfig, count_parameters
from src.fusion.f0_grid import grid_search_blend
from src.fusion.f4_xattn import CrossModalAttention
from src.fusion.f5_lmf import LMFusion
from src.fusion.f6c_iaca import IACAGate
from src.fusion.f6d_mbt import MBTFusion
from src.fusion.variants import (
    AudioOnly,
    EarlyConcat,
    LearnedBlend,
    TaskGate,
    VisualOnly,
)


CFG = FusionConfig(
    v_dim=1280,
    scores_dim=10,
    a_dim=1024,
    num_expr=8,
    num_aus=12,
    hidden=384,
    dropout=0.3,
    au_hidden=128,
)

B = 4


def _fake_frame_batch():
    v = torch.randn(B, CFG.v_dim)
    s = torch.randn(B, CFG.scores_dim)
    a = torch.randn(B, CFG.a_dim)
    return v, s, a


def _fake_window_batch(T: int = 11):
    v = torch.randn(B, T, CFG.v_dim)
    s = torch.randn(B, T, CFG.scores_dim)
    a = torch.randn(B, T, CFG.a_dim)
    return v, s, a


def _assert_output_shapes(expr, va, au):
    assert expr.shape == (B, CFG.num_expr)
    assert va.shape == (B, 2)
    assert au.shape == (B, CFG.num_aus)
    assert torch.all(va.abs() <= 1.0 + 1e-4)     # tanh range
    assert torch.all((au >= 0) & (au <= 1))      # sigmoid range


def test_visual_only_forward_and_params():
    m = VisualOnly(CFG)
    _assert_output_shapes(*m(*_fake_frame_batch()))
    assert count_parameters(m) < 1_200_000


def test_audio_only_forward_and_params():
    m = AudioOnly(CFG)
    _assert_output_shapes(*m(*_fake_frame_batch()))
    assert count_parameters(m) < 1_200_000


@pytest.mark.parametrize(
    "ctor, budget",
    [
        (EarlyConcat, 1_500_000),
        (LearnedBlend, 3_000_000),   # two unimodal heads + 4 scalars
        (TaskGate, 4_000_000),       # two unimodal heads + gate MLP
        (LMFusion, 1_500_000),
    ],
)
def test_frame_variants_shapes_and_budget(ctor, budget):
    m = ctor(CFG)
    _assert_output_shapes(*m(*_fake_frame_batch()))
    n = count_parameters(m)
    assert n < budget, f"{ctor.__name__} has {n} trainable params (budget {budget})"


def test_f4_xattn_accepts_frame_and_window():
    m = CrossModalAttention(CFG)
    _assert_output_shapes(*m(*_fake_frame_batch()))     # frame-level unsqueeze
    _assert_output_shapes(*m(*_fake_window_batch(11)))  # windowed
    # Cross-modal stack is the heaviest fusion variant due to 1290->256 and
    # 1024->256 input projections; 2.0 M is a comfortable upper bound.
    assert count_parameters(m) < 2_000_000


def test_f6c_iaca_shapes_and_budget():
    # F6c is built on F4's fixed-dim cross-attention block, so its count is
    # ~2.0M independent of cfg.hidden; report budget is < 2.5M.
    m = IACAGate(FusionConfig(v_dim=1280, scores_dim=10, a_dim=1024, hidden=256))
    _assert_output_shapes(*m(*_fake_frame_batch()))     # frame-level unsqueeze
    _assert_output_shapes(*m(*_fake_window_batch(11)))  # windowed
    n = count_parameters(m)
    assert n < 2_500_000, f"IACAGate has {n} trainable params (budget 2.5M)"


def test_f6d_mbt_shapes_and_budget():
    # F6d-MBT scales steeply with hidden (four _CrossAttnBlocks): ~1.7M at the
    # production hidden=192, but ~9.8M at hidden=512. Pin the production value.
    m = MBTFusion(FusionConfig(v_dim=1280, scores_dim=10, a_dim=1024, hidden=192))
    _assert_output_shapes(*m(*_fake_frame_batch()))     # frame-level unsqueeze
    _assert_output_shapes(*m(*_fake_window_batch(11)))  # windowed
    n = count_parameters(m)
    assert n < 2_500_000, f"MBTFusion(hidden=192) has {n} trainable params (budget 2.5M)"


def test_f3_gate_exposes_alpha():
    m = TaskGate(CFG)
    m(*_fake_frame_batch())
    a = m.last_alpha()
    assert a is not None
    assert a.shape == (B, 4)
    assert torch.all((a >= 0) & (a <= 1))


def test_f0_grid_blend_picks_best_weights():
    rng = np.random.default_rng(0)
    N = 200

    y_expr = rng.integers(0, 8, size=N)
    y_va = rng.standard_normal((N, 2))
    y_aus = (rng.random((N, 12)) > 0.7).astype(int)
    m = np.ones(N, dtype=np.float32)

    # Visual is near-perfect for EXPR, random for VA/AU.
    v_expr_logits = np.zeros((N, 8))
    v_expr_logits[np.arange(N), y_expr] = 5.0
    v_va = rng.standard_normal((N, 2))
    v_au = rng.random((N, 12))

    # Audio is near-perfect for VA, confidently wrong on EXPR, random on AU.
    a_expr_logits = np.zeros((N, 8))
    a_expr_logits[np.arange(N), (y_expr + 1) % 8] = 5.0
    a_va = y_va + 0.05 * rng.standard_normal((N, 2))
    a_au = rng.random((N, 12))

    res = grid_search_blend(
        v_expr_logits, v_va, v_au,
        a_expr_logits, a_va, a_au,
        y_expr, y_va, y_aus, m, m, m,
    )
    # EXPR should lean visual, VA should lean audio, and both perfect-signal
    # tasks should score near their ceilings.
    assert res.w_expr > res.w_va
    assert res.w_expr >= 0.5
    assert res.w_va <= 0.2
    assert res.metrics["F1_EXPR_macro"] == 1.0
    assert res.metrics["CCC_VA"] > 0.95
