"""Workstream E — Megatron TP rewrite + cross-rank gradient equivalence.

Proves the automatic rewrite of a plain linear into column/row/sequence parallel
is forward-correct AND that the sharded backward gradients, recombined, equal the
single-rank gradients (the specific gap P2 named). Runs over real MockRankGroup
threads exercising the collectives.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream E).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.tensor_parallel import (
    TPMode, TPSpec, rewrite_linear, verify_tp_gradient_equivalence)
from tessera.testing.mock_collective import MockRankGroup


_MODES = [TPMode.COLUMN, TPMode.ROW, TPMode.SEQUENCE]


def _xw(n=8, cin=8, cout=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, cin)), rng.standard_normal((cin, cout)))


# ── forward correctness across modes ─────────────────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("world_size", [2, 4])
def test_forward_matches_single_rank(mode, world_size):
    X, W = _xw(n=8, cin=8, cout=8, seed=1)
    pl = rewrite_linear(W, TPSpec(mode, world_size))
    group = MockRankGroup(world_size, {"tp": world_size})
    ys = group.run(lambda mr: pl.forward(mr, X))
    Y_ref = X @ W
    for y in ys:
        np.testing.assert_allclose(y, Y_ref, rtol=1e-9, atol=1e-9)


# ── the E2 oracle: cross-rank gradient equivalence ───────────────────────────


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("world_size", [2, 4])
def test_gradient_equivalence(mode, world_size):
    X, W = _xw(n=8, cin=8, cout=8, seed=2)
    verdict = verify_tp_gradient_equivalence(X, W, TPSpec(mode, world_size))
    assert verdict.is_equivalent, verdict.detail
    assert verdict.max_dx_err < 1e-9
    assert verdict.max_dw_err < 1e-9


@pytest.mark.parametrize("mode", _MODES)
def test_gradient_equivalence_rectangular(mode):
    # Non-square shapes: N=12, C_in=16, C_out=8, world=4.
    X, W = _xw(n=12, cin=16, cout=8, seed=3)
    verdict = verify_tp_gradient_equivalence(X, W, TPSpec(mode, 4))
    assert verdict.is_equivalent, verdict.detail


# ── weight sharding shapes ────────────────────────────────────────────────────


def test_column_shards_output_dim():
    _, W = _xw(cin=8, cout=8)
    pl = rewrite_linear(W, TPSpec(TPMode.COLUMN, 4))
    assert pl.weight_shard(0).shape == (8, 2)


def test_row_shards_input_dim():
    _, W = _xw(cin=8, cout=8)
    pl = rewrite_linear(W, TPSpec(TPMode.ROW, 4))
    assert pl.weight_shard(0).shape == (2, 8)


def test_sequence_replicates_weight():
    _, W = _xw(cin=8, cout=8)
    pl = rewrite_linear(W, TPSpec(TPMode.SEQUENCE, 4))
    np.testing.assert_array_equal(pl.weight_shard(2), W)


# ── guards ────────────────────────────────────────────────────────────────────


def test_non_divisible_dim_raises():
    _, W = _xw(cin=8, cout=7)
    pl = rewrite_linear(W, TPSpec(TPMode.COLUMN, 4))
    with pytest.raises(ValueError):
        pl.weight_shard(0)  # 7 cols not divisible by 4


def test_spec_accepts_string_mode():
    assert TPSpec("column", 2).mode is TPMode.COLUMN
