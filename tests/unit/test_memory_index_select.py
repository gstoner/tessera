"""LSA selector — ``memory_index_select`` (sigmoid-threshold block selection).

Covers the Phase-1 contract (D3): threshold selection, union across indexer
layers, empty-selection fallback, deterministic tie behaviour (``>=``), causal
masking, and the non-differentiable coverage status. See
``docs/audit/domain/archive/lsa_scope.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import lsa
from tessera.compiler.op_catalog import get_op_spec
from tessera.compiler.primitive_coverage import coverage_for


def _keys_query(seed=0, B=2, H=2, S=16, D=8, block_size=4):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((B, H, S, D))
    Q = rng.standard_normal((B, H, S, D))
    keys = lsa.compress_block_keys(K, block_size=block_size)
    return keys, Q


def test_in_op_catalog():
    spec = get_op_spec("memory_index_select")
    assert spec is not None
    assert spec.graph_name == "tessera.memory_index_select"


def test_coverage_is_non_differentiable_selector():
    cov = coverage_for("memory_index_select")
    cs = cov.contract_status
    # Boolean-mask output ⇒ gradient is undefined on the primary output.
    assert cs["vjp"] == "non_differentiable"
    assert cs["jvp"] == "non_differentiable"
    # The closed-form selector has determinate math/shape/dtype + shipped tests.
    assert cs["math_semantics"] == "complete"
    assert cs["shape_rule"] == "complete"
    assert cs["tests"] == "complete"


def test_returns_bool_mask_of_expected_shape():
    keys, Q = _keys_query()
    mask = ts.ops.memory_index_select(keys, Q, block_size=4)
    assert mask.dtype == np.bool_
    assert mask.shape == (2, 2, 16, 4)  # (B, H, S_q, num_blocks)


def test_threshold_monotone_lower_selects_superset():
    keys, Q = _keys_query(seed=1)
    low = lsa.memory_index_select(keys, Q, block_size=4, threshold=0.2, fallback_local=False).mask
    high = lsa.memory_index_select(keys, Q, block_size=4, threshold=0.8, fallback_local=False).mask
    # Every block selected at the higher threshold is also selected at the lower.
    assert np.all(high <= low)


def test_tie_at_threshold_is_selected():
    # Orthogonal key/query → score 0 → sigmoid(0) == 0.5 exactly. With `>=`
    # the exact-threshold tie must be selected; a hair above must deselect.
    keys = np.zeros((1, 1, 2, 2))
    keys[0, 0, 0] = [1.0, 0.0]   # block 0 key
    keys[0, 0, 1] = [1.0, 0.0]   # block 1 key
    Q = np.zeros((1, 1, 1, 2))
    Q[0, 0, 0] = [0.0, 1.0]      # orthogonal to both keys → score 0
    selected = lsa.memory_index_select(
        keys, Q, block_size=1, threshold=0.5, causal=False, fallback_local=False).mask
    assert bool(selected.all())  # tie (sigmoid==0.5) selected under `>=`
    above = lsa.memory_index_select(
        keys, Q, block_size=1, threshold=0.5 + 1e-9, causal=False, fallback_local=False).mask
    assert not bool(above.any())


def test_union_across_indexer_layers():
    keys, Q = _keys_query(seed=2)
    keys2, _ = _keys_query(seed=99)
    m1 = lsa.memory_index_select(keys, Q, block_size=4, fallback_local=False).mask
    m2 = lsa.memory_index_select(keys2, Q, block_size=4, fallback_local=False).mask
    union = lsa.memory_index_select([keys, keys2], Q, block_size=4, fallback_local=False).mask
    np.testing.assert_array_equal(union, m1 | m2)


def test_empty_selection_falls_back_to_own_block():
    keys, Q = _keys_query(seed=3)
    # threshold=1.0 selects nothing → fallback puts each query in its own block.
    mask = lsa.memory_index_select(keys, Q, block_size=4, threshold=1.0, causal=True).mask
    assert bool(mask.any(axis=-1).all())          # no empty rows
    assert bool((mask.sum(axis=-1) == 1).all())   # exactly the own block
    B, H, S, nb = mask.shape
    own = (np.arange(S) // 4)
    for b in range(B):
        for h in range(H):
            for sq in range(S):
                assert mask[b, h, sq, own[sq]]


def test_causal_never_selects_future_blocks():
    keys, Q = _keys_query(seed=4)
    mask = lsa.memory_index_select(keys, Q, block_size=4, threshold=0.0, causal=True).mask
    B, H, S, nb = mask.shape
    qb = np.arange(S) // 4
    future = np.arange(nb)[None, None, None, :] > qb[None, None, :, None]
    assert int((mask & future).sum()) == 0


def test_deterministic():
    keys, Q = _keys_query(seed=5)
    a = ts.ops.memory_index_select(keys, Q, block_size=4)
    b = ts.ops.memory_index_select(keys, Q, block_size=4)
    np.testing.assert_array_equal(a, b)


def test_rejects_bad_threshold():
    keys, Q = _keys_query()
    with pytest.raises(ValueError):
        lsa.memory_index_select(keys, Q, block_size=4, threshold=1.5)
