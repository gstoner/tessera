from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit.executable_layout import (
    DynamicAttentionContract,
    DynamicKVCacheContract,
    DynamicReductionContract,
    DynamicShapeGuardError,
    DynamicSoftmaxContract,
    ExecutableLayout,
    LayoutOrder,
    guard_dynamic_attention,
    guard_dynamic_kv_cache,
    guard_dynamic_last_axis_reduction,
    guard_dynamic_last_axis_softmax,
    guard_dynamic_matmul,
    materialize_layout,
)


def test_row_and_column_layouts_are_physically_materialized():
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    strided = base[:, ::2]
    row = materialize_layout(
        strided, ExecutableLayout("A", LayoutOrder.ROW_MAJOR, 2)
    )
    col = materialize_layout(
        strided, ExecutableLayout("A", LayoutOrder.COLUMN_MAJOR, 2)
    )
    np.testing.assert_array_equal(row, strided)
    np.testing.assert_array_equal(col, strided)
    assert row.flags.c_contiguous
    assert col.flags.f_contiguous


def test_layout_rank_is_a_launch_contract():
    with pytest.raises(ValueError, match="requires rank 2"):
        materialize_layout(
            np.zeros(8, np.float32),
            ExecutableLayout("A", LayoutOrder.ROW_MAJOR, 2),
        )


def test_dynamic_matmul_guard_accepts_ragged_runtime_shape():
    a = np.zeros((13, 7), np.float32)
    b = np.zeros((7, 11), np.float32)
    bias = np.zeros(11, np.float32)
    residual = np.zeros((13, 11), np.float32)
    assert guard_dynamic_matmul(
        a, b, bias=bias, residual=residual,
        require_bias=True, require_residual=True,
    ) == (13, 11, 7)


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "message"),
    [
        ((2, 3, 4), (4, 5), "rank-2"),
        ((2, 3), (4, 5), "contracting dimensions"),
        ((0, 3), (3, 5), "must be positive"),
    ],
)
def test_dynamic_matmul_guard_rejects_unsafe_shapes(a_shape, b_shape, message):
    with pytest.raises(DynamicShapeGuardError, match=message):
        guard_dynamic_matmul(
            np.zeros(a_shape, np.float32), np.zeros(b_shape, np.float32)
        )


def test_dynamic_matmul_guard_rejects_side_buffer_shape():
    with pytest.raises(DynamicShapeGuardError, match="bias must have shape"):
        guard_dynamic_matmul(
            np.zeros((3, 4), np.float32),
            np.zeros((4, 5), np.float32),
            bias=np.zeros(4, np.float32),
        )


def test_dynamic_last_axis_reduction_contract_is_shape_exact():
    x = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    assert guard_dynamic_last_axis_reduction(x) == DynamicReductionContract(
        outer=6, axis_extent=5, output_shape=(2, 3)
    )
    assert guard_dynamic_last_axis_reduction(
        x, keepdims=True
    ).output_shape == (2, 3, 1)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.array(1.0, dtype=np.float32), "rank >= 1"),
        (np.empty((2, 0), dtype=np.float32), "must be positive"),
    ],
)
def test_dynamic_last_axis_reduction_rejects_invalid_shapes(value, message):
    with pytest.raises(DynamicShapeGuardError, match=message):
        guard_dynamic_last_axis_reduction(value)


def test_dynamic_last_axis_softmax_contract_preserves_shape():
    x = np.zeros((2, 3, 7), np.float32)
    assert guard_dynamic_last_axis_softmax(x) == DynamicSoftmaxContract(
        outer=6, axis_extent=7, output_shape=(2, 3, 7)
    )


@pytest.mark.parametrize(
    "value",
    [np.array(1.0, np.float32), np.empty((2, 0), np.float32)],
)
def test_dynamic_last_axis_softmax_rejects_invalid_shapes(value):
    with pytest.raises(DynamicShapeGuardError):
        guard_dynamic_last_axis_softmax(value)


def test_dynamic_attention_contract_accepts_ragged_gqa():
    q = np.zeros((2, 8, 13, 16), np.float32)
    k = np.zeros((2, 2, 21, 16), np.float32)
    v = np.zeros((2, 2, 21, 24), np.float32)
    assert guard_dynamic_attention(q, k, v) == DynamicAttentionContract(
        batch_heads=16,
        query_extent=13,
        key_extent=21,
        query_key_width=16,
        value_width=24,
        output_shape=(2, 8, 13, 24),
    )


@pytest.mark.parametrize(
    ("q", "k", "v", "message"),
    [
        (
            np.zeros((0, 8), np.float32),
            np.zeros((4, 8), np.float32),
            np.zeros((4, 8), np.float32),
            "positive",
        ),
        (
            np.zeros((2, 4, 8), np.float32),
            np.zeros((2, 4, 7), np.float32),
            np.zeros((2, 4, 8), np.float32),
            "Q/K widths",
        ),
        (
            np.zeros((1, 3, 4, 8), np.float32),
            np.zeros((1, 2, 4, 8), np.float32),
            np.zeros((1, 2, 4, 8), np.float32),
            "divisible GQA",
        ),
    ],
)
def test_dynamic_attention_rejects_invalid_shapes(q, k, v, message):
    with pytest.raises(DynamicShapeGuardError, match=message):
        guard_dynamic_attention(q, k, v)


def test_dynamic_kv_cache_contract_tracks_growing_capacity():
    cache = np.zeros((31, 3, 8), np.float32)
    rows = np.zeros((7, 3, 8), np.float32)
    assert guard_dynamic_kv_cache(
        cache, rows=rows, start=11
    ) == DynamicKVCacheContract(31, 24, (31, 3, 8))
    assert guard_dynamic_kv_cache(
        cache, start=5, end=19
    ).output_shape == (14, 3, 8)


def test_dynamic_kv_cache_rejects_tail_and_bounds_mismatches():
    cache = np.zeros((16, 2, 4), np.float32)
    with pytest.raises(DynamicShapeGuardError, match="tail shape"):
        guard_dynamic_kv_cache(
            cache, rows=np.zeros((2, 3, 4), np.float32), start=0
        )
    with pytest.raises(DynamicShapeGuardError, match="out of bounds"):
        guard_dynamic_kv_cache(cache, start=14, end=17)
    with pytest.raises(DynamicShapeGuardError, match="current sequence"):
        guard_dynamic_kv_cache(cache, current_sequence=17, limit=4)
