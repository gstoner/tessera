from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit.executable_layout import (
    DynamicShapeGuardError,
    ExecutableLayout,
    LayoutOrder,
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
