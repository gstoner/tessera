"""Scatter lane on x86 AVX-512 (P8 of S_SERIES_GAP_CLOSURE_PLAN) — the
0-reduce / indexed-store companion to the P4 gather lane: scatter (set) /
scatter_add (sum) / scatter_reduce (min/max), row-wise along an axis with 1-D
indices (tessera_x86_scatter_f32). Reachable via
`compiler_path="x86_scatter_compiled"`. Validated vs the numpy scatter
reference. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_scatter_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x", "i", "u"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["x", "i", "u"],
                 "kwargs": kwargs}]})


def _run(rt, op, x, idx, upd, **kwargs):
    res = rt.launch(_art(rt, op, kwargs), (x, idx, upd))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_scatter_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(17)


def test_scatter_set_unique():
    rt = _rt_or_skip()
    x = _RNG.standard_normal((6, 4)).astype(np.float32)
    idx = np.array([5, 0, 3], np.int64)          # unique → deterministic set
    upd = _RNG.standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_allclose(_run(rt, "tessera.scatter", x, idx, upd),
                               tessera.ops.scatter(x, idx, upd), rtol=0, atol=0)


def test_scatter_add_duplicates():
    rt = _rt_or_skip()
    x = _RNG.standard_normal((4, 5)).astype(np.float32)
    idx = np.array([1, 3, 1, 1, 3], np.int64)    # duplicates accumulate
    upd = _RNG.standard_normal((5, 5)).astype(np.float32)
    np.testing.assert_allclose(_run(rt, "tessera.scatter_add", x, idx, upd),
                               tessera.ops.scatter_add(x, idx, upd),
                               rtol=1e-5, atol=1e-5)


def test_scatter_reduce_min_max():
    rt = _rt_or_skip()
    x = _RNG.standard_normal((4, 3)).astype(np.float32)
    idx = np.array([0, 0, 2, 2, 1], np.int64)
    upd = _RNG.standard_normal((5, 3)).astype(np.float32)
    np.testing.assert_allclose(
        _run(rt, "tessera.scatter_reduce", x, idx, upd, reduce="min"),
        tessera.ops.scatter_reduce(x, idx, upd, reduce="min"),
        rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        _run(rt, "tessera.scatter_reduce", x, idx, upd, reduce="max"),
        tessera.ops.scatter_reduce(x, idx, upd, reduce="max"),
        rtol=1e-5, atol=1e-5)


def test_scatter_add_axis1():
    rt = _rt_or_skip()
    x = _RNG.standard_normal((3, 6)).astype(np.float32)
    idx = np.array([0, 5, 0], np.int64)
    upd = _RNG.standard_normal((3, 3)).astype(np.float32)
    np.testing.assert_allclose(
        _run(rt, "tessera.scatter_add", x, idx, upd, axis=1),
        tessera.ops.scatter_add(x, idx, upd, axis=1), rtol=1e-5, atol=1e-5)


def test_scatter_add_broadcast_row_update():
    """A single (row_len,) update broadcasts across all selected rows —
    matching numpy np.add.at / scatter assignment."""
    rt = _rt_or_skip()
    x = _RNG.standard_normal((4, 2)).astype(np.float32)
    idx = np.array([1, 3, 1], np.int64)
    upd = _RNG.standard_normal((2,)).astype(np.float32)     # one shared row
    np.testing.assert_allclose(_run(rt, "tessera.scatter_add", x, idx, upd),
                               tessera.ops.scatter_add(x, idx, upd),
                               rtol=1e-5, atol=1e-5)


def test_scatter_set_broadcast_keepdim_update():
    """A (1, row_len) update broadcasts across selected rows for set mode."""
    rt = _rt_or_skip()
    x = _RNG.standard_normal((5, 3)).astype(np.float32)
    idx = np.array([0, 4], np.int64)                        # unique
    upd = _RNG.standard_normal((1, 3)).astype(np.float32)
    np.testing.assert_allclose(_run(rt, "tessera.scatter", x, idx, upd),
                               tessera.ops.scatter(x, idx, upd), rtol=0, atol=0)


def test_out_of_range_index_skipped():
    rt = _rt_or_skip()
    x = np.zeros((3, 2), np.float32)
    idx = np.array([0, 9, 2], np.int64)          # 9 is OOB → skipped
    upd = np.ones((3, 2), np.float32)
    got = _run(rt, "tessera.scatter_add", x, idx, upd)
    expect = np.zeros((3, 2), np.float32)
    expect[0] = 1.0
    expect[2] = 1.0
    np.testing.assert_array_equal(got, expect)
