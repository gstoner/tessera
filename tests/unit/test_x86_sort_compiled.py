"""Sort lane on x86 AVX-512 (P9 of S_SERIES_GAP_CLOSURE_PLAN) — sort / argsort /
top_k via the data-independent BITONIC sort network kernel
(tessera_x86_bitonic_sort_kv_f32; AVX-512 wide stages + scalar tail). The host
moves the sort axis last, pads each row to a power of two with +INF sentinels,
runs the device key+index sort ascending, then trims/flips host-side. Reachable
via `compiler_path="x86_sort_compiled"`. Validated vs numpy (distinct values so
the non-stable network agrees with numpy's argsort). Skip-clean:
libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_sort_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a0"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["a0"],
                 "kwargs": kwargs}]})


def _run(rt, op, kwargs, x):
    res = rt.launch(_art(rt, op, kwargs), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_sort_compiled"
    return res["output"]


# distinct values per row (a permutation) so argsort is unambiguous
_RNG = np.random.default_rng(7)
X = _RNG.permutation(60).astype(np.float32).reshape(4, 15)
X1 = _RNG.permutation(31).astype(np.float32)        # non-power-of-two row


def test_sort_ascending():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.sort", {}, X),
                                  np.sort(X, axis=-1))
    np.testing.assert_array_equal(_run(rt, "tessera.sort", {}, X1),
                                  np.sort(X1, axis=-1))


def test_sort_descending():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(
        _run(rt, "tessera.sort", {"descending": True}, X),
        np.flip(np.sort(X, axis=-1), axis=-1))


def test_sort_axis0():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.sort", {"axis": 0}, X),
                                  np.sort(X, axis=0))


def test_argsort():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.argsort", {}, X),
                                  np.argsort(X, axis=-1))
    np.testing.assert_array_equal(
        _run(rt, "tessera.argsort", {"descending": True}, X),
        np.flip(np.argsort(X, axis=-1), axis=-1))


def test_top_k():
    rt = _rt_or_skip()
    vals, idx = _run(rt, "tessera.top_k", {"k": 4}, X)
    order = np.argsort(X, axis=-1)
    ref_idx = np.flip(np.take(order, np.arange(15 - 4, 15), axis=-1), axis=-1)
    ref_vals = np.take_along_axis(X, ref_idx, axis=-1)
    np.testing.assert_array_equal(vals, ref_vals)
    np.testing.assert_array_equal(idx, ref_idx)
