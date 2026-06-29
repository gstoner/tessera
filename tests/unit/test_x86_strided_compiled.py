"""Compiler-generated 0-move / strided-copy lane on x86 AVX-512 (P4 of
S_SERIES_GAP_CLOSURE_PLAN) — pad / cat / roll / flip / tile / repeat / stack.
The host computes the integer index map from the op's numpy semantics on an
arange grid (shape arithmetic only); the AVX-512 masked-gather kernel
(tessera_x86_gather_f32) does the f32 data movement. Reachable via
`compiler_path="x86_strided_compiled"`. Validated vs numpy. Skip-clean:
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


def _art(rt, op, kwargs, n_operands):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_strided_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, kwargs, *arrs):
    res = rt.launch(_art(rt, op, kwargs, len(arrs)), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_strided_compiled"
    return np.asarray(res["output"])


X = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
Y = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)


def test_roll():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.roll", {"shift": 2, "axis": 1}, X),
                                  np.roll(X, 2, axis=1))
    np.testing.assert_array_equal(_run(rt, "tessera.roll", {"shift": -3}, X),
                                  np.roll(X, -3))


def test_flip():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.flip", {"axis": 0}, X),
                                  np.flip(X, 0))


def test_tile_and_repeat():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.tile", {"reps": [2, 3]}, X),
                                  np.tile(X, [2, 3]))
    np.testing.assert_array_equal(
        _run(rt, "tessera.repeat", {"repeats": 2, "axis": 1}, X),
        np.repeat(X, 2, axis=1))


def test_pad_constant_and_edge():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(
        _run(rt, "tessera.pad",
             {"pad_width": [[1, 2], [0, 1]], "mode": "constant",
              "constant_values": 5.0}, X),
        np.pad(X, [[1, 2], [0, 1]], constant_values=5.0))
    np.testing.assert_array_equal(
        _run(rt, "tessera.pad", {"pad_width": [[1, 1], [1, 1]], "mode": "edge"}, X),
        np.pad(X, [[1, 1], [1, 1]], mode="edge"))


def test_cat_and_stack():
    rt = _rt_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.cat", {"axis": 0}, X, Y),
                                  np.concatenate([X, Y], 0))
    np.testing.assert_array_equal(_run(rt, "tessera.stack", {"axis": 1}, X, X),
                                  np.stack([X, X], 1))
