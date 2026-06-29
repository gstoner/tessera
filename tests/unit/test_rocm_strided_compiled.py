"""Compiler-generated 0-move / strided-copy lane on gfx1151 (P4 of
S_SERIES_GAP_CLOSURE_PLAN) — pad / cat / roll / flip / tile / repeat / stack via
the COMPILER-GENERATED masked-gather kernel (generate-rocm-gather-kernel; host
index map). Reachable via `compiler_path="rocm_strided_compiled"`. Validated vs
numpy on gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, kwargs, n_operands):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_strided_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, kwargs, *arrs):
    res = rt.launch(_art(rt, op, kwargs, len(arrs)), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_strided_compiled"
    return np.asarray(res["output"])


X = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
Y = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)


def test_roll_flip():
    rt = _rocm_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.roll", {"shift": 2, "axis": 1}, X),
                                  np.roll(X, 2, axis=1))
    np.testing.assert_array_equal(_run(rt, "tessera.flip", {"axis": 0}, X),
                                  np.flip(X, 0))


def test_tile_repeat():
    rt = _rocm_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.tile", {"reps": [2, 3]}, X),
                                  np.tile(X, [2, 3]))
    np.testing.assert_array_equal(
        _run(rt, "tessera.repeat", {"repeats": 2, "axis": 1}, X),
        np.repeat(X, 2, axis=1))


def test_pad():
    rt = _rocm_or_skip()
    np.testing.assert_array_equal(
        _run(rt, "tessera.pad",
             {"pad_width": [[1, 2], [0, 1]], "mode": "constant",
              "constant_values": 5.0}, X),
        np.pad(X, [[1, 2], [0, 1]], constant_values=5.0))
    np.testing.assert_array_equal(
        _run(rt, "tessera.pad", {"pad_width": [[1, 1], [1, 1]], "mode": "edge"}, X),
        np.pad(X, [[1, 1], [1, 1]], mode="edge"))


def test_cat_stack():
    rt = _rocm_or_skip()
    np.testing.assert_array_equal(_run(rt, "tessera.cat", {"axis": 0}, X, Y),
                                  np.concatenate([X, Y], 0))
    np.testing.assert_array_equal(_run(rt, "tessera.stack", {"axis": 1}, X, X),
                                  np.stack([X, X], 1))
