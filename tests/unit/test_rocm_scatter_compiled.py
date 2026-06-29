"""Scatter lane on AMD ROCm gfx1151 (P8 of S_SERIES_GAP_CLOSURE_PLAN) — the
0-reduce / indexed-store companion to the P4 gather lane: scatter (set) /
scatter_add (sum) / scatter_reduce (min/max) via the COMPILER-GENERATED kernel
(generate-rocm-scatter-kernel; one thread per element; atomic_rmw for
add/min/max). Reachable via `compiler_path="rocm_scatter_compiled"`. Validated
vs the numpy scatter reference on gfx1151. Skip-clean: tessera-opt not built /
no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_scatter_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x", "i", "u"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["x", "i", "u"],
                 "kwargs": kwargs}]})


def _run(rt, op, x, idx, upd, **kwargs):
    res = rt.launch(_art(rt, op, kwargs), (x, idx, upd))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_scatter_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(17)


def test_scatter_set_unique():
    rt = _rocm_or_skip()
    x = _RNG.standard_normal((6, 4)).astype(np.float32)
    idx = np.array([5, 0, 3], np.int64)          # unique → no store race
    upd = _RNG.standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_allclose(_run(rt, "tessera.scatter", x, idx, upd),
                               tessera.ops.scatter(x, idx, upd), rtol=0, atol=0)


def test_scatter_add_duplicates():
    rt = _rocm_or_skip()
    x = _RNG.standard_normal((4, 5)).astype(np.float32)
    idx = np.array([1, 3, 1, 1, 3], np.int64)    # atomic add over duplicates
    upd = _RNG.standard_normal((5, 5)).astype(np.float32)
    np.testing.assert_allclose(_run(rt, "tessera.scatter_add", x, idx, upd),
                               tessera.ops.scatter_add(x, idx, upd),
                               rtol=1e-5, atol=1e-5)


def test_scatter_reduce_min_max():
    rt = _rocm_or_skip()
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
