"""Apple GPU scatter lane — scatter / scatter_add / scatter_reduce.

Row-wise indexed store along an axis (set / sum / min / max). Dense contiguous
f32 inputs run through the deterministic Apple Metal scatter ABI; unsupported
contracts truthfully use the reference override. Validated against tessera.ops
and the x86/ROCm scatter semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops as O
from tessera import runtime as rt
from tests._support.apple import assert_native_apple_gpu, assert_reference_cpu

_RNG = np.random.default_rng(0)


def _run(op, x, idx, upd, **kwargs):
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_scatter_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x", "i", "u"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["x", "i", "u"],
                 "kwargs": dict(kwargs)}]})
    res = rt.launch(art, (x, idx, upd))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_scatter_compiled"
    return np.asarray(res["output"]), res


def test_scatter_set_unique():
    x = _RNG.standard_normal((6, 4)).astype(np.float32)
    idx = np.array([5, 0, 3], np.int64)
    upd = _RNG.standard_normal((3, 4)).astype(np.float32)
    out, _ = _run("tessera.scatter", x, idx, upd)
    np.testing.assert_allclose(out,
                               O.scatter(x, idx, upd), rtol=0, atol=0)


def test_scatter_add_duplicates():
    x = _RNG.standard_normal((4, 5)).astype(np.float32)
    idx = np.array([1, 3, 1, 1, 3], np.int64)
    upd = _RNG.standard_normal((5, 5)).astype(np.float32)
    out, _ = _run("tessera.scatter_add", x, idx, upd)
    np.testing.assert_allclose(out,
                               O.scatter_add(x, idx, upd), rtol=1e-5, atol=1e-5)


def test_scatter_reduce_min_max():
    x = _RNG.standard_normal((4, 3)).astype(np.float32)
    idx = np.array([0, 0, 2, 2, 1], np.int64)
    upd = _RNG.standard_normal((5, 3)).astype(np.float32)
    for reduce in ("min", "max"):
        out, _ = _run("tessera.scatter_reduce", x, idx, upd, reduce=reduce)
        np.testing.assert_allclose(out,
            O.scatter_reduce(x, idx, upd, reduce=reduce), rtol=1e-5, atol=1e-5)


@pytest.mark.hardware_apple_gpu
def test_scatter_f32_reports_native_gpu_on_metal():
    x = np.zeros((4, 3), np.float32)
    idx = np.array([1, 1, 3], np.int64)
    upd = np.ones((3, 3), np.float32)
    out, res = _run("tessera.scatter_add", x, idx, upd)
    assert_native_apple_gpu(res, compiler_path="apple_gpu_scatter_compiled")
    np.testing.assert_allclose(out, O.scatter_add(x, idx, upd))


def test_scatter_non_f32_uses_reference_cpu_override():
    x = np.zeros((3, 2), np.float64)
    idx = np.array([1], np.int64)
    upd = np.ones((1, 2), np.float64)
    _, res = _run("tessera.scatter", x, idx, upd)
    assert_reference_cpu(res)


def test_scatter_out_of_range_preserves_reference_error():
    x = np.zeros((3, 2), np.float32)
    idx = np.array([3], np.int64)
    upd = np.ones((1, 2), np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_scatter_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x", "i", "u"], "output_name": "o",
        "ops": [{"op_name": "tessera.scatter", "result": "o",
                 "operands": ["x", "i", "u"], "kwargs": {}}],
    })
    res = rt.launch(art, (x, idx, upd))
    assert res["ok"] is False
    assert "out of bounds" in res["reason"]
