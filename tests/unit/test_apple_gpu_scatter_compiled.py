"""Apple GPU scatter lane — scatter / scatter_add / scatter_reduce.

Row-wise indexed store along an axis (set / sum / min / max). Apple ships no
device scatter kernel, so it runs the numpy reference the x86/ROCm device
kernels are matched against. Reachable via
`compiler_path="apple_gpu_scatter_compiled"`; execution_kind=reference_cpu.
Validated vs tessera.ops — parity with test_x86_scatter_compiled.
"""

from __future__ import annotations

import numpy as np

from tessera import ops as O
from tessera import runtime as rt

_RNG = np.random.default_rng(0)


def _run(op, x, idx, upd, **kwargs):
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_scatter_compiled",
        "executable": True, "execution_kind": "reference_cpu",
        "arg_names": ["x", "i", "u"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["x", "i", "u"],
                 "kwargs": dict(kwargs)}]})
    res = rt.launch(art, (x, idx, upd))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_scatter_compiled"
    assert res["execution_kind"] == "reference_cpu"
    return np.asarray(res["output"])


def test_scatter_set_unique():
    x = _RNG.standard_normal((6, 4)).astype(np.float32)
    idx = np.array([5, 0, 3], np.int64)
    upd = _RNG.standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.scatter", x, idx, upd),
                               O.scatter(x, idx, upd), rtol=0, atol=0)


def test_scatter_add_duplicates():
    x = _RNG.standard_normal((4, 5)).astype(np.float32)
    idx = np.array([1, 3, 1, 1, 3], np.int64)
    upd = _RNG.standard_normal((5, 5)).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.scatter_add", x, idx, upd),
                               O.scatter_add(x, idx, upd), rtol=1e-5, atol=1e-5)


def test_scatter_reduce_min_max():
    x = _RNG.standard_normal((4, 3)).astype(np.float32)
    idx = np.array([0, 0, 2, 2, 1], np.int64)
    upd = _RNG.standard_normal((5, 3)).astype(np.float32)
    for reduce in ("min", "max"):
        np.testing.assert_allclose(
            _run("tessera.scatter_reduce", x, idx, upd, reduce=reduce),
            O.scatter_reduce(x, idx, upd, reduce=reduce), rtol=1e-5, atol=1e-5)
