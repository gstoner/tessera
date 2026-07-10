"""Apple GPU reduce lane — sum.

sum over an axis (default: all) with keepdims runs genuinely on the MPSGraph
reduce lane (numpy fallback when Metal is unavailable). Reachable via
`compiler_path="apple_gpu_reduce_compiled"`; execution_kind=native_gpu.
Validated vs numpy.sum — parity with x86/rocm_reduce_compiled.
"""

from __future__ import annotations

import numpy as np

from tessera import runtime as rt


def _run(x, kwargs):
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_reduce_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": "tessera.sum", "result": "o", "operands": ["x"],
                 "kwargs": dict(kwargs)}]})
    res = rt.launch(art, (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_reduce_compiled"
    assert res["execution_kind"] == "native_gpu"
    return np.asarray(res["output"])


def test_sum_all_and_axis():
    x = np.random.default_rng(0).standard_normal((3, 4, 5)).astype(np.float32)
    np.testing.assert_allclose(_run(x, {}), x.sum(), atol=1e-3, rtol=1e-5)
    for axis in (0, 1, 2, -1):
        np.testing.assert_allclose(_run(x, {"axis": axis}), x.sum(axis=axis),
                                   atol=1e-4, rtol=1e-5)


def test_sum_keepdims_and_multiaxis():
    x = np.random.default_rng(1).standard_normal((2, 3, 4)).astype(np.float32)
    np.testing.assert_allclose(_run(x, {"axis": 1, "keepdims": True}),
                               x.sum(axis=1, keepdims=True), atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(_run(x, {"axis": (0, 2)}), x.sum(axis=(0, 2)),
                               atol=1e-4, rtol=1e-5)
