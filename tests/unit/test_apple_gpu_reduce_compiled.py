"""Apple GPU reduce lane — sum.

sum over an axis (default: all) with keepdims runs genuinely on the MPSGraph
reduce lane (numpy fallback when Metal is unavailable). Reachable via
`compiler_path="apple_gpu_reduce_compiled"`; execution_kind=native_gpu.
Validated vs numpy.sum — parity with x86/rocm_reduce_compiled.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tests._support.apple import assert_native_apple_gpu, assert_reference_cpu


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
    return res


def _output(x, kwargs):
    return np.asarray(_run(x, kwargs)["output"])


def test_sum_all_and_axis():
    x = np.random.default_rng(0).standard_normal((3, 4, 5)).astype(np.float32)
    np.testing.assert_allclose(_output(x, {}), x.sum(), atol=1e-3, rtol=1e-5)
    for axis in (0, 1, 2, -1):
        np.testing.assert_allclose(_output(x, {"axis": axis}), x.sum(axis=axis),
                                   atol=1e-4, rtol=1e-5)


def test_sum_keepdims_and_multiaxis():
    x = np.random.default_rng(1).standard_normal((2, 3, 4)).astype(np.float32)
    np.testing.assert_allclose(_output(x, {"axis": 1, "keepdims": True}),
                               x.sum(axis=1, keepdims=True), atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(_output(x, {"axis": (0, 2)}), x.sum(axis=(0, 2)),
                               atol=1e-4, rtol=1e-5)


@pytest.mark.hardware_apple_gpu
def test_sum_f32_reports_native_gpu_on_metal():
    x = np.random.default_rng(20260716).standard_normal((3, 4, 5)).astype(np.float32)
    result = _run(x, {"axis": (0, 2)})
    np.testing.assert_allclose(result["output"], x.sum(axis=(0, 2)), atol=1e-4, rtol=1e-5)
    assert_native_apple_gpu(result, compiler_path="apple_gpu_reduce_compiled")


def test_mpsgraph_reduce_and_mse_forced_miss_is_reference_cpu(monkeypatch):
    monkeypatch.setattr(rt, "_apple_gpu_mpsgraph_reduce_f32", lambda: None)
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = _run(x, {"axis": -1})
    np.testing.assert_allclose(result["output"], x.sum(axis=-1))
    assert_reference_cpu(result)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("x, why", [
    (np.arange(12, dtype=np.int32).reshape(3, 4), "non-float dtype"),
    (np.float32(3.5), "0-d input"),
])
def test_inputs_the_dispatch_computes_in_numpy_report_reference_cpu(x, why):
    """Placement must follow what ran, not whether a device was present.

    The dispatch falls back to numpy per input, so a bare `is_metal()` probe
    reported `native_gpu` for these — work no GPU touched.
    """
    result = _run(x, {})
    np.testing.assert_allclose(np.asarray(result["output"]), np.asarray(x).sum(),
                               atol=1e-4, rtol=1e-5)
    assert_reference_cpu(result), why


def test_device_predicate_agrees_with_the_dispatch_fallback(monkeypatch):
    """The label predicate and the dispatch guard are one function.

    They were two, and drifted: the guard rejected non-float and 0-d input
    while the label only asked whether Metal existed.
    """
    f32 = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert not rt._apple_gpu_reduce_runs_on_device("tessera.reduce",
                                                   f32.astype(np.int32), {}, np)
    assert not rt._apple_gpu_reduce_runs_on_device("tessera.reduce",
                                                   np.float32(1.0), {}, np)
    monkeypatch.setattr(rt, "_apple_gpu_mpsgraph_reduce_f32", lambda: None)
    assert not rt._apple_gpu_reduce_runs_on_device("tessera.reduce", f32, {}, np)
