"""Close the last two normalization backend_kernel gaps through runtime.launch().

The normalization device kernels already exist; these two ops were numerically
identical to existing kernels but not yet dispatched to them:

* ``online_softmax`` (no streaming state) == ``softmax`` over the last axis — it
  rides the compiled softmax lane on ROCm gfx1151 (rocm_softmax_compiled) and x86
  AVX-512 (x86_softmax_compiled). The streaming (``state``) form is declined
  (Decision #21 — never a silent wrong answer).
* ``rmsnorm_safe`` == ``rmsnorm`` (tighter eps default) — already dispatched on
  ROCm; this closes the x86 lane (x86_norm_compiled).

Execute-compare fixtures for the manifest ``compiled`` claims. Skip-clean off the
respective hardware/toolchain.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, target, compiler_path, op_name, kind):
    return rt.RuntimeArtifact(metadata={
        "target": target, "compiler_path": compiler_path,
        "executable": True, "execution_kind": kind,
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["x"], "kwargs": {"axis": -1}}]})


@pytest.mark.parametrize("shape", [(4, 16), (3, 33), (8, 7)])
def test_rocm_online_softmax_matches_reference(shape):
    rt = _rocm_or_skip()
    x = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)
    res = rt.launch(_artifact(rt, "rocm", "rocm_softmax_compiled",
                              "tessera.online_softmax", "native_gpu"), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_softmax_compiled"
    got = np.asarray(res["output"])
    ref = np.asarray(ts.ops.softmax(x, axis=-1))
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-3)


@pytest.mark.parametrize("shape", [(4, 16), (3, 33), (8, 7)])
def test_x86_online_softmax_matches_reference(shape):
    rt = _x86_or_skip()
    x = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)
    res = rt.launch(_artifact(rt, "x86", "x86_softmax_compiled",
                              "tessera.online_softmax", "native_cpu"), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_softmax_compiled"
    got = np.asarray(res["output"])
    ref = np.asarray(ts.ops.softmax(x, axis=-1))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


@pytest.mark.parametrize("shape", [(4, 16), (3, 33), (8, 7)])
def test_x86_rmsnorm_safe_matches_reference(shape):
    rt = _x86_or_skip()
    x = np.random.default_rng(sum(shape) + 1).standard_normal(shape).astype(np.float32)
    res = rt.launch(_artifact(rt, "x86", "x86_norm_compiled",
                              "tessera.rmsnorm_safe", "native_cpu"), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_norm_compiled"
    got = np.asarray(res["output"])
    ref = np.asarray(ts.ops.rmsnorm_safe(x))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)


def test_online_softmax_streaming_state_declined():
    """Decision #21: the compiled softmax lane must NOT silently mishandle the
    streaming (state) form — it declines with a clear error."""
    from tessera import runtime as rt
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_softmax_compiled",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": "tessera.online_softmax", "operands": ["x"],
                 "kwargs": {"axis": -1, "state": (np.zeros((1,)), np.ones((1,)))}}]})
    with pytest.raises(ValueError):
        rt._execute_x86_compiled_softmax(art, (np.zeros((2, 3), np.float32),))
