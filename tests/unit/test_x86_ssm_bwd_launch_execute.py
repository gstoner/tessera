"""The AVX-512 x86 selective_ssm (Mamba2) BACKWARD through ``runtime.launch()``.

The x86 analog of test_rocm_ssm_bwd_launch_execute.py: an artifact stamped
``compiler_path = "x86_selective_ssm_bwd_compiled"`` routes through
``runtime.launch()`` → the execution matrix → ``_execute_x86_selective_ssm_bwd``,
which runs the AVX-512 fused backward kernel and returns
``(dx, dA, dB, dC, ddelta)``. This makes x86 the **second native backward target**
(after ROCm gfx1151) in the autodiff ledger (AUTODIFF_UNIFICATION_PLAN §9a). Fully
host-verifiable — no GPU. The compare test skips only if the x86 lib isn't built.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.vjp import vjp_selective_ssm
from tessera.compiler import execution_matrix as em


def test_execution_matrix_has_x86_ssm_bwd_row():
    row = em.lookup("x86", "x86_selective_ssm_bwd_compiled")
    assert row is not None
    assert row.executable and row.execution_kind == "native_cpu"
    assert row.direction == "backward" and row.op_family == "selective_ssm"
    assert row.device_proof == "device_verified_abi"
    assert row.evidence_target == "x86_avx512"
    assert row.proof_build == "x86-runtime-avx512"


def test_x86_is_a_native_backward_target():
    assert em.has_native_backward("selective_ssm", "x86")
    info = em.native_backward_targets()["selective_ssm"]
    assert "x86_avx512" in info["device_verified_abi"]
    assert "rocm_gfx1151" in info["device_verified_jit"]


def test_x86_ssm_bwd_executor_rejects_bad_arity():
    from tessera import runtime as rt
    bad = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_selective_ssm_bwd_compiled",
        "arg_names": ["dout"], "ops": [
            {"op_name": "tessera.selective_ssm_bwd", "operands": ["dout"]}]})
    with pytest.raises(ValueError):
        rt._execute_x86_selective_ssm_bwd(bad, (np.zeros((2, 3, 4)),))


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, names):
    return rt.RuntimeArtifact(metadata={
        "target": "x86",
        "compiler_path": "x86_selective_ssm_bwd_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "grads",
        "ops": [{"op_name": "tessera.selective_ssm_bwd", "result": "grads",
                 "operands": names, "kwargs": {}}],
    })


@pytest.mark.parametrize("n,a_1d,gated", [(16, False, False), (10, True, True)])
def test_launch_x86_ssm_bwd_matches_vjp(n, a_1d, gated):
    rt = _rt_or_skip()
    rng = np.random.default_rng(n * 7 + a_1d + (gated << 2))
    b, s, d = 2, 7, 4
    x = rng.standard_normal((b, s, d)).astype(np.float32)
    A = (-np.abs(rng.standard_normal((d,) if a_1d else (d, n)))).astype(np.float32)
    B = rng.standard_normal((b, s, n)).astype(np.float32)
    C = rng.standard_normal((b, s, n)).astype(np.float32)
    delta = np.abs(rng.standard_normal((b, s, d)) * 0.1).astype(np.float32)
    dout = rng.standard_normal((b, s, d)).astype(np.float32)
    gate = rng.standard_normal((b, s, d)).astype(np.float32) if gated else None
    state = rng.standard_normal((b, d, n)).astype(np.float32) if gated else None

    # Operands are cotangent-first: (dout, x, A, B, C, delta[, gate, state]).
    names = ["dout", "x", "A", "B", "C", "delta"]
    inputs = [dout, x, A, B, C, delta]
    if gated:
        names += ["gate", "state"]
        inputs += [gate, state]

    res = rt.launch(_artifact(rt, names), tuple(inputs))
    assert res["ok"] is True, res.get("reason")
    assert res["runtime_status"] == "success"
    assert res["compiler_path"] == "x86_selective_ssm_bwd_compiled"
    assert res["execution_kind"] == "native_cpu"
    dx, dA, dB, dC, dd = res["output"]

    ref = vjp_selective_ssm(
        np.asarray(dout, np.float64), np.asarray(x, np.float64),
        np.asarray(A, np.float64), np.asarray(B, np.float64),
        np.asarray(C, np.float64), np.asarray(delta, np.float64),
        gate=None if gate is None else np.asarray(gate, np.float64),
        state=None if state is None else np.asarray(state, np.float64))
    for name, g, r in zip(("dx", "dA", "dB", "dC", "ddelta"),
                          (dx, dA, dB, dC, dd), ref):
        np.testing.assert_allclose(np.asarray(g, np.float64), r, rtol=0,
                                   atol=5e-3, err_msg=f"{name} mismatch")
