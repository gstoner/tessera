"""The COMPILER-GENERATED ROCm selective_ssm (Mamba2) BACKWARD through
``runtime.launch()``.

The reverse-mode analog of the flash_attn backward launch lane
(test_rocm_flash_attn_launch_execute.py): an artifact stamped
``compiler_path = "rocm_selective_ssm_bwd_compiled"`` routes through
``runtime.launch()`` → the execution matrix → ``_execute_rocm_compiled_selective_
ssm_bwd``, which HIP-launches the generate-rocm-selective-ssm-bwd-kernel hsaco
and returns ``(dx, dA, dB, dC, ddelta)`` — the second native backward launch lane
(AUTODIFF_UNIFICATION_PLAN §9a). The first two tests are GPU-free; the compare
test runs on real gfx1151 and matches the numpy VJP oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.vjp import vjp_selective_ssm
from tessera.compiler import execution_matrix as em


def test_execution_matrix_has_rocm_ssm_bwd_row():
    row = em.lookup("rocm", "rocm_selective_ssm_bwd_compiled")
    assert row is not None
    assert row.executable and row.execution_kind == "native_gpu"
    assert row.direction == "backward" and row.op_family == "selective_ssm"


def test_rocm_ssm_bwd_executor_registered():
    from tessera import runtime as rt
    art = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_selective_ssm_bwd_compiled"})
    # An artifact with a bad op arity is rejected cleanly (not silent / OOB).
    bad = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_selective_ssm_bwd_compiled",
        "arg_names": ["dout"], "ops": [
            {"op_name": "tessera.selective_ssm_bwd", "operands": ["dout"]}]})
    with pytest.raises(ValueError):
        rt._execute_rocm_compiled_selective_ssm_bwd(bad, (np.zeros((2, 3, 4)),))
    assert art.metadata["compiler_path"] == "rocm_selective_ssm_bwd_compiled"


def _ssm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, names):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_selective_ssm_bwd_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "grads",
        "ops": [{"op_name": "tessera.selective_ssm_bwd", "result": "grads",
                 "operands": names, "kwargs": {}}],
    })


@pytest.mark.parametrize("n,a_1d,gated", [(16, False, False), (10, True, True)])
def test_launch_rocm_ssm_bwd_matches_vjp(n, a_1d, gated):
    rt = _ssm_or_skip()
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
    assert res["compiler_path"] == "rocm_selective_ssm_bwd_compiled"
    assert res["execution_kind"] == "native_gpu"
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
