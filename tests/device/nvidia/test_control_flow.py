"""Exact-device NVIDIA bounded-control execute/compare proofs."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_runtime_available
from tessera.compiler.emit import nvidia_cuda as nv


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_bounded_control_cuda_matches_numpy():
    x = np.linspace(1.0, 2.0, 32, dtype=np.float32)
    np.testing.assert_allclose(nv.run_control_for_f32(x, trip=4, alpha=2.0, beta=1.0), (((x * 2 + 1) * 2 + 1) * 2 + 1) * 2 + 1)
    for flag, ref in ((1.0, 3 * x + 2), (-1.0, -2 * x + 4)):
        np.testing.assert_allclose(nv.run_control_if_f32(np.array([flag], np.float32), x, then_alpha=3, then_beta=2, else_alpha=-2, else_beta=4), ref)
    np.testing.assert_array_equal(nv.run_control_while_f32(np.ones(32, np.float32), max_iters=8, limit=10, alpha=2, beta=0), np.full(32, 16, np.float32))
    xs = np.arange(6 * 32, dtype=np.float32).reshape(6, 32) / 100
    carry = x.copy(); expected = []
    for row in xs:
        carry = np.float32(0.5) * carry + row; expected.append(carry.copy())
    got_carry, got_ys = nv.run_control_scan_f32(x, xs, alpha=0.5)
    np.testing.assert_allclose(got_carry, carry, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_ys, np.stack(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_control_scan_runtime_binding_matches_numpy():
    from tessera import runtime as rt
    init = np.arange(8, dtype=np.float32); xs = np.ones((4, 8), np.float32)
    artifact = rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_control_flow_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["init", "xs"], "output_name": "result", "ops": [{"op_name": "tessera.control_scan", "result": "result", "operands": ["init", "xs"], "kwargs": {"alpha": 0.5}}]})
    result = rt.launch(artifact, (init, xs)); assert result["ok"] is True, result.get("reason")
    carry, ys = result["output"]; expected = init.copy(); expected_ys = []
    for row in xs:
        expected = .5 * expected + row; expected_ys.append(expected.copy())
    np.testing.assert_allclose(carry, expected); np.testing.assert_allclose(ys, np.stack(expected_ys))
