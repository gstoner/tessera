"""NVIDIA bounded-control correctness ABI: one CUDA launch per construct."""

from __future__ import annotations

import numpy as np
import pytest

from _nvidia_testutil import nvidia_mma_runtime_available
from tessera.compiler.emit import nvidia_cuda as nv


def test_control_source_has_four_single_launch_entries():
    src = nv._synthesize_control_flow_cuda()
    for name in ("control_for", "control_if", "control_while", "control_scan"):
        assert f'tessera_nvidia_{name}_f32' in src
    assert src.count("<<<") == 4


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_bounded_control_cuda_matches_numpy():
    x = np.linspace(1.0, 2.0, 32, dtype=np.float32)
    np.testing.assert_allclose(
        nv.run_control_for_f32(x, trip=4, alpha=2.0, beta=1.0),
        (((x * 2 + 1) * 2 + 1) * 2 + 1) * 2 + 1)

    for flag, ref in ((1.0, 3 * x + 2), (-1.0, -2 * x + 4)):
        got = nv.run_control_if_f32(
            np.array([flag], np.float32), x, then_alpha=3, then_beta=2,
            else_alpha=-2, else_beta=4)
        np.testing.assert_allclose(got, ref)

    w = nv.run_control_while_f32(
        np.ones(32, np.float32), max_iters=8, limit=10, alpha=2, beta=0)
    np.testing.assert_array_equal(w, np.full(32, 16, np.float32))

    xs = np.arange(6 * 32, dtype=np.float32).reshape(6, 32) / 100
    carry = x.copy(); ref_ys = []
    for row in xs:
        carry = np.float32(0.5) * carry + row
        ref_ys.append(carry.copy())
    got_carry, got_ys = nv.run_control_scan_f32(x, xs, alpha=0.5)
    np.testing.assert_allclose(got_carry, carry, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_ys, np.stack(ref_ys), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_control_scan_runtime_binding_matches_numpy():
    from tessera import runtime as rt
    init = np.arange(8, dtype=np.float32)
    xs = np.ones((4, 8), np.float32)
    artifact = rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_control_flow_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["init", "xs"], "output_name": "result",
        "ops": [{"op_name": "tessera.control_scan", "result": "result",
                 "operands": ["init", "xs"],
                 "kwargs": {"alpha": 0.5}}],
    })
    result = rt.launch(artifact, (init, xs))
    assert result["ok"] is True, result.get("reason")
    carry, ys = result["output"]
    ref = init.copy(); ref_ys = []
    for row in xs:
        ref = 0.5 * ref + row; ref_ys.append(ref.copy())
    np.testing.assert_allclose(carry, ref)
    np.testing.assert_allclose(ys, np.stack(ref_ys))


def test_control_abi_rejects_bad_shapes_before_cuda():
    with pytest.raises(ValueError, match="rank-1 f32"):
        nv.run_control_for_f32(np.zeros((2, 2), np.float32), trip=2)
    with pytest.raises(ValueError, match="N<=1024"):
        nv.run_control_while_f32(
            np.zeros(1025, np.float32), max_iters=2, limit=1)
    with pytest.raises(ValueError, match=r"\[trip,N\]"):
        nv.run_control_scan_f32(
            np.zeros(4, np.float32), np.zeros((3, 5), np.float32))
