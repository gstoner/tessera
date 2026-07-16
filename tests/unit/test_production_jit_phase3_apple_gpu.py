"""Phase 3 Sprint 3.1 — Apple GPU back-half + cross-target oracle
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

Apple GPU has no upstream MLIR backend, so it's a bespoke Metal back-half. This
sprint establishes the foundation: the production lane reaches the Apple GPU via
hand-tuned kernels, and the **compiled CPU lane is the cross-target oracle** —
the GPU result must match the CPU `_jit_boundary` result, which matches numpy.

Skips on non-Darwin / when the Apple GPU runtime can't load.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb

pytestmark = [
    pytest.mark.hardware_apple_gpu,
    pytest.mark.usefixtures("apple_gpu_jit_runtime"),
]


def _np_softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ── matmul: GPU == CPU lane == numpy ────────────────────────────────────────


@pytest.mark.parametrize("M,K,N", [(8, 16, 4), (1, 1, 1), (32, 32, 32), (7, 5, 9)])
def test_gpu_matmul_matches_cpu_lane_and_numpy(M, K, N):
    rng = np.random.default_rng(M + K + N)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)

    gpu = agb.gpu_matmul(a, b)
    cpu = jb.jit_matmul(a, b)  # CPU production lane (linalg→LLVM→ORC)

    np.testing.assert_allclose(gpu, a @ b, rtol=1e-4, atol=1e-4)  # GPU vs numpy
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)  # GPU vs CPU lane


# ── softmax: GPU == CPU lane ────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(4, 8), (1, 256), (16, 64)])
def test_gpu_softmax_matches_cpu_lane(shape):
    rng = np.random.default_rng(abs(hash(shape)) & 0xFFFF)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    gpu = agb.gpu_softmax(x)
    cpu = jb.jit_softmax(x, axis=-1)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gpu, _np_softmax(x), rtol=1e-4, atol=1e-4)


# ── fused matmul→softmax: ONE Metal kernel == un-fused CPU composition ───────


@pytest.mark.parametrize("M,K,N", [(4, 8, 16), (8, 8, 64), (2, 4, 128)])
def test_gpu_fused_matmul_softmax_matches_cpu_composition(M, K, N):
    """The D2 fused-chain target override: one fused Metal kernel computes
    softmax(A@B); it must equal the CPU lane's un-fused matmul-then-softmax."""
    rng = np.random.default_rng(M * 7 + N)
    a = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)

    gpu_fused = agb.gpu_matmul_softmax(a, b)  # single fused kernel
    cpu_composed = jb.jit_softmax(jb.jit_matmul(a, b), axis=-1)  # two CPU ops

    np.testing.assert_allclose(gpu_fused, cpu_composed, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gpu_fused, _np_softmax(a @ b), rtol=1e-4, atol=1e-4)


# ── gelu: GPU == CPU lane ───────────────────────────────────────────────────


def test_gpu_gelu_matches_cpu_lane():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((4, 16)) * 2.0).astype(np.float32)
    gpu = agb.gpu_gelu(x)
    cpu = jb.jit_gelu(x)
    # Both are the tanh-approx GELU family; tolerate kernel-vs-lowering spread.
    np.testing.assert_allclose(gpu, cpu, rtol=2e-2, atol=2e-2)


# ── envelope: non-f32 rejected (no silent cast) ─────────────────────────────


def test_gpu_matmul_rejects_non_f32():
    with pytest.raises(agb.AppleGpuError):
        agb.gpu_matmul(
            np.ones((4, 4), np.float64), np.ones((4, 4), np.float64)
        )
