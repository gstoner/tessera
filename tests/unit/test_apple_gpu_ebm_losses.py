"""Close-out gap #5 — EBM training losses on Metal (CD / PCD / SM / ISM / DSM).

The four EBM training losses previously had VJP/JVP but no kernel path
(backend_kernel=planned) and weren't reachable from the canonical tessera.ops /
@jit surface. This suite locks: the dedicated MPSGraph reduction kernels are
numerically correct, the ops are in the apple_gpu envelope, and a
@jit(target="apple_gpu") loss call is classified metal_runtime.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu

_LOSS_GRAPH_OPS = [
    "tessera.loss.contrastive_divergence",
    "tessera.loss.persistent_cd",
    "tessera.loss.score_matching",
    "tessera.loss.implicit_score_matching",
    "tessera.loss.denoising_score_matching",
]


def test_ebm_loss_ops_in_envelope():
    for op in _LOSS_GRAPH_OPS:
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_EBM_LOSS_OPS, op


def test_driver_runtime_ebm_loss_envelopes_match():
    assert _driver._APPLE_GPU_EBM_LOSS_OPS == _runtime._APPLE_GPU_EBM_LOSS_OPS


def test_losses_on_canonical_surface_with_autodiff():
    from tessera.autodiff import get_vjp, get_jvp
    for n in ("contrastive_divergence_loss", "persistent_cd_loss",
              "score_matching_loss", "implicit_score_matching_loss",
              "denoising_score_matching_loss"):
        assert hasattr(ts.ops, n), n
        assert get_vjp(n) is not None and get_jvp(n) is not None, n


# ── kernel numerics ─────────────────────────────────────────────────────────
@gpu
def test_cd_pcd_kernel_matches_reference():
    rng = np.random.default_rng(0)
    ep = rng.standard_normal(128).astype(np.float32)
    en = rng.standard_normal(128).astype(np.float32)
    np.testing.assert_allclose(agb.gpu_ebm_energy_diff_mean(ep, en), np.mean(ep - en), atol=1e-4)


@gpu
def test_score_matching_kernel_matches_reference():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 7)).astype(np.float32)
    b = rng.standard_normal((16, 7)).astype(np.float32)
    np.testing.assert_allclose(agb.gpu_ebm_half_mse(a, b), 0.5 * np.mean((a - b) ** 2), atol=1e-4)


@gpu
def test_ism_kernel_matches_reference():
    rng = np.random.default_rng(2)
    score = rng.standard_normal((16, 7)).astype(np.float32)
    div = rng.standard_normal(16).astype(np.float32)
    expect = np.mean(0.5 * (score ** 2).sum(-1) + div)
    np.testing.assert_allclose(agb.gpu_ebm_ism(score, div), expect, atol=1e-4)


@gpu
def test_dsm_kernel_matches_reference():
    rng = np.random.default_rng(3)
    score = rng.standard_normal((16, 7)).astype(np.float32)
    yc = rng.standard_normal((16, 7)).astype(np.float32)
    yn = rng.standard_normal((16, 7)).astype(np.float32)
    sigma = 0.5
    target = -(yn - yc) / (sigma * sigma)
    expect = 0.5 * np.mean(((score - target) ** 2).sum(-1))
    np.testing.assert_allclose(agb.gpu_ebm_dsm(score, yc, yn, 1.0 / (sigma * sigma)), expect, atol=1e-4)


# ── @jit(apple_gpu) → metal_runtime ─────────────────────────────────────────
@gpu
def test_cd_jit_metal_runtime():
    rng = np.random.default_rng(4)
    ep = rng.standard_normal(64).astype(np.float32)
    en = rng.standard_normal(64).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def cd(ep, en):
        return ts.ops.contrastive_divergence_loss(ep, en)

    np.testing.assert_allclose(float(np.asarray(cd(ep, en))), np.mean(ep - en), atol=1e-4)
    assert cd.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_dsm_jit_metal_runtime():
    rng = np.random.default_rng(5)
    score = rng.standard_normal((8, 6)).astype(np.float32)
    yc = rng.standard_normal((8, 6)).astype(np.float32)
    yn = rng.standard_normal((8, 6)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def dsm(s, c, n):
        return ts.ops.denoising_score_matching_loss(s, c, n, sigma=0.7)

    target = -(yn - yc) / (0.7 * 0.7)
    expect = 0.5 * np.mean(((score - target) ** 2).sum(-1))
    np.testing.assert_allclose(float(np.asarray(dsm(score, yc, yn))), expect, atol=1e-4)
    assert dsm.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


def test_non_mean_reduction_falls_back_correct():
    # sum/none reductions aren't GPU-kerneled; must still be numerically correct.
    rng = np.random.default_rng(6)
    ep = rng.standard_normal(32).astype(np.float32)
    en = rng.standard_normal(32).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def cd_sum(ep, en):
        return ts.ops.contrastive_divergence_loss(ep, en, reduction="sum")

    np.testing.assert_allclose(float(np.asarray(cd_sum(ep, en))), np.sum(ep - en), atol=1e-3)
