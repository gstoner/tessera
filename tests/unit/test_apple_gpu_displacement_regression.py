"""Phase B (runtime half) + Phase D2 — the displacement no-regression lock.

B1-runtime: ``apple_gpu_coverage.fallback_histogram`` runs a model under
``@jit(target="apple_gpu")`` and reports the *failure-class* dispatch fallbacks
(shape/dtype/Metal-failure reasons + frequency) via the purpose-built
``runtime.dispatch_fallback_log``. This complements the static no-lane worklist.

D2: run a representative decoder-MLP block on apple_gpu and assert it does NOT
silently rot to numpy — i.e. the failure-class fallback histogram is empty. A
kernel that quietly degrades (the "displacement rots" failure mode) would make
the histogram non-empty and trip this guard. Darwin-gated, since "did it run on
the GPU" is only meaningful where Metal is present; off-Darwin we only assert the
program runs and is finite.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import apple_gpu_coverage as cov

DARWIN = sys.platform == "darwin"
_RNG = np.random.default_rng(20260617)


@ts.jit(target="apple_gpu")
def _decoder_mlp_block(x, Wu, Wd):
    """Representative pre-norm decoder MLP block: rmsnorm -> up-proj -> silu ->
    down-proj -> residual. Exercises the rowop (norm), mps (matmul), unary/fused
    (silu), and pointwise (residual add) lanes — all GPU-capable."""
    n = ts.ops.rmsnorm(x)
    u = ts.ops.matmul(n, Wu)
    g = ts.ops.silu(u)
    d = ts.ops.matmul(g, Wd)
    return ts.ops.add(x, d)


def _run_block():
    T, D, F = 8, 16, 32
    x = _RNG.standard_normal((T, D)).astype(np.float32)
    Wu = (_RNG.standard_normal((D, F)) / D**0.5).astype(np.float32)
    Wd = (_RNG.standard_normal((F, D)) / F**0.5).astype(np.float32)
    return np.asarray(_decoder_mlp_block(x, Wu, Wd))


def test_block_runs_and_is_finite():
    out = _run_block()
    assert out.shape == (8, 16)
    assert np.all(np.isfinite(out))


def test_fallback_histogram_returns_a_dict():
    hist = cov.fallback_histogram(_run_block)
    assert isinstance(hist, dict)
    # Keys (when present) are (op_name, reason) pairs with int counts.
    for key, count in hist.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(count, int)


@pytest.mark.skipif(not DARWIN, reason="GPU-residency check requires Metal (Darwin)")
def test_decoder_block_does_not_silently_rot_to_numpy():
    """D2 — the real regression lock: on Metal, a representative decoder block
    must run entirely on the GPU lanes; any failure-class fallback to numpy is a
    silent-rot regression and fails here."""
    hist = cov.fallback_histogram(_run_block)
    assert hist == {}, f"decoder block fell back to numpy for: {sorted(hist)}"
