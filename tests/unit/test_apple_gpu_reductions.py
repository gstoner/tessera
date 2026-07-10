"""Apple GPU Tier-3 reduction lane (2026-05-29).

Reductions / scans via the MetalPerformanceShadersGraph reduce lane:
  * reduce:  sum / mean / amax / amin / prod / var / std  (op_code per row)
  * arg:     argmax / argmin  (int indices)
  * scan:    cumsum / cumprod  (cumulative, same shape)

Arbitrary `axis` (None / int / tuple) + `keepdims` + `ddof` are normalized in
`runtime.py` by transposing the reduced axes to the end and folding to
[rows, cols]; f16/bf16 upcast to f32 (fp32 reduction numerics). See
docs/audit/backend/apple/archive/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as R

DARWIN = sys.platform == "darwin"


def _disp(op_name, x, **kw):
    return np.asarray(R._apple_gpu_dispatch_reduce(op_name, [x], kw, np))


_REDUCERS = [
    ("tessera.mean", np.mean),
    ("tessera.reduce", np.sum),
    ("tessera.amax", np.amax),
    ("tessera.amin", np.amin),
    ("tessera.prod", np.prod),
]


@pytest.mark.parametrize("op_name,npf", _REDUCERS)
@pytest.mark.parametrize("axis", [None, -1, 0, 1, (0, 2)])
def test_reduce_axis_variants(op_name, npf, axis):
    rng = np.random.RandomState(0)
    x = rng.randn(4, 6, 5).astype(np.float32)
    np.testing.assert_allclose(_disp(op_name, x, axis=axis),
                               npf(x, axis=axis), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("axis", [-1, 1, (1, 2)])
def test_reduce_keepdims(axis):
    rng = np.random.RandomState(1)
    x = rng.randn(3, 5, 7).astype(np.float32)
    out = _disp("tessera.mean", x, axis=axis, keepdims=True)
    ref = np.mean(x, axis=axis, keepdims=True)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op_name,npf", [("tessera.var", np.var), ("tessera.std", np.std)])
@pytest.mark.parametrize("ddof", [0, 1])
def test_var_std_ddof(op_name, npf, ddof):
    rng = np.random.RandomState(2)
    x = rng.randn(4, 16).astype(np.float32)
    np.testing.assert_allclose(_disp(op_name, x, axis=-1, ddof=ddof),
                               npf(x, axis=-1, ddof=ddof), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op_name,npf", [("tessera.argmax", np.argmax), ("tessera.argmin", np.argmin)])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_argreduce(op_name, npf, axis):
    rng = np.random.RandomState(3)
    x = rng.randn(4, 6, 5).astype(np.float32)
    out = _disp(op_name, x, axis=axis)
    np.testing.assert_array_equal(out, npf(x, axis=axis))


@pytest.mark.parametrize("op_name,npf", [("tessera.cumsum", np.cumsum), ("tessera.cumprod", np.cumprod)])
@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_scan(op_name, npf, axis):
    rng = np.random.RandomState(4)
    x = rng.randn(4, 6, 5).astype(np.float32)
    np.testing.assert_allclose(_disp(op_name, x, axis=axis),
                               npf(x, axis=axis), rtol=1e-4, atol=1e-4)


def test_reduce_no_n_limit():
    # N>256 proves the MPSGraph reduce has no per-thread envelope limit.
    rng = np.random.RandomState(5)
    x = rng.randn(3, 1024).astype(np.float32)
    np.testing.assert_allclose(_disp("tessera.sum" if False else "tessera.reduce", x, axis=-1),
                               x.sum(-1), rtol=1e-4, atol=1e-3)


def test_reduce_f16_upcasts():
    rng = np.random.RandomState(6)
    x = (rng.randn(4, 32) * 0.5).astype(np.float16)
    out = _disp("tessera.mean", x, axis=-1)
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32),
                               x.astype(np.float32).mean(-1), rtol=2e-2, atol=2e-3)


def test_reduce_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    for sym in ("tessera_apple_gpu_mpsgraph_reduce_f32",
                "tessera_apple_gpu_mpsgraph_argreduce_f32",
                "tessera_apple_gpu_mpsgraph_scan_f32"):
        assert hasattr(rt, sym), sym


# ── on-device dispatch gates via @jit ───────────────────────────────────────
@ts.jit(target="apple_gpu")
def _jit_mean(x):
    return ts.ops.mean(x, axis=-1)


@ts.jit(target="apple_gpu")
def _jit_cumsum(x):
    return ts.ops.cumsum(x, axis=-1)


def test_jit_reductions_runtime_executable():
    _jit_mean(np.zeros((4, 8), np.float32))
    _jit_cumsum(np.zeros((4, 8), np.float32))
    for fn in (_jit_mean, _jit_cumsum):
        meta = fn.runtime_artifact().metadata
        assert meta["compiler_path"] == "apple_gpu_mps"
        assert meta["runtime_status"] == "ready"
        assert meta["execution_mode"] in ("metal_runtime", "metal_artifact")


@pytest.mark.skipif(not DARWIN, reason="metal_runtime dispatch is Darwin-only")
def test_jit_mean_metal_runtime_on_darwin():
    rng = np.random.RandomState(7)
    x = rng.randn(4, 16).astype(np.float32)
    out = np.asarray(_jit_mean(x))
    np.testing.assert_allclose(out, x.mean(-1), rtol=1e-4, atol=1e-4)
    assert _jit_mean.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
