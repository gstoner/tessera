"""Phase-G close-out, Phase A — AST ``@tessera.jit`` → ``tessera.control_for`` bridge.

A plain Python ``for`` loop in ``@jit(target="apple_gpu")`` now lowers to the
GraphFn ``tessera.control_for`` path and executes on Apple GPU (control_for →
tessera-opt → ``tessera_apple.gpu.control_loop`` → ``run_graph_loop_f32``).

Detection + the hard-diagnostic path are pure (no runtime needed); the numeric
end-to-end cases need the Apple GPU runtime + a built ``tessera-opt`` and skip
otherwise.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import TesseraJitError, _find_tessera_opt

_GPU = agb.is_available() and jb.is_available() and _find_tessera_opt() is not None
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / tessera-opt unavailable")


# --- module-level decorated functions (source must be file-inspectable) ----- #
@ts.jit(target="apple_gpu")
def linear_loop(x, w):
    for _ in range(5):
        x = ts.ops.matmul(x, w)
    return x


@ts.jit(target="apple_gpu")
def silu_loop(x, w):
    for _ in range(4):
        x = ts.ops.silu(ts.ops.matmul(x, w))
    return x


@ts.jit(target="apple_gpu")
def prenorm_residual_loop(x, w):
    for _ in range(5):
        x = ts.ops.add(x, ts.ops.matmul(ts.ops.rmsnorm(x), w))
    return x


@ts.jit(target="apple_gpu")
def no_carry_loop(x, w):  # y is not fed back -> NOT a carry loop
    for _ in range(3):
        y = ts.ops.relu(x)
    return y


@ts.jit(target="apple_gpu")
def untranslatable_loop(x, w):  # tessera.sqrt has no GraphFn builder
    for _ in range(2):
        x = ts.ops.sqrt(ts.ops.matmul(x, w))
    return x


@ts.jit(target="apple_gpu")
def divergent_if(flag, x, w):
    if flag:
        y = ts.ops.silu(ts.ops.matmul(x, w))
    else:
        y = ts.ops.relu(ts.ops.matmul(x, w))
    return y


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _rms(z, eps=1e-5):
    return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + eps)


# --- detection (no runtime) ------------------------------------------------- #
def test_detects_single_carry_loop():
    sh = silu_loop._loop_shape
    assert sh is not None
    assert sh.carry_arg_index == 0 and sh.carry_base == "x"
    assert sh.trip == 4 and sh.next_carry_ssa == "x__1"
    assert sh.arg_names == ("x", "w")


def test_non_carry_loop_not_detected():
    # `y = relu(x)` never feeds back -> not the supported shape -> existing path.
    assert no_carry_loop._loop_shape is None


def test_untranslatable_op_still_matches_shape():
    # Shape matches (x is a carry); the diagnostic fires at call/build time.
    assert untranslatable_loop._loop_shape is not None


def test_untranslatable_op_raises_hard_diagnostic():
    x = np.ones((1, 8), np.float32)
    w = np.eye(8, dtype=np.float32)
    with pytest.raises(TesseraJitError, match="tessera.sqrt"):
        untranslatable_loop(x, w)


# --- apple_gpu execution ---------------------------------------------------- #
@gpu
@pytest.mark.parametrize("d", [8, 16])
def test_bridge_linear_recurrence_matches_numpy(d):
    rng = np.random.default_rng(d)
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = linear_loop(x, w)  # one decorated fn, shape-specialized per call
    ref = x.copy()
    for _ in range(5):
        ref = ref @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_bridge_silu_matmul_matches_numpy():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((1, 16)) / 16).astype(np.float32)
    w = (rng.standard_normal((16, 16)) / 4).astype(np.float32)
    out = silu_loop(x, w)
    ref = x.copy()
    for _ in range(4):
        ref = _silu(ref @ w)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_bridge_prenorm_residual_matches_numpy():
    rng = np.random.default_rng(2026)
    x = (rng.standard_normal((1, 32)) * 0.3).astype(np.float32)
    w = (rng.standard_normal((32, 32)) / np.sqrt(32)).astype(np.float32)
    out = prenorm_residual_loop(x, w)
    ref = x.copy()
    for _ in range(5):
        ref = ref + _rms(ref) @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# --- AST if-bridge (C2) ----------------------------------------------------- #
def test_detects_divergent_if():
    sh = divergent_if._cond_shape
    assert sh is not None
    assert sh.flag_arg_index == 0 and sh.flag_base == "flag"
    assert sh.arg_names == ("flag", "x", "w")
    assert divergent_if._loop_shape is None  # not a loop


@gpu
@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_bridge_if_selects_branch_matches_numpy(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 3)
    d = 8
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = divergent_if(np.array([flagv], np.float32), x, w)
    z = x @ w
    ref = _silu(z) if flagv > 0 else np.maximum(z, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
    g = next(iter(divergent_if._bridge_cache.values()))
    assert g.last_dispatch() == ["control_if"]


@gpu
def test_bridge_drives_control_loop_and_caches_per_shape():
    x = np.ones((1, 16), np.float32) / 16
    w = (np.random.default_rng(3).standard_normal((16, 16)) / 4).astype(np.float32)
    silu_loop(x, w)
    g = next(iter(silu_loop._bridge_cache.values()))
    assert g.last_dispatch() == ["control_loop"]
    # Second call, same shape -> cache reused (no new entry).
    n = len(silu_loop._bridge_cache)
    silu_loop(x, w)
    assert len(silu_loop._bridge_cache) == n
