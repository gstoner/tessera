"""Compiler-generated rotary position embedding (RoPE) on gfx1151.

The `tessera_rocm.rope` directive expands (via `generate-rocm-rope-kernel`) into
an elementwise-per-pair kernel over a rank-2-or-more `[..., D]` input (D even):
for pair p (e = X[...,2p], o = X[...,2p+1], angle a = Theta[...,2p]):

    O[...,2p]   = e·cos(a) − o·sin(a)
    O[...,2p+1] = e·sin(a) + o·cos(a)

One workgroup per row; cos/sin in f32. Reachable through `runtime.launch()` via
`compiler_path="rocm_rope_compiled"` (operands x, theta), f32/f16/bf16. Validated
vs the reference `_runtime_rope` (interleaved pairs).

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _rope_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_rope_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x", "theta"], "output_name": "o",
        "ops": [{"op_name": "tessera.rope", "result": "o",
                 "operands": ["x", "theta"], "kwargs": {}}],
    })


def _ref(x, theta):
    f = x.astype(np.float32)
    th = theta.astype(np.float32)
    even = f[..., 0::2]
    odd = f[..., 1::2]
    a = th[..., 0::2]
    out = np.empty_like(f)
    out[..., 0::2] = even * np.cos(a) - odd * np.sin(a)
    out[..., 1::2] = even * np.sin(a) + odd * np.cos(a)
    return out


def _make_theta(shape, base, np_):
    # A typical RoPE angle table: theta[..., 2p] = pos / base^(2p/D); the kernel
    # only reads the even-indexed entries, so fill pairs consistently.
    M = int(np_.prod(shape[:-1])) if len(shape) > 1 else 1
    D = shape[-1]
    pos = np_.arange(M).reshape(-1, 1)
    freq = base ** (-(np_.arange(0, D, 2) / D))
    ang = pos * freq                      # [M, D/2]
    full = np_.repeat(ang, 2, axis=1)     # [M, D] (even/odd share the pair angle)
    return full.reshape(shape)


@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 1e-5), (np.float16, 4e-3), ("bf16", 3e-2),
])
@pytest.mark.parametrize("shape", [
    (1, 16), (8, 64), (4, 320),     # D > one 256-thread pass over D/2 pairs
    (2, 3, 32),                     # rank-3 (folded to [M, D])
])
def test_launch_rope_matches_numpy(dtype, tol, shape):
    rt = _rope_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(5 + len(shape) + shape[-1])
    x = (rng.standard_normal(shape) * 0.5).astype(dtype)
    theta = _make_theta(shape, 10000.0, np).astype(dtype)

    res = rt.launch(_artifact(rt), (x, theta))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_rope_compiled"
    out = res["output"].astype(np.float32).reshape(shape)
    np.testing.assert_allclose(out, _ref(x, theta), atol=tol, rtol=0)


@pytest.mark.parametrize("dtype,tol", [(np.float16, 4e-3), ("bf16", 3e-2)])
def test_fp32_theta_with_half_precision_x(dtype, tol):
    """A common setup: x is f16/bf16 but the angle table stays fp32 (e.g.
    nn.RotaryEmbedding defaults theta to fp32). The lane must accept the mixed
    storage — theta is cast to x's dtype on the device copy — rather than reject
    it as invalid_artifact."""
    rt = _rope_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(73)
    shape = (8, 64)
    x = (rng.standard_normal(shape) * 0.5).astype(dtype)
    theta = _make_theta(shape, 10000.0, np).astype(np.float32)  # fp32 angle table

    res = rt.launch(_artifact(rt), (x, theta))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_rope_compiled"
    out = res["output"].astype(np.float32).reshape(shape)
    np.testing.assert_allclose(out, _ref(x, theta), atol=tol, rtol=0)


def test_non_float_theta_rejected():
    """A non-floating angle table would silently produce wrong angles after the
    storage cast — reject it with a clear message instead."""
    from tessera import runtime as rt
    x = np.zeros((4, 16), np.float32)
    th = np.zeros((4, 16), np.int32)
    with pytest.raises(ValueError, match="theta must be a floating dtype"):
        rt._execute_rocm_compiled_rope(_artifact(rt), (x, th))


def test_rope_norm_preserved_zero_angle():
    """At angle 0 RoPE is the identity — a sanity anchor independent of cos/sin."""
    rt = _rope_or_skip()
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((4, 32)) * 0.5).astype(np.float32)
    theta = np.zeros((4, 32), np.float32)
    out = rt.launch(_artifact(rt), (x, theta))["output"]
    np.testing.assert_allclose(out, x, atol=1e-6, rtol=0)


def test_odd_dim_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 7), np.float32)
    with pytest.raises(ValueError, match="even innermost"):
        rt._execute_rocm_compiled_rope(_artifact(rt), (x, x))


def test_shape_mismatch_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 16), np.float32)
    th = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="theta to match x"):
        rt._execute_rocm_compiled_rope(_artifact(rt), (x, th))
