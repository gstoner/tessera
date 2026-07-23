"""Compiler-generated row normalization (rmsnorm / layer_norm) on gfx1151 —
siblings of the softmax reduction kernel.

The `tessera_rocm.norm` directive expands (via `generate-rocm-norm-kernel`) into
a row-reduction kernel: one workgroup per row, a single X-pass reduces the row
sum and centered sum-of-squares (LDS tree-reduce), then a write pass applies
the normalize and optional channel affine transform over the last axis:

  * rmsnorm    — `O = X / sqrt(mean(X²) + eps)`
  * layer_norm — `O = (X − μ) / sqrt(mean((X − μ)²) + eps)`

Reachable through `runtime.launch()` via `compiler_path="rocm_norm_compiled"` —
op names `tessera.rmsnorm` / `tessera.rmsnorm_safe` / `tessera.layer_norm`,
f32/f16/bf16. Validated against NumPy for unary and affine forms.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _norm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, eps=None, *, affine=False):
    kwargs = {} if eps is None else {"eps": float(eps)}
    operands = ["x"]
    if affine:
        operands.append("gamma")
        if op_name == "tessera.layer_norm":
            operands.append("beta")
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_norm_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": operands, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": operands, "kwargs": kwargs}],
    })


def _ref(x, op_name, eps):
    f = x.astype(np.float32)
    if op_name == "tessera.layer_norm":
        mu = f.mean(axis=-1, keepdims=True)
        return (f - mu) / np.sqrt(f.var(axis=-1, keepdims=True) + eps)
    return f / np.sqrt(np.mean(f * f, axis=-1, keepdims=True) + eps)


@pytest.mark.parametrize("op_name,default_eps", [
    ("tessera.rmsnorm", 1e-5),
    ("tessera.rmsnorm_safe", 1e-6),
    ("tessera.layer_norm", 1e-5),
])
@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 2e-4), (np.float16, 4e-3), ("bf16", 3e-2),
])
@pytest.mark.parametrize("shape", [
    (1, 16), (8, 64), (4, 300),     # K > one 256-lane stride pass
    (32, 17),                       # ragged small K
    (2, 3, 48),                     # rank-3 → folded to [M, K]
])
def test_launch_norm_matches_numpy(op_name, default_eps, dtype, tol, shape):
    rt = _norm_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(13 + len(shape) + shape[-1] + len(op_name))
    x = (rng.standard_normal(shape) * 2.0 + 0.5).astype(dtype)

    res = rt.launch(_artifact(rt, op_name), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_norm_compiled"
    out = res["output"].astype(np.float32).reshape(shape)

    ref = _ref(x, op_name, default_eps)
    np.testing.assert_allclose(out, ref, atol=tol, rtol=0)


def test_explicit_eps_is_honored():
    rt = _norm_or_skip()
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((4, 64)) * 2.0).astype(np.float32)
    out = rt.launch(_artifact(rt, "tessera.rmsnorm", eps=0.1), (x,))["output"]
    np.testing.assert_allclose(out, _ref(x, "tessera.rmsnorm", 0.1),
                               atol=2e-4, rtol=0)


@pytest.mark.parametrize("op_name", ["tessera.rmsnorm", "tessera.layer_norm"])
@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 3e-4), (np.float16, 5e-3), ("bf16", 3e-2),
])
@pytest.mark.parametrize("shape", [(3, 17), (2, 5, 64), (7, 300)])
def test_dynamic_affine_norm_matches_numpy(op_name, dtype, tol, shape):
    rt = _norm_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(101 + len(shape) + shape[-1])
    x = rng.standard_normal(shape).astype(dtype)
    gamma = rng.uniform(0.5, 1.5, shape[-1]).astype(dtype)
    beta = rng.standard_normal(shape[-1]).astype(dtype)
    args = (x, gamma, beta) if op_name == "tessera.layer_norm" else (x, gamma)
    out = rt.launch(_artifact(rt, op_name, affine=True), args)["output"]
    ref = _ref(x, op_name, 1e-5) * gamma
    if op_name == "tessera.layer_norm":
        ref = ref + beta
    np.testing.assert_allclose(np.asarray(out, np.float32), ref,
                               atol=tol, rtol=0)


def test_norm_hsaco_cache_identity_is_shape_and_affine_independent():
    rt = _norm_or_skip()
    rt._rocm_norm_hsaco_cache.clear()
    rng = np.random.default_rng(113)
    for shape, affine in [((2, 17), False), ((7, 64), True), ((3, 5, 300), True)]:
        x = rng.standard_normal(shape).astype(np.float32)
        gamma = np.ones(shape[-1], np.float32)
        args = (x, gamma) if affine else (x,)
        rt.launch(_artifact(rt, "tessera.rmsnorm", affine=affine), args)
    assert len(rt._rocm_norm_hsaco_cache) == 1


@pytest.mark.parametrize("offset,scale", [(1e4, 1.0), (1e3, 0.1), (-5e3, 2.0)])
def test_layer_norm_large_offset_small_variance(offset, scale):
    """PR#123 review: a large common offset with small variance (e.g. ~1e4 ± 1)
    cancels catastrophically under E[x²]−E[x]² but is exact under the two-pass
    squared-deviation variance. The near-zero random tests don't exercise this."""
    rt = _norm_or_skip()
    rng = np.random.default_rng(2024)
    x = (offset + scale * rng.standard_normal((4, 128))).astype(np.float32)
    out = rt.launch(_artifact(rt, "tessera.layer_norm"), (x,))["output"]
    ref = _ref(x, "tessera.layer_norm", 1e-5)
    # Output is O(1) (unit variance); demand a tight absolute match + finiteness.
    assert np.all(np.isfinite(out)), "layer_norm produced non-finite output"
    np.testing.assert_allclose(out, ref, atol=2e-3, rtol=0)


def test_unknown_op_name_rejected():
    """A non-norm op routed here is a clean error (GPU-free validation)."""
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="handles exactly one"):
        rt._execute_rocm_compiled_norm(_artifact(rt, "tessera.softmax"), (x,))
