"""Compiler-generated row-wise softmax on gfx1151 — the first non-matmul /
non-WMMA device_verified_jit ROCm kernel.

The `tessera_rocm.softmax` directive expands (via `generate-rocm-softmax-kernel`)
into a row-reduction kernel: one workgroup per row, the lanes stride over the
last axis and tree-reduce (through LDS) the row max then the row sum, computing
the numerically-stable softmax `O[m,:] = exp(X[m,:] - max) / Σ exp(...)` in f32
regardless of storage dtype. Reachable through `runtime.launch()` via
`compiler_path="rocm_softmax_compiled"` — f32/f16/bf16, axis=-1.

Validated vs the numpy reference (`_apple_gpu_dispatch_softmax` math).

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _sm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name="tessera.softmax"):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_softmax_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["x"], "kwargs": {"axis": -1}}],
    })


def _ref_softmax(x):
    xf = x.astype(np.float32)
    e = np.exp(xf - xf.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 1e-5), (np.float16, 3e-3), ("bf16", 2e-2),
])
@pytest.mark.parametrize("shape", [
    (1, 16),
    (8, 64),
    (4, 300),       # K > one stride pass of 256 lanes
    (32, 17),       # ragged small K
    (2, 3, 48),     # rank-3 → reshaped to [M, K] over the last axis
])
def test_launch_softmax_matches_numpy(dtype, tol, shape):
    rt = _sm_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(7 + len(shape) + shape[-1])
    x = (rng.standard_normal(shape) * 2.0).astype(dtype)

    res = rt.launch(_artifact(rt), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_softmax_compiled"
    out = res["output"].astype(np.float32).reshape(shape)

    ref = _ref_softmax(x)
    np.testing.assert_allclose(out, ref, atol=tol, rtol=0)
    # softmax rows must sum to 1.
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(shape[:-1]), atol=tol * 4)


def test_softmax_safe_op_name_also_runs():
    """`tessera.softmax_safe` is the same stable formula — the lane accepts it."""
    rt = _sm_or_skip()
    rng = np.random.default_rng(99)
    x = (rng.standard_normal((4, 64)) * 3.0).astype(np.float32)
    out = rt.launch(_artifact(rt, "tessera.softmax_safe"), (x,))["output"]
    np.testing.assert_allclose(out, _ref_softmax(x), atol=1e-5, rtol=0)


def test_unknown_op_name_rejected():
    """A non-softmax op routed here is a clean error (GPU-free validation)."""
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="handles exactly one"):
        rt._execute_rocm_compiled_softmax(_artifact(rt, "tessera.matmul"), (x,))


def test_nonlast_axis_rejected():
    """Only axis=-1 (last) is supported — a clean error otherwise (GPU-free)."""
    from tessera import runtime as rt
    art = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_softmax_compiled",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": "tessera.softmax", "result": "o",
                 "operands": ["x"], "kwargs": {"axis": 0}}],
    })
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="axis=-1"):
        rt._execute_rocm_compiled_softmax(art, (x,))
