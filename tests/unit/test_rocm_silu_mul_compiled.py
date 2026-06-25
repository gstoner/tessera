"""Compiler-generated SwiGLU gate-multiply (silu_mul) on gfx1151 — the
standalone analog of the fused SwiGLU gate-multiply.

The `tessera_rocm.silu_mul` directive expands (via
`generate-rocm-silu-mul-kernel`) into a flat 2-operand elementwise kernel
computing `silu(a)·b = (a / (1 + exp(-a)))·b`, in f32 regardless of storage
dtype. Reachable through `runtime.launch()` via
`compiler_path="rocm_silu_mul_compiled"`, op name `tessera.silu_mul`,
f32/f16/bf16.

Validated vs the numpy reference (the same one the Apple GPU lane uses).

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _silu_mul_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_silu_mul_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": "tessera.silu_mul", "result": "o",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


def _ref(a, b):
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    return (af / (1.0 + np.exp(-af))) * bf


@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 1e-5), (np.float16, 3e-3), ("bf16", 2e-2),
])
@pytest.mark.parametrize("shape", [
    (16,), (300,),              # 1-D, incl. > one 256-thread block
    (8, 64), (4, 3, 33),        # multi-D (flattened)
])
def test_launch_silu_mul_matches_numpy(dtype, tol, shape):
    rt = _silu_mul_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(11 + len(shape) + shape[-1])
    a = (rng.standard_normal(shape) * 1.5).astype(dtype)
    b = (rng.standard_normal(shape) * 1.5).astype(dtype)

    res = rt.launch(_artifact(rt), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_silu_mul_compiled"
    out = res["output"].astype(np.float32).reshape(shape)
    # silu_mul products reach larger magnitudes than a single activation, so the
    # half-precision OUTPUT rounding is relative — pair atol with rtol=tol.
    np.testing.assert_allclose(out, _ref(a, b), atol=tol, rtol=tol)


def test_silu_mul_zero_gate_is_zero():
    """silu(0)=0 ⇒ silu_mul(0, b) = 0 for any b — an anchor independent of exp."""
    rt = _silu_mul_or_skip()
    a = np.zeros((4, 32), np.float32)
    b = (np.random.default_rng(3).standard_normal((4, 32))).astype(np.float32)
    out = rt.launch(_artifact(rt), (a, b))["output"]
    np.testing.assert_allclose(out, np.zeros_like(b), atol=1e-6, rtol=0)


def test_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 16), np.float32)
    b = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_rocm_compiled_silu_mul(_artifact(rt), (a, b))
