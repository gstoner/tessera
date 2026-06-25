"""Compiler-generated standalone elementwise activations (gelu/silu/relu) on
gfx1151 — the standalone analog of the GEMM fused epilogue.

The `tessera_rocm.activation` directive expands (via
`generate-rocm-activation-kernel`) into a flat per-element kernel applying a
pointwise activation, computed in f32 regardless of storage dtype. Reachable
through `runtime.launch()` via `compiler_path="rocm_activation_compiled"`, op
names `tessera.gelu` / `tessera.silu` / `tessera.relu`, f32/f16/bf16.

Validated vs the numpy references (gelu = tanh approximation, silu = x·σ(x)).

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _act_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_activation_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["x"], "kwargs": {}}],
    })


def _ref(x, op_name):
    f = x.astype(np.float32)
    if op_name == "tessera.relu":
        return np.maximum(f, 0.0)
    if op_name == "tessera.silu":
        return f / (1.0 + np.exp(-f))
    c = np.sqrt(2.0 / np.pi)  # gelu tanh approximation
    return 0.5 * f * (1.0 + np.tanh(c * (f + 0.044715 * f ** 3)))


@pytest.mark.parametrize("op_name", ["tessera.gelu", "tessera.silu",
                                     "tessera.relu"])
@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 1e-5), (np.float16, 3e-3), ("bf16", 2e-2),
])
@pytest.mark.parametrize("shape", [
    (16,), (300,),              # 1-D, incl. > one 256-thread block
    (8, 64), (4, 3, 33),        # multi-D (flattened)
])
def test_launch_activation_matches_numpy(op_name, dtype, tol, shape):
    rt = _act_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(3 + len(shape) + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 2.0).astype(dtype)

    res = rt.launch(_artifact(rt, op_name), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_activation_compiled"
    out = res["output"].astype(np.float32).reshape(shape)
    np.testing.assert_allclose(out, _ref(x, op_name), atol=tol, rtol=0)


def test_unknown_op_name_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="handles exactly one"):
        rt._execute_rocm_compiled_activation(_artifact(rt, "tessera.softmax"),
                                             (x,))
