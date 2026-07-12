"""Fused-epilogue GEMM through ``runtime.launch()`` on gfx1151.

The same `rocm_compiled` executor that runs a plain matmul also runs the fused
epilogue, opt-in via the op kwargs: an optional third operand is the per-output-
column bias (shape [N]), and `activation` selects an in-kernel pointwise
activation (relu/gelu/silu). The kernel adds the bias and applies the activation
on the f32 accumulator before the store — no intermediate D round-trip, no new
executor / matrix row. Validated vs a numpy gemm+bias+activation reference.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _compiled_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, with_bias, activation, op_name="tessera.matmul"):
    operands = ["a", "b"] + (["bias"] if with_bias else [])
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": operands, "output_name": "c",
        "ops": [{"op_name": op_name, "result": "c",
                 "operands": operands,
                 "kwargs": {"activation": activation}}],
    })


def test_fused_epilogue_op_name_accepted():
    """The `fused_epilogue` op name (a `compiled` rocm_target_map row) must be
    accepted by the rocm_compiled executor — else the dashboard overstates
    runtime.launch() support. GPU-free: rank-1 operands trip the rank-2 check,
    which is AFTER the op-name gate, so reaching it proves acceptance."""
    from tessera import runtime as rt
    bad = np.zeros((4,), np.float16)
    art = _artifact(rt, False, "relu", op_name="tessera.fused_epilogue")
    with pytest.raises(ValueError, match="rank-2 operands"):
        rt._execute_rocm_compiled_gemm(art, (bad, bad))


def test_unknown_op_name_rejected_by_gemm_lane():
    from tessera import runtime as rt
    z = np.zeros((16, 16), np.float16)
    art = _artifact(rt, False, "none", op_name="tessera.flash_attn")
    with pytest.raises(ValueError, match="handles exactly one"):
        rt._execute_rocm_compiled_gemm(art, (z, z))


def _act_ref(x, activation):
    if activation == "none":
        return x
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "silu":
        return x / (1.0 + np.exp(-x))
    if activation == "gelu":
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
    raise AssertionError(activation)


@pytest.mark.parametrize("activation", ["none", "relu", "gelu", "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("m,n,k", [(16, 16, 16), (64, 48, 32)])
def test_launch_fused_epilogue_matches_numpy(activation, with_bias, m, n, k):
    if activation == "none" and not with_bias:
        pytest.skip("plain GEMM is the matmul-lane test")
    rt = _compiled_or_skip()
    rng = np.random.default_rng(11 + m + n + k + len(activation) + with_bias)
    a = (rng.standard_normal((m, k)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.4).astype(np.float16)
    inputs = [a, b]
    if with_bias:
        bias = (rng.standard_normal((n,)) * 0.5).astype(np.float32)
        inputs.append(bias)

    res = rt.launch(_artifact(rt, with_bias, activation), tuple(inputs))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_compiled"
    out = res["output"]
    assert out.shape == (m, n)

    ref = a.astype(np.float32) @ b.astype(np.float32)
    if with_bias:
        ref = ref + bias[None, :]
    ref = _act_ref(ref, activation)
    maxerr = float(np.max(np.abs(out - ref)))
    np.testing.assert_allclose(out, ref, rtol=0, atol=5e-2)
    assert maxerr < 5e-2, (
        f"launch epilogue act={activation} bias={with_bias} "
        f"{m}x{n}x{k} maxerr={maxerr}")


def test_launch_fused_epilogue_rejects_integer_dtype():
    """bias/activation on an int dtype is a structured invalid_artifact (the
    epilogue is float-only), never a silent miscompute."""
    rt = _compiled_or_skip()
    a = np.ones((16, 16), np.int8)
    b = np.ones((16, 16), np.int8)
    res = rt.launch(_artifact(rt, False, "gelu"), (a, b))
    assert res["ok"] is False
    assert res["runtime_status"] == "invalid_artifact"
    assert "float-only" in res["reason"]
