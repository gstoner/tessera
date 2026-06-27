"""x86 low-precision float quantize lane — quantize/dequantize fp8 / fp6 / fp4,
loaded from libtessera_x86_elementwise.so. Per-tensor symmetric grid-snap on the
AVX-512 fpquant kernel (fake-quant in f32 storage). The CPU lane for these S9
quantize ops (previously reference-only on both devices).

Reachable through `runtime.launch()` via `compiler_path="x86_fpquant_compiled"`.
f32; the quantize path matches tessera.ops exactly (0 abs err).

Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, operands, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_fpquant_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs}],
    })


_QUANT = {
    "tessera.quantize_fp8": (ops.quantize_fp8, ["e4m3", "e5m2"]),
    "tessera.quantize_fp6": (ops.quantize_fp6, ["e2m3", "e3m2"]),
    "tessera.quantize_fp4": (ops.quantize_fp4, ["e2m1"]),
}


@pytest.mark.parametrize("op_name", list(_QUANT))
@pytest.mark.parametrize("shape", [(64,), (8, 16), (3, 5, 7)])
def test_quantize_matches_reference(op_name, shape):
    rt = _x86_or_skip()
    ref_fn, formats = _QUANT[op_name]
    rng = np.random.default_rng(7 + len(shape) + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 3).astype(np.float32)
    for fmt in formats:
        res = rt.launch(_artifact(rt, op_name, ("x",), {"format": fmt}), (x,))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "x86_fpquant_compiled"
        ref_q, _ = ref_fn(x, format=fmt)
        out = np.asarray(res["output"]).astype(np.float32)
        np.testing.assert_array_equal(out, np.asarray(ref_q, np.float32))


def test_quantize_with_explicit_scale():
    rt = _x86_or_skip()
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((4, 16)) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.quantize_fp8", ("x",),
                              {"format": "e4m3", "scale": 0.05}), (x,))
    assert res["ok"] is True, res.get("reason")
    ref_q, _ = ops.quantize_fp8(x, format="e4m3", scale=0.05)
    np.testing.assert_array_equal(np.asarray(res["output"]).astype(np.float32),
                                  np.asarray(ref_q, np.float32))


@pytest.mark.parametrize("dq_op,q_op,fmt", [
    ("tessera.dequantize_fp8", ops.quantize_fp8, "e4m3"),
    ("tessera.dequantize_fp6", ops.quantize_fp6, "e3m2"),
    ("tessera.dequantize_fp4", ops.quantize_fp4, "e2m1"),
])
def test_dequantize_passthrough(dq_op, q_op, fmt):
    rt = _x86_or_skip()
    rng = np.random.default_rng(9)
    x = (rng.standard_normal((4, 16)) * 3).astype(np.float32)
    q, scale = q_op(x, format=fmt)
    q = np.asarray(q, np.float32)
    res = rt.launch(_artifact(rt, dq_op, ("xq",),
                              {"format": fmt, "scale": float(scale)}), (q,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(np.asarray(res["output"]).astype(np.float32),
                                  q)


def test_fp8_subnormal_matches_ml_dtypes():
    """e4m3 with scale=1 and small (subnormal-range) inputs — the AVX-512 grid
    must follow IEEE gradual underflow like the ml_dtypes reference, not the
    pure mantissa-snap (the divergence the review flagged)."""
    rt = _x86_or_skip()
    x = np.array([1e-3, 2e-3, 5e-3, 1e-2, 1.7e-2, -3e-3, 8e-3, 4.5e-3],
                 np.float32)
    res = rt.launch(_artifact(rt, "tessera.quantize_fp8", ("x",),
                              {"format": "e4m3", "scale": 1.0}), (x,))
    assert res["ok"] is True, res.get("reason")
    ref_q, _ = ops.quantize_fp8(x, format="e4m3", scale=1.0)
    np.testing.assert_array_equal(np.asarray(res["output"]).astype(np.float32),
                                  np.asarray(ref_q, np.float32))


def test_quantize_propagates_nan():
    rt = _x86_or_skip()
    x = np.array([1.0, np.nan, -2.0, np.nan, 0.5, 3.0, np.nan, -1.0],
                 np.float32)
    res = rt.launch(_artifact(rt, "tessera.quantize_fp6", ("x",),
                              {"format": "e3m2", "scale": 1.0}), (x,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.float32)
    assert np.isnan(out[[1, 3, 6]]).all()       # NaN sentinels preserved
    assert not np.isnan(out[[0, 2, 4, 5, 7]]).any()


def test_quantize_bad_format_rejected():
    rt = _x86_or_skip()
    x = np.zeros((8,), np.float32)
    with pytest.raises(ValueError, match="format must be"):
        rt._execute_x86_compiled_fpquant(
            _artifact(rt, "tessera.quantize_fp4", ("x",), {"format": "e3m2"}),
            (x,))


def test_fpquant_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((8,), np.float32)
    with pytest.raises(ValueError, match="x86_fpquant_compiled executor"):
        rt._execute_x86_compiled_fpquant(
            _artifact(rt, "tessera.mse_loss", ("x",), {}), (x,))
