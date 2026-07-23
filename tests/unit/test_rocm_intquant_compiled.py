"""ROCm integer quantization composite lane.

Uses generated ROCm unary/binary kernels for round/max/min/mul, with host-side
qparam selection and int8 container conversion.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import quantization as q


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, operands=("x",), kwargs=None, **metadata):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_intquant_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": list(operands),
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": list(operands),
            "kwargs": dict(kwargs or {}),
        }],
        **metadata,
    })


@pytest.mark.parametrize("op_name,ref_fn,bits", [
    ("tessera.quantize_int8", q.quantize_int8, 8),
    ("tessera.quantize_int4", q.quantize_int4, 4),
])
def test_quantize_matches_reference_on_gpu(op_name, ref_fn, bits):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(200 + bits)
    x = (rng.standard_normal((4, 17)) * 2.5).astype(np.float32)
    res = rt.launch(_artifact(rt, op_name), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_intquant_compiled"
    got_q, got_scale, got_zp = res["output"]
    ref_q, ref_scale, ref_zp = ref_fn(x)
    np.testing.assert_array_equal(got_q, ref_q)
    assert got_scale == ref_scale
    assert got_zp == ref_zp


def test_dequantize_and_fake_quantize_match_reference_on_gpu():
    rt = _rocm_or_skip()
    x = np.linspace(-2.0, 2.0, 33, dtype=np.float32)
    q4, scale4, zp4 = q.quantize_int4(x)
    res = rt.launch(
        _artifact(rt, "tessera.dequantize_int4", kwargs={
            "scale": float(scale4),
            "zero_point": int(zp4),
        }),
        (q4,),
    )
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(res["output"], q.dequantize_int4(q4, scale4, zp4))

    fq = rt.launch(
        _artifact(rt, "tessera.fake_quantize", kwargs={"num_bits": 8}),
        (x,),
    )
    assert fq["ok"] is True, fq.get("reason")
    np.testing.assert_array_equal(fq["output"], q.fake_quantize(x, num_bits=8))


@pytest.mark.parametrize("shape", [(17,), (3, 8), (2, 5, 7)])
def test_compiled_terminal_int4_pack_roundtrip(shape):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(401 + int(np.prod(shape)))
    x = rng.standard_normal(shape).astype(np.float32)
    quantized = rt.launch(
        _artifact(
            rt, "tessera.quantize_int4", terminal_packed=True
        ),
        (x,),
    )["output"]
    packed, scale, zero_point, logical_shape = quantized
    assert packed.dtype == np.int8
    assert packed.size == (x.size + 1) // 2
    assert tuple(logical_shape) == shape
    result = rt.launch(
        _artifact(
            rt,
            "tessera.dequantize_int4",
            kwargs={"scale": float(scale), "zero_point": int(zero_point)},
            terminal_packed=True,
            logical_shape=shape,
        ),
        (packed,),
    )["output"]
    ref_q, ref_scale, ref_zero = q.quantize_int4(x)
    np.testing.assert_array_equal(
        result, q.dequantize_int4(ref_q, ref_scale, ref_zero)
    )
