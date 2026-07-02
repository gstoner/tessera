"""x86 integer quantization composite lane.

The lane uses AVX-512 round/max/min/mul kernels with host qparam selection and
int8 container conversion. int4 means signed int4 values stored in int8
containers, matching tessera.quantization.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import quantization as q


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, operands=("x",), kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "x86",
        "compiler_path": "x86_intquant_compiled",
        "executable": True,
        "execution_kind": "native_cpu",
        "arg_names": list(operands),
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": list(operands),
            "kwargs": dict(kwargs or {}),
        }],
    })


@pytest.mark.parametrize("op_name,ref_fn,bits", [
    ("tessera.quantize_int8", q.quantize_int8, 8),
    ("tessera.quantize_int4", q.quantize_int4, 4),
])
@pytest.mark.parametrize("shape", [(64,), (4, 17)])
def test_quantize_matches_reference(op_name, ref_fn, bits, shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(100 + bits + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 2.5).astype(np.float32)
    res = rt.launch(_artifact(rt, op_name), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_intquant_compiled"
    got_q, got_scale, got_zp = res["output"]
    ref_q, ref_scale, ref_zp = ref_fn(x)
    np.testing.assert_array_equal(got_q, ref_q)
    assert got_scale == ref_scale
    assert got_zp == ref_zp


def test_dequantize_and_fake_quantize_match_reference():
    rt = _x86_or_skip()
    x = np.linspace(-2.0, 2.0, 33, dtype=np.float32)
    q8, scale8, zp8 = q.quantize_int8(x, scale=0.03125)
    res = rt.launch(
        _artifact(rt, "tessera.dequantize_int8", kwargs={
            "scale": float(scale8),
            "zero_point": int(zp8),
        }),
        (q8,),
    )
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(res["output"], q.dequantize_int8(q8, scale8, zp8))

    fq = rt.launch(
        _artifact(rt, "tessera.fake_quantize", kwargs={"num_bits": 4}),
        (x,),
    )
    assert fq["ok"] is True, fq.get("reason")
    np.testing.assert_array_equal(fq["output"], q.fake_quantize(x, num_bits=4))


def test_bad_fake_quant_bits_rejected():
    rt = _x86_or_skip()
    with pytest.raises(ValueError, match="num_bits 4 or 8"):
        rt._execute_x86_compiled_intquant(
            _artifact(rt, "tessera.fake_quantize", kwargs={"num_bits": 6}),
            (np.zeros((4,), np.float32),),
        )
