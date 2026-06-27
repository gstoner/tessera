"""x86 NVFP4 lane — block-scaled fp4 (E2M1 codes + per-block fp8-E4M3 scale),
composed from the AVX-512 fpquant kernel + host block structure. The CPU lane
for the S9 nvfp4 ops (previously reference-only on both devices).

Reachable through `runtime.launch()` via `compiler_path="x86_nvfp4_compiled"`.
f32 fake-quant; the quantize path matches compiler/microscaling (via
tessera.ops.quantize_nvfp4) exactly.

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
        "target": "x86", "compiler_path": "x86_nvfp4_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs}],
    })


@pytest.mark.parametrize("shape", [(4, 32), (2, 3, 64), (8, 16), (6, 48)])
def test_quantize_nvfp4_matches_reference(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(1 + len(shape) + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.quantize_nvfp4", ("x",),
                              {"block_size": 16}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_nvfp4_compiled"
    ref_dq, _ = ops.quantize_nvfp4(x, block_size=16)
    np.testing.assert_array_equal(np.asarray(res["output"]).astype(np.float32),
                                  np.asarray(ref_dq, np.float32))


def test_dequantize_nvfp4_passthrough():
    rt = _x86_or_skip()
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((4, 32)) * 3).astype(np.float32)
    dq, _ = ops.quantize_nvfp4(x, block_size=16)
    dq = np.asarray(dq, np.float32)
    res = rt.launch(_artifact(rt, "tessera.dequantize_nvfp4", ("xq",),
                              {"block_size": 16}), (dq,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(np.asarray(res["output"]).astype(np.float32),
                                  dq)


def test_nvfp4_bad_block_rejected():
    rt = _x86_or_skip()
    x = np.zeros((4, 30), np.float32)   # 30 not divisible by 16
    with pytest.raises(ValueError, match="divisible by block_size"):
        rt._execute_x86_compiled_nvfp4(
            _artifact(rt, "tessera.quantize_nvfp4", ("x",), {"block_size": 16}),
            (x,))


def test_nvfp4_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 16), np.float32)
    with pytest.raises(ValueError, match="x86_nvfp4_compiled executor"):
        rt._execute_x86_compiled_nvfp4(
            _artifact(rt, "tessera.quantize_fp4", ("x",), {}), (x,))
