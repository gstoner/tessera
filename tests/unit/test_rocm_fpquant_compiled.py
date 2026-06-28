"""Compiler-generated low-precision float quantization on gfx1151 — the ROCm
mirror of the x86 fpquant / nvfp4 lanes. quantize/dequantize fp8 / fp6 / fp4 on
generate-rocm-fpquant-kernel; nvfp4 = block-scaled fp4.

Reachable via `compiler_path="rocm_fpquant_compiled"` / `"rocm_nvfp4_compiled"`.
Validated vs tessera.ops on gfx1151. Skip-clean: tessera-opt not built/no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, path, operands, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": path,
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs}],
    })


# fp4/fp6 use the deterministic mantissa-snap; fp8 uses the ml_dtypes reference,
# so the GPU log2/roundeven path matches to a small grid tolerance.
_TOL = dict(atol=2e-3, rtol=2e-3)

_QUANT = {
    "tessera.quantize_fp8": (ops.quantize_fp8, ["e4m3", "e5m2"]),
    "tessera.quantize_fp6": (ops.quantize_fp6, ["e2m3", "e3m2"]),
    "tessera.quantize_fp4": (ops.quantize_fp4, ["e2m1"]),
}


@pytest.mark.parametrize("op_name", list(_QUANT))
@pytest.mark.parametrize("shape", [(64,), (8, 16)])
def test_quantize_matches_reference(op_name, shape):
    rt = _rocm_or_skip()
    ref_fn, formats = _QUANT[op_name]
    rng = np.random.default_rng(7 + len(shape) + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 3).astype(np.float32)
    for fmt in formats:
        res = rt.launch(_artifact(rt, op_name, "rocm_fpquant_compiled", ("x",),
                                  {"format": fmt}), (x,))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "rocm_fpquant_compiled"
        ref_q, _ = ref_fn(x, format=fmt)
        np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                                   np.asarray(ref_q, np.float32), **_TOL)


def test_dequantize_passthrough():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(9)
    x = (rng.standard_normal((4, 16)) * 3).astype(np.float32)
    q, scale = ops.quantize_fp8(x, format="e4m3")
    q = np.asarray(q, np.float32)
    res = rt.launch(_artifact(rt, "tessera.dequantize_fp8",
                              "rocm_fpquant_compiled", ("xq",),
                              {"format": "e4m3", "scale": float(scale)}), (q,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(np.asarray(res["output"]).astype(np.float32), q)


@pytest.mark.parametrize("shape", [(4, 32), (2, 3, 64)])
def test_nvfp4_matches_reference(shape):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3 + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.quantize_nvfp4", "rocm_nvfp4_compiled",
                              ("x",), {"block_size": 16}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_nvfp4_compiled"
    ref_dq, _ = ops.quantize_nvfp4(x, block_size=16)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               np.asarray(ref_dq, np.float32), **_TOL)


def test_fpquant_codegen_lowers():
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = ('module {\n  "tessera_rocm.fpquant"() {name = "fq", dtype = "f32", '
         'max_normal = 448.0 : f32, mantissa_bits = 3 : i64, '
         'min_exp = -6 : i64} : () -> ()\n}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-fpquant-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
