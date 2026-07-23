"""Compiler-generated elementwise comparison (eq/ne/lt/le/gt/ge) on gfx1151 —
the S2 comparison family, producing a boolean (i8 0/1) result.

The `tessera_rocm.compare` directive expands (via `generate-rocm-compare-kernel`)
into a flat 2-operand per-element kernel (one thread per element). Reachable
through `runtime.launch()` via `compiler_path="rocm_compare_compiled"`; op names
tessera.eq / ne / lt / le / gt / ge; f16/bf16/f32 input storage, f32 compare,
bool output. NaN semantics match numpy: ordered everywhere except `ne`.

Validated vs numpy. Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _compare_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_compare_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["a", "b"]}],
    })


_CASES = {
    "tessera.eq": np.equal,
    "tessera.ne": np.not_equal,
    "tessera.lt": np.less,
    "tessera.le": np.less_equal,
    "tessera.gt": np.greater,
    "tessera.ge": np.greater_equal,
}


@pytest.mark.parametrize("op_name", list(_CASES))
@pytest.mark.parametrize("dtype", [np.float32, np.float16, "bf16"])
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_compare_matches_numpy(op_name, dtype, shape):
    rt = _compare_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    ref = _CASES[op_name]
    rng = np.random.default_rng(31 + len(shape) + int(np.prod(shape)))
    a = (rng.standard_normal(shape) * 1.5).astype(dtype)
    # Overlap some values so eq/le/ge actually fire.
    b = a.copy()
    mask = rng.random(shape) < 0.5
    b[mask] = (rng.standard_normal(int(mask.sum())) * 1.5).astype(dtype)
    res = rt.launch(_artifact(rt, op_name), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_compare_compiled"
    out = np.asarray(res["output"])
    assert out.dtype == np.bool_
    expect = ref(a.astype(np.float32), b.astype(np.float32))
    np.testing.assert_array_equal(out, expect)


def test_compare_nan_semantics():
    rt = _compare_or_skip()
    a = np.array([1.0, np.nan, np.nan, 2.0], np.float32)
    b = np.array([1.0, 1.0, np.nan, 3.0], np.float32)
    for op_name, ref in _CASES.items():
        res = rt.launch(_artifact(rt, op_name), (a, b))
        assert res["ok"] is True, res.get("reason")
        out = np.asarray(res["output"])
        # numpy: every comparison with NaN is False, except not_equal → True.
        np.testing.assert_array_equal(out, ref(a, b),
                                      err_msg=f"{op_name} NaN mismatch")


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("op_name", list(_CASES))
def test_integer_compare_signedness_matches_numpy(dtype, op_name):
    rt = _compare_or_skip()
    if dtype == np.int32:
        a = np.array([-2**31, -7, -1, 0, 1, 7, 2**31 - 1], dtype=dtype)
        b = np.array([0, -8, 1, 0, -1, 8, 2**31 - 1], dtype=dtype)
    else:
        a = np.array([0, 1, 7, 2**31, 2**32 - 1], dtype=dtype)
        b = np.array([1, 0, 8, 2**31 - 1, 2**32 - 1], dtype=dtype)
    out = rt.launch(_artifact(rt, op_name), (a, b))["output"]
    np.testing.assert_array_equal(out, _CASES[op_name](a, b))


def test_compare_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_rocm_compiled_compare(_artifact(rt, "tessera.lt"), (a, b))


def test_compare_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="rocm_compare_compiled executor"):
        rt._execute_rocm_compiled_compare(_artifact(rt, "tessera.softmax"), (a, b))


# ── GPU-free codegen gate (needs only tessera-opt, not a GPU) ────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"

_KINDS = ["eq", "ne", "lt", "le", "gt", "ge"]


def _opt(directive, *passes):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(_OPT), "-", *passes], input=directive,
                          capture_output=True, text=True)


@pytest.mark.parametrize("kind", _KINDS)
def test_compare_codegen_and_lowers(kind):
    import re
    d = ('module {\n  "tessera_rocm.compare"() {name = "c", '
         f'kind = "{kind}", dtype = "f32"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-compare-kernel")
    assert ir.returncode == 0, ir.stderr
    # 4 args (a, b : f32; o : i8; n : index); output memref is i8.
    m = re.search(r"gpu\.func @c\(([^)]*)\)", ir.stdout)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == 4
    assert "memref<?xi8>" in ir.stdout and "arith.cmpf" in ir.stdout
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-compare-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


@pytest.mark.parametrize("dtype,predicate", [("i32", "slt"), ("u32", "ult")])
def test_integer_compare_codegen_has_explicit_order(dtype, predicate):
    d = ('module {\n  "tessera_rocm.compare"() {name = "c", kind = "lt", '
         f'dtype = "{dtype}"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-compare-kernel")
    assert ir.returncode == 0, ir.stderr
    assert f"arith.cmpi {predicate}" in ir.stdout


def test_compare_codegen_bad_kind_rejected():
    d = ('module {\n  "tessera_rocm.compare"() {name = "c", kind = "zz"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-compare-kernel")
    assert r.returncode != 0 and "unknown kind" in r.stderr
