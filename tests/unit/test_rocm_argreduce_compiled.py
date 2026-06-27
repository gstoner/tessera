"""Compiler-generated argmax/argmin (warp-shuffle arg-reduce) on gfx1151 — the
CUB `DeviceReduce::ArgMax`/`ArgMin` pattern reimplemented in Tessera codegen.

The `tessera_rocm.argreduce` directive expands (via
`generate-rocm-argreduce-kernel`) into a row arg-reduction kernel: each thread
carries the best (value, index) pair, then a `gpu.shuffle xor` butterfly reduces
the pair within a 32-lane subgroup. Reachable through `runtime.launch()` via
`compiler_path="rocm_argreduce_compiled"`; op names tessera.argmax / argmin;
f16/bf16/f32 input, i32 index output; numpy first-occurrence tie-break.

Validated vs np.argmax/np.argmin. Skip-clean: tessera-opt not built / no AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _arg_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, axis):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_argreduce_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": {"axis": axis}}],
    })


_REFS = {"tessera.argmax": np.argmax, "tessera.argmin": np.argmin}


@pytest.mark.parametrize("op_name", list(_REFS))
@pytest.mark.parametrize("dtype", [np.float32, np.float16, "bf16"])
@pytest.mark.parametrize("shape,axis", [
    ((8, 64), -1), ((8, 64), 0), ((130,), None), ((3, 5, 7), -1),
    ((3, 5, 7), 1), ((4, 6), None),
])
def test_argreduce_matches_numpy(op_name, dtype, shape, axis):
    rt = _arg_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    ref = _REFS[op_name]
    # distinct values so the argmax/argmin winner is unambiguous (avoids ties,
    # which the kernel breaks first-occurrence but bf16 rounding could perturb).
    rng = np.random.default_rng(71 + len(shape) + int(np.prod(shape)))
    flat = rng.permutation(int(np.prod(shape))).astype(np.float32)
    x = flat.reshape(shape).astype(dtype)
    res = rt.launch(_artifact(rt, op_name, axis), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_argreduce_compiled"
    out = np.asarray(res["output"]).astype(np.int64)
    expect = ref(x.astype(np.float32), axis=axis).astype(np.int64)
    np.testing.assert_array_equal(out, expect)


def test_argreduce_first_occurrence_tiebreak():
    rt = _arg_or_skip()
    # repeated max value → numpy returns the FIRST index; so must the kernel.
    x = np.array([[1.0, 5.0, 5.0, 2.0, 5.0]], np.float32)
    res = rt.launch(_artifact(rt, "tessera.argmax", -1), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(
        np.asarray(res["output"]).astype(np.int64), np.argmax(x, axis=-1))


def test_argreduce_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="rocm_argreduce_compiled executor"):
        rt._execute_rocm_compiled_argreduce(
            _artifact(rt, "tessera.softmax", -1), (x,))


# ── GPU-free codegen gate ────────────────────────────────────────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"


def _opt(directive, *passes):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(_OPT), "-", *passes], input=directive,
                          capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["argmax", "argmin"])
def test_argreduce_codegen_and_lowers(kind):
    d = ('module {\n  "tessera_rocm.argreduce"() {name = "ar", '
         f'kind = "{kind}", dtype = "f32"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-argreduce-kernel")
    assert ir.returncode == 0, ir.stderr
    assert "memref<?xi32>" in ir.stdout and "gpu.shuffle" in ir.stdout
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-argreduce-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_argreduce_codegen_bad_kind_rejected():
    d = ('module {\n  "tessera_rocm.argreduce"() {name = "ar", kind = "zz"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-argreduce-kernel")
    assert r.returncode != 0 and "argmax or argmin" in r.stderr
