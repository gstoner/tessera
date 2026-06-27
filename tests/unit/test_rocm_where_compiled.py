"""Compiler-generated ternary select where(cond, a, b) on gfx1151.

The `tessera_rocm.where` directive expands (via `generate-rocm-where-kernel`)
into a flat 3-operand per-element kernel computing numpy `where`/select (one
thread per element). Reachable through `runtime.launch()` via
`compiler_path="rocm_where_compiled"`; op name tessera.where. `cond` is an i8
boolean normalized via != 0; `a`/`b`/the output share a float storage dtype
(f16/bf16/f32).

Validated vs np.where. Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_where_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["cond", "a", "b"], "output_name": "o",
        "ops": [{"op_name": "tessera.where", "result": "o",
                 "operands": ["cond", "a", "b"]}],
    })


@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_where_matches_numpy(shape):
    rt = _skip()
    rng = np.random.default_rng(71 + len(shape) + int(np.prod(shape)))
    cond = rng.integers(0, 4, size=shape).astype(np.uint8)
    a = rng.standard_normal(shape).astype(np.float32)
    b = rng.standard_normal(shape).astype(np.float32)
    res = rt.launch(_artifact(rt), (cond, a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_where_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_array_equal(out, np.where(cond != 0, a, b))


def test_where_shape_mismatch_rejected():
    from tessera import runtime as rt
    cond = np.zeros((4, 8), np.uint8)
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_rocm_compiled_where(_artifact(rt), (cond, a, b))


def test_where_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_where_compiled",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.softmax", "result": "o",
                 "operands": ["a"]}],
    })
    with pytest.raises(ValueError, match="rocm_where_compiled executor"):
        rt._execute_rocm_compiled_where(art, (a,))


# ── GPU-free codegen gate (needs only tessera-opt, not a GPU) ────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"


def _opt(directive, *passes):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(_OPT), "-", *passes], input=directive,
                          capture_output=True, text=True)


def test_where_codegen_and_lowers():
    import re
    d = ('module {\n  "tessera_rocm.where"() {name = "w", dtype = "f32"} '
         ': () -> ()\n}\n')
    ir = _opt(d, "--generate-rocm-where-kernel")
    assert ir.returncode == 0, ir.stderr
    m = re.search(r"gpu\.func @w\(([^)]*)\)", ir.stdout)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == 5
    assert "memref<?xi8>" in ir.stdout and "arith.select" in ir.stdout
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-where-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_where_codegen_bad_dtype_rejected():
    d = ('module {\n  "tessera_rocm.where"() {name = "w", dtype = "i7"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-where-kernel")
    assert r.returncode != 0 and "dtype must be" in r.stderr
