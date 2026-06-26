"""Compiler-generated elementwise logical (and/or/xor/not) on gfx1151 — the S2
logical family over i8 booleans.

The `tessera_rocm.logical` directive expands (via `generate-rocm-logical-kernel`)
into a flat per-element kernel (one thread per element). Reachable through
`runtime.launch()` via `compiler_path="rocm_logical_compiled"`; op names
tessera.logical_and / logical_or / logical_xor / logical_not. Inputs are
normalized to bool via `!= 0` (numpy semantics); `not` is unary, the rest
binary. bool in/out.

Validated vs numpy. Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _logical_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, nin):
    operands = ["a", "b"][:nin]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_logical_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": operands, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": operands}],
    })


_BINARY = {
    "tessera.logical_and": np.logical_and,
    "tessera.logical_or": np.logical_or,
    "tessera.logical_xor": np.logical_xor,
}


@pytest.mark.parametrize("op_name", list(_BINARY))
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_logical_binary_matches_numpy(op_name, shape):
    rt = _logical_or_skip()
    ref = _BINARY[op_name]
    rng = np.random.default_rng(41 + len(shape) + int(np.prod(shape)))
    a = (rng.random(shape) < 0.5)
    b = (rng.random(shape) < 0.5)
    res = rt.launch(_artifact(rt, op_name, 2), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_logical_compiled"
    out = np.asarray(res["output"])
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, ref(a, b))


@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_logical_not_matches_numpy(shape):
    rt = _logical_or_skip()
    rng = np.random.default_rng(7 + int(np.prod(shape)))
    a = (rng.random(shape) < 0.5)
    res = rt.launch(_artifact(rt, "tessera.logical_not", 1), (a,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"])
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, np.logical_not(a))


def test_logical_normalizes_nonzero_inputs():
    # numpy treats any nonzero as true; the kernel normalizes via != 0.
    rt = _logical_or_skip()
    a = np.array([0, 1, 2, 0, 5], dtype=np.uint8)
    b = np.array([0, 0, 3, 7, 0], dtype=np.uint8)
    res = rt.launch(_artifact(rt, "tessera.logical_and", 2), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(
        np.asarray(res["output"]), np.logical_and(a, b))


def test_logical_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), bool)
    b = np.zeros((4, 9), bool)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_rocm_compiled_logical(
            _artifact(rt, "tessera.logical_or", 2), (a, b))


def test_logical_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), bool)
    with pytest.raises(ValueError, match="rocm_logical_compiled executor"):
        rt._execute_rocm_compiled_logical(
            _artifact(rt, "tessera.softmax", 1), (a,))


# ── GPU-free codegen gate (needs only tessera-opt, not a GPU) ────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"

_KINDS = ["and", "or", "xor", "not"]


def _opt(directive, *passes):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(_OPT), "-", *passes], input=directive,
                          capture_output=True, text=True)


@pytest.mark.parametrize("kind", _KINDS)
def test_logical_codegen_and_lowers(kind):
    import re
    d = ('module {\n  "tessera_rocm.logical"() {name = "l", '
         f'kind = "{kind}"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-logical-kernel")
    assert ir.returncode == 0, ir.stderr
    # not -> 3 args (a, o, n); and/or/xor -> 4 args. All i8 memrefs.
    nargs = 3 if kind == "not" else 4
    m = re.search(r"gpu\.func @l\(([^)]*)\)", ir.stdout)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == nargs
    assert "memref<?xi8>" in ir.stdout
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-logical-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_logical_codegen_bad_kind_rejected():
    d = ('module {\n  "tessera_rocm.logical"() {name = "l", kind = "zz"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-logical-kernel")
    assert r.returncode != 0 and "unknown kind" in r.stderr
