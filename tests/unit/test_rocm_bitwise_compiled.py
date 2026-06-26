"""Compiler-generated elementwise bitwise (and/or/xor/not) on gfx1151 — the S2
bitwise family over i32 integers.

The `tessera_rocm.bitwise` directive expands (via `generate-rocm-bitwise-kernel`)
into a flat per-element kernel (one thread per element). Reachable through
`runtime.launch()` via `compiler_path="rocm_bitwise_compiled"`; op names
tessera.bitwise_and / bitwise_or / bitwise_xor / bitwise_not. Acts on the full
i32 bit pattern (no normalization); `not` is unary (~a), the rest binary.

Validated vs numpy. Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _bitwise_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, nin):
    operands = ["a", "b"][:nin]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_bitwise_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": operands, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": operands}],
    })


_BINARY = {
    "tessera.bitwise_and": np.bitwise_and,
    "tessera.bitwise_or": np.bitwise_or,
    "tessera.bitwise_xor": np.bitwise_xor,
}


@pytest.mark.parametrize("op_name", list(_BINARY))
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_bitwise_binary_matches_numpy(op_name, shape):
    rt = _bitwise_or_skip()
    ref = _BINARY[op_name]
    rng = np.random.default_rng(53 + len(shape) + int(np.prod(shape)))
    a = rng.integers(-(1 << 20), 1 << 20, size=shape, dtype=np.int32)
    b = rng.integers(-(1 << 20), 1 << 20, size=shape, dtype=np.int32)
    res = rt.launch(_artifact(rt, op_name, 2), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_bitwise_compiled"
    out = np.asarray(res["output"]).astype(np.int32)
    np.testing.assert_array_equal(out, ref(a, b))


@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_bitwise_not_matches_numpy(shape):
    rt = _bitwise_or_skip()
    rng = np.random.default_rng(9 + int(np.prod(shape)))
    a = rng.integers(-(1 << 20), 1 << 20, size=shape, dtype=np.int32)
    res = rt.launch(_artifact(rt, "tessera.bitwise_not", 1), (a,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.int32)
    np.testing.assert_array_equal(out, np.bitwise_not(a))


def test_bitwise_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.int32)
    b = np.zeros((4, 9), np.int32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_rocm_compiled_bitwise(
            _artifact(rt, "tessera.bitwise_or", 2), (a, b))


def test_bitwise_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.int32)
    with pytest.raises(ValueError, match="rocm_bitwise_compiled executor"):
        rt._execute_rocm_compiled_bitwise(
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
def test_bitwise_codegen_and_lowers(kind):
    import re
    d = ('module {\n  "tessera_rocm.bitwise"() {name = "w", '
         f'kind = "{kind}"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-bitwise-kernel")
    assert ir.returncode == 0, ir.stderr
    nargs = 3 if kind == "not" else 4
    m = re.search(r"gpu\.func @w\(([^)]*)\)", ir.stdout)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == nargs
    assert "memref<?xi32>" in ir.stdout
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-bitwise-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_bitwise_codegen_bad_kind_rejected():
    d = ('module {\n  "tessera_rocm.bitwise"() {name = "w", kind = "zz"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-bitwise-kernel")
    assert r.returncode != 0 and "unknown kind" in r.stderr
