"""GPU-free codegen proof for the fused-epilogue WMMA GEMM.

Complements ``test_rocm_fused_epilogue_compiled.py`` (which executes on a real
gfx1151): here we only run ``generate-wmma-gemm-kernel`` + lower to ROCDL via
tessera-opt and check the *structure* — so CI without a GPU still gates the
epilogue codegen. We assert that:

  * ``bias = true`` appends a trailing memref operand to the kernel signature,
  * each activation emits its portable arithmetic (relu directly; silu/gelu
    through the shared bounded-tanh polynomial),
  * the polynomial lowers completely to LLVM arithmetic without unresolved
    device-math libcalls, and
  * the epilogue on an integer dtype is a named error, never a silent no-op.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _directive(dtype="f16", bias=False, activation="none"):
    epi = ""
    if bias:
        epi += ", bias = true"
    if activation != "none":
        epi += f', activation = "{activation}"'
    return (
        'module {\n  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : i64, '
        'n = 16 : i64, k = 16 : i64, '
        f'dtype = "{dtype}"{epi}}} : () -> ()\n}}\n')


def _opt(directive, *passes):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(TESSERA_OPT), "-", *passes],
                          input=directive, capture_output=True, text=True)


def _gen(directive):
    r = _opt(directive, "--generate-wmma-gemm-kernel")
    assert r.returncode == 0, r.stderr
    return r.stdout


def _sig(ir):
    import re
    m = re.search(r"gpu\.func @gemm\(([^)]*)\)", ir)
    assert m, "no gpu.func @gemm signature"
    return m.group(1)


def test_bias_appends_trailing_memref_operand():
    plain = _sig(_gen(_directive()))
    biased = _sig(_gen(_directive(bias=True)))
    # Plain signature: A, B, D, M, N, K (3 memref + 3 index).
    assert plain.count("memref<?x") == 3
    # Bias signature: one extra memref operand (the per-column bias).
    assert biased.count("memref<?x") == 4
    # ... and the body loads it and adds it to the accumulator.
    body = _gen(_directive(bias=True))
    assert "memref.load" in body and "arith.addf" in body


def test_relu_activation_emits_maximum() -> None:
    assert "arith.maximumf" in _gen(_directive(activation="relu"))


@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_transcendental_activation_emits_bounded_portable_arithmetic(activation):
    ir = _gen(_directive(activation=activation))
    for op in ("arith.maximumf", "arith.minimumf", "arith.divf"):
        assert op in ir, f"{activation} should emit {op}"
    assert "math.exp" not in ir and "math.tanh" not in ir


@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_activation_lowers_to_native_portable_arithmetic(activation):
    """The shared bounded approximation reaches native LLVM arithmetic."""
    r = _opt(_directive(activation=activation),
             "--pass-pipeline=builtin.module(generate-wmma-gemm-kernel,"
             "lower-tessera-target-to-rocdl,gpu.module(convert-scf-to-cf,"
             "convert-gpu-to-rocdl,reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    for op in ("llvm.fadd", "llvm.fmul", "llvm.fdiv"):
        assert op in r.stdout, f"{activation} did not lower to {op}"
    assert "math." not in r.stdout
    assert "__ocml_" not in r.stdout


@pytest.mark.parametrize("dtype", ["int8", "int4"])
def test_epilogue_on_integer_dtype_is_a_named_error(dtype):
    r = _opt(_directive(dtype=dtype, activation="gelu"),
             "--generate-wmma-gemm-kernel")
    assert r.returncode != 0
    assert "float-only" in r.stderr
