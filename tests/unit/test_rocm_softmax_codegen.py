"""GPU-free codegen proof for the compiler-generated ROCm softmax kernel.

Complements ``test_rocm_softmax_compiled.py`` (which executes on a real gfx1151):
here we only run ``generate-rocm-softmax-kernel`` (+ ROCDL lowering) via
tessera-opt and check structure, so CI without a GPU still gates the codegen:

  * the kernel signature is (X, O : memref, M, K : index);
  * it emits the reduction math (math.exp + arith.maximumf for the row max);
  * f16/bf16 round-trip through f32 (arith.extf / arith.truncf);
  * an unknown dtype is a named error;
  * it lowers cleanly to ROCDL (no WMMA path needed).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._support.compiler_tool import run_tessera_opt

REPO = Path(__file__).resolve().parents[2]


def _directive(dtype="f32"):
    return ('module {\n  "tessera_rocm.softmax"() {name = "sm", '
            f'dtype = "{dtype}"}} : () -> ()\n}}\n')


def _opt(directive, *passes):
    return run_tessera_opt(directive, *passes)


def _gen(directive):
    r = _opt(directive, "--generate-rocm-softmax-kernel")
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_signature_and_reduction_math():
    ir = _gen(_directive("f32"))
    m = re.search(r"gpu\.func @sm\(([^)]*)\)", ir)
    assert m, "no gpu.func @sm signature"
    args = [a.strip() for a in m.group(1).split(",") if a.strip()]
    assert len(args) == 4, f"expected (X, O, M, K), got {args}"
    assert "math.exp" in ir            # the exp pass
    assert "arith.maximumf" in ir      # the row-max reduction
    assert "arith.divf" in ir          # the final normalize
    # f32 storage: no extend/truncate round-trip.
    assert "arith.extf" not in ir and "arith.truncf" not in ir


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_low_precision_roundtrips_through_f32(dtype):
    ir = _gen(_directive(dtype))
    assert "arith.extf" in ir and "arith.truncf" in ir


def test_unknown_dtype_is_named_error():
    """The rejection MOVED to ODS (W1.1b) — it did not weaken.

    This used to assert the generator's own "dtype must be f32, f16, or bf16".
    `ROCM_FloatDTypeAttr` now rejects `int8` at verification, before the pass
    runs, so the generator's message is never reached. That is the point of the
    item rather than a regression: the same program is still refused, with a
    message that still names the attribute and its legal set, one layer earlier.

    Both messages are accepted so the test states the CONTRACT — an illegal
    dtype is a named error, not a silent fallthrough — instead of pinning which
    layer happens to catch it. Asserting only the ODS text would break again if
    the constraint is ever narrowed to leave `int8` for the generator to reject.
    """
    r = _opt(_directive("int8"), "--generate-rocm-softmax-kernel")
    assert r.returncode != 0
    assert ("dtype must be f32, f16, or bf16" in r.stderr
            or "attribute 'dtype' failed to satisfy constraint" in r.stderr), \
        r.stderr
    # Whichever layer refuses it, the diagnostic must name the attribute.
    assert "dtype" in r.stderr


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_lowers_to_rocdl(dtype):
    r = _opt(_directive(dtype),
             "--pass-pipeline=builtin.module(generate-rocm-softmax-kernel,"
             "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
             "reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    assert "llvm." in r.stdout
