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
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _directive(dtype="f32"):
    return ('module {\n  "tessera_rocm.softmax"() {name = "sm", '
            f'dtype = "{dtype}"}} : () -> ()\n}}\n')


def _opt(directive, *passes):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(TESSERA_OPT), "-", *passes],
                          input=directive, capture_output=True, text=True)


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
    r = _opt(_directive("int8"), "--generate-rocm-softmax-kernel")
    assert r.returncode != 0
    assert "dtype must be f32, f16, or bf16" in r.stderr


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_lowers_to_rocdl(dtype):
    r = _opt(_directive(dtype),
             "--pass-pipeline=builtin.module(generate-rocm-softmax-kernel,"
             "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
             "reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    assert "llvm." in r.stdout
