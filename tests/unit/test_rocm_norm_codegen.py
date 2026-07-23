"""GPU-free codegen proof for the compiler-generated ROCm norm kernels.

Complements ``test_rocm_norm_compiled.py`` (which executes on a real gfx1151):
here we only run ``generate-rocm-norm-kernel`` (+ ROCDL lowering) via tessera-opt
and check structure, so CI without a GPU still gates the codegen:

  * the kernel signature carries X/gamma/beta/O plus runtime M/K/eps and
    uniform affine-presence flags;
  * both kinds emit math.sqrt (the denominator) over the reduced stat;
  * layer_norm subtracts the row mean (arith.subf), rmsnorm does not;
  * an unknown kind is a named error;
  * both lower cleanly to ROCDL.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _directive(kind="rmsnorm", dtype="f32", *, backward=False):
    backward_attr = ", backward = true" if backward else ""
    return ('module {\n  "tessera_rocm.norm"() {name = "nm", '
            f'kind = "{kind}", dtype = "{dtype}"{backward_attr}}} : () -> ()\n}}\n')


def _opt(directive, *passes):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(TESSERA_OPT), "-", *passes],
                          input=directive, capture_output=True, text=True)


def _gen(directive):
    r = _opt(directive, "--generate-rocm-norm-kernel")
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_signature_and_sqrt():
    ir = _gen(_directive("rmsnorm"))
    m = re.search(r"gpu\.func @nm\(([^)]*)\)", ir)
    assert m, "no gpu.func @nm signature"
    args = [a.strip() for a in m.group(1).split(",") if a.strip()]
    assert len(args) == 9, f"expected affine runtime ABI, got {args}"
    assert args[-3].endswith("f32"), "eps must precede the affine flags"
    assert args[-2].endswith("i1") and args[-1].endswith("i1")
    assert "math.sqrt" in ir
    assert "arith.divf" in ir


def test_rmsnorm_has_no_mean_subtraction():
    # rmsnorm does not subtract the row mean; layer_norm does.
    assert "arith.subf" not in _gen(_directive("rmsnorm"))
    assert "arith.subf" in _gen(_directive("layer_norm"))


def test_unknown_kind_is_named_error():
    r = _opt(_directive("groupnorm"), "--generate-rocm-norm-kernel")
    assert r.returncode != 0
    assert "kind must be rmsnorm or layer_norm" in r.stderr


@pytest.mark.parametrize("kind", ["rmsnorm", "layer_norm"])
def test_backward_signature_and_deterministic_affine_reduction(kind):
    ir = _gen(_directive(kind, backward=True))
    match = re.search(r"gpu\.func @nm\(([^)]*)\)", ir)
    assert match, "no backward gpu.func @nm signature"
    args = [arg.strip() for arg in match.group(1).split(",") if arg.strip()]
    assert len(args) == 9, f"expected paired backward row ABI, got {args}"
    reduce_match = re.search(r"gpu\.func @nm_reduce\(([^)]*)\)", ir)
    assert reduce_match, "no deterministic affine-reduction kernel"
    reduce_args = [arg.strip() for arg in reduce_match.group(1).split(",")
                   if arg.strip()]
    assert len(reduce_args) == 8, f"expected reduction ABI, got {reduce_args}"
    assert "memref.atomic_rmw" not in ir
    assert "scf.for" in ir
    assert "math.sqrt" in ir


@pytest.mark.parametrize("kind", ["rmsnorm", "layer_norm"])
@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_backward_lowers_to_rocdl(kind, dtype):
    result = _opt(
        _directive(kind, dtype, backward=True),
        "--pass-pipeline=builtin.module(generate-rocm-norm-kernel,"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts))",
    )
    assert result.returncode == 0, result.stderr
    assert "llvm.func @nm_reduce" in result.stdout
    assert "llvm.atomicrmw" not in result.stdout


@pytest.mark.parametrize("kind", ["rmsnorm", "layer_norm"])
@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_lowers_to_rocdl(kind, dtype):
    r = _opt(_directive(kind, dtype),
             "--pass-pipeline=builtin.module(generate-rocm-norm-kernel,"
             "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
             "reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    assert "llvm." in r.stdout
