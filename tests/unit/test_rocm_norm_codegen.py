"""GPU-free codegen proof for the compiler-generated ROCm norm kernels.

Complements ``test_rocm_norm_compiled.py`` (which executes on a real gfx1151):
here we only run ``generate-rocm-norm-kernel`` (+ ROCDL lowering) via tessera-opt
and check structure, so CI without a GPU still gates the codegen:

  * the kernel signature is (X, O : memref, M, K : index, eps : f32);
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


def _directive(kind="rmsnorm", dtype="f32"):
    return ('module {\n  "tessera_rocm.norm"() {name = "nm", '
            f'kind = "{kind}", dtype = "{dtype}"}} : () -> ()\n}}\n')


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
    assert len(args) == 5, f"expected (X, O, M, K, eps), got {args}"
    assert args[-1].endswith("f32"), "eps must be the trailing f32 arg"
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
@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_lowers_to_rocdl(kind, dtype):
    r = _opt(_directive(kind, dtype),
             "--pass-pipeline=builtin.module(generate-rocm-norm-kernel,"
             "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
             "reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    assert "llvm." in r.stdout
