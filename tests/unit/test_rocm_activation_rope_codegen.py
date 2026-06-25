"""GPU-free codegen proof for the compiler-generated ROCm activation + rope
kernels. Complements the *_compiled.py GPU oracle tests: here we only run the
generators (+ ROCDL lowering) via tessera-opt and check structure, so CI without
a GPU still gates the codegen.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _opt(directive, *passes):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(TESSERA_OPT), "-", *passes],
                          input=directive, capture_output=True, text=True)


def _act(kind, dtype="f32"):
    return ('module {\n  "tessera_rocm.activation"() {name = "a", '
            f'kind = "{kind}", dtype = "{dtype}"}} : () -> ()\n}}\n')


def _rope(dtype="f32"):
    return ('module {\n  "tessera_rocm.rope"() {name = "r", '
            f'dtype = "{dtype}"}} : () -> ()\n}}\n')


def _gen(directive, pass_name):
    r = _opt(directive, f"--{pass_name}")
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.mark.parametrize("kind,op", [
    ("gelu", "math.tanh"), ("silu", "math.exp"), ("relu", "arith.maximumf"),
])
def test_activation_emits_its_math(kind, op):
    ir = _gen(_act(kind), "generate-rocm-activation-kernel")
    m = re.search(r"gpu\.func @a\(([^)]*)\)", ir)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == 3
    assert op in ir


def test_activation_unknown_kind_named_error():
    r = _opt(_act("mish"), "--generate-rocm-activation-kernel")
    assert r.returncode != 0
    assert "kind must be gelu, silu, or relu" in r.stderr


def test_rope_signature_and_trig():
    ir = _gen(_rope(), "generate-rocm-rope-kernel")
    m = re.search(r"gpu\.func @r\(([^)]*)\)", ir)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == 5
    assert "math.cos" in ir and "math.sin" in ir


@pytest.mark.parametrize("directive,pass_name", [
    (_act("gelu", "f16"), "generate-rocm-activation-kernel"),
    (_rope("f16"), "generate-rocm-rope-kernel"),
])
def test_lowers_to_rocdl(directive, pass_name):
    r = _opt(directive,
             f"--pass-pipeline=builtin.module({pass_name},"
             "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
             "reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    assert "llvm." in r.stdout
