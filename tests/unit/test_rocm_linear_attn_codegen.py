"""GPU-free codegen proof for compiler-generated linear attention.

Complements ``test_rocm_linear_attn_compiled.py`` (which executes on a real
gfx1151): here we only run ``generate-wmma-linear-attn-kernel`` (+ ROCDL
lowering) via tessera-opt and check structure, so CI without a GPU still gates
the codegen:

  * the base kernel takes 7 args (Q,K,V,O,Sq,Sk,causal) — no scale/softmax;
  * the base form emits NO ``math.exp`` (no softmax);
  * ``feature_map="relu"`` emits ``arith.maximumf`` (φ on the loaded frags);
  * ``decay=true`` appends a trailing f32 (log λ) arg + emits ``math.exp``;
  * an unknown feature_map is a named error;
  * the kernel lowers cleanly to ROCDL (``rocdl.wmma``).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _directive(feature_map="identity", dtype="f16", decay=False):
    fm = (f', feature_map = "{feature_map}"'
          if feature_map != "identity" else "")
    dc = ", decay = true" if decay else ""
    return (
        'module {\n  "tessera_rocm.linear_attn"() {name = "la", '
        f'head_dim = 64 : i64, dtype = "{dtype}"{fm}{dc}}} : () -> ()\n}}\n')


def _opt(directive, *passes):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(TESSERA_OPT), "-", *passes],
                          input=directive, capture_output=True, text=True)


def _gen(directive):
    r = _opt(directive, "--generate-wmma-linear-attn-kernel")
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_signature_is_seven_args_no_softmax():
    ir = _gen(_directive())
    m = re.search(r"gpu\.func @la\(([^)]*)\)", ir)
    assert m, "no gpu.func @la signature"
    args = [a.strip() for a in m.group(1).split(",") if a.strip()]
    assert len(args) == 7, f"expected 7 args (Q,K,V,O,Sq,Sk,causal), got {args}"
    # No softmax: the linear form computes no exponential.
    assert "math.exp" not in ir


def test_relu_feature_map_emits_maximumf():
    assert "arith.maximumf" not in _gen(_directive("identity"))
    assert "arith.maximumf" in _gen(_directive("relu"))


def test_polynomial_2_is_accepted():
    # poly2 (x²) is a valid feature map (retention); just must not error.
    r = _opt(_directive("polynomial_2"), "--generate-wmma-linear-attn-kernel")
    assert r.returncode == 0, r.stderr


def _sig_args(ir):
    m = re.search(r"gpu\.func @la\(([^)]*)\)", ir)
    assert m, "no gpu.func @la signature"
    return [a.strip() for a in m.group(1).split(",") if a.strip()]


def test_decay_appends_f32_arg_and_emits_exp():
    base = _gen(_directive())
    dec = _gen(_directive(decay=True))
    assert len(_sig_args(base)) == 7
    da = _sig_args(dec)
    assert len(da) == 8 and da[-1].endswith("f32"), da
    assert "math.exp" not in base       # base linear form has no softmax/decay
    assert "math.exp" in dec            # decay = exp((i-j)·log λ)


def test_unknown_feature_map_is_named_error():
    r = _opt(_directive("elu"), "--generate-wmma-linear-attn-kernel")
    assert r.returncode != 0
    assert "feature_map must be" in r.stderr


def test_linear_attn_lowers_to_rocdl():
    r = _opt(_directive("relu"),
             "--pass-pipeline=builtin.module(generate-wmma-linear-attn-kernel,"
             "lower-tessera-target-to-rocdl,gpu.module(convert-scf-to-cf,"
             "convert-gpu-to-rocdl,reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    assert "rocdl.wmma" in r.stdout
