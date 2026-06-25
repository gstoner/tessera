"""GPU-free codegen proof for compiler-generated linear attention.

Complements ``test_rocm_linear_attn_compiled.py`` (which executes on a real
gfx1151): here we only run ``generate-wmma-linear-attn-kernel`` (+ ROCDL
lowering) via tessera-opt and check structure, so CI without a GPU still gates
the codegen:

  * the kernel takes 7 args (Q,K,V,O,Sq,Sk,causal) — no scale, no softmax stats;
  * it emits NO ``math.exp`` (the linear form has no softmax);
  * ``feature_map="relu"`` emits ``arith.maximumf`` (φ on the loaded fragments);
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


def _directive(feature_map="identity", dtype="f16"):
    fm = f', feature_map = "{feature_map}"' if feature_map != "identity" else ""
    return (
        'module {\n  "tessera_rocm.linear_attn"() {name = "la", '
        f'head_dim = 64 : i64, dtype = "{dtype}"{fm}}} : () -> ()\n}}\n')


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
