"""GPU-free codegen proof for Gemma-2 logit soft-capping in flash attention.

Complements ``test_rocm_logit_softcap_compiled.py`` (which executes on a real
gfx1151): here we only run ``generate-wmma-flash-attn-kernel`` (+ ROCDL lowering)
via tessera-opt and check structure, so CI without a GPU still gates the
soft-cap codegen:

  * `logit_softcap = true` appends a trailing `f32` (cap) arg to the signature,
  * it emits a `math.tanh` (the soft-cap transform),
  * it composes with `gqa` + `sliding_window` (cap is the last arg → 12 total),
  * the soft-capped kernel lowers cleanly (`tanh` → `__ocml_tanh`).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _directive(gqa=False, window=False, softcap=False):
    attrs = ""
    if gqa:
        attrs += ", gqa = true"
    if window:
        attrs += ", sliding_window = true"
    if softcap:
        attrs += ", logit_softcap = true"
    return (
        'module {\n  "tessera_rocm.flash_attn"() {name = "fa", '
        f'head_dim = 64 : i64, dtype = "f16"{attrs}}} : () -> ()\n}}\n')


def _opt(directive, *passes):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(TESSERA_OPT), "-", *passes],
                          input=directive, capture_output=True, text=True)


def _gen(directive):
    r = _opt(directive, "--generate-wmma-flash-attn-kernel")
    assert r.returncode == 0, r.stderr
    return r.stdout


def _sig(ir):
    m = re.search(r"gpu\.func @fa\(([^)]*)\)", ir)
    assert m, "no gpu.func @fa signature"
    return [a.strip() for a in m.group(1).split(",") if a.strip()]


def test_softcap_appends_f32_cap_arg_and_tanh():
    base = _sig(_gen(_directive()))
    capped = _sig(_gen(_directive(softcap=True)))
    assert len(base) == 8
    assert len(capped) == 9
    assert capped[-1].endswith("f32"), "cap arg must be the trailing f32"
    assert "math.tanh" in _gen(_directive(softcap=True))


def test_softcap_composes_with_gqa_and_window():
    # gqa (+2) + window (+1 W) + softcap (+1 cap) = 12 args; cap is last (f32).
    sig = _sig(_gen(_directive(gqa=True, window=True, softcap=True)))
    assert len(sig) == 12
    assert sig[-1].endswith("f32")


def test_softcap_kernel_lowers_to_rocdl():
    r = _opt(_directive(softcap=True),
             "--pass-pipeline=builtin.module(generate-wmma-flash-attn-kernel,"
             "lower-tessera-target-to-rocdl,gpu.module(convert-scf-to-cf,"
             "convert-gpu-to-rocdl,reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    assert "__ocml_tanh" in r.stdout
