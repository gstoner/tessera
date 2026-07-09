"""GPU-free codegen proof for additive attention bias in flash attention.

Complements ``test_rocm_flash_attn_bias_compiled.py`` (which executes on a real
gfx1151): here we only run ``generate-wmma-flash-attn-kernel`` (+ ROCDL lowering)
via tessera-opt and check structure, so CI without a GPU still gates the bias
codegen:

  * `attn_bias = true` appends a trailing `memref<?xf32>` arg to the signature,
  * the kernel actually LOADS from that trailing bias arg (the additive read),
  * it composes with gqa + sliding_window + softcap (bias is LAST → 13 total),
  * the bias kernel lowers cleanly through ROCDL.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _directive(gqa=False, window=False, softcap=False, bias=False):
    attrs = ""
    if gqa:
        attrs += ", gqa = true"
    if window:
        attrs += ", sliding_window = true"
    if softcap:
        attrs += ", logit_softcap = true"
    if bias:
        attrs += ", attn_bias = true"
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


def test_bias_appends_trailing_f32_memref_and_is_loaded():
    ir = _gen(_directive(bias=True))
    base = _sig(_gen(_directive()))
    withbias = _sig(ir)
    assert len(base) == 8
    assert len(withbias) == 9
    # The bias arg is the trailing f32 memref (like O, but last).
    assert withbias[-1].endswith("memref<?xf32>"), withbias[-1]
    # It must actually be read — a memref.load from the last arg (%arg8 here).
    bias_arg = re.match(r"(%\w+)", withbias[-1]).group(1)
    assert re.search(rf"memref\.load {re.escape(bias_arg)}\[", ir), \
        "bias arg is never loaded (the additive read is missing)"


def test_bias_composes_with_gqa_window_softcap():
    # gqa (+2) + window (+1 W) + softcap (+1 cap) + bias (+1 memref) = 13 args;
    # the bias memref is LAST so every prior variant's arg index is unchanged.
    sig = _sig(_gen(_directive(gqa=True, window=True, softcap=True, bias=True)))
    assert len(sig) == 13
    assert sig[-1].endswith("memref<?xf32>")


def test_bias_kernel_lowers_to_rocdl():
    r = _opt(_directive(bias=True),
             "--pass-pipeline=builtin.module(generate-wmma-flash-attn-kernel,"
             "lower-tessera-target-to-rocdl,gpu.module(convert-scf-to-cf,"
             "convert-gpu-to-rocdl,reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
