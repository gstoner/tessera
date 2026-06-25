"""GPU-free codegen proof for sliding-window flash attention.

Complements ``test_rocm_sliding_window_compiled.py`` (which executes on a real
gfx1151): here we only run ``generate-wmma-flash-attn-kernel`` (+ ROCDL lowering)
via tessera-opt and check structure, so CI without a GPU still gates the
windowed codegen:

  * `sliding_window = true` appends a trailing `index` (W) arg to the signature,
  * it composes with `gqa` (window arg follows heads/kv_ratio — 11 args total),
  * the windowed + gqa kernels still lower cleanly to ROCDL.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _directive(gqa=False, window=False):
    attrs = ""
    if gqa:
        attrs += ", gqa = true"
    if window:
        attrs += ", sliding_window = true"
    return (
        'module {\n  "tessera_rocm.flash_attn"() {name = "fa", '
        f'head_dim = 64 : i64, dtype = "f16"{attrs}}} : () -> ()\n}}\n')


def _opt(directive, *passes):
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(TESSERA_OPT), "-", *passes],
                          input=directive, capture_output=True, text=True)


def _nargs(directive):
    r = _opt(directive, "--generate-wmma-flash-attn-kernel")
    assert r.returncode == 0, r.stderr
    m = re.search(r"gpu\.func @fa\(([^)]*)\)", r.stdout)
    assert m, "no gpu.func @fa signature"
    return len([a for a in m.group(1).split(",") if a.strip()])


def test_window_appends_w_arg():
    # Base flash_attn: Q,K,V,O,Sq,Sk,scale,causal = 8 args.
    assert _nargs(_directive()) == 8
    # Sliding window adds the trailing W arg.
    assert _nargs(_directive(window=True)) == 9


def test_window_composes_with_gqa():
    # gqa adds heads + kv_ratio (10); window adds W after them (11).
    assert _nargs(_directive(gqa=True)) == 10
    assert _nargs(_directive(gqa=True, window=True)) == 11


@pytest.mark.parametrize("gqa", [False, True])
def test_windowed_kernel_lowers_to_rocdl(gqa):
    r = _opt(_directive(gqa=gqa, window=True),
             "--pass-pipeline=builtin.module(generate-wmma-flash-attn-kernel,"
             "lower-tessera-target-to-rocdl,gpu.module(convert-scf-to-cf,"
             "convert-gpu-to-rocdl,reconcile-unrealized-casts))")
    assert r.returncode == 0, r.stderr
    assert "rocdl.wmma" in r.stdout or "llvm" in r.stdout
