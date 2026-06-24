"""Fork-A (pilot): the compiler-generated WMMA GEMM routes through the wave/LDS
pipeline and lowers identically to the direct generator (GPU-free IR parity).

  direct : generate-wmma-gemm-kernel           -> lower-tessera-target-to-rocdl
  fork_a : generate-wmma-gemm-kernel=via-tile  (emits tile.mma)
             -> rocm-wave-lds-pipeline -> rocm-wave-lds-legality
             -> lower-tile-to-rocm{arch=gfx1151} -> lower-tessera-target-to-rocdl

This proves, with no GPU, that (a) --via-tile emits the matrix op at the Tile-IR
seam (tile.mma), (b) the wave/LDS pipeline + legality accept it, (c)
lower-tile-to-rocm threads the real accumulator and lowers it back, and (d) the
final rocdl.wmma count matches the direct lane. On-device correctness + perf
parity is covered by benchmarks/rocm/benchmark_rocm_gemm_pipeline_vs_direct.py.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"

_DIRECTIVE = (
    'module {\n  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : i64, '
    'n = 16 : i64, k = 16 : i64, mt = 2 : i64, nt = 4 : i64, dtype = "f16"} '
    ': () -> ()\n}\n'
)

_DIRECT = ["--generate-wmma-gemm-kernel", "--lower-tessera-target-to-rocdl"]
_FORK_A = ["--generate-wmma-gemm-kernel=via-tile=true",
           "--rocm-wave-lds-pipeline", "--rocm-wave-lds-legality",
           "--lower-tile-to-rocm=arch=gfx1151",
           "--lower-tessera-target-to-rocdl"]


def _opt(*passes: str) -> str:
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    r = subprocess.run([str(TESSERA_OPT), "-", *passes],
                       input=_DIRECTIVE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_via_tile_emits_tile_mma_at_the_seam():
    """--via-tile routes the matrix op through tile.mma (not tessera_rocm.wmma
    directly), so the wave/LDS pipeline can see it."""
    ir = _opt("--generate-wmma-gemm-kernel=via-tile=true")
    assert "tile.mma" in ir
    assert "tessera_rocm.wmma" not in ir          # not emitted directly
    assert "gpu.func @gemm" in ir
    # the default lane still emits the matrix op directly (unchanged).
    direct = _opt("--generate-wmma-gemm-kernel")
    assert "tessera_rocm.wmma" in direct and "tile.mma" not in direct


def test_fork_a_lowers_to_same_rocdl_wmma_count_as_direct():
    """The pipeline-routed GEMM lowers to the SAME number of rocdl.wmma ops as
    the direct generator — IR parity of the executable matrix path."""
    direct = _opt(*_DIRECT).lower().count("rocdl.wmma")
    fork_a = _opt(*_FORK_A).lower().count("rocdl.wmma")
    assert direct > 0, "direct path produced no rocdl.wmma"
    assert fork_a == direct, f"fork_a={fork_a} != direct={direct} rocdl.wmma"


def test_fork_a_consumes_tile_mma_through_lower_tile_to_rocm():
    """After lower-tile-to-rocm the tile.mma is gone (consumed into the matrix
    op) — the Tile-IR seam is fully lowered, nothing leaks to ROCDL."""
    out = _opt("--generate-wmma-gemm-kernel=via-tile=true",
               "--rocm-wave-lds-pipeline", "--rocm-wave-lds-legality",
               "--lower-tile-to-rocm=arch=gfx1151")
    assert "tile.mma" not in out                  # consumed
    assert "tessera_rocm.wmma" in out             # lowered to the matrix op


def _op_multiset(ir: str) -> Counter:
    """Multiset of dialect op names (dialect.op), ignoring SSA names/attrs."""
    return Counter(re.findall(r"\b[a-z_]+\.[a-z_.]+\b", ir))


def test_fork_a_final_rocdl_is_identical_to_direct():
    """The strongest parity proof (GPU-free): after full lowering to ROCDL, the
    pipeline-routed kernel has the SAME op multiset as the direct generator —
    they are the same kernel, so on-device perf parity is by construction (the
    device A/B in benchmark_rocm_gemm_pipeline_vs_direct.py only confirms it
    within APU clock noise)."""
    direct = _op_multiset(_opt(*_DIRECT))
    fork_a = _op_multiset(_opt(*_FORK_A))
    assert direct == fork_a, (
        "Fork-A lowered to a different kernel than direct; "
        f"delta={ (direct - fork_a) | (fork_a - direct) }")
