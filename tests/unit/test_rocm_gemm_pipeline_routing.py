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

from pathlib import Path

from tests._support.compiler_tool import run_tessera_opt

REPO = Path(__file__).resolve().parents[2]

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
    r = run_tessera_opt(_DIRECTIVE, *passes)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_via_tile_emits_tile_mma_at_the_seam():
    """The 2x4 pilot emits the complete typed Tile fragment chain."""
    ir = _opt("--generate-wmma-gemm-kernel=via-tile=true")
    assert ir.count("tile.view") == 24
    assert ir.count("tile.fragment_pack") == 24
    assert ir.count("tile.fragment_zero") == 8
    assert ir.count("tile.mma") == 32
    assert ir.count("tile.fragment_unpack") == 16
    assert ir.count("tile.store") == 16
    assert "!tile.fragment<m = 16, n = 16, k = 16" in ir
    assert 'leading_dim = 0' in ir
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


def test_fork_a_final_rocdl_has_no_tile_boundary_leaks():
    """The typed address path differs structurally, but fully lowers to ROCDL.

    Numerical identity is the exact-device gate; op-multiset identity ceased to
    be a valid claim once fragment_pack became the owner of lane addressing.
    """
    out = _opt(*_FORK_A)
    assert out.lower().count("rocdl.wmma") == 32
    for leaked in ("tile.view", "tile.fragment_", "tile.mma", "tile.store",
                   "unrealized_conversion_cast"):
        assert leaked not in out
