"""Apple GEMM schedule candidates mined from MLX — seeds + axes, and the bridge
to the steel emitter.
"""

from __future__ import annotations

import pytest

from tessera.compiler.apple_gemm_schedules import (
    MLX_SEED_TILES,
    AppleGemmTile,
    DeviceClass,
    schedule_axes_for,
    select_seed_tile,
)
from tessera.compiler.msl_gemm_emit import (
    emit_steel_gemm_msl,
    validate_steel_gemm_structure,
)


def test_all_seed_tiles_are_steel_compatible():
    # Every MLX seed tile is a whole 8x8-fragment multiple -> feeds emit_steel_gemm_msl.
    for t in MLX_SEED_TILES:
        assert t.is_steel_compatible(), t


def test_seed_tiles_match_grounded_mlx_values():
    # Grounded from mlx/backend/metal/matmul.cpp GEMM_TPARAM_MACRO.
    assert AppleGemmTile(64, 64, 16, 1, 2) in MLX_SEED_TILES
    assert AppleGemmTile(64, 32, 32, 2, 2) in MLX_SEED_TILES
    assert AppleGemmTile(64, 32, 8, 4, 1) in MLX_SEED_TILES   # complex64


@pytest.mark.parametrize("dc,dtype,tb,large,want", [
    (DeviceClass.MEDIUM, "bf16", True, False, AppleGemmTile(64, 64, 16, 2, 2)),
    (DeviceClass.SMALL, "bf16", True, False, AppleGemmTile(64, 32, 32, 2, 2)),   # nt half
    (DeviceClass.SMALL, "bf16", False, False, AppleGemmTile(64, 64, 16, 1, 2)),  # nn
    (DeviceClass.LARGE, "bf16", True, True, AppleGemmTile(64, 64, 16, 1, 2)),    # large matmul
    (DeviceClass.LARGE, "f32", True, True, AppleGemmTile(32, 64, 16, 1, 2)),     # f32 large
    (DeviceClass.SMALL, "complex64", True, False, AppleGemmTile(64, 32, 8, 4, 1)),
])
def test_select_seed_tile_matches_mlx_heuristic(dc, dtype, tb, large, want):
    assert select_seed_tile(dc, dtype, transpose_b=tb, large=large) == want


def test_seed_tile_feeds_steel_emitter():
    # The integration the user asked for: an MLX-derived tile drives the steel GEMM.
    tile = select_seed_tile(DeviceClass.MEDIUM, "bf16")
    msl = emit_steel_gemm_msl("bf16", tile.bm, tile.bn, tile.bk)
    assert validate_steel_gemm_structure(msl, dtype="bf16").ok
    # 64x64 tile -> 8x8 = 64 output fragments.
    assert "simdgroup_matrix<float, 8, 8> acc[8 * 8]" in msl


def test_schedule_axes_alignment_and_swizzle():
    tile = AppleGemmTile(64, 64, 16, 2, 2)
    # M=128,N=128,K=64 all divisible -> aligned; tm=2 (<=3) -> swizzle_log 0.
    ax = schedule_axes_for(128, 128, 64, tile)
    assert ax.align_m and ax.align_n and ax.align_k
    assert ax.swizzle_log == 0
    assert not ax.do_axpby
    # ragged M, big M -> unaligned + swizzle_log 1.
    ax2 = schedule_axes_for(1000, 130, 70, tile)   # tm=ceil(1000/64)=16 >3
    assert not ax2.align_m and not ax2.align_n and not ax2.align_k
    assert ax2.swizzle_log == 1


def test_schedule_axes_axpby_epilogue():
    tile = AppleGemmTile(64, 64, 16, 2, 2)
    assert schedule_axes_for(64, 64, 64, tile, alpha=2.0).do_axpby
    assert schedule_axes_for(64, 64, 64, tile, beta=1.0).do_axpby
    assert not schedule_axes_for(64, 64, 64, tile, alpha=1.0, beta=0.0).do_axpby
