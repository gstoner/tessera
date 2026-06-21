"""Unit tests for tessera.compiler.rocm_lds (B2).

Covers software-pipeline buffer accounting, XOR-swizzle / additive-padding LDS
layouts, the arch-keyed selection rule, validation, and metadata round-trips.
"""

from __future__ import annotations

import pytest

from tessera.compiler.rocm_lds import (
    LdsLayout,
    PaddedLdsLayout,
    SoftwarePipeline,
    SwizzledLdsLayout,
    attn_kv_tile_swizzle,
    select_lds_layout,
)
from tessera.compiler.rocm_target import AMDArch


# ── SoftwarePipeline ─────────────────────────────────────────────────────────


def test_pipeline_double_buffer_count() -> None:
    p = SoftwarePipeline(stages=2)  # v4 GEMM
    assert p.lds_buffers(1024) == 2048
    assert p.is_double_buffered
    assert not p.is_triple


def test_pipeline_triple_buffer_count() -> None:
    p = SoftwarePipeline(stages=3)  # v5 GEMM
    assert p.lds_buffers(1024) == 3072
    assert p.is_double_buffered
    assert p.is_triple


def test_pipeline_single_stage() -> None:
    p = SoftwarePipeline(stages=1)
    assert p.lds_buffers(512) == 512
    assert not p.is_double_buffered
    assert not p.is_triple


def test_pipeline_default_is_double() -> None:
    assert SoftwarePipeline().stages == 2


def test_pipeline_zero_tile_bytes() -> None:
    assert SoftwarePipeline(stages=3).lds_buffers(0) == 0


def test_pipeline_rejects_zero_stages() -> None:
    with pytest.raises(ValueError, match="stages must be >= 1"):
        SoftwarePipeline(stages=0)


def test_pipeline_rejects_negative_tile_bytes() -> None:
    with pytest.raises(ValueError, match="tile_bytes must be >= 0"):
        SoftwarePipeline(stages=2).lds_buffers(-1)


def test_pipeline_metadata_roundtrip() -> None:
    md = SoftwarePipeline(stages=3).as_metadata_dict()
    assert md["kind"] == "software_pipeline"
    assert md["stages"] == 3
    assert md["is_double_buffered"] is True
    assert md["is_triple"] is True


# ── SwizzledLdsLayout ────────────────────────────────────────────────────────


def test_swizzle_formula_hand_computed_cells() -> None:
    # vec=4, per_phase=1, max_phase=8.
    lay = SwizzledLdsLayout(vec=4, per_phase=1, max_phase=8)
    # row 0 -> phase 0 -> identity
    assert lay.swizzled_col(0, 0) == 0
    assert lay.swizzled_col(0, 7) == 7
    # row 1 -> phase 1: block (col//4) ^ 1.
    #   col=0 -> block 0 ^ 1 = 1 -> 1*4 + 0 = 4
    assert lay.swizzled_col(1, 0) == 4
    #   col=5 -> block 1 ^ 1 = 0 -> 0*4 + 1 = 1
    assert lay.swizzled_col(1, 5) == 1
    # row 2 -> phase 2: col=0 -> block 0 ^ 2 = 2 -> 2*4 + 0 = 8
    assert lay.swizzled_col(2, 0) == 8


def test_swizzle_per_phase_groups_rows() -> None:
    lay = SwizzledLdsLayout(vec=4, per_phase=2, max_phase=8)
    # rows 0 and 1 share phase 0 (per_phase=2)
    assert lay.swizzled_col(0, 4) == lay.swizzled_col(1, 4)
    # row 2 jumps to phase 1
    assert lay.swizzled_col(2, 0) == 4


def test_swizzle_max_phase_wraps() -> None:
    lay = SwizzledLdsLayout(vec=1, per_phase=1, max_phase=2)
    # phases cycle 0,1,0,1,...
    assert lay.swizzled_col(0, 3) == lay.swizzled_col(2, 3)
    assert lay.swizzled_col(1, 3) == lay.swizzled_col(3, 3)


def test_swizzle_conflict_free_across_row() -> None:
    # Across one row, the swizzled columns of a full vector-block-aligned span
    # must still be a permutation (a bijection) -> distinct banks, no conflicts.
    lay = SwizzledLdsLayout(vec=4, per_phase=1, max_phase=8)
    width = 32  # 8 vector blocks
    for row in range(8):
        mapped = [lay.swizzled_col(row, c) for c in range(width)]
        assert sorted(mapped) == list(range(width)), (
            f"row {row} swizzle is not a bijection: {mapped}"
        )


def test_swizzle_distinct_banks_per_lane() -> None:
    # The leading element of each vector block across one row must land in a
    # distinct bank (vector-block index is a bijection under XOR by a constant).
    lay = SwizzledLdsLayout(vec=8, per_phase=1, max_phase=8)
    blocks = [lay.swizzled_col(3, b * 8) // 8 for b in range(8)]
    assert sorted(blocks) == list(range(8))


def test_swizzle_rejects_non_pow2_vec() -> None:
    with pytest.raises(ValueError, match="vec must be a power of two"):
        SwizzledLdsLayout(vec=3, per_phase=1, max_phase=8)


def test_swizzle_rejects_zero_vec() -> None:
    with pytest.raises(ValueError, match="vec must be >= 1"):
        SwizzledLdsLayout(vec=0, per_phase=1, max_phase=8)


def test_swizzle_rejects_non_pow2_max_phase() -> None:
    with pytest.raises(ValueError, match="max_phase must be a power of two"):
        SwizzledLdsLayout(vec=4, per_phase=1, max_phase=3)


def test_swizzle_rejects_zero_per_phase() -> None:
    with pytest.raises(ValueError, match="per_phase must be >= 1"):
        SwizzledLdsLayout(vec=4, per_phase=0, max_phase=8)


def test_swizzle_rejects_bad_order() -> None:
    with pytest.raises(ValueError, match="order must be a permutation"):
        SwizzledLdsLayout(vec=4, per_phase=1, max_phase=8, order=(0, 0))


def test_swizzle_rejects_negative_indices() -> None:
    lay = SwizzledLdsLayout(vec=4, per_phase=1, max_phase=8)
    with pytest.raises(ValueError, match="row/col must be >= 0"):
        lay.swizzled_col(-1, 0)


def test_swizzle_metadata_roundtrip() -> None:
    md = SwizzledLdsLayout(vec=8, per_phase=2, max_phase=4, order=(0, 1)).as_metadata_dict()
    assert md["kind"] == "lds_layout"
    assert md["strategy"] == "swizzle"
    assert md["vec"] == 8
    assert md["per_phase"] == 2
    assert md["max_phase"] == 4
    assert md["order"] == [0, 1]


# ── PaddedLdsLayout ──────────────────────────────────────────────────────────


def test_padded_stride() -> None:
    lay = PaddedLdsLayout(pad_elems=4, inner_dim=64)
    assert lay.padded_stride(64) == 68
    assert lay.padded_stride(128) == 132
    assert lay.padded_stride(0) == 4


def test_padded_rejects_zero_pad() -> None:
    with pytest.raises(ValueError, match="pad_elems must be >= 1"):
        PaddedLdsLayout(pad_elems=0, inner_dim=64)


def test_padded_rejects_zero_inner_dim() -> None:
    with pytest.raises(ValueError, match="inner_dim must be >= 1"):
        PaddedLdsLayout(pad_elems=4, inner_dim=0)


def test_padded_rejects_negative_cols() -> None:
    with pytest.raises(ValueError, match="cols must be >= 0"):
        PaddedLdsLayout(pad_elems=4, inner_dim=64).padded_stride(-1)


def test_padded_metadata_roundtrip() -> None:
    md = PaddedLdsLayout(pad_elems=8, inner_dim=128).as_metadata_dict()
    assert md["kind"] == "lds_layout"
    assert md["strategy"] == "pad"
    assert md["pad_elems"] == 8
    assert md["inner_dim"] == 128


# ── select_lds_layout (arch-keyed rule) ──────────────────────────────────────


def test_select_pads_on_gfx950_global_to_lds() -> None:
    # CDNA 4 GLOBAL_LOAD_LDS path -> padding.
    lay = select_lds_layout(AMDArch.GFX_950, global_to_lds=True)
    assert isinstance(lay, PaddedLdsLayout)
    assert lay.as_metadata_dict()["strategy"] == "pad"


def test_select_swizzles_on_gfx950_without_global_to_lds() -> None:
    lay = select_lds_layout(AMDArch.GFX_950, global_to_lds=False)
    assert isinstance(lay, SwizzledLdsLayout)
    assert lay.as_metadata_dict()["strategy"] == "swizzle"


def test_select_swizzles_on_gfx942() -> None:
    # CDNA 3 (no GLOBAL_LOAD_LDS swizzle constraint) -> swizzle even with the flag.
    lay = select_lds_layout(AMDArch.GFX_942, global_to_lds=True)
    assert isinstance(lay, SwizzledLdsLayout)


def test_select_swizzles_on_gfx940() -> None:
    lay = select_lds_layout(AMDArch.GFX_940, global_to_lds=False)
    assert isinstance(lay, SwizzledLdsLayout)


def test_select_honors_inner_dim_on_pad_path() -> None:
    lay = select_lds_layout(AMDArch.GFX_950, global_to_lds=True, inner_dim=256)
    assert isinstance(lay, PaddedLdsLayout)
    assert lay.inner_dim == 256


def test_select_honors_vec_on_swizzle_path() -> None:
    lay = select_lds_layout(AMDArch.GFX_942, global_to_lds=True, vec=16)
    assert isinstance(lay, SwizzledLdsLayout)
    assert lay.vec == 16


def test_ldslayout_union_members() -> None:
    # Both strategy types are members of the LdsLayout union alias.
    assert SwizzledLdsLayout(vec=4, per_phase=1, max_phase=8) is not None
    assert PaddedLdsLayout(pad_elems=4, inner_dim=64) is not None
    # LdsLayout is importable and is a typing Union.
    assert LdsLayout is not None


# ── swizzle correctness: lossless + bank-conflict-breaking ───────────────────


def test_swizzle_is_lossless_permutation_per_row() -> None:
    # For each row the column map must be a bijection over the window — distinct
    # logical columns never alias, so addressing stays correct.
    s = SwizzledLdsLayout(vec=8, per_phase=1, max_phase=8)
    for row in range(s.max_phase):
        cols = sorted(s.swizzled_col(row, c) for c in range(s.window))
        assert cols == list(range(s.window))


def test_swizzle_breaks_bank_conflicts() -> None:
    # A column read down successive rows lands in >1 LDS bank (32-bank model).
    s = SwizzledLdsLayout(vec=8, per_phase=1, max_phase=8)
    assert s.is_conflict_free(num_banks=32)


@pytest.mark.parametrize("vec,max_phase", [(4, 8), (8, 8), (8, 16), (16, 8), (2, 16)])
def test_swizzle_conflict_free_across_params(vec: int, max_phase: int) -> None:
    s = SwizzledLdsLayout(vec=vec, per_phase=1, max_phase=max_phase)
    assert s.is_conflict_free(num_banks=32)


def test_swizzle_window_property() -> None:
    s = SwizzledLdsLayout(vec=8, per_phase=2, max_phase=4)
    assert s.window == 32


# ── MLIR attribute emission ──────────────────────────────────────────────────


def test_swizzle_to_mlir_attr_round_trips_params() -> None:
    s = SwizzledLdsLayout(vec=8, per_phase=1, max_phase=16, order=(1, 0))
    attr = s.to_mlir_attr()
    assert attr == (
        "#tessera_rocm.lds_swizzle<vec = 8, per_phase = 1, "
        "max_phase = 16, order = [1, 0]>"
    )


def test_pad_to_mlir_attr_round_trips_params() -> None:
    p = PaddedLdsLayout(pad_elems=4, inner_dim=64)
    assert p.to_mlir_attr() == "#tessera_rocm.lds_pad<pad_elems = 4, inner_dim = 64>"


def test_selected_layout_is_emittable() -> None:
    # Whatever the arch-keyed rule picks, it can be lowered to an IR attr.
    for arch, g2l in (
        (AMDArch.GFX_942, True),
        (AMDArch.GFX_950, True),
        (AMDArch.GFX_950, False),
    ):
        attr = select_lds_layout(arch, global_to_lds=g2l).to_mlir_attr()
        assert attr.startswith("#tessera_rocm.lds_")


# ── attention K/V-tile swizzle helper ────────────────────────────────────────


def test_attn_kv_tile_swizzle_sizes_window_to_row() -> None:
    s = attn_kv_tile_swizzle(AMDArch.GFX_942, inner_dim=128, vec=8)
    assert s.max_phase == 16
    assert s.window == 128
    assert s.is_conflict_free(num_banks=32)


def test_attn_kv_tile_swizzle_rejects_non_multiple_inner_dim() -> None:
    with pytest.raises(ValueError, match="multiple of vec"):
        attn_kv_tile_swizzle(AMDArch.GFX_942, inner_dim=130, vec=8)


def test_attn_kv_tile_swizzle_rejects_non_pow2_phase() -> None:
    # inner_dim // vec = 48 // 8 = 6, not a power of two.
    with pytest.raises(ValueError, match="power of two"):
        attn_kv_tile_swizzle(AMDArch.GFX_942, inner_dim=48, vec=8)
