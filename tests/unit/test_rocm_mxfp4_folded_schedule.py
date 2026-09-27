"""Host contracts for the folded gfx1201 prefill load schedule.

Sync GFX1201-LANES-2026-09-27. The schedule is a set of performance keys
carried in Target IR; these tests pin what each key does to the emitted
source and launch grid. Device proof that every schedule preserves BF16
output bits lives in ``tests/device/rocm/test_mxfp4_folded_prefill.py``.
"""
from __future__ import annotations

import itertools

import pytest

from tessera.compiler.rocm_mxfp4_folded import (
    FOLDED_PREFILL_SCHEDULE_V1,
    FoldedPrefillSchedule,
    MAX_RASTER_GROUP_M,
    emit_mxfp4_folded_prefill_hip,
    folded_prefill_grid,
)

V2 = FoldedPrefillSchedule(
    raster_group_m=4, workgroup_mode="cu",
    staging_prefetch="register_next_slab",
    epilogue="complete_tile_vector_scales",
)


def test_default_schedule_is_the_original_kernel() -> None:
    assert FOLDED_PREFILL_SCHEDULE_V1 == FoldedPrefillSchedule()
    assert FOLDED_PREFILL_SCHEDULE_V1.compile_flags() == ()
    assert FOLDED_PREFILL_SCHEDULE_V1.raster == "n_major_2d"
    for full_k64 in (False, True):
        source = emit_mxfp4_folded_prefill_hip(full_k64=full_k64)
        assert source == emit_mxfp4_folded_prefill_hip(
            full_k64=full_k64, schedule=FOLDED_PREFILL_SCHEDULE_V1,
        )
        assert "const long m0 = (long)blockIdx.y * 256;" in source
        for marker in ("fetch_slab", "per_group", "scale_f32x4"):
            assert marker not in source


def test_selected_schedule_edits_only_its_named_mechanisms() -> None:
    source = emit_mxfp4_folded_prefill_hip(full_k64=True, schedule=V2)
    base = emit_mxfp4_folded_prefill_hip(full_k64=True)
    # Raster: a 1-D grid decoded into grouped row/column tile origins.
    assert "const long per_group = (long)4 * tiles_n;" in source
    assert "const long m0 = (long)blockIdx.y * 256;" not in source
    # Prefetch: one register fetch before the loop, one per slab after the
    # copy barrier, and the stash replaces the in-loop global copy.
    assert source.count("fetch_slab(0);") == 1
    assert source.count("if (kb + 64 < K) fetch_slab(kb + 64);") == 1
    assert "copy_u32x4 value = {};" not in source
    assert source.index("= next_b;") < source.index("fetch_slab(kb + 64)")
    # Epilogue: a CTA-uniform complete-tile fast path, then the unchanged
    # predicated production epilogue for every other launch.
    assert source.count("m0 + 256 <= M && n0 + 64 <= N") == 1
    assert "(reinterpret_cast<unsigned long>(As) & 15) == 0" in source
    fast, _, predicated = source.partition("    return;\n  }\n")
    assert "if (m < M && n < N) {" in predicated
    assert "if (m < M && n < N)" not in fast
    # The WMMA loop, its K-step guard and barriers are untouched.
    assert source.count("__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12") == 1
    assert "step < 4 && kb + step * 16 < K" in source
    assert source.count("__syncthreads()") == base.count("__syncthreads()")
    assert "__builtin_amdgcn_sched_barrier(6);" in source
    assert V2.compile_flags() == ("-mcumode",)
    assert V2.raster == "grouped_m4_1d"


def test_vector_epilogue_keeps_every_element_expression() -> None:
    source = emit_mxfp4_folded_prefill_hip(full_k64=True, schedule=V2)
    fast = source.partition("    return;\n  }\n")[0]
    assert "const float combined_scale = row_scale[jn] * activation_scale;" in fast
    assert "float scaled = partial * combined_scale;" in fast
    assert "!__builtin_isfinite(combined_scale) || combined_scale == 0.0f" in fast
    assert "if (partial == 0.0f && __builtin_isfinite(activation_scale))" in fast
    assert "(double)partial * (double)row_scale[jn] *" in fast


def test_schedule_refuses_incompatible_or_undeclared_choices() -> None:
    with pytest.raises(ValueError, match="complete K64"):
        emit_mxfp4_folded_prefill_hip(full_k64=False, schedule=V2)
    with pytest.raises(ValueError, match="safe-scale"):
        emit_mxfp4_folded_prefill_hip(
            full_k64=True, safe_epilogue=True,
            schedule=FoldedPrefillSchedule(epilogue="complete_tile_vector_scales"),
        )
    # A prefetch-free vector epilogue is legal on the K32-tail path.
    assert "scale_f32x4" in emit_mxfp4_folded_prefill_hip(
        full_k64=False,
        schedule=FoldedPrefillSchedule(epilogue="complete_tile_vector_scales"),
    )
    for bad in (
        {"raster_group_m": -1}, {"raster_group_m": MAX_RASTER_GROUP_M + 1},
        {"raster_group_m": True}, {"workgroup_mode": "wave"},
        {"staging_prefetch": "lds_double_buffer"}, {"epilogue": "fast"},
        {"row_guard": "lane"},
    ):
        with pytest.raises(ValueError):
            FoldedPrefillSchedule(**bad)  # type: ignore[arg-type]


def test_wave_row_guard_skips_only_waves_with_no_stored_row() -> None:
    """GFX1201-PERF-2026-09-27: the per-wave M guard is wave-uniform, keeps
    every barrier and the staging, and changes no element expression."""
    selected = FoldedPrefillSchedule(
        raster_group_m=4, staging_prefetch="register_next_slab",
        epilogue="complete_tile_vector_scales",
    )
    guarded = FoldedPrefillSchedule(
        raster_group_m=4, staging_prefetch="register_next_slab",
        epilogue="complete_tile_vector_scales", row_guard="wave",
    )
    base = emit_mxfp4_folded_prefill_hip(full_k64=True, schedule=selected)
    source = emit_mxfp4_folded_prefill_hip(full_k64=True, schedule=guarded)
    assert "wave_rows_live" not in base
    assert source.count("const bool wave_rows_live = m0 + wm * 64 < M;") == 1
    # Only the WMMA step loop and the epilogue are guarded; the slab copy,
    # the prefetch and both barriers are unchanged.
    assert "for (int step = 0; wave_rows_live && step < 4" in source
    assert source.count("__syncthreads()") == base.count("__syncthreads()")
    assert source.count("fetch_slab(") == base.count("fetch_slab(")
    assert source.index("if (!wave_rows_live) return;") > source.rindex("__syncthreads()")
    # The vector epilogue's completeness test is the wave's own block.
    assert "m0 + wm * 64 + 64 <= M && n0 + wn * 32 + 32 <= N" in source
    assert "m0 + 256 <= M && n0 + 64 <= N" not in source
    # Every element expression is the base kernel's.
    for expression in ("float scaled = partial * combined_scale;",
                       "(double)partial * (double)row_scale[jn] *",
                       "O[m * N + n] = (__bf16)scaled;"):
        assert source.count(expression) == base.count(expression)
    # Without the vector epilogue the guard still returns before the
    # predicated epilogue, and the default stays the original kernel.
    scalar = emit_mxfp4_folded_prefill_hip(
        full_k64=True, schedule=FoldedPrefillSchedule(row_guard="wave"))
    assert scalar.count("if (!wave_rows_live) return;") == 1
    assert FOLDED_PREFILL_SCHEDULE_V1.row_guard == "cta"
    assert guarded.as_dict()["row_guard"] == "wave"


def _grouped_origin(pid: int, m: int, n: int, group: int) -> tuple[int, int]:
    """Python transcription of the kernel's grouped raster decode."""
    tiles_m, tiles_n = (m + 255) // 256, (n + 63) // 64
    per_group = group * tiles_n
    first_m = (pid // per_group) * group
    group_rows = min(tiles_m - first_m, group)
    local = pid % per_group
    return (first_m + local % group_rows) * 256, (local // group_rows) * 64


@pytest.mark.parametrize("m,n,group", [
    (65, 48, 4), (256, 5120, 4), (257, 80, 4), (1024, 17408, 4),
    (1300, 200, 3), (2048, 5120, 2), (768, 128, 64), (4096, 64, 1),
])
def test_grouped_raster_covers_every_tile_once(m: int, n: int, group: int) -> None:
    schedule = FoldedPrefillSchedule(raster_group_m=group)
    grid = folded_prefill_grid(m, n, schedule)
    tiles_m, tiles_n = (m + 255) // 256, (n + 63) // 64
    assert grid == (tiles_m * tiles_n, 1, 1)
    origins = [_grouped_origin(pid, m, n, group) for pid in range(grid[0])]
    assert sorted(origins) == sorted(
        (tm * 256, tn * 64) for tm, tn in itertools.product(range(tiles_m), range(tiles_n))
    )
    # Consecutive CTAs share a weight column tile within each full group.
    if tiles_m >= group > 1:
        assert origins[0][1] == origins[1][1] and origins[0][0] != origins[1][0]
    assert folded_prefill_grid(m, n) == (tiles_n, tiles_m, 1)
