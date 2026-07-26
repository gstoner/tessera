from benchmarks.rocm.benchmark_rocm_ssa_lds_pipeline import (
    _median_mad,
    _module,
    _validate_lowered,
    _validate_plan,
)


def test_ssa_lds_benchmark_source_scales_independent_stages():
    source = _module(4)
    assert source.count('"tile.async_copy"') == 4
    assert source.count('"tile.wait_async"') == 4
    assert "#tile.buffer_ref" not in source


def test_ssa_lds_benchmark_validates_planner_structure():
    plan = (
        "tile.pipeline_init " * 2
        + "tile.alloc " * 3
        + "tile.async_copy " * 3
        + "tile.wait_async " * 3
        + "tile.pipeline_advance " * 6
        + "tile.dealloc " * 3
    )
    assert _validate_plan(plan, 3)["legacy_buffer_refs"] == 0


def test_ssa_lds_benchmark_validates_target_consumption():
    lowered = "tessera_rocm.async_copy " * 2 + "tessera_rocm.wait " * 2 + 'buffer_identity = "ssa_allocation" ' * 2
    counts = _validate_lowered(lowered, 2)
    assert counts["portable_allocations"] == 0
    assert counts["portable_pipeline_ops"] == 0


def test_ssa_lds_benchmark_uses_median_and_mad():
    assert _median_mad([1.0, 2.0, 100.0]) == (2.0, 1.0)
