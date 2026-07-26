from benchmarks.rocm.benchmark_rocm_attention_backward_program import (
    PROGRAM_WALL_BASELINE_MS,
    PROGRAM_WALL_MAX_REGRESSION,
    _record,
)


def test_backward_program_benchmark_separates_resident_program_wall() -> None:
    record = _record(
        package_ms=500.0,
        operation_total_ms=4.0,
        result={
            "kernel_wall_samples_ms": [0.4, 0.3, 0.5],
            "workspace_bytes": 4096,
            "entry_symbols": ("forward", "pre", "dkdv", "reduce", "dq"),
        },
        max_abs_error={"dq": 0.001, "dk": 0.002, "dv": 0.003},
        image_bytes=32768,
    )

    assert record["timing"]["program_wall"]["median_ms"] == 0.4
    assert record["timing"]["program_wall"]["resident_module"] is True
    assert record["timing"]["program_wall"]["resident_buffers"] is True
    assert record["timing"]["program_wall"]["launch_count_per_sample"] == 5
    assert record["timing"]["program_wall"]["selector_eligible"] is False
    assert record["timing"]["program_wall"]["passes_ratchet"] is True
    assert record["timing"]["operation_total_ms"] == 4.0
    assert record["workspace_bytes"] == 4096
    assert PROGRAM_WALL_BASELINE_MS == 0.368203
    assert PROGRAM_WALL_MAX_REGRESSION == 0.10
