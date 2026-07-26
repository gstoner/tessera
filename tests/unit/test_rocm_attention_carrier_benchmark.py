from benchmarks.rocm.benchmark_rocm_attention_carrier import (
    KERNEL_WALL_BASELINE_MS,
    KERNEL_WALL_MAX_REGRESSION,
    _timing_domains,
)


def test_attention_carrier_benchmark_keeps_timing_domains_separate():
    timing = _timing_domains(
        cold_operation_total_ms=10.0,
        operation_total_samples_ms=[3.0, 2.0, 4.0],
        kernel_wall_samples_ms=[0.3, 0.2, 0.4],
        warmup=5,
    )

    assert timing["operation_total"]["median_ms"] == 3.0
    assert timing["kernel_wall"]["median_ms"] == 0.3
    assert timing["kernel_wall"]["resident_module"] is True
    assert timing["kernel_wall"]["resident_buffers"] is True
    assert timing["kernel_wall"]["warmup_iterations"] == 5
    assert timing["kernel_wall"]["launch_api"] == "hipModuleLaunchKernel"
    assert timing["kernel_wall"]["completion_api"] == "hipDeviceSynchronize"
    assert timing["kernel_wall"]["selector_eligible"] is False
    assert "device_allocation" in timing["operation_total"]["includes"]
    assert "device_allocation" not in timing["kernel_wall"]["includes"]


def test_attention_carrier_wall_ratchet_is_tied_to_gfx1151_baseline():
    assert KERNEL_WALL_BASELINE_MS == 0.097763
    assert KERNEL_WALL_MAX_REGRESSION == 0.10
