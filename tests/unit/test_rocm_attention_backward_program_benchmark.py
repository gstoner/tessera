import json
from pathlib import Path

from benchmarks.rocm.benchmark_rocm_attention_backward_program import (
    PROGRAM_WALL_BASELINE_MS,
    PROGRAM_WALL_BASELINE_MS_BY_DTYPE,
    PROGRAM_WALL_MAX_REGRESSION,
    _record,
)


def test_backward_program_benchmark_separates_resident_program_wall() -> None:
    record = _record(
        package_ms=500.0,
        operation_total_ms=4.0,
        result={
            "kernel_wall_samples_ms": [0.39, 0.3, 0.5],
            "workspace_bytes": 4096,
            "entry_symbols": ("forward", "pre", "dkdv", "reduce", "dq"),
        },
        max_abs_error={"dq": 0.001, "dk": 0.002, "dv": 0.003},
        image_bytes=32768,
        dtype="bf16",
        dropout_p=0.25,
        dropout_seed=37,
    )

    assert record["timing"]["program_wall"]["median_ms"] == 0.39
    assert record["timing"]["program_wall"]["resident_module"] is True
    assert record["timing"]["program_wall"]["resident_buffers"] is True
    assert record["timing"]["program_wall"]["launch_count_per_sample"] == 5
    assert record["timing"]["program_wall"]["selector_eligible"] is False
    assert record["timing"]["program_wall"]["passes_ratchet"] is True
    assert record["timing"]["operation_total_ms"] == 4.0
    assert record["workspace_bytes"] == 4096
    assert record["storage"] == "bf16"
    assert record["dropout"] == {
        "probability": 0.25,
        "seed": 37,
        "counter": "lcg32_counter_v1",
        "replay": "forward_backward_identical",
    }
    assert PROGRAM_WALL_BASELINE_MS == 0.368203
    assert PROGRAM_WALL_BASELINE_MS_BY_DTYPE["bf16"] == 0.362481
    assert PROGRAM_WALL_MAX_REGRESSION == 0.10


def test_bf16_backward_program_exact_device_packet_is_passing() -> None:
    root = Path(__file__).resolve().parents[2]
    packet = json.loads((root / "benchmarks/baselines/rocm_gfx1151_bf16_attention_backward.json").read_text())
    assert packet["storage"] == "bf16"
    assert packet["entry_count"] == 5
    assert packet["dropout"]["replay"] == "forward_backward_identical"
    assert packet["timing"]["sample_count"] == 21
    assert packet["ratchet"]["passed"] is True
    assert packet["numerics"]["passed"] is True
    assert packet["passed"] is True
