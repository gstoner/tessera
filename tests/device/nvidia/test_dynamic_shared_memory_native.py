from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tessera.compiler.nvidia_dynamic_smem import (
    align_up,
    evaluate_launch_expression,
    package_local_expression_probe,
    package_path_max_probe,
    path_max_launch_bytes,
)
from tessera.runtime import (
    _nvidia_device_memory_envelope,
    _nvidia_native_descriptor_device_latency,
    _nvidia_native_descriptor_resources,
    _submit_nvidia_sm120_native,
)
from tests._support.nvidia import nvidia_cuda_host_ready


def _require_cuda() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")


@pytest.mark.hardware_nvidia
def test_exact_cuda_context_reports_capacity_for_rematerialization_policy() -> None:
    _require_cuda()
    envelope = _nvidia_device_memory_envelope()
    assert envelope["capacity_bytes"] > 0
    assert 0 < envelope["free_bytes"] <= envelope["capacity_bytes"]


def test_path_max_expression_aligns_paths_without_summing_them() -> None:
    assert align_up(12_289) == 12_304
    assert path_max_launch_bytes(((12_289,), (32_001,))) == 32_016
    assert path_max_launch_bytes(((8_192, 4_111), (32_001,))) == 32_016
    with pytest.raises(ValueError, match="non-negative"):
        path_max_launch_bytes(((-1,),))


def test_serialized_launch_expression_evaluates_checked_local_arithmetic() -> None:
    expression = {
        "kind": "path_max",
        "operands": [
            {
                "kind": "add",
                "operands": [
                    {
                        "kind": "multiply",
                        "operands": [
                            {"kind": "argument", "name": "Base"},
                            {"kind": "constant", "value": 4},
                        ],
                    },
                    {"kind": "constant", "value": 17},
                ],
            },
            {"kind": "argument", "name": "Fallback"},
        ],
    }
    assert evaluate_launch_expression(
        expression, {"Base": 4096, "Fallback": 12_289}
    ) == 16_401
    with pytest.raises(ValueError, match="lacks argument"):
        evaluate_launch_expression(expression, {"Base": 4096})


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(
    ("branch", "expected_value", "expected_extent"),
    [(0, 72.0, 32_001.0), (1, 48.0, 12_289.0)],
)
def test_dynamic_path_max_arena_executes_both_sm120_paths(
    branch: int, expected_value: float, expected_extent: float
) -> None:
    _require_cuda()
    package = package_path_max_probe(
        then_bytes=12_289, else_bytes=32_001, branch=branch
    )
    output = np.zeros(2, np.float32)
    got = _submit_nvidia_sm120_native(
        package.image,
        package.descriptor,
        {"output": output},
        {"ThenBytes": 12_289, "ElseBytes": 32_001, "Branch": branch},
        None,
    )
    np.testing.assert_array_equal(
        got, np.array([expected_value, expected_extent], np.float32)
    )
    assert ".extern .shared" in package.ptx
    assert package.descriptor.dynamic_local_memory_bytes == 32_016
    metrics = package.image.resource_record.metrics
    assert metrics["spill_store_bytes"] == metrics["spill_load_bytes"] == 0


@pytest.mark.hardware_nvidia
def test_dynamic_path_max_resources_occupancy_timing_and_rejection() -> None:
    _require_cuda()
    package = package_path_max_probe(
        then_bytes=12_289, else_bytes=32_001, branch=1
    )
    args = {
        "output": np.zeros(2, np.float32),
        "ThenBytes": 12_289,
        "ElseBytes": 32_001,
        "Branch": 1,
    }
    resources = _nvidia_native_descriptor_resources(
        package.image, package.descriptor, block_size=1
    )
    assert resources["dynamic_shared_memory_bytes"] == 32_016
    assert resources["static_shared_memory_bytes"] == 0
    assert resources["local_memory_bytes"] == 0
    assert resources["registers_per_thread"] > 0
    assert resources["active_blocks_per_sm"] > 0
    latency = _nvidia_native_descriptor_device_latency(
        package.image, package.descriptor, args, reps=20, warmup=5
    )
    assert latency > 0.0

    undersized = replace(
        package.descriptor,
        dynamic_local_memory_bytes=package.descriptor.dynamic_local_memory_bytes - 16,
    )
    with pytest.raises(RuntimeError, match="path-max descriptor"):
        _submit_nvidia_sm120_native(
            package.image,
            undersized,
            {"output": np.zeros(2, np.float32)},
            {"ThenBytes": 12_289, "ElseBytes": 32_001, "Branch": 1},
            None,
        )


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(
    ("branch", "expected_value", "expected_extent"),
    [(0, 114.0, 12_289.0), (1, 106.0, 16_401.0)],
)
def test_local_expression_dynamic_arena_executes_both_sm120_paths(
    branch: int, expected_value: float, expected_extent: float
) -> None:
    _require_cuda()
    package = package_local_expression_probe(
        base=4096,
        factor=4,
        bias=17,
        fallback=12_289,
        branch=branch,
    )
    scalars = {
        "Base": 4096,
        "Factor": 4,
        "Bias": 17,
        "Fallback": 12_289,
        "Branch": branch,
    }
    output = np.zeros(2, np.float32)
    got = _submit_nvidia_sm120_native(
        package.image,
        package.descriptor,
        {"output": output},
        scalars,
        None,
    )
    np.testing.assert_array_equal(
        got, np.array([expected_value, expected_extent], np.float32)
    )
    assert package.descriptor.dynamic_local_memory_bytes == 16_416
    resources = _nvidia_native_descriptor_resources(
        package.image, package.descriptor, block_size=1
    )
    assert resources["dynamic_shared_memory_bytes"] == 16_416
    assert resources["local_memory_bytes"] == 0
