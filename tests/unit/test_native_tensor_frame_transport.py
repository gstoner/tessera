"""Host-only frame transport checks match CUDA and arena ABI limits."""
import copy

import pytest

from tessera.compiler.prepared_native_tensor_frame import validate_frame


def frame():
    return {
        "schema": "tessera.native.tensor_program_frame.v1",
        "target": "nvidia_sm120", "architecture": "sm_120", "function": "test",
        "buffers": [
            {"id": 0, "role": "input", "shape": [1, 1], "storage": "f16", "bytes": 2},
            {"id": 1, "role": "input", "shape": [1, 1], "storage": "f16", "bytes": 2},
            {"id": 2, "role": "output", "shape": [1, 1], "storage": "f32", "bytes": 4},
        ],
        "roots": [0, 1], "output": 2,
        "stages": [{"member": "test", "entry": "test", "inputs": [0, 1],
                    "output": 2, "scalars": [1, 1, 1], "geometry": [1, 1, 1, 32, 1, 1]}],
    }


@pytest.mark.parametrize("geometry", [
    [2**31 - 1, 65535, 65535, 1024, 1, 1],
    [1, 1, 1, 16, 1, 64],
])
def test_transport_accepts_cuda_boundary_geometry_without_allocating(geometry):
    value = frame()
    value["stages"][0]["geometry"] = geometry
    validate_frame(value)


@pytest.mark.parametrize("geometry", [
    [2**31, 1, 1, 32, 1, 1],
    [1, 65536, 1, 32, 1, 1],
    [1, 1, 65536, 32, 1, 1],
    [1, 1, 1, 1025, 1, 1],
    [1, 1, 1, 32, 1025, 1],
    [1, 1, 1, 32, 1, 65],
    [1, 1, 1, 1024, 2, 1],
])
def test_transport_rejects_grid_axis_block_axis_and_total_threads(geometry):
    value = frame()
    value["stages"][0]["geometry"] = geometry
    with pytest.raises(ValueError, match="geometry exceeds CUDA limits"):
        validate_frame(value)


@pytest.mark.parametrize("extra", [2, 3])
def test_transport_checks_aggregate_aligned_arena_without_hardware(extra):
    value = frame()
    for _ in range(extra):
        index = len(value["buffers"])
        value["buffers"].append({
            "id": index, "role": "input", "storage": "f16",
            "shape": [2**31 - 1, 2**31 - 1], "bytes": 2 * (2**31 - 1)**2})
        value["roots"].append(index)
    if extra == 2:
        validate_frame(copy.deepcopy(value))
    else:
        with pytest.raises(ValueError, match="aligned arena exceeds its 64-bit capacity"):
            validate_frame(value)
