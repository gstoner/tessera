"""Bounded transport rejects malformed row/launch metadata without hardware."""
import copy

import pytest

from tessera.compiler.prepared_native_tensor_frame import validate_frame
from tests.unit.test_native_tensor_frame_transport import frame


def bounded_frame():
    value = frame()
    value.update(schema="tessera.native.tensor_program_frame.v2",
                 row_bound=17, row_root=0, row_buffers=[1, 0, 1])
    for index in (0, 2):
        spec = value["buffers"][index]
        spec["shape"][0] = 17
        spec["bytes"] *= 17
    value["stages"][0].update(
        scalars=[17, 1, 1], geometry=[1, 2, 1, 32, 1, 1],
        scalar_row_multipliers=[1, 0, 0], grid_row_multipliers=[0, 1, 0],
        grid_row_divisors=[1, 16, 1])
    return value


def test_bounded_metadata_validates_without_allocating():
    value = bounded_frame()
    before = copy.deepcopy(value)
    validate_frame(value)
    assert value == before


@pytest.mark.parametrize("key,value", [
    ("row_bound", 0), ("row_bound", -1), ("row_bound", 2**31), ("row_bound", True),
    ("row_root", 2), ("row_root", -1), ("row_root", True),
    ("row_buffers", [1, 0]), ("row_buffers", [0, 0, 1]),
    ("row_buffers", [1, 2, 1]), ("row_buffers", [True, 0, 1]),
    ("row_buffers", [1, 1, 1]),
])
def test_bounded_metadata_rejects_invalid_capacity_bindings(key, value):
    packet = bounded_frame()
    packet[key] = value
    with pytest.raises(ValueError):
        validate_frame(packet)


@pytest.mark.parametrize("key,value", [
    ("scalar_row_multipliers", [1]),
    ("scalar_row_multipliers", [2, 0, 0]),
    ("scalar_row_multipliers", [-1, 0, 0]),
    ("scalar_row_multipliers", [2**62, 0, 0]),
    ("scalar_row_multipliers", [True, 0, 0]),
    ("grid_row_multipliers", [0, 2, 0]),
    ("grid_row_multipliers", [0, 2**62, 0]),
    ("grid_row_divisors", [1, 0, 1]),
    ("grid_row_divisors", [2, 16, 1]),
    ("grid_row_divisors", [1, -16, 1]),
])
def test_bounded_metadata_rejects_inconsistent_or_overflowing_launch_expressions(key, value):
    packet = bounded_frame()
    packet["stages"][0][key] = value
    with pytest.raises(ValueError):
        validate_frame(packet)


def test_static_schema_does_not_accept_bounded_extension_fields():
    packet = bounded_frame()
    packet["schema"] = "tessera.native.tensor_program_frame.v1"
    with pytest.raises(ValueError):
        validate_frame(packet)
