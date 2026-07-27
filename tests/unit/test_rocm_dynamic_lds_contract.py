import pytest

from tessera.compiler.rocm_dynamic_lds import (
    align_up,
    evaluate_launch_expression,
    interference_slot_launch_bytes,
    interference_slot_layout,
    packed_path_layout,
    path_max_launch_bytes,
)


def test_path_max_reuses_mutually_exclusive_storage():
    assert packed_path_layout((12_289, 4_111)) == ((0, 12_304), 16_416)
    assert path_max_launch_bytes(((8_192, 8_192), (32_001,))) == 32_016
    assert path_max_launch_bytes(((16_384,), (32_768,))) == 32_768


def test_dynamic_lds_contract_rejects_invalid_sizes_and_alignment():
    with pytest.raises(ValueError, match="non-negative"):
        path_max_launch_bytes(((4, -1),))
    with pytest.raises(ValueError, match="power of two"):
        align_up(4, 12)


def test_interference_slots_reuse_nested_and_loop_local_storage():
    offsets, launch_bytes = interference_slot_layout(
        ((8_192, 32_001, 16_384), (4_097, 8_192))
    )
    assert offsets == (0, 32_016)
    assert launch_bytes == 40_208
    assert interference_slot_launch_bytes(((16_384, 32_768),)) == 32_768


def test_serialized_expression_handles_runtime_arithmetic_and_sequential_arenas():
    expression = {
        "kind": "align_up",
        "alignment": 16,
        "operands": [
            {
                "kind": "add",
                "operands": [
                    {
                        "kind": "multiply",
                        "operands": [
                            {"kind": "argument", "name": "Rows"},
                            {"kind": "argument", "name": "Stride"},
                        ],
                    },
                    {
                        "kind": "max",
                        "operands": [
                            {"kind": "argument", "name": "ThenBytes"},
                            {"kind": "argument", "name": "ElseBytes"},
                        ],
                    },
                ],
            }
        ],
    }
    assert evaluate_launch_expression(
        expression,
        {"Rows": 7, "Stride": 130, "ThenBytes": 4097, "ElseBytes": 8192},
    ) == 9_104


def test_serialized_expression_rejects_overflow_and_missing_arguments():
    expression = {
        "kind": "multiply",
        "operands": [
            {"kind": "argument", "name": "Rows"},
            {"kind": "constant", "value": 1 << 62},
        ],
    }
    with pytest.raises(ValueError, match="signed-i64"):
        evaluate_launch_expression(expression, {"Rows": 4})
    with pytest.raises(ValueError, match="lacks argument"):
        evaluate_launch_expression(expression, {})
