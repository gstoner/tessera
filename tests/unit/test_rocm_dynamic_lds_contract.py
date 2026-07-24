import pytest

from tessera.compiler.rocm_dynamic_lds import (
    align_up,
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
