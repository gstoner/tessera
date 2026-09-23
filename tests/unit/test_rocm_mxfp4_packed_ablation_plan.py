"""Every opt-in packed-producer run includes its immediate permute control."""
from __future__ import annotations

import pytest

from benchmarks.rocm.benchmark_gfx1201_mxfp4_packed_folded import (
    _include_permute_control,
)


@pytest.mark.parametrize("include_permute,include_vector_pair,include_a_base,expected", [
    (False, False, False, False),
    (True, False, False, True),
    (False, True, False, True),
    (False, False, True, True),
    (True, True, True, True),
])
def test_packed_producer_control_plan(
    include_permute: bool, include_vector_pair: bool,
    include_a_base: bool, expected: bool,
) -> None:
    assert _include_permute_control(
        include_permute, include_vector_pair, include_a_base,
    ) is expected
