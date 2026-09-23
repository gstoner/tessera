"""Every opt-in packed-producer run includes its immediate permute control."""
from __future__ import annotations

import pytest

from benchmarks.rocm.benchmark_gfx1201_mxfp4_packed_folded import (
    _include_permute_control,
    _selected_integer_alu,
)


@pytest.mark.parametrize("include_permute,include_vector_pair,include_a_base,include_a_offset32,expected", [
    (False, False, False, False, False),
    (True, False, False, False, True),
    (False, True, False, False, True),
    (False, False, True, False, True),
    (False, False, False, True, True),
    (True, True, True, True, True),
])
def test_packed_producer_control_plan(
    include_permute: bool, include_vector_pair: bool,
    include_a_base: bool, include_a_offset32: bool, expected: bool,
) -> None:
    assert _include_permute_control(
        include_permute, include_vector_pair, include_a_base, include_a_offset32,
    ) is expected


def test_a_offset32_implies_64_bit_a_base_control() -> None:
    from benchmarks.rocm.benchmark_gfx1201_mxfp4_packed_folded import (
        _include_a_base_control,
    )

    assert _include_a_base_control(False, False) is False
    assert _include_a_base_control(True, False) is True
    assert _include_a_base_control(False, True) is True


def test_integer_alu_census_is_bound_to_selected_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks.rocm import benchmark_gfx1201_mxfp4_packed_folded as bench

    disassembly = """00000000 <selected>:
    v_add_co_u32 v0, v1, v2 // 0000
    v_mul_lo_u32 v0, v1, v2 // 0004
    v_mul_f32_e32 v0, v1, v2 // 0006
    v_mul_f64_e32 v0, v1, v2 // 0007
00000008 <other>:
    v_add_co_u32 v0, v1, v2 // 0008
"""
    monkeypatch.setattr(bench.rocm_isa, "disassemble", lambda *_args, **_kwargs: disassembly)
    assert _selected_integer_alu(b"payload", "selected") == {
        "v_add_co_u32": 1, "v_mul_lo_u32": 1,
    }
