"""Pure guards for the static gfx1201 folded staging comparison."""
from __future__ import annotations

import pytest

from benchmarks.rocm.inspect_gfx1201_folded_prefill import (
    _mnemonics, requested_bytes,
)


def test_requested_bytes_distinguish_expanded_and_packed_weights() -> None:
    first = requested_bytes(256, 5120, 8704)
    assert first["tessera_b_folded"] == 5120 * 8704
    assert first["radiance_b_packed"] == 5120 * 8704 // 2
    assert first["radiance_block_scales"] == 5120 * 8704 // 32
    second = requested_bytes(1024, 17408, 5120)
    assert second["tessera_b_folded"] == 4 * 17408 * 5120
    assert second["radiance_b_packed"] == second["tessera_b_folded"] // 2
    with pytest.raises(ValueError, match="divisible by 64"):
        requested_bytes(256, 80, 96)


def test_isa_census_counts_only_instruction_mnemonics() -> None:
    isa = """
      global_load_b128 v[1:4], v[5:6], off
      ds_store_2addr_b64 v1, v[2:5]
      s_wait_loadcnt 0x0
      v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[9:10], v[11:12], v[1:8]
      unrelated_label:
    """
    assert _mnemonics(isa) == {
        "ds_store_2addr_b64": 1,
        "global_load_b128": 1,
        "s_wait_loadcnt": 1,
        "v_wmma_f32_16x16x16_fp8_fp8": 1,
    }
