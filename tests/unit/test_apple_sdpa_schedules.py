"""Apple SDPA schedule candidates mined from MLX — routing + block seeds."""

from __future__ import annotations

import pytest

from tessera.compiler.apple_sdpa_schedules import (
    AttnPath,
    AttnTile,
    full_attn_tile,
    select_attn_schedule,
    vector_2pass_blocks,
)


def test_full_tile_nax_vs_metal_bk():
    # NAX path: bk=32. Metal path: bk = head_dim<128 ? 32 : 16. bq=64,wm=4,wn=1.
    assert full_attn_tile(128, nax=True) == AttnTile(64, 32, 128, 4, 1)
    assert full_attn_tile(128, nax=False) == AttnTile(64, 16, 128, 4, 1)
    assert full_attn_tile(64, nax=False) == AttnTile(64, 32, 64, 4, 1)


def test_decode_routes_to_vector():
    s = select_attn_schedule(q_len=1, kv_len=2048, head_dim=128, q_heads=32, kv_heads=8)
    assert s.path is AttnPath.VECTOR
    assert s.tile is None
    assert s.spec.gqa_factor == 4   # 32 / 8


def test_long_context_decode_routes_to_2pass():
    s = select_attn_schedule(q_len=1, kv_len=16384, head_dim=128, q_heads=8, kv_heads=8)
    assert s.path is AttnPath.VECTOR_2PASS
    assert s.vector_blocks == 256   # 8192 < 16384 <= 32768


@pytest.mark.parametrize("kv,want", [(4096, 128), (16384, 256), (100000, 1024)])
def test_vector_2pass_blocks(kv, want):
    assert vector_2pass_blocks(kv) == want


def test_prefill_uses_nax_when_available():
    s = select_attn_schedule(q_len=512, kv_len=512, head_dim=128, q_heads=16, kv_heads=16,
                             nax_available=True)
    assert s.path is AttnPath.FULL_NAX
    assert s.tile == AttnTile(64, 32, 128, 4, 1)


def test_prefill_falls_back_to_metal_for_head_dim_80():
    # NAX SDPA is disabled for head_dim 80 even when NAX is available.
    s = select_attn_schedule(q_len=512, kv_len=512, head_dim=80, q_heads=16, kv_heads=16,
                             nax_available=True)
    assert s.path is AttnPath.FULL_METAL


def test_prefill_metal_when_no_nax():
    s = select_attn_schedule(q_len=512, kv_len=512, head_dim=128, q_heads=8, kv_heads=2,
                             nax_available=False, do_causal=True)
    assert s.path is AttnPath.FULL_METAL
    assert s.spec.do_causal
    assert s.spec.gqa_factor == 4


def test_spec_carries_mask_causal_sinks():
    s = select_attn_schedule(q_len=128, kv_len=128, head_dim=64, q_heads=4, kv_heads=4,
                             has_mask=True, do_causal=True, has_sinks=True)
    assert s.spec.has_mask and s.spec.do_causal and s.spec.has_sinks


def test_rejects_bad_gqa():
    with pytest.raises(ValueError):
        select_attn_schedule(q_len=1, kv_len=64, head_dim=64, q_heads=7, kv_heads=2)
