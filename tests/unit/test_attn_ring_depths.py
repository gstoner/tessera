"""C5 scaffold — independent per-ring pipeline depths (TIRx review item C5).

The hardware-free half of C5: the FA-4 Q/KV/TMEM rings are double-buffered
*independently* (book defaults 2/3/2), exposed as config fields + emitted attrs +
an autotuner sweep surface. The measured per-ring sweep + the kernel consuming
the depths stay hardware-gated, so these tests cover the *surface* only.
"""

import pytest

from tessera.compiler.attn_lower import (
    FlashAttnLoweringConfig,
    SM90_DEFAULT,
    TesseraAttnConfigError,
)


def test_ring_depth_defaults_match_book():
    c = FlashAttnLoweringConfig()
    assert c.ring_depths() == (2, 3, 2)
    assert (c.q_depth, c.kv_depth, c.tmem_depth) == (2, 3, 2)


def test_ring_depths_emitted_as_attrs():
    attrs = FlashAttnLoweringConfig(q_depth=1, kv_depth=4, tmem_depth=2).to_mlir_attrs()
    assert "tessera.q_depth = 1 : i32" in attrs
    assert "tessera.kv_depth = 4 : i32" in attrs
    assert "tessera.tmem_depth = 2 : i32" in attrs
    # Legacy single knob still present (back-compat).
    assert "tessera.pipeline_stages = 2 : i32" in attrs


@pytest.mark.parametrize("field", ["q_depth", "kv_depth", "tmem_depth"])
def test_nonpositive_ring_depth_rejected(field):
    with pytest.raises(TesseraAttnConfigError):
        FlashAttnLoweringConfig(**{field: 0})


def test_sweep_surface_is_legal_and_default_first():
    space = FlashAttnLoweringConfig.ring_depth_search_space()
    assert space[0] == (2, 3, 2), "default must be first"
    assert len(space) == len(set(space)), "no duplicates"
    for q, kv, tmem in space:
        # Every candidate must form a legal config (each depth >= 1).
        cfg = FlashAttnLoweringConfig(q_depth=q, kv_depth=kv, tmem_depth=tmem)
        assert cfg.ring_depths() == (q, kv, tmem)


def test_sweep_surface_respects_choices():
    space = FlashAttnLoweringConfig.ring_depth_search_space(
        q_choices=(2,), kv_choices=(3,), tmem_choices=(2,)
    )
    assert space == [(2, 3, 2)]


def test_pipeline_stages_back_compat_unchanged():
    # The legacy knob and the LDS budget it drives are untouched by C5.
    assert SM90_DEFAULT.pipeline_stages == 2
    assert SM90_DEFAULT.lds_bytes(head_dim=64) == 2 * 64 * 64 * 2 * 2
