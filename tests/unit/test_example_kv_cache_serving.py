"""Regression coverage for the reworked examples/advanced/kv_cache_serving demo.

The example used to only *model* prefix-affine routing + KV-cache accounting.
The rework runs those decisions against the real ``tessera.cache.MLABlockPagedCache``
block-paged manager.  These tests lock the behaviors the demo claims: prefix
sharing reclaims pages, the non-contiguous block gather decodes correctly, and
freed pages are reused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EX = Path(__file__).resolve().parents[2] / "examples" / "advanced" / "kv_cache_serving"


@pytest.fixture(scope="module")
def serving_mod():
    if not _EX.exists():
        pytest.skip("kv_cache_serving example not present")
    if str(_EX) not in sys.path:
        sys.path.insert(0, str(_EX))
    from kv_cache_serving import serving  # noqa: E402
    return serving


def test_serving_demo_validates(serving_mod):
    s = serving_mod.run_serving_demo()
    # Block-paged (non-contiguous) decode matches the contiguous reference.
    assert s.validated
    assert s.max_abs_error <= 1e-4


def test_prefix_sharing_saves_blocks(serving_mod):
    # 12 requests over 4 prefixes => 8 cache hits, and we allocate pages for
    # only the 4 unique prefixes instead of all 12 requests.
    s = serving_mod.run_serving_demo(num_requests=12, num_prefixes=4)
    assert s.num_unique_prefixes == 4
    assert s.num_cache_hits == 8
    assert s.blocks_saved > 0
    assert s.blocks_allocated < s.blocks_without_sharing


def test_eviction_reclaims_and_reuses_pages(serving_mod):
    s = serving_mod.run_serving_demo()
    assert s.free_blocks_after_evict > 0
    assert s.reused_after_evict


def test_real_footprint_is_reported(serving_mod):
    s = serving_mod.run_serving_demo()
    # Real per-token MLA latent footprint vs the policy's full-scale estimate.
    assert s.cache_bytes_per_token_real > 0
    assert s.cache_bytes_per_request_estimate > 0
