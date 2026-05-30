"""Regression coverage for the reworked examples/advanced/mla GPU-decode demo.

Ensures `mla.run_gpu_decode_demo` keeps validating the shipped Apple GPU MLA
decode surfaces (weight absorption, paged + block-paged serving, the
GPU-resident decode loop) against numpy as the runtime evolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EX = Path(__file__).resolve().parents[2] / "examples" / "advanced" / "mla"


@pytest.fixture(scope="module")
def mla_mod():
    if not _EX.exists():
        pytest.skip("mla example not present")
    if str(_EX) not in sys.path:
        sys.path.insert(0, str(_EX))
    import mla  # noqa: E402
    return mla


def test_gpu_decode_demo_validates(mla_mod):
    g = mla_mod.run_gpu_decode_demo(mla_mod.tiny_config())
    assert g.absorbed_matches_explicit
    assert g.paged_matches_reference
    assert g.block_paged_matches_reference
    assert g.resident_loop_tokens == 4
    assert g.backend in ("metal", "numpy")


def test_gpu_decode_cache_footprint(mla_mod):
    g = mla_mod.run_gpu_decode_demo(mla_mod.tiny_config())
    assert g.cache_bytes_per_token_explicit > g.cache_bytes_per_token_latent
