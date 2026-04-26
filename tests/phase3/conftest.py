"""
Phase 3 test fixtures — GPU target profiles and mock WGMMA verifiers.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import tessera
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.attn_lower import FlashAttnLoweringConfig


@pytest.fixture
def sm90_profile():
    """Standard SM_90 Hopper profile used by most Phase 3 tests."""
    return GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4)


@pytest.fixture
def sm80_profile():
    """SM_80 Ampere profile — exercises WMMA fallback path."""
    return GPUTargetProfile(isa=ISA.SM_80, warps_per_cta=4)


@pytest.fixture
def causal_attn_config():
    """Causal FlashAttention config with default SM_90 tile sizes."""
    return FlashAttnLoweringConfig(tile_q=64, tile_kv=64, causal=True)


@pytest.fixture
def flash_attn_ir(sm90_profile):
    """Pre-built Graph IR for a simple causal flash_attn function."""
    from tessera.compiler.graph_ir import GraphIRBuilder

    @tessera.jit(target=sm90_profile)
    def flash_attn_fwd(
        Q: tessera.Tensor["B", "H", "S", "D"],
        K: tessera.Tensor["B", "H", "S", "D"],
        V: tessera.Tensor["B", "H", "S", "D"],
    ):
        tessera.require(tessera.constraint.Divisible("D", 64))
        return tessera.ops.flash_attn(Q, K, V, causal=True)

    return flash_attn_fwd.graph_ir
