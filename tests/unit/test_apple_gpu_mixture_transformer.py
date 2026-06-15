"""Apple GPU proof for the Cosmos-3-style dual-tower MoT attention.

Executes the dual-stream (Reasoner causal-self / Generator bidirectional-over-
[AR;DM]) attention on the Apple GPU ``metal_runtime`` lane by injecting the
real Metal ``flash_attn`` / ``flash_attn_bias`` dispatcher
(``tessera.dflash.apple_gpu_attention_fn``) as the attention core, and checks:

  * the Metal dense path matches the numpy reference (fp32 rtol);
  * the Metal "two-way flat attention" varlen path (Cosmos §5.2.2) matches the
    same reference — proving the perf-shaped formulation is substitutable on a
    real backend, not just on numpy.

Guarded to Darwin; on non-Apple hosts the numeric oracle in
``test_varlen_sdpa.py`` already covers the formulation equivalence.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.models import mixture_transformer as MT

DARWIN = sys.platform == "darwin"


def _setup(seq_roles):
    cfg = MT.MixtureTransformerConfig(
        hidden_size=32, num_heads=4,
        reasoner_intermediate=64, generator_intermediate=64,
    )
    r = MT.synthetic_tower_weights(cfg, cfg.reasoner_intermediate, seed=10)
    g = MT.synthetic_tower_weights(cfg, cfg.generator_intermediate, seed=20)
    S = len(seq_roles)
    x = np.random.default_rng(7).standard_normal((1, S, cfg.hidden_size)).astype(np.float32)
    return cfg, r, g, x


@pytest.mark.skipif(not DARWIN, reason="metal_runtime dispatch is Darwin-only")
@pytest.mark.parametrize("roles", [
    [0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1],
])
def test_dual_stream_attention_on_metal_matches_reference(roles):
    from tessera.dflash import apple_gpu_attention_fn

    cfg, r, g, x = _setup(roles)
    ref = MT.dual_stream_attention_dense(x, roles, r, g, cfg)  # numpy reference

    gpu_dense = MT.dual_stream_attention_dense(
        x, roles, r, g, cfg, attention_fn=apple_gpu_attention_fn)
    gpu_varlen = MT.dual_stream_attention_varlen(
        x, roles, r, g, cfg, attention_fn=apple_gpu_attention_fn)

    np.testing.assert_allclose(gpu_dense, ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gpu_varlen, ref, rtol=1e-4, atol=1e-5)
    # The two on-GPU formulations agree (derive validates declare, on Metal).
    np.testing.assert_allclose(gpu_dense, gpu_varlen, rtol=1e-4, atol=1e-5)
