"""Host-free NVIDIA paged-KV provenance and source contracts."""
from __future__ import annotations

import numpy as np

from tessera.cache import KVCacheHandle
from tessera.compiler.evaluator import paged_kv_native_equivalence


def _case(seed=3):
    rng = np.random.default_rng(seed); h = KVCacheHandle(num_heads=4, head_dim=32, max_seq=72, page_size=8)
    h.append(rng.standard_normal((48, 4, 32)).astype(np.float32), rng.standard_normal((48, 4, 32)).astype(np.float32))
    return h, rng.standard_normal((4, 3, 32)).astype(np.float32)


def test_nvidia_paged_fallback_cannot_earn_native_rung(monkeypatch):
    import tessera.cache.paged_kv as pk
    monkeypatch.setattr(pk, "_paged_attention_nvidia", lambda q, abi, idx, scale, causal: (pk._reference_attention(q, np.transpose(abi.gather(idx)[0], (1, 0, 2)), np.transpose(abi.gather(idx)[1], (1, 0, 2)), scale, causal), "reference"))
    state, q = _case(9)
    assert paged_kv_native_equivalence(state, q, backend="nvidia").relation == "inconclusive"


def test_nvidia_paged_attention_source_is_directly_fused():
    from tessera.compiler.emit import nvidia_cuda
    source = nvidia_cuda._synthesize_resident_ops_cuda()
    assert "__global__ void paged_attn" in source
    assert "tessera_nvidia_resident_paged_attention" in source
    assert "const float*kp,const float*vp,const int*table" in source
