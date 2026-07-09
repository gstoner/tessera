"""@jit(target="rocm") msa_sparse_attention executes on gfx1151.

Proves the Target-IR `status = "compiled"` on `tessera_rocm.msa_block_sparse` is
honest end-to-end: the @jit ROCm path stamps the artifact executable
(compiler_path="rocm_sparse_attn_compiled"), and runtime.launch() runs the
compiler-generated block-sparse WMMA + GPU-top-k lane, matching the numpy MSA
reference. Skip-clean without a live gfx1151.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt
from tessera.stdlib import attention as A


# Module-level so @jit can inspect the source.
@ts.jit(target="rocm")
def _msa_jit(Q, K, V):
    return ts.ops.msa_sparse_attention(Q, K, V, block_size=8, top_k=2)


def _rocm_or_skip():
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")


def test_jit_rocm_msa_is_stamped_executable():
    # Even off-device the stamping is host-gated, so only assert the metadata
    # shape when the compiled lane is actually available here.
    _rocm_or_skip()
    rng = np.random.default_rng(0)
    b, hq, hkv, sq, sk, d = 1, 4, 2, 16, 32, 16
    Q = rng.standard_normal((b, hq, sq, d)).astype(np.float32)
    K = rng.standard_normal((b, hkv, sk, d)).astype(np.float32)
    V = rng.standard_normal((b, hkv, sk, d)).astype(np.float32)
    out = np.asarray(_msa_jit(Q, K, V))

    meta = _msa_jit.runtime_artifact().metadata
    assert meta.get("execution_kind") == "native_gpu"
    assert meta.get("compiler_path") == "rocm_sparse_attn_compiled"
    assert meta.get("executable") is True

    ref = np.asarray(A.msa_sparse_attention(Q, K, V, block_size=8, top_k=2))
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-3)
