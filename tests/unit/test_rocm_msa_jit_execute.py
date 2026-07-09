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


def test_jit_rocm_msa_executes_via_launch():
    _rocm_or_skip()
    rng = np.random.default_rng(0)
    b, hq, hkv, sq, sk, d = 1, 4, 2, 16, 32, 16
    Q = rng.standard_normal((b, hq, sq, d)).astype(np.float32)
    K = rng.standard_normal((b, hkv, sk, d)).astype(np.float32)
    V = rng.standard_normal((b, hkv, sk, d)).astype(np.float32)

    # The @jit(target="rocm") path stamps the artifact executable.
    art = _msa_jit.runtime_artifact()
    meta = art.metadata
    assert meta.get("execution_kind") == "native_gpu"
    assert meta.get("compiler_path") == "rocm_sparse_attn_compiled"
    assert meta.get("executable") is True

    # JitFn.__call__ fast-dispatches only CPU/Apple targets and otherwise falls
    # through to the eager Python body — so calling _msa_jit(...) would NOT run
    # the compiled lane. Exercise it the way it is actually reached: launch the
    # stamped artifact and assert THAT result (the block-sparse WMMA + GPU top-k
    # lane) matches the reference. A broken rocm_sparse_attn_compiled artifact
    # now fails here instead of hiding behind the eager fallback.
    res = rt.launch(art, (Q, K, V))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_sparse_attn_compiled"
    ref = np.asarray(A.msa_sparse_attention(Q, K, V, block_size=8, top_k=2))
    np.testing.assert_allclose(np.asarray(res["output"]), ref, rtol=0, atol=1e-3)
