"""MSA (MiniMax Sparse Attention) lane on x86 AVX-512.

`msa_sparse_attention` is exact attention over a per-GQA-group Top-k of KV blocks:
the exp-free index scoring + block selection run on the host (bit-identical to the
reference ops), and the exact attend runs on the AVX-512 flash_attn kernel as
dense attention with the non-selected / causal-invalid keys folded into an
additive -inf mask. Reachable via `compiler_path="x86_msa_compiled"`. Validated vs
`tessera.stdlib.attention.msa_sparse_attention`, incl. the dense-equivalence
anchor (top_k == num_blocks reproduces dense GQA). Skip-clean:
libtessera_x86_elementwise.so not built.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.stdlib import attention as A


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _run(rt, kwargs, *arrs):
    names = [f"a{i}" for i in range(len(arrs))]
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_msa_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.msa_sparse_attention", "result": "o",
                 "operands": names, "kwargs": kwargs}]})
    res = rt.launch(art, arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_msa_compiled"
    return np.asarray(res["output"])


def _qkv(rng, b, hq, hkv, sq, sk, d):
    return (rng.standard_normal((b, hq, sq, d)).astype(np.float32),
            rng.standard_normal((b, hkv, sk, d)).astype(np.float32),
            rng.standard_normal((b, hkv, sk, d)).astype(np.float32))


@pytest.mark.parametrize("b,hq,hkv,sq,sk,d,bs,tk,causal", [
    (1, 4, 2, 16, 32, 16, 8, 2, True),    # GQA, causal
    (2, 4, 4, 12, 24, 16, 8, 3, True),    # MHA
    (1, 8, 2, 8, 16, 16, 4, 4, False),    # MQA-ish, non-causal
    (1, 4, 2, 10, 20, 16, 4, 5, True),    # top_k = num_blocks (dense-equiv path)
    (1, 4, 1, 12, 30, 16, 8, 3, True),    # non-block-divisible Sk (padded)
])
def test_x86_msa_matches_reference(b, hq, hkv, sq, sk, d, bs, tk, causal):
    rt = _rt_or_skip()
    rng = np.random.default_rng(b * 100 + sk + tk)
    q, k, v = _qkv(rng, b, hq, hkv, sq, sk, d)
    kw = dict(block_size=bs, top_k=tk, causal=causal, force_local_block=True)
    out = _run(rt, kw, q, k, v)
    ref = np.asarray(A.msa_sparse_attention(q, k, v, **kw))
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-4)


def test_x86_msa_dense_equivalence_is_dense_gqa():
    # top_k == num_blocks: MSA collapses to dense (causal) GQA attention, the
    # first correctness oracle — independent of the approximate index scores.
    rt = _rt_or_skip()
    rng = np.random.default_rng(7)
    b, hq, hkv, sq, sk, d, bs = 1, 4, 2, 16, 32, 16, 8
    nb = (sk + bs - 1) // bs
    q, k, v = _qkv(rng, b, hq, hkv, sq, sk, d)
    out = _run(rt, dict(block_size=bs, top_k=nb, causal=True,
                        force_local_block=True), q, k, v)
    dense = np.asarray(A.dense_causal_attention(q, k, v))
    np.testing.assert_allclose(out, dense, rtol=0, atol=1e-4)


def test_x86_msa_explicit_selected_block_ids():
    # The KV-outer worklist ABI: feed selected_block_ids directly (backend/
    # lowering-test path) and match the reference given the same selection.
    rt = _rt_or_skip()
    rng = np.random.default_rng(3)
    b, hq, hkv, sq, sk, d, bs, tk = 1, 4, 2, 12, 24, 16, 8, 2
    nb = sk // bs
    q, k, v = _qkv(rng, b, hq, hkv, sq, sk, d)
    sel = np.sort(rng.integers(0, nb, (b, hkv, sq, tk)).astype(np.int64), axis=-1)
    # Pass as a nested list — the launch path JSON-serializes the artifact
    # metadata (the executor does np.asarray on it either way); the real
    # ops.msa_sparse_attention path passes the array directly.
    kw = dict(block_size=bs, top_k=tk, causal=True,
              selected_block_ids=sel.tolist())
    out = _run(rt, kw, q, k, v)
    ref = np.asarray(A.msa_sparse_attention(
        q, k, v, block_size=bs, top_k=tk, causal=True, selected_block_ids=sel))
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-4)
