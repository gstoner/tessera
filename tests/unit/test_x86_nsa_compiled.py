"""NSA (DeepSeek native sparse attention) lane on x86 AVX-512 (P11 of
S_SERIES_GAP_CLOSURE_PLAN) — deepseek_sparse_attention blends a sliding-window
branch (the P10 windowed flash_attn), a compressed-block branch (dense
flash_attn over per-block mean summaries), and a top-k-block branch (host top-k
block select + gather + dense flash_attn) through a learned gate. The attention
FLOPs in every branch run on the AVX-512 flash_attn kernels. Reachable via
`compiler_path="x86_nsa_compiled"`. Validated vs the dense-masked reference
(tessera.ops.deepseek_sparse_attention). Skip-clean:
libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, n_operands, kwargs):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_nsa_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.deepseek_sparse_attention", "result": "o",
                 "operands": names, "kwargs": kwargs}]})


def _run(rt, kwargs, *arrs):
    res = rt.launch(_art(rt, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_nsa_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(43)


def _qkv(B, H, S, D):
    return (_RNG.standard_normal((B, H, S, D)).astype(np.float32),
            _RNG.standard_normal((B, H, S, D)).astype(np.float32),
            _RNG.standard_normal((B, H, S, D)).astype(np.float32))


def test_nsa_uniform_gate():
    rt = _rt_or_skip()
    Q, K, V = _qkv(2, 2, 16, 8)
    kw = dict(window_size=4, block_size=4, top_k=2, causal=True)
    got = _run(rt, kw, Q, K, V)
    ref = tessera.ops.deepseek_sparse_attention(Q, K, V, **kw)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_nsa_with_gate_logits():
    rt = _rt_or_skip()
    Q, K, V = _qkv(2, 3, 12, 8)
    gate = _RNG.standard_normal((2, 3, 12, 3)).astype(np.float32)
    kw = dict(window_size=3, block_size=3, top_k=2, causal=True)
    got = _run(rt, kw, Q, K, V, gate)
    ref = tessera.ops.deepseek_sparse_attention(Q, K, V, gate, **kw)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_nsa_topk_equals_all_blocks():
    rt = _rt_or_skip()
    # top_k == num_blocks → the top-k branch attends every block (dense).
    Q, K, V = _qkv(1, 2, 12, 8)
    kw = dict(window_size=12, block_size=4, top_k=3, causal=True)  # nb = 3
    got = _run(rt, kw, Q, K, V)
    ref = tessera.ops.deepseek_sparse_attention(Q, K, V, **kw)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)
