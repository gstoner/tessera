"""Varlen two-stream attention — the metamorphic / "derive validates declare"
oracle for the packed-sequence attention primitive.

Motivation: NVIDIA Cosmos 3 (§5.2.2) replaces a masked FlexAttention pass with
"two-way flat attention" — two variable-length SDPA launches keyed on
``cu_seqlens`` — for a 22% training-throughput win. This suite proves the
Tessera ``varlen_sdpa`` primitive and the Cosmos-3 dual-tower MoT reference
agree across formulations, so the perf path can later be substituted with
confidence:

  1. ``cu_seqlens`` packing contract.
  2. ``varlen_sdpa`` == a single dense masked ``flash_attn`` (block_diagonal_bias)
     — including the rectangular ``cu_seqlens_q != cu_seqlens_k`` case.
  3. Cosmos join semantics: dual_stream dense == dual_stream varlen.
  4. The Cosmos metamorphic invariant: the Reasoner output is *identical* whether
     or not Generator tokens are present (reasoning is self-contained; AR never
     attends to DM).
"""
import numpy as np
import pytest

from tessera import ops
from tessera.nn import varlen as V
from tessera.models import mixture_transformer as MT


# ---------------------------------------------------------------------------
# 1. cu_seqlens contract
# ---------------------------------------------------------------------------

def test_cu_seqlens_from_lengths_matches_report_example():
    # Cosmos Fig. 14(a): blocks (3, 2, 4) -> cu_seqlens [0, 3, 5, 9].
    cu = V.cu_seqlens_from_lengths([3, 2, 4])
    assert cu.tolist() == [0, 3, 5, 9]
    assert cu.dtype == np.int32


def test_cu_seqlens_roundtrip():
    lens = [5, 0, 3, 7]
    cu = V.cu_seqlens_from_lengths(lens)
    assert V.lengths_from_cu_seqlens(cu).tolist() == lens


def test_cu_seqlens_rejects_negative_and_bad_start():
    with pytest.raises(ValueError):
        V.cu_seqlens_from_lengths([1, -2])
    with pytest.raises(ValueError):
        V.lengths_from_cu_seqlens([1, 2, 3])  # must start at 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dense_masked(Q, K, Vv, bias, scale):
    """Single dense flash_attn over the fully-packed streams with an additive
    block-diagonal bias — the FlexAttention-equivalent reference."""
    H = Q.shape[0]
    Lq, Lk = Q.shape[1], K.shape[1]
    b = np.broadcast_to(bias, (H, Lq, Lk)).astype(np.float32)
    return np.asarray(ops.flash_attn(Q, K, Vv, scale=scale, causal=False, attn_bias=b))


# ---------------------------------------------------------------------------
# 2. varlen == dense masked flash_attn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("causal", [False, True])
def test_varlen_equals_dense_square_blocks(causal):
    rng = np.random.default_rng(0)
    H, Dh = 3, 8
    lens = [4, 2, 5]
    cu = V.cu_seqlens_from_lengths(lens)
    total = int(cu[-1])
    Q = rng.standard_normal((H, total, Dh)).astype(np.float32)
    K = rng.standard_normal((H, total, Dh)).astype(np.float32)
    Vv = rng.standard_normal((H, total, Dh)).astype(np.float32)
    scale = Dh ** -0.5

    o_varlen = V.varlen_sdpa(Q, K, Vv, cu_seqlens_q=cu, cu_seqlens_k=cu,
                             causal=causal, scale=scale)
    bias = V.block_diagonal_bias(cu, cu, causal=causal)
    o_dense = _dense_masked(Q, K, Vv, bias, scale)

    np.testing.assert_allclose(o_varlen, o_dense, rtol=1e-5, atol=1e-5)


def test_varlen_equals_dense_rectangular_blocks():
    """The Generator-pathway shape: cu_seqlens_q != cu_seqlens_k (each query
    block attends over a longer [R_i; G_i] key block). Cosmos Fig. 14(b)."""
    rng = np.random.default_rng(1)
    H, Dh = 2, 16
    # Three samples (|G_i|, |R_i|+|G_i|) = (2,5), (3,5), (1,5)  -> like the report.
    q_lens = [2, 3, 1]
    k_lens = [5, 5, 5]
    cu_q = V.cu_seqlens_from_lengths(q_lens)
    cu_k = V.cu_seqlens_from_lengths(k_lens)
    tq, tk = int(cu_q[-1]), int(cu_k[-1])
    assert cu_q.tolist() == [0, 2, 5, 6]
    assert cu_k.tolist() == [0, 5, 10, 15]

    Q = rng.standard_normal((H, tq, Dh)).astype(np.float32)
    K = rng.standard_normal((H, tk, Dh)).astype(np.float32)
    Vv = rng.standard_normal((H, tk, Dh)).astype(np.float32)
    scale = Dh ** -0.5

    o_varlen = V.varlen_sdpa(Q, K, Vv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                             causal=False, scale=scale)
    bias = V.block_diagonal_bias(cu_q, cu_k, causal=False)
    o_dense = _dense_masked(Q, K, Vv, bias, scale)

    np.testing.assert_allclose(o_varlen, o_dense, rtol=1e-5, atol=1e-5)


def test_varlen_rejects_mismatched_sample_count():
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((1, 4, 8)).astype(np.float32)
    K = rng.standard_normal((1, 6, 8)).astype(np.float32)
    Vv = K.copy()
    with pytest.raises(ValueError):
        V.varlen_sdpa(Q, K, Vv,
                      cu_seqlens_q=[0, 4],          # 1 sample
                      cu_seqlens_k=[0, 3, 6])       # 2 samples


def test_block_diagonal_bias_blocks_cross_sample_attention():
    cu = V.cu_seqlens_from_lengths([2, 3])
    bias = V.block_diagonal_bias(cu, cu, causal=False)
    # rows 0-1 (sample 0) must be -inf on cols 2-4 (sample 1) and vice versa.
    assert np.all(bias[0:2, 2:5] < -1e29)
    assert np.all(bias[2:5, 0:2] < -1e29)
    assert np.all(bias[0:2, 0:2] == 0.0)
    assert np.all(bias[2:5, 2:5] == 0.0)


# ---------------------------------------------------------------------------
# 3. Cosmos dual-stream: dense == varlen
# ---------------------------------------------------------------------------

def _cfg():
    return MT.MixtureTransformerConfig(
        hidden_size=32, num_heads=4,
        reasoner_intermediate=64, generator_intermediate=64,
    )


def _weights(cfg):
    r = MT.synthetic_tower_weights(cfg, cfg.reasoner_intermediate, seed=10)
    g = MT.synthetic_tower_weights(cfg, cfg.generator_intermediate, seed=20)
    return r, g


@pytest.mark.parametrize("role_ids", [
    [0, 0, 0, 1, 1],            # contiguous AR then DM (the canonical Cosmos layout)
    [0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1],               # interleaved roles (relative order preserved)
])
def test_dual_stream_dense_equals_varlen(role_ids):
    cfg = _cfg()
    r, g = _weights(cfg)
    rng = np.random.default_rng(3)
    S = len(role_ids)
    x = rng.standard_normal((1, S, cfg.hidden_size)).astype(np.float32)

    o_dense = MT.dual_stream_attention_dense(x, role_ids, r, g, cfg)
    o_varlen = MT.dual_stream_attention_varlen(x, role_ids, r, g, cfg)

    np.testing.assert_allclose(o_dense, o_varlen, rtol=1e-4, atol=1e-5)


def test_cosmos_join_bias_ar_never_attends_dm():
    role_ids = np.array([0, 0, 1, 1])
    bias = MT.cosmos_join_bias(role_ids)
    # AR queries (rows 0,1) must be masked on DM keys (cols 2,3).
    assert np.all(bias[0:2, 2:4] < -1e29)
    # AR causal within AR: row 0 cannot see col 1.
    assert bias[0, 1] < -1e29
    assert bias[1, 0] == 0.0
    # DM queries (rows 2,3) attend everywhere (bidirectional over [AR; DM]).
    assert np.all(bias[2:4, :] == 0.0)


# ---------------------------------------------------------------------------
# 4. The Cosmos metamorphic invariant — reasoning is self-contained
# ---------------------------------------------------------------------------

def test_reasoner_output_invariant_to_generator_presence():
    """Cosmos §2.3.1: "AR remains autoregressively self-contained." Adding,
    removing, or perturbing Generator tokens must NOT change the Reasoner
    outputs — AR queries never attend to DM keys/values."""
    cfg = _cfg()
    r, g = _weights(cfg)
    rng = np.random.default_rng(4)
    n_ar = 4

    # Build a sequence with AR tokens followed by some DM tokens.
    ar_x = rng.standard_normal((1, n_ar, cfg.hidden_size)).astype(np.float32)
    dm_x_a = rng.standard_normal((1, 3, cfg.hidden_size)).astype(np.float32)
    dm_x_b = rng.standard_normal((1, 5, cfg.hidden_size)).astype(np.float32)  # different count + values

    def run(dm_x):
        S = n_ar + dm_x.shape[1]
        x = np.concatenate([ar_x, dm_x], axis=1)
        roles = [MT.REASONER] * n_ar + [MT.GENERATOR] * dm_x.shape[1]
        out = MT.dual_stream_attention_dense(x, roles, r, g, cfg)
        return out[:, :n_ar, :]  # the Reasoner slice

    out_a = run(dm_x_a)
    out_b = run(dm_x_b)
    # Reasoner outputs must be bit-stable across totally different DM context.
    np.testing.assert_allclose(out_a, out_b, rtol=0, atol=1e-6)


def test_public_ops_varlen_sdpa_matches_reference():
    """The contract is exercised by the trace: tessera.ops.varlen_sdpa (the
    public op surface, lazy-routed to nn.varlen) equals the direct reference."""
    import tessera as ts

    rng = np.random.default_rng(11)
    H, Dh = 3, 8
    cu = V.cu_seqlens_from_lengths([4, 2, 3])
    total = int(cu[-1])
    Q = rng.standard_normal((H, total, Dh)).astype(np.float32)
    K = rng.standard_normal((H, total, Dh)).astype(np.float32)
    Vv = rng.standard_normal((H, total, Dh)).astype(np.float32)

    o_ops = ts.ops.varlen_sdpa(Q, K, Vv, cu_seqlens_q=cu, cu_seqlens_k=cu, causal=True)
    o_ref = V.varlen_sdpa(Q, K, Vv, cu_seqlens_q=cu, cu_seqlens_k=cu, causal=True)
    np.testing.assert_allclose(o_ops, o_ref, rtol=0, atol=0)


def test_runtime_op_dispatch_varlen_sdpa_by_name():
    """The runtime executes tessera.varlen_sdpa by op-name with cu_seqlens as
    operands 3/4 (the Graph IR / trace form)."""
    from tessera.runtime import _execute_runtime_cpu_op

    rng = np.random.default_rng(12)
    H, Dh = 2, 16
    cu_q = V.cu_seqlens_from_lengths([2, 3, 1]).astype(np.int32)
    cu_k = V.cu_seqlens_from_lengths([5, 5, 5]).astype(np.int32)
    tq, tk = int(cu_q[-1]), int(cu_k[-1])
    Q = rng.standard_normal((H, tq, Dh)).astype(np.float32)
    K = rng.standard_normal((H, tk, Dh)).astype(np.float32)
    Vv = rng.standard_normal((H, tk, Dh)).astype(np.float32)

    o_rt = _execute_runtime_cpu_op(
        "tessera.varlen_sdpa", [Q, K, Vv, cu_q, cu_k], {"causal": False}, np)
    o_ref = V.varlen_sdpa(Q, K, Vv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, causal=False)
    np.testing.assert_allclose(np.asarray(o_rt), o_ref, rtol=1e-6, atol=1e-6)


def test_varlen_path_reports_apple_gpu_when_injected():
    """Smoke: varlen_sdpa accepts an injected attention_fn (the seam where the
    Apple GPU metal_runtime dispatcher — or a future varlen kernel — plugs in)."""
    rng = np.random.default_rng(5)
    H, Dh = 2, 8
    cu = V.cu_seqlens_from_lengths([3, 2])
    total = int(cu[-1])
    Q = rng.standard_normal((H, total, Dh)).astype(np.float32)
    K = rng.standard_normal((H, total, Dh)).astype(np.float32)
    Vv = rng.standard_normal((H, total, Dh)).astype(np.float32)

    calls = {"n": 0}

    def spy(q, k, v, *, scale, causal, attn_bias):
        calls["n"] += 1
        return ops.flash_attn(q, k, v, scale=scale, causal=causal, attn_bias=attn_bias)

    out = V.varlen_sdpa(Q, K, Vv, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        causal=False, attention_fn=spy)
    assert calls["n"] == 2          # one launch per packed sample
    assert out.shape == (H, total, Dh)
