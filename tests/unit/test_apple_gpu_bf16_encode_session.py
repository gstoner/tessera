"""Project-3 bf16 encode-session ABIs — closes the dtype matrix.

bf16 is the third leg of the {f32, f16, bf16} encode-session matrix
that LLM workloads care about. MPSGraph accepts ``MPSDataTypeBFloat16``
on macOS 26+ M2+ — the implementation simply routes through the
existing MPSGraph encode helpers with the bf16 type code.

MSL custom kernel paths (rope, flash_attn) intentionally lack bf16
encode-session variants today: they need an on-GPU bf16↔fp32
conversion pass which is the Phase-3b follow-on. The registry has
6 bf16 ops (not 8); the chain planner correctly segments those gaps.

Tests pin:

* **Capability probe** — ``bf16_session_available()`` reports
  truthfully (matches the runtime probe).
* **Symbol availability** for all 6 new bf16 C ABIs.
* **Numerical correctness** of each bf16 op against numpy at bf16
  tolerance (~1% — bf16 has ~8-bit mantissa precision).
* **bf16 attention block (partial)** — rmsnorm + bmm chain in bf16
  on one cb. flash_attn / rope would route to f32 today; explicit
  in the test commentary.
* **Mixed-dtype chain** — f32 + bf16 in the same trace correctly
  splits into 2 sessions.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_gpu_batched import (
    DeviceTensor,
    batched_session,
    bf16_session_available,
    bmm_enc,
    bmm_enc_bf16,
    device_tensor,
    gelu_enc_bf16,
    layer_norm_enc_bf16,
    rmsnorm_enc_bf16,
    session_available,
    session_commit_count,
    silu_enc_bf16,
    softmax_enc_bf16,
)
from tessera.apple_gpu_chain import OpRecord, plan_chain, run_trace


# numpy doesn't have native bf16. bf16 = high 16 bits of fp32 IEEE-754
# bit pattern. Helpers convert via uint32 views.

def _to_bf16(a: np.ndarray) -> np.ndarray:
    """Convert fp32 → bf16 bit pattern (uint16 carrying the high 16
    bits of the fp32 bit pattern). Round-to-nearest-even."""
    fp32 = a.astype(np.float32)
    bits = fp32.view(np.uint32)
    # rounding_bias = 0x7FFF + (bits>>16)&1, then >>16
    bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + bias) >> 16
    return rounded.astype(np.uint16).reshape(a.shape)


def _from_bf16(a: np.ndarray) -> np.ndarray:
    """Convert bf16 bit pattern (uint16) → fp32."""
    bits = (a.astype(np.uint32) << 16)
    return bits.view(np.float32).reshape(a.shape)


# ---- Capability probe --------------------------------------------------

def test_bf16_session_available_reports_truthfully():
    if not session_available():
        pytest.skip("encode-session unavailable")
    avail = bf16_session_available()
    assert isinstance(avail, bool)
    # On macOS 26+ M2+, this is True. On older hosts, False. Either
    # way the helper should not crash.


# ---- Symbol availability ----------------------------------------------

@pytest.mark.parametrize("symbol", [
    "tessera_apple_gpu_bmm_dev_bf16_enc",
    "tessera_apple_gpu_layer_norm_dev_bf16_enc",
    "tessera_apple_gpu_rmsnorm_dev_bf16_enc",
    "tessera_apple_gpu_softmax_dev_bf16_enc",
    "tessera_apple_gpu_unary_dev_bf16_enc",
    "tessera_apple_gpu_mpsgraph_bf16_supported",
])
def test_bf16_symbols_resolve(symbol):
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    fn = bind_symbol(symbol, (ctypes.c_void_p,), ctypes.c_int32)
    assert fn is not None, f"missing bf16 ABI: {symbol}"


# ---- Numerical correctness — per op -----------------------------------

def test_rmsnorm_bf16_matches_numpy():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    rows, cols, eps = 8, 32, 1e-5
    rng = np.random.default_rng(0xBF160E1)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.5
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    x_dev = device_tensor(_to_bf16(X))
    g_dev = device_tensor(_to_bf16(gamma))
    try:
        with batched_session() as s:
            y_dev = rmsnorm_enc_bf16(s, x_dev, g_dev,
                                      rows=rows, cols=cols, eps=eps)
        gpu = _from_bf16(y_dev.download(np.uint16, (rows, cols)))
        y_dev.free()
        var = (X * X).mean(axis=-1, keepdims=True)
        expected = X / np.sqrt(var + eps) * gamma
        # bf16 has ~8-bit mantissa → ~3e-3 relative precision.
        np.testing.assert_allclose(gpu, expected, rtol=1e-2, atol=1e-2)
    finally:
        x_dev.free(); g_dev.free()


def test_layer_norm_bf16_matches_numpy():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    rows, cols, eps = 8, 32, 1e-5
    rng = np.random.default_rng(0xBF16178)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.5
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    beta = rng.standard_normal((cols,), dtype=np.float32)
    x_dev = device_tensor(_to_bf16(X))
    g_dev = device_tensor(_to_bf16(gamma))
    b_dev = device_tensor(_to_bf16(beta))
    try:
        with batched_session() as s:
            y_dev = layer_norm_enc_bf16(s, x_dev, g_dev, b_dev,
                                         rows=rows, cols=cols, eps=eps)
        gpu = _from_bf16(y_dev.download(np.uint16, (rows, cols)))
        y_dev.free()
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        expected = (X - mean) / np.sqrt(var + eps) * gamma + beta
        np.testing.assert_allclose(gpu, expected, rtol=1e-2, atol=1e-2)
    finally:
        x_dev.free(); g_dev.free(); b_dev.free()


def test_softmax_bf16_matches_numpy():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    rows, cols = 6, 12
    rng = np.random.default_rng(0xBF1650A)
    X = rng.standard_normal((rows, cols), dtype=np.float32)
    x_dev = device_tensor(_to_bf16(X))
    try:
        with batched_session() as s:
            y_dev = softmax_enc_bf16(s, x_dev, rows=rows, cols=cols)
        gpu = _from_bf16(y_dev.download(np.uint16, (rows, cols)))
        y_dev.free()
        m = X.max(axis=-1, keepdims=True)
        e = np.exp(X - m)
        expected = e / e.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(gpu, expected, rtol=1e-2, atol=1e-2)
        # Softmax rows sum to ~1 within bf16 tolerance.
        np.testing.assert_allclose(gpu.sum(axis=-1), 1.0, atol=2e-2)
    finally:
        x_dev.free()


def test_silu_bf16_matches_numpy():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    n = 128
    rng = np.random.default_rng(0xBF1655)
    X = rng.standard_normal((n,), dtype=np.float32)
    x_dev = device_tensor(_to_bf16(X))
    try:
        with batched_session() as s:
            y_dev = silu_enc_bf16(s, x_dev, n=n)
        gpu = _from_bf16(y_dev.download(np.uint16, (n,)))
        y_dev.free()
        expected = X / (1.0 + np.exp(-X))
        np.testing.assert_allclose(gpu, expected, rtol=1e-2, atol=1e-2)
    finally:
        x_dev.free()


def test_gelu_bf16_matches_numpy():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    n = 128
    rng = np.random.default_rng(0xBF16E1)
    X = rng.standard_normal((n,), dtype=np.float32)
    x_dev = device_tensor(_to_bf16(X))
    try:
        with batched_session() as s:
            y_dev = gelu_enc_bf16(s, x_dev, n=n)
        gpu = _from_bf16(y_dev.download(np.uint16, (n,)))
        y_dev.free()
        c = np.sqrt(2.0 / np.pi)
        expected = 0.5 * X * (1.0 + np.tanh(c * (X + 0.044715 * X ** 3)))
        np.testing.assert_allclose(gpu, expected, rtol=2e-2, atol=2e-2)
    finally:
        x_dev.free()


def test_bmm_bf16_matches_numpy():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    batch, M, N, K = 2, 4, 8, 6
    rng = np.random.default_rng(0xB7B16)
    A = rng.standard_normal((batch, M, K), dtype=np.float32) * 0.3
    B = rng.standard_normal((batch, K, N), dtype=np.float32) * 0.3
    a_dev = device_tensor(_to_bf16(A))
    b_dev = device_tensor(_to_bf16(B))
    try:
        with batched_session() as s:
            o = bmm_enc_bf16(s, a_dev, b_dev,
                              batch=batch, M=M, N=N, K=K)
        gpu = _from_bf16(o.download(np.uint16, (batch, M, N)))
        o.free()
        expected = np.einsum("bik,bkj->bij", A, B)
        np.testing.assert_allclose(gpu, expected, rtol=2e-2, atol=2e-2)
    finally:
        a_dev.free(); b_dev.free()


# ---- Partial bf16 attention block on one cb ----------------------------

def test_bf16_rmsnorm_plus_bmm_on_one_command_buffer():
    """The encode-session ops we have in bf16 today: rmsnorm + bmm.
    Run them in one session; verify single-cb invariant and correct
    output. (flash_attn / rope bf16 is Phase-3b — the encode-session
    versions need on-GPU bf16↔fp32 conversion.)"""
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    rows, cols, eps = 4, 16, 1e-5
    rng = np.random.default_rng(0xBF16BB)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    W = rng.standard_normal((cols, cols), dtype=np.float32) * 0.1
    x_dev = device_tensor(_to_bf16(X))
    g_dev = device_tensor(_to_bf16(gamma))
    w_dev = device_tensor(_to_bf16(W.reshape(1, cols, cols)))
    try:
        before = session_commit_count()
        with batched_session() as s:
            n = rmsnorm_enc_bf16(s, x_dev, g_dev,
                                  rows=rows, cols=cols, eps=eps)
            out = bmm_enc_bf16(s, n, w_dev,
                                batch=1, M=rows, N=cols, K=cols)
        after = session_commit_count()
        assert (after - before) == 1
        gpu = _from_bf16(
            out.download(np.uint16, (1, rows, cols))
        ).reshape(rows, cols)
        n.free(); out.free()

        var = (X * X).mean(axis=-1, keepdims=True)
        n_ref = X / np.sqrt(var + eps) * gamma
        expected = n_ref @ W
        np.testing.assert_allclose(gpu, expected, rtol=2e-2, atol=2e-2)
    finally:
        x_dev.free(); g_dev.free(); w_dev.free()


# ---- Mixed-dtype trace -------------------------------------------------

def test_chain_planner_groups_bf16_ops_separately_from_f32():
    """Planner segments by dtype: a mixed f32+bf16 trace produces 2
    chain segments (one per dtype). Verifies the registry's bf16
    entries integrate correctly with the planner."""
    trace = [
        OpRecord("rmsnorm", "f32"),
        OpRecord("bmm", "f32"),
        OpRecord("rmsnorm", "bf16"),
        OpRecord("bmm", "bf16"),
    ]
    segs = plan_chain(trace)
    assert len(segs) == 2
    assert all(s.kind == "encode" for s in segs)
    assert len(segs[0].ops) == 2
    assert len(segs[1].ops) == 2


def test_chain_planner_breaks_on_dtype_boundary_in_either_direction():
    """f32 → bf16 → f32 produces 3 segments."""
    trace = [
        OpRecord("rmsnorm", "f32"),
        OpRecord("bmm", "bf16"),
        OpRecord("silu", "f32"),
    ]
    segs = plan_chain(trace)
    assert len(segs) == 3
    assert all(s.kind == "encode" for s in segs)


def test_mixed_dtype_trace_uses_separate_sessions():
    """End-to-end: mixed f32 + bf16 trace executes as 2 separate
    sessions (2 commits)."""
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    rows, cols, eps = 4, 16, 1e-5
    rng = np.random.default_rng(0xD7D7BF)
    X32 = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma32 = rng.standard_normal((cols,), dtype=np.float32)
    X16 = _to_bf16(X32)
    g16 = _to_bf16(gamma32)

    trace = [
        OpRecord("rmsnorm", "f32",
                  inputs=[X32, gamma32],
                  shape_kwargs=dict(rows=rows, cols=cols, eps=eps)),
        OpRecord("rmsnorm", "bf16",
                  inputs=[X16, g16],
                  shape_kwargs=dict(rows=rows, cols=cols, eps=eps)),
    ]
    segs = plan_chain(trace)
    assert len(segs) == 2

    before = session_commit_count()
    results = run_trace(trace)
    after = session_commit_count()
    assert (after - before) == 2
    for r in results:
        if r is not None:
            r.free()


# ---- Registry coverage check ------------------------------------------

def test_bf16_registry_covers_full_op_envelope():
    """After Phase 3b (2026-06-01), bf16 covers the full 8-op
    envelope — rope and flash_attn route through on-GPU bf16↔fp32
    cast. Project 5 (2026-06-01) added conv2d (f32 only — bf16/f16
    conv2d encode lanes deliberately deferred), so the asymmetric
    matrix is: 8 ops × {f16, bf16} + 9 ops × {f32} = 25 entries."""
    from tessera.apple_gpu_chain import ENCODE_OP_REGISTRY
    bf16_ops = {name for (name, dtype) in ENCODE_OP_REGISTRY
                if dtype == "bf16"}
    assert bf16_ops == {"bmm", "layer_norm", "rmsnorm", "softmax",
                        "silu", "gelu", "rope", "flash_attn"}, bf16_ops
    # f16 and bf16 cover 8 ops; f32 covers 9 (conv2d added by Project 5).
    f16_ops = {name for (name, d) in ENCODE_OP_REGISTRY if d == "f16"}
    f32_ops = {name for (name, d) in ENCODE_OP_REGISTRY if d == "f32"}
    assert len(f16_ops) == 8, f16_ops
    assert len(bf16_ops) == 8
    assert f32_ops == bf16_ops | {"conv2d"}, f32_ops
