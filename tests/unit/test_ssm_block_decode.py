"""Track-R — block decode kernels (dispatch-overhead fix + GDN fused).

The per-token fused decode kernel pays one command-buffer commit+wait *per
token*, which dominates wall-clock at decode-1.  The block decode kernels
process a whole block of T tokens (prefill / speculative verification /
benchmark, where the inputs are known up front) in a SINGLE dispatch — one
commit+wait for the block.  These tests validate both the Mamba-2 scalar-A
block kernel and the gated delta-rule (GDN) block kernel against the eager
references, and that the block path is materially faster than per-token.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import tessera
from tessera import runtime as rt
from tessera.cache import SSMStateHandle, DeltaNetStateHandle
from tessera.stdlib.delta_rule import gated_delta_rule_recurrent


# ── Mamba-2 scalar-A block decode ───────────────────────────────────────

def _ssm_inputs(rng, B, D, N, T):
    x = rng.standard_normal((T, B, D))
    a = -np.abs(rng.standard_normal(D))
    b = rng.standard_normal((T, B, N))
    c = rng.standard_normal((T, B, N))
    dt = np.abs(rng.standard_normal((T, B, D))) * 0.5
    return x, a, b, c, dt


@pytest.mark.parametrize("B,D,N,T", [(1, 4, 3, 16), (2, 8, 4, 24), (1, 16, 8, 32)])
def test_ssm_block_decode_matches_eager(B, D, N, T):
    rng = np.random.default_rng(B * 13 + D + N + T)
    x, a, b, c, dt = _ssm_inputs(rng, B, D, N, T)
    y_eager = np.moveaxis(np.asarray(tessera.ops.selective_ssm(
        np.moveaxis(x, 0, 1), a, np.moveaxis(b, 0, 1),
        np.moveaxis(c, 0, 1), np.moveaxis(dt, 0, 1))), 1, 0)
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=8)
    y = h.decode_block(dt, x, b, c)
    assert y.shape == (T, B, D)
    assert np.max(np.abs(y - y_eager)) < 5e-3        # f32 GPU tolerance


def test_ssm_block_matches_per_token_and_is_pure():
    rng = np.random.default_rng(1)
    B, D, N, T = 2, 8, 4, 20
    x, a, b, c, dt = _ssm_inputs(rng, B, D, N, T)
    block = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=8)
    y_block = block.decode_block(dt, x, b, c)
    assert block.count == 0                          # pure: handle unchanged
    # Per-token decode from zero state (reference handle) must agree.
    ref = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=T + 2)
    y_pt = np.stack([ref.step(dt[t], x[t], b[t], c[t]) for t in range(T)], 0)
    assert np.max(np.abs(y_block - y_pt)) < 5e-3


def test_ssm_block_numpy_fallback_is_exact():
    rng = np.random.default_rng(2)
    B, D, N, T = 1, 6, 4, 18
    x, a, b, c, dt = _ssm_inputs(rng, B, D, N, T)
    y_eager = np.moveaxis(np.asarray(tessera.ops.selective_ssm(
        np.moveaxis(x, 0, 1), a, np.moveaxis(b, 0, 1),
        np.moveaxis(c, 0, 1), np.moveaxis(dt, 0, 1))), 1, 0)
    h = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=8)  # no block_fn
    y = h.decode_block(dt, x, b, c)                  # numpy reference path
    assert np.max(np.abs(y - y_eager)) < 1e-9


# ── Gated DeltaNet (GDN) block decode ───────────────────────────────────

def _gdn_inputs(rng, B, H, dk, dv, S):
    Q = rng.standard_normal((B, H, S, dk))
    K = rng.standard_normal((B, H, S, dk))
    V = rng.standard_normal((B, H, S, dv))
    beta = np.abs(rng.standard_normal((B, H, S))) * 0.5
    decay = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, S))))
    return Q, K, V, beta, decay


@pytest.mark.parametrize("B,H,dk,dv,S", [(1, 2, 8, 16, 24), (2, 2, 6, 8, 32), (1, 1, 4, 4, 16)])
def test_gdn_block_decode_matches_eager(B, H, dk, dv, S):
    rng = np.random.default_rng(B * 19 + H + dk + dv + S)
    Q, K, V, beta, decay = _gdn_inputs(rng, B, H, dk, dv, S)
    O_eager = np.asarray(gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay))
    h = rt.apple_gpu_delta_state_handle(B, H, dk, dv, capacity=8)
    O = h.decode_block(Q, K, V, beta=beta, decay=decay)
    assert O.shape == (B, H, S, dv)
    assert np.max(np.abs(O - O_eager)) < 5e-3        # f32 GPU tolerance


def test_gdn_block_matches_per_token_and_is_pure():
    rng = np.random.default_rng(3)
    B, H, dk, dv, S = 1, 2, 6, 8, 20
    Q, K, V, beta, decay = _gdn_inputs(rng, B, H, dk, dv, S)
    block = rt.apple_gpu_delta_state_handle(B, H, dk, dv, capacity=8)
    O_block = block.decode_block(Q, K, V, beta=beta, decay=decay)
    assert block.count == 0
    ref = DeltaNetStateHandle(batch=B, num_heads=H, key_dim=dk, value_dim=dv,
                              capacity=S + 2)
    O_pt = np.zeros((B, H, S, dv))
    for t in range(S):
        O_pt[:, :, t, :] = ref.step(Q[:, :, t, :], K[:, :, t, :], V[:, :, t, :],
                                    beta_t=beta[:, :, t], decay_t=decay[:, :, t])
    assert np.max(np.abs(O_block - O_pt)) < 5e-3


def test_gdn_block_numpy_fallback_is_exact():
    rng = np.random.default_rng(4)
    B, H, dk, dv, S = 1, 2, 5, 4, 16
    Q, K, V, beta, decay = _gdn_inputs(rng, B, H, dk, dv, S)
    O_eager = np.asarray(gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay))
    h = DeltaNetStateHandle(batch=B, num_heads=H, key_dim=dk, value_dim=dv, capacity=8)
    O = h.decode_block(Q, K, V, beta=beta, decay=decay)   # numpy reference path
    assert np.max(np.abs(O - O_eager)) < 1e-9


# ── f16 block kernels (half I/O, f32 accumulation) ──────────────────────

def test_ssm_block_f16_matches_eager():
    rng = np.random.default_rng(11)
    B, D, N, T = 2, 16, 8, 24
    x, a, b, c, dt = _ssm_inputs(rng, B, D, N, T)
    y_eager = np.moveaxis(np.asarray(tessera.ops.selective_ssm(
        np.moveaxis(x, 0, 1), a, np.moveaxis(b, 0, 1),
        np.moveaxis(c, 0, 1), np.moveaxis(dt, 0, 1))), 1, 0)
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=8,
                                            compute_dtype="fp16")
    y = h.decode_block(dt, x, b, c)
    assert np.max(np.abs(y - y_eager)) < 5e-2        # f16 tolerance


def test_gdn_block_f16_matches_eager():
    rng = np.random.default_rng(12)
    B, H, dk, dv, S = 1, 2, 8, 16, 20
    Q, K, V, beta, decay = _gdn_inputs(rng, B, H, dk, dv, S)
    O_eager = np.asarray(gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay))
    h = rt.apple_gpu_delta_state_handle(B, H, dk, dv, capacity=8,
                                        compute_dtype="fp16")
    O = h.decode_block(Q, K, V, beta=beta, decay=decay)
    assert np.max(np.abs(O - O_eager)) < 5e-2        # f16 tolerance


# ── bf16 block kernels (storage bf16 via fp32-conversion, f32 accum) ─────

_bf16 = rt._bfloat16_dtype()


@pytest.mark.skipif(_bf16 is None, reason="ml_dtypes (bfloat16) not installed")
def test_ssm_block_bf16_matches_eager():
    rng = np.random.default_rng(21)
    B, D, N, T = 2, 16, 8, 24
    x, a, b, c, dt = _ssm_inputs(rng, B, D, N, T)
    y_eager = np.moveaxis(np.asarray(tessera.ops.selective_ssm(
        np.moveaxis(x, 0, 1), a, np.moveaxis(b, 0, 1),
        np.moveaxis(c, 0, 1), np.moveaxis(dt, 0, 1))), 1, 0)
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=8,
                                            compute_dtype="bf16")
    y = h.decode_block(dt, x, b, c)
    assert np.max(np.abs(y - y_eager)) < 2e-1        # bf16 (8-bit mantissa)


@pytest.mark.skipif(_bf16 is None, reason="ml_dtypes (bfloat16) not installed")
def test_gdn_block_bf16_matches_eager():
    rng = np.random.default_rng(22)
    B, H, dk, dv, S = 1, 2, 8, 16, 20
    Q, K, V, beta, decay = _gdn_inputs(rng, B, H, dk, dv, S)
    O_eager = np.asarray(gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay))
    h = rt.apple_gpu_delta_state_handle(B, H, dk, dv, capacity=8,
                                        compute_dtype="bf16")
    O = h.decode_block(Q, K, V, beta=beta, decay=decay)
    assert np.max(np.abs(O - O_eager)) < 2e-1        # bf16


# ── Multi-head GDN beyond the register envelope (threadgroup state) ──────

@pytest.mark.parametrize("dk,dv", [(16, 32), (16, 64), (12, 48)])  # dk*dv > 256
def test_gdn_block_big_state_matches_eager(dk, dv):
    rng = np.random.default_rng(dk * 100 + dv)
    B, H, S = 1, 2, 18
    Q, K, V, beta, decay = _gdn_inputs(rng, B, H, dk, dv, S)
    assert dk * dv > 256                              # register kernel can't hold this
    O_eager = np.asarray(gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay))
    h = rt.apple_gpu_delta_state_handle(B, H, dk, dv, capacity=8)
    O = h.decode_block(Q, K, V, beta=beta, decay=decay)
    assert O.shape == (B, H, S, dv)
    assert np.max(np.abs(O - O_eager)) < 5e-3        # threadgroup f32


def test_delta_block_callable_routes_by_size():
    """Small state → register kernel; large state → big (threadgroup) kernel;
    both return a result (not None)."""
    cb = rt.apple_gpu_delta_block_callable("fp32")
    rng = np.random.default_rng(0)
    for dk, dv in [(4, 4), (16, 32)]:                # 16 (reg) and 512 (big)
        B, H, S = 1, 1, 8
        Q, K, V, beta, decay = _gdn_inputs(rng, B, H, dk, dv, S)
        res = cb(Q, K, V, beta, decay, np.zeros((B, H, dk, dv)),
                 B, H, S, dk, dv, 1)
        assert res is not None, f"no kernel for dk*dv={dk*dv}"
        O, Sout = res
        assert O.shape == (B, H, S, dv) and Sout.shape == (B, H, dk, dv)


# ── The dispatch-overhead fix: block beats per-token ────────────────────

def test_block_prefill_optin_matches_default(monkeypatch):
    """The opt-in TESSERA_SSM_BLOCK_PREFILL path routes the stateless
    `selective_ssm` graph dispatch through the block kernel and must agree with
    the default bmm path and the eager reference."""
    rng = np.random.default_rng(0)
    B, S, D, N = 2, 24, 16, 8
    x = rng.standard_normal((B, S, D))
    a = -np.abs(rng.standard_normal(D))
    Bp = rng.standard_normal((B, S, N))
    Cp = rng.standard_normal((B, S, N))
    dt = np.abs(rng.standard_normal((B, S, D))) * 0.5
    operands = (x, a, Bp, Cp, dt)
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))

    monkeypatch.delenv("TESSERA_SSM_BLOCK_PREFILL", raising=False)
    y_default = np.asarray(rt._apple_gpu_dispatch_selective_ssm(operands, {}, np))
    monkeypatch.setenv("TESSERA_SSM_BLOCK_PREFILL", "1")
    y_block = np.asarray(rt._apple_gpu_dispatch_selective_ssm(operands, {}, np))

    assert np.max(np.abs(y_default - y_eager)) < 5e-3
    assert np.max(np.abs(y_block - y_eager)) < 5e-3
    assert np.max(np.abs(y_block - y_default)) < 5e-3


def test_block_prefill_size_heuristic_default(monkeypatch):
    """With no env override, the default auto-routes by size: a large shape
    (T>=128 or B*D>=256) uses the block kernel; a tiny shape uses bmm.  Both
    must match eager regardless of which path runs."""
    monkeypatch.delenv("TESSERA_SSM_BLOCK_PREFILL", raising=False)
    rng = np.random.default_rng(1)
    for B, S, D, N in [(1, 16, 8, 4), (1, 160, 32, 16)]:   # tiny (bmm), large (block)
        x = rng.standard_normal((B, S, D)).astype(np.float32)
        a = -np.abs(rng.standard_normal(D)).astype(np.float32)
        Bp = rng.standard_normal((B, S, N)).astype(np.float32)
        Cp = rng.standard_normal((B, S, N)).astype(np.float32)
        dt = (np.abs(rng.standard_normal((B, S, D))) * 0.4).astype(np.float32)
        y = np.asarray(rt._apple_gpu_dispatch_selective_ssm((x, a, Bp, Cp, dt), {}, np))
        y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))
        assert np.max(np.abs(y - y_eager)) < 5e-3, f"shape {(B,S,D,N)}"


@pytest.mark.performance
def test_block_decode_beats_per_token():
    """One dispatch for T tokens must be materially faster than T per-token
    dispatches (the documented decode-1 dispatch-overhead fix)."""
    B, D, N, T = 1, 64, 64, 48
    rng = np.random.default_rng(0)
    x, a, b, c, dt = _ssm_inputs(rng, B, D, N, T)

    def per_token():
        h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=T + 2)
        for t in range(T):
            h.step(dt[t], x[t], b[t], c[t])

    def block():
        h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=8)
        h.decode_block(dt, x, b, c)

    per_token(); block()                              # warm up
    t0 = time.perf_counter_ns(); per_token(); pt = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns(); block(); bl = time.perf_counter_ns() - t0
    assert bl < pt, f"block ({bl/1e6:.2f}ms) not faster than per-token ({pt/1e6:.2f}ms)"
