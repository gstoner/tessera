"""Track-R (ReplaySSM) Phase 1 — ``SSMStateHandle`` contract tests.

The headline oracle is **replay output ≡ eager ``selective_ssm``**: decoding a
sequence token-by-token through the handle (checkpoint state + ring-buffered
replay inputs, flushing when the buffer fills) must reproduce, per token, the
output of the dense eager reference — the metamorphic identity that makes the
ABI safe to lower per-backend.  The rest lock the ring-buffer ABI: flush route
selection, speculative rollback as a cursor move, checkpoint/restore, clone
isolation, and shape guards.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera.cache import SSMStateHandle


def _decode(handle, dt, x, Bp, Cp, gate=None):
    """Token-by-token decode through the handle → ``y`` (B, S, D)."""
    B, S, D = x.shape
    y = np.zeros((B, S, D))
    for t in range(S):
        gt = None if gate is None else gate[:, t, :]
        y[:, t, :] = handle.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :], gate_t=gt)
    return y


def _inputs(rng, B, S, D, N, scalar_a=True, gate=False):
    x = rng.standard_normal((B, S, D))
    a = -np.abs(rng.standard_normal(D if scalar_a else (D, N)))
    Bp = rng.standard_normal((B, S, N))
    Cp = rng.standard_normal((B, S, N))
    dt = np.abs(rng.standard_normal((B, S, D))) * 0.5
    g = np.abs(rng.standard_normal((B, S, D))) if gate else None
    return x, a, Bp, Cp, dt, g


# ── The headline oracle: replay ≡ eager ─────────────────────────────────

@pytest.mark.parametrize(
    "B,S,D,N,cap,scalar_a,spec,gate",
    [
        (1, 20, 4, 3, 64, True, 0, False),   # capacity >> S: no flush
        (1, 20, 4, 3, 5, True, 0, False),    # small ring: forces flushes
        (2, 30, 8, 5, 7, True, 2, False),    # spec window reserved
        (1, 16, 4, 3, 4, False, 0, False),   # full (D, N) A + flush
        (2, 24, 6, 4, 6, True, 0, True),     # output gate
        (3, 18, 5, 4, 9, True, 1, True),     # batched + spec + gate
    ],
)
def test_replay_equals_eager(B, S, D, N, cap, scalar_a, spec, gate):
    rng = np.random.default_rng(B * 100 + S + D + N + cap)
    x, a, Bp, Cp, dt, g = _inputs(rng, B, S, D, N, scalar_a, gate)
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt, gate=g))
    h = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a,
                       capacity=cap, spec_window=spec)
    y_replay = _decode(h, dt, x, Bp, Cp, gate=g)
    assert np.max(np.abs(y_eager - y_replay)) < 1e-9


def test_materialize_state_equals_eager_state():
    rng = np.random.default_rng(7)
    B, S, D, N = 2, 14, 5, 3
    x, a, Bp, Cp, dt, _ = _inputs(rng, B, S, D, N)
    h = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=5)
    _decode(h, dt, x, Bp, Cp)
    # Eager final state from the raw recurrence.
    a2 = np.broadcast_to(a[:, None], (D, N))
    hs = np.zeros((B, D, N))
    for t in range(S):
        ab = np.exp(dt[:, t, :, None] * a2[None])
        bb = dt[:, t, :, None] * Bp[:, t, None, :]
        hs = ab * hs + bb * x[:, t, :, None]
    assert np.max(np.abs(h.materialize_state() - hs)) < 1e-9


def test_output_only_matches_state_and_output():
    """The two routes must agree: output-only read ≡ read against the
    explicitly materialized (flushed) state."""
    rng = np.random.default_rng(11)
    B, S, D, N = 1, 10, 4, 3
    x, a, Bp, Cp, dt, _ = _inputs(rng, B, S, D, N)
    out_only = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=64)
    flushing = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=64)
    for t in range(S):
        y_oo = out_only.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
        # force the state-and-output route: append, flush, then read S0 directly
        flushing.append(dt[:, t, :], x[:, t, :], Bp[:, t, :])
        flushing.flush()
        y_so = np.einsum("bdn,bn->bd", flushing.checkpoint_state, Cp[:, t, :])
        assert np.max(np.abs(y_oo - y_so)) < 1e-9


# ── Ring-buffer ABI ─────────────────────────────────────────────────────

def test_route_and_flush_policy():
    h = SSMStateHandle(batch=1, num_channels=2, state_dim=2, a=np.array([-1.0, -1.0]),
                       capacity=8, spec_window=2)
    # count=0: 0 + 2*2 + 1 = 5 <= 8 → output_only
    assert h.route_for(1) == "output_only"
    assert not h.should_flush(1)
    # Fill to count=4: 4 + 4 + 1 = 9 > 8 → flush
    for _ in range(4):
        h.append(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
    assert h.should_flush(1)
    assert h.route_for(1) == "state_and_output"


def test_rollback_full_is_a_noop_on_outputs():
    """Appending speculative drafts then rolling them all back must leave the
    handle producing identical outputs to one that never saw the drafts."""
    rng = np.random.default_rng(3)
    B, S, D, N = 1, 6, 4, 3
    x, a, Bp, Cp, dt, _ = _inputs(rng, B, S, D, N)
    # Warm up both handles identically; large spec_window so drafts never flush.
    base = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=64, spec_window=4)
    for t in range(3):
        base.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
    spec = base.clone()
    # Draft 3 speculative tokens on `spec`, then reject all → rollback(3).
    draft = rng.standard_normal((B, 3, D))
    for j in range(3):
        spec.step(draft[:, j, :], draft[:, j, :], rng.standard_normal((B, N)),
                  rng.standard_normal((B, N)))
    spec.rollback(3)
    assert spec.count == base.count
    # Continue both with the real continuation; outputs must match.
    for t in range(3, S):
        y_b = base.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
        y_s = spec.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
        assert np.max(np.abs(y_b - y_s)) < 1e-12


def test_rollback_partial_keeps_accepted_drafts():
    """Accept j of T drafts: rollback(T-j) must equal having only ever
    decoded the j accepted tokens."""
    rng = np.random.default_rng(5)
    B, S, D, N = 1, 4, 4, 3
    x, a, Bp, Cp, dt, _ = _inputs(rng, B, S, D, N)
    spec = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=64, spec_window=4)
    truth = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=64, spec_window=4)
    for t in range(S):  # decode prefix identically
        spec.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
        truth.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
    # 3 drafts, accept first 1.
    drafts_d = rng.standard_normal((B, 3, D))
    drafts_b = rng.standard_normal((B, 3, N))
    drafts_c = rng.standard_normal((B, 3, N))
    for j in range(3):
        spec.step(drafts_d[:, j, :], drafts_d[:, j, :], drafts_b[:, j, :], drafts_c[:, j, :])
    spec.rollback(2)  # reject last 2
    # truth replays only the 1 accepted draft.
    y_truth = truth.step(drafts_d[:, 0, :], drafts_d[:, 0, :], drafts_b[:, 0, :], drafts_c[:, 0, :])
    # A fresh read at the accepted token on `spec` must match.
    y_spec = spec.read_output(drafts_c[:, 0, :])
    assert spec.count == truth.count
    assert np.max(np.abs(y_truth - y_spec)) < 1e-12


def test_checkpoint_restore_roundtrip():
    rng = np.random.default_rng(9)
    B, S, D, N = 2, 13, 5, 4
    x, a, Bp, Cp, dt, _ = _inputs(rng, B, S, D, N)
    h = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=6, spec_window=1)
    _decode(h, dt, x, Bp, Cp)
    blob = h.checkpoint()
    h2 = SSMStateHandle.restore(blob)
    assert h2.count == h.count
    assert np.array_equal(h2.checkpoint_state, h.checkpoint_state)
    assert np.max(np.abs(h2.materialize_state() - h.materialize_state())) < 1e-12
    # Both continue identically.
    extra_d, extra_x = rng.standard_normal((B, D)), rng.standard_normal((B, D))
    extra_b, extra_c = rng.standard_normal((B, N)), rng.standard_normal((B, N))
    y1 = h.step(extra_d, extra_x, extra_b, extra_c)
    y2 = h2.step(extra_d, extra_x, extra_b, extra_c)
    assert np.max(np.abs(y1 - y2)) < 1e-12


def test_clone_is_isolated():
    a = np.array([-1.0, -0.5, -0.3, -0.2])
    h = SSMStateHandle(batch=1, num_channels=4, state_dim=2, a=a, capacity=8)
    h.append(np.ones((1, 4)), np.ones((1, 4)), np.ones((1, 2)))
    c = h.clone()
    c.append(np.ones((1, 4)) * 9, np.ones((1, 4)), np.ones((1, 2)))
    assert h.count == 1 and c.count == 2  # mutation on clone does not leak


def test_flush_does_not_change_state():
    rng = np.random.default_rng(2)
    B, D, N = 1, 4, 3
    a = -np.abs(rng.standard_normal(D))
    h = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=8)
    for _ in range(5):
        h.append(rng.standard_normal((B, D)), rng.standard_normal((B, D)),
                 rng.standard_normal((B, N)))
    before = h.materialize_state()
    h.flush()
    assert h.count == 0
    assert np.max(np.abs(h.materialize_state() - before)) < 1e-12  # S0 == old live state


# ── Guards ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [
    dict(batch=0, num_channels=4, state_dim=3, a=np.zeros(4)),
    dict(batch=1, num_channels=4, state_dim=3, a=np.zeros(4), capacity=0),
    dict(batch=1, num_channels=4, state_dim=3, a=np.zeros(3)),       # a leading != D
    dict(batch=1, num_channels=4, state_dim=3, a=np.zeros((4, 5))),  # a (D,N) wrong N
    dict(batch=1, num_channels=4, state_dim=3, a=np.zeros(4), spec_window=-1),
])
def test_construction_guards(kwargs):
    with pytest.raises(ValueError):
        SSMStateHandle(**kwargs)


def test_dtype_canonicalized():
    h = SSMStateHandle(batch=1, num_channels=4, state_dim=3, a=np.zeros(4), dtype="float32")
    assert h.dtype == "fp32"
