"""SD1-3 — cache_commit / cache_rollback are typed-effect (state) cache cursor
ops, the IR-visible form of the speculative-decode commit/rollback. This proves
two things: (1) the EffectLattice classifies them as `state` (>= state, not pure),
so a speculative loop that touches the cache is no longer "pure" — the mutation is
visible to the compiler (CF0: "cache mutation only through typed cache handles");
(2) the registered references faithfully advance/rewind a real KVCacheHandle /
SSMStateHandle cursor, matching speculative.advance_kv / advance_ssm.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera.compiler.effects import Effect, EffectLattice, _OP_EFFECTS


# ── (1) typed state effect ──────────────────────────────────────────────────
def test_cache_ops_are_state_effect_in_the_table():
    for name in ("cache_commit", "cache_rollback"):
        assert _OP_EFFECTS[name] == Effect.state
        assert _OP_EFFECTS[name] > Effect.pure  # not pure → compiler-visible


def test_effect_lattice_infers_state_for_a_cache_committing_region():
    lat = EffectLattice()

    def pure_region(x):
        return x + 1

    def committing_region(cache, n):
        return cache_commit(cache, n)  # noqa: F821 — name resolved by the AST walk

    def rolling_region(cache, n):
        return cache_rollback(cache, n)  # noqa: F821

    assert lat.infer(pure_region) == Effect.pure
    # The cache cursor ops lift the region's effect to >= state (out of pure).
    assert lat.infer(committing_region) >= Effect.state
    assert lat.infer(rolling_region) >= Effect.state


def test_cache_ops_registered_and_dispatchable():
    for name in ("cache_commit", "cache_rollback"):
        assert name in tessera.ops.registry.list()
        assert hasattr(tessera.ops, name)


# ── (2) faithful KV cursor behavior ─────────────────────────────────────────
def _kv(seq):
    from tessera.cache import KVCacheHandle
    h = KVCacheHandle(num_heads=2, head_dim=4, max_seq=64)
    h.append(np.ones((seq, 2, 4), np.float32), np.ones((seq, 2, 4), np.float32))
    return h


def test_cache_commit_keeps_accepted_prefix_on_kv():
    from tessera.speculative import advance_kv
    # commit(n) keeps n tokens — identical to advance_kv(cache, n).
    h = _kv(7)
    tessera.ops.cache_commit(h, 4)
    assert h.current_seq == 4
    ref = _kv(7)
    advance_kv(ref, 4)
    assert h.current_seq == ref.current_seq


def test_cache_rollback_rewinds_rejected_on_kv():
    # rollback(n) rewinds the newest n tokens (KVCacheHandle.trim).
    h = _kv(7)
    tessera.ops.cache_rollback(h, 3)
    assert h.current_seq == 4  # 7 - 3


# ── (2) faithful SSM cursor behavior ────────────────────────────────────────
def test_cache_rollback_rewinds_rejected_on_ssm():
    from tessera.cache import SSMStateHandle
    h = SSMStateHandle(batch=1, num_channels=3, state_dim=2,
                       a=np.ones((3, 2), np.float32))
    for _ in range(5):
        h.append(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 2)))
    assert h.count == 5
    # rollback(num_rejected) == handle.rollback — the SSM speculative undo.
    tessera.ops.cache_rollback(h, 2)
    assert h.count == 3


def test_cache_rollback_requires_a_cursor_handle():
    with pytest.raises(TypeError):
        tessera.ops.cache_rollback(object(), 1)
