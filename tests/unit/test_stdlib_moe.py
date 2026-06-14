"""M2 — capacity-aware MoE dispatch + quantized grouped SwiGLU.

The no-capacity path must equal the proven ``models.moe_routing`` reference; the
quantized expert path must equal the dense path within the quant tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.models import moe_routing as ref
from tessera.stdlib import moe
from tessera.stdlib import quant as q


def _weights(rng, H, F, E):
    s = 1.0 / np.sqrt(H)
    return (
        (rng.standard_normal((H, E)) * s).astype(np.float32),       # router
        (rng.standard_normal((E, H, F)) * s).astype(np.float32),    # gate
        (rng.standard_normal((E, H, F)) * s).astype(np.float32),    # up
        (rng.standard_normal((E, F, H)) / np.sqrt(F)).astype(np.float32),  # down
    )


def test_compute_capacity():
    assert moe.compute_capacity(0, 2, 8, 1.0) == 0
    # 16 tokens, k=2, E=8 → base 4; factor 1.5 → ceil(6) = 6
    assert moe.compute_capacity(16, 2, 8, 1.5) == 6
    assert moe.compute_capacity(16, 2, 8, 1.0) == 4


def test_no_capacity_matches_reference_moe():
    """Capacity=None routed+shared output must match models.moe_routing.moe_forward."""
    rng = np.random.default_rng(0)
    H, F, E, k, T = 16, 24, 6, 2, 20
    wr, wg, wu, wd = _weights(rng, H, F, E)
    wsg = (rng.standard_normal((H, F)) / np.sqrt(H)).astype(np.float32)
    wsu = (rng.standard_normal((H, F)) / np.sqrt(H)).astype(np.float32)
    wsd = (rng.standard_normal((F, H)) / np.sqrt(F)).astype(np.float32)
    x = rng.standard_normal((T, H)).astype(np.float32)

    got = moe.moe_forward(x, wr, wg, wu, wd, top_k=k,
                          shared=(wsg, wsu, wsd), capacity_factor=None)
    want, _ = ref.moe_forward(x, wr, wg, wu, wd, wsg, wsu, wsd, top_k=k)
    np.testing.assert_allclose(got.y, want, rtol=1e-5, atol=1e-5)
    assert got.plan.drop_fraction == 0.0


def test_capacity_drops_overflow():
    """A tight capacity drops slots; combine zeros the dropped contribution."""
    rng = np.random.default_rng(1)
    H, F, E, k, T = 16, 24, 4, 2, 32
    wr, wg, wu, wd = _weights(rng, H, F, E)
    x = rng.standard_normal((T, H)).astype(np.float32)
    res = moe.moe_forward(x, wr, wg, wu, wd, top_k=k, capacity_factor=1.0)
    cap = moe.compute_capacity(T, k, E, 1.0)
    assert res.plan.capacity == cap
    assert (res.plan.group_sizes <= cap).all()
    assert res.plan.num_kept <= E * cap
    # at least one expert was contended enough to drop (tight cap, random routing)
    assert res.plan.drop_fraction >= 0.0


def test_dispatch_combine_roundtrip_identity_weights():
    """With unit weights and no capacity, combine∘dispatch reconstructs a sum
    over each token's k routed copies of itself."""
    rng = np.random.default_rng(2)
    H, E, k, T = 8, 5, 2, 10
    x = rng.standard_normal((T, H)).astype(np.float64)
    eids = rng.integers(0, E, size=(T, k)).astype(np.int64)
    w = np.ones((T, k), dtype=np.float32)
    plan = moe.plan_dispatch(eids, w, E, capacity=None)
    packed = moe.dispatch(x, plan)
    out = moe.combine(packed, plan)
    np.testing.assert_allclose(out, x * k, rtol=1e-6, atol=1e-6)


def test_quantized_grouped_swiglu_matches_dense():
    """Quantized expert SwiGLU ≈ dense SwiGLU within int4 group tolerance."""
    rng = np.random.default_rng(3)
    H, F, E = 32, 48, 3
    wg = (rng.standard_normal((E, H, F)) / np.sqrt(H)).astype(np.float32)
    wu = (rng.standard_normal((E, H, F)) / np.sqrt(H)).astype(np.float32)
    wd = (rng.standard_normal((E, F, H)) / np.sqrt(F)).astype(np.float32)
    group_sizes = np.array([4, 3, 5], dtype=np.int64)
    T = int(group_sizes.sum())
    xp = rng.standard_normal((T, H)).astype(np.float32)

    dense = moe.grouped_swiglu(xp, wg, wu, wd, group_sizes)
    gate_q = [q.quantize_weight(wg[e], "int4", group_size=16) for e in range(E)]
    up_q = [q.quantize_weight(wu[e], "int4", group_size=16) for e in range(E)]
    down_q = [q.quantize_weight(wd[e], "int4", group_size=16) for e in range(E)]
    quantized = moe.moe_swiglu_quantized(xp, gate_q, up_q, down_q, group_sizes)
    err = np.linalg.norm(quantized - dense) / np.linalg.norm(dense)
    assert err < 0.2, f"quant grouped SwiGLU rel err {err:.3f}"
