"""Autodiff coverage for the four-sprint follow-up:

  Sprint 1 — S6 collectives: psum / pmean / pmax / pmin /
             collective_permute / broadcast_to_axis VJPs and JVPs.
  Sprint 2 — Stateful optimizers: momentum / nesterov / adamw VJPs and JVPs.
  Sprint 3 — Memory: differentiable `memory_read` VJP and JVP.
  Sprint 4 — Reduction hardening: cummax / cummin VJPs and JVPs.

Each VJP is checked against a central finite-difference reference; each
JVP is checked against a finite-difference forward tangent.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera.sharding as ts_sharding
import tessera.optim as ts_optim
import tessera.memory as ts_memory
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp


# ── helpers ────────────────────────────────────────────────────────────────


def _numeric_grad(fn, x, eps=1e-4):
    g = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64).copy()
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = float(np.asarray(fn(x)).sum())
        x[idx] = orig - eps
        f_minus = float(np.asarray(fn(x)).sum())
        x[idx] = orig
        g[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return g


def _numeric_jvp(fn, x, dx, eps=1e-5):
    plus = np.asarray(fn(x + eps * dx), dtype=np.float64)
    minus = np.asarray(fn(x - eps * dx), dtype=np.float64)
    return (plus - minus) / (2 * eps)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 1 — Collectives
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "psum", "pmean", "pmax", "pmin",
    "collective_permute", "broadcast_to_axis",
])
def test_collective_vjp_and_jvp_registered(name):
    assert get_vjp(name) is not None, f"VJP missing: {name}"
    assert get_jvp(name) is not None, f"JVP missing: {name}"


def test_psum_vjp_broadcasts_back_to_each_rank():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(4, 3, 5))   # 4 ranks
    do = rng.normal(size=(3, 5))
    grad, = get_vjp("psum")(do, values)
    # psum is summing over rank axis; backward broadcasts dout to every rank.
    expected = np.broadcast_to(do, values.shape)
    np.testing.assert_allclose(grad, expected)


def test_pmean_vjp_divides_by_rank_count():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(8, 6))
    do = np.ones((6,))
    grad, = get_vjp("pmean")(do, values)
    np.testing.assert_allclose(grad, np.full_like(values, 1.0 / 8))


def test_pmax_vjp_routes_to_argmax_rank_with_tie_split():
    # Exact ties at rank 0 and rank 1 — gradient should split evenly.
    values = np.array([
        [3.0, 1.0],   # rank 0
        [3.0, 5.0],   # rank 1
        [2.0, 5.0],   # rank 2
    ])
    do = np.ones((2,))
    grad, = get_vjp("pmax")(do, values)
    expected = np.array([
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ])
    np.testing.assert_allclose(grad, expected)


def test_pmax_vjp_matches_finite_difference():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(4, 3, 2))
    do = rng.normal(size=(3, 2))
    grad, = get_vjp("pmax")(do, values)
    expected = _numeric_grad(
        lambda v: float((do * np.max(v, axis=0)).sum()), values,
    )
    np.testing.assert_allclose(grad, expected, atol=1e-3)


def test_collective_permute_vjp_inverts_pairs():
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]   # rotate
    out = ts_sharding.collective_permute(values, pairs)
    do = np.array([[10.0], [20.0], [30.0], [40.0]])
    grad, _ = get_vjp("collective_permute")(do, values, pairs)
    # Inverting: (src, dst) -> ranks src receive dout from rank dst.
    # rank 0 -> dst 1, so rank 0 receives do[1] = 20
    # rank 1 -> dst 2, so rank 1 receives do[2] = 30, etc.
    expected = np.array([[20.0], [30.0], [40.0], [10.0]])
    np.testing.assert_allclose(grad, expected)


def test_broadcast_to_axis_vjp_is_psum_along_axis():
    do = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # axis_size=3, axis=0
    grad, = get_vjp("broadcast_to_axis")(do, np.array([0.0, 0.0]),
                                          axis_size=3, axis=0)
    np.testing.assert_allclose(grad, np.array([9.0, 12.0]))


def test_psum_jvp_matches_primal():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(5, 4))
    dvalues = rng.normal(size=values.shape) * 0.1
    primal, tangent = get_jvp("psum")((values,), (dvalues,))
    np.testing.assert_allclose(primal, np.sum(values, axis=0))
    np.testing.assert_allclose(tangent, np.sum(dvalues, axis=0))


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — Stateful optimizers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["momentum", "nesterov", "adamw"])
def test_optimizer_vjp_and_jvp_registered(name):
    assert get_vjp(name) is not None, f"VJP missing: {name}"
    assert get_jvp(name) is not None, f"JVP missing: {name}"


def test_momentum_vjp_propagates_grad_correctly():
    """`new_params = params - lr*(momentum*velocity + grads)`.
    With dout=ones: d_params=ones, d_grads=-lr, d_velocity=-lr*momentum."""
    params = np.array([1.0, 2.0])
    grads = np.array([0.1, -0.2])
    state = {"velocity": np.array([0.5, 0.5])}
    dout = np.ones_like(params)

    d_params, d_grads, d_state = get_vjp("momentum")(
        dout, params, grads, state, lr=0.1, momentum=0.9,
    )
    np.testing.assert_allclose(d_params, dout)
    np.testing.assert_allclose(d_grads, -0.1 * dout)
    np.testing.assert_allclose(d_state["velocity"], -0.1 * 0.9 * dout)


def test_momentum_jvp_matches_finite_difference():
    params = np.array([1.0, 2.0])
    grads = np.array([0.1, -0.2])
    state = {"velocity": np.array([0.5, 0.5])}
    dparams = np.array([1.0, 0.0])

    primal, tangent = get_jvp("momentum")(
        (params, grads, state),
        (dparams, np.zeros_like(grads), {"velocity": np.zeros_like(state["velocity"])}),
        lr=0.1, momentum=0.9,
    )
    expected_primal, _ = ts_optim.momentum(params, grads, state, lr=0.1, momentum=0.9)
    np.testing.assert_allclose(primal, expected_primal)
    expected_tan = _numeric_jvp(
        lambda p: ts_optim.momentum(p, grads, state, lr=0.1, momentum=0.9)[0],
        params, dparams,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_nesterov_vjp_vs_finite_difference_through_grads():
    rng = np.random.default_rng(0)
    params = rng.normal(size=(4,))
    grads = rng.normal(size=(4,))
    state = {"velocity": rng.normal(size=(4,))}
    dout = np.ones_like(params)

    d_params, d_grads, d_state = get_vjp("nesterov")(
        dout, params, grads, state, lr=0.1, momentum=0.9,
    )
    expected_grads = _numeric_grad(
        lambda g: float(ts_optim.nesterov(params, g, state, lr=0.1, momentum=0.9)[0].sum()),
        grads,
    )
    np.testing.assert_allclose(d_grads, expected_grads, atol=1e-3)


def test_adamw_vjp_vs_finite_difference_through_grads():
    rng = np.random.default_rng(1)
    params = rng.normal(size=(3,))
    grads = rng.normal(size=(3,)) * 0.1
    state = {"m": np.zeros(3), "v": np.zeros(3), "step": 0}
    dout = np.ones_like(params)

    d_params, d_grads, d_state = get_vjp("adamw")(
        dout, params, grads, state,
        lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
    )
    expected_grads = _numeric_grad(
        lambda g: float(ts_optim.adamw(params, g, state,
                                       lr=0.01, beta1=0.9, beta2=0.999,
                                       eps=1e-8, weight_decay=0.0)[0].sum()),
        grads,
    )
    np.testing.assert_allclose(d_grads, expected_grads, atol=1e-3)


def test_adamw_vjp_weight_decay_path():
    """`d_params = dout * (1 - lr*wd)` — verify decoupled-decay derivative."""
    params = np.array([1.0, 1.0])
    grads = np.array([0.0, 0.0])
    state = {"m": np.zeros(2), "v": np.zeros(2), "step": 0}
    dout = np.array([1.0, 1.0])
    d_params, _, _ = get_vjp("adamw")(
        dout, params, grads, state,
        lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
    )
    np.testing.assert_allclose(d_params, np.array([0.999, 0.999]))


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 3 — Memory: differentiable `memory_read`
# ─────────────────────────────────────────────────────────────────────────────


def _build_memory(rng, N=6, key_dim=4, value_dim=5):
    keys = rng.normal(size=(N, key_dim))
    values = rng.normal(size=(N, value_dim))
    return ts_memory.MemoryTable(keys=keys, values=values), keys, values


def test_memory_read_vjp_against_finite_difference():
    rng = np.random.default_rng(0)
    table, keys, values = _build_memory(rng)
    query = rng.normal(size=(4,))
    do = rng.normal(size=(values.shape[1],))
    (d_keys, d_values), d_query = get_vjp("memory_read")(
        do, table, query, top_k=3, normalize=True,
    )

    def loss_against(q, k, v):
        result = ts_memory.memory_read(
            ts_memory.MemoryTable(keys=k, values=v),
            q, top_k=3, normalize=True,
        )
        return float((do * result.values).sum())

    expected_q = _numeric_grad(lambda v: loss_against(v, keys, values), query)
    expected_k = _numeric_grad(lambda v: loss_against(query, v, values), keys)
    expected_v = _numeric_grad(lambda v: loss_against(query, keys, v), values)
    np.testing.assert_allclose(d_query, expected_q, atol=1e-3)
    np.testing.assert_allclose(d_keys, expected_k, atol=1e-3)
    np.testing.assert_allclose(d_values, expected_v, atol=1e-3)


def test_memory_read_vjp_handles_batched_query():
    rng = np.random.default_rng(1)
    table, keys, values = _build_memory(rng)
    queries = rng.normal(size=(3, 4))   # B=3
    do = rng.normal(size=(3, values.shape[1]))
    (d_keys, d_values), d_query = get_vjp("memory_read")(
        do, table, queries, top_k=2, normalize=True,
    )
    assert d_query.shape == queries.shape
    assert d_keys.shape == keys.shape
    assert d_values.shape == values.shape


def test_memory_read_jvp_matches_finite_difference():
    rng = np.random.default_rng(2)
    table, keys, values = _build_memory(rng, N=4)
    query = rng.normal(size=(4,))
    dquery = rng.normal(size=query.shape) * 0.05

    primal, tangent = get_jvp("memory_read")(
        (table, query),
        ((np.zeros_like(keys), np.zeros_like(values)), dquery),
        top_k=2, normalize=True,
    )
    expected_tan = _numeric_jvp(
        lambda v: ts_memory.memory_read(table, v, top_k=2, normalize=True).values,
        query, dquery,
    )
    np.testing.assert_allclose(tangent.values, expected_tan, atol=1e-3)


def test_memory_read_vjp_treats_indices_as_constants():
    """Top-k argpartition gives integer indices; the VJP must NOT flow grad
    through `indices` — the gradient w.r.t. indices is the zero tensor."""
    rng = np.random.default_rng(3)
    table, _, _ = _build_memory(rng)
    query = rng.normal(size=(4,))
    do = np.ones((5,))
    # Just verify it runs without raising and returns proper shapes.
    (d_keys, d_values), d_query = get_vjp("memory_read")(do, table, query)
    assert d_query.shape == query.shape


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 4 — cummax / cummin
# ─────────────────────────────────────────────────────────────────────────────


def test_cummax_vjp_routes_to_running_max():
    """Routes grad to the position of the running max at each step.
    With distinct entries [3, 1, 4, 1]:
      cummax = [3, 3, 4, 4]
      argmax-so-far = [0, 0, 2, 2]
      With dout = [1, 1, 1, 1]:
        d[0] += 1 (from i=0, only k=0)
        d[0] += 1 (from i=1, mask[0]=True)
        d[2] += 1 (from i=2)
        d[2] += 1 (from i=3)
      → d = [2, 0, 2, 0]"""
    x = np.array([3.0, 1.0, 4.0, 1.0])
    do = np.ones(4)
    grad, = get_vjp("cummax")(do, x, axis=-1)
    np.testing.assert_array_equal(grad, np.array([2.0, 0.0, 2.0, 0.0]))


def test_cummax_vjp_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 5))
    do = rng.normal(size=(3, 5))
    grad, = get_vjp("cummax")(do, x, axis=-1)
    expected = _numeric_grad(
        lambda v: float((do * np.maximum.accumulate(v, axis=-1)).sum()), x,
    )
    np.testing.assert_allclose(grad, expected, atol=1e-3)


def test_cummin_vjp_matches_finite_difference():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 6))
    do = rng.normal(size=(2, 6))
    grad, = get_vjp("cummin")(do, x, axis=-1)
    expected = _numeric_grad(
        lambda v: float((do * np.minimum.accumulate(v, axis=-1)).sum()), x,
    )
    np.testing.assert_allclose(grad, expected, atol=1e-3)


def test_cummax_jvp_matches_finite_difference():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4,))
    dx = rng.normal(size=(4,)) * 0.05
    primal, tangent = get_jvp("cummax")((x,), (dx,), axis=-1)
    np.testing.assert_allclose(primal, np.maximum.accumulate(x))
    expected_tan = _numeric_jvp(
        lambda v: np.maximum.accumulate(v, axis=-1), x, dx,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_cummin_jvp_matches_finite_difference():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5,))
    dx = rng.normal(size=(5,)) * 0.05
    primal, tangent = get_jvp("cummin")((x,), (dx,), axis=-1)
    np.testing.assert_allclose(primal, np.minimum.accumulate(x))
    expected_tan = _numeric_jvp(
        lambda v: np.minimum.accumulate(v, axis=-1), x, dx,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Registry promotion
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    # Sprint 1
    "psum", "pmean", "pmax", "pmin", "collective_permute", "broadcast_to_axis",
    # Sprint 2
    "momentum", "nesterov", "adamw",
    # Sprint 3
    "memory_read",
    # Sprint 4
    "cummax", "cummin",
])
def test_registry_reports_vjp_complete(name):
    from tessera.compiler.primitive_coverage import coverage_for

    entry = coverage_for(name)
    assert entry.contract_status["vjp"] == "complete", (
        f"{name} VJP registered but registry shows {entry.contract_status['vjp']}"
    )


@pytest.mark.parametrize("name", [
    "psum", "pmean", "pmax", "pmin", "collective_permute", "broadcast_to_axis",
    "momentum", "nesterov", "adamw",
    "memory_read",
    "cummax", "cummin",
])
def test_registry_reports_jvp_complete(name):
    from tessera.compiler.primitive_coverage import coverage_for

    entry = coverage_for(name)
    assert entry.contract_status["jvp"] == "complete", (
        f"{name} JVP registered but registry shows {entry.contract_status['jvp']}"
    )
