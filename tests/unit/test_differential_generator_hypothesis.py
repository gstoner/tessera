"""Compiler-testing program #2 (hypothesis variant) — property-based differential.

Same contract as ``test_differential_generator.py`` (eager numpy oracle vs the
real trace → GraphFn / ``execute_traced`` Metal path over the executable lane),
but driven by **hypothesis** instead of stdlib fixed seeds. The payoff is
**automatic shrinking**: when a generated program trips a Metal miscompile,
hypothesis reduces it to the *minimal* failing program — typically the 1–2 ops
that actually diverge — instead of a 5-op chain you'd have to bisect by hand.

CI without hypothesis installed skips this whole module (``importorskip``); the
stdlib harness still provides dependency-free coverage everywhere. The shared
grammar + oracle live in ``_diff_lane`` so the two harnesses can't drift.
"""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import tessera as ts  # noqa: E402
from tessera.compiler.trace import run_traced, trace  # noqa: E402

from _diff_lane import (  # noqa: E402
    N, _BINARY, _EXPECT, _UNARY, apply_op, gpu, inputs, stable, straightline_fn,
)

# Apple-GPU dispatch dominates wall-time and isn't hypothesis's concern; give it
# room and silence the "too slow" health check (each example is a real Metal run).
_SETTINGS = settings(
    max_examples=40, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ── program strategy ──────────────────────────────────────────────────────── #
@st.composite
def _programs(draw, max_ops=5):
    """Draw a ``[(op, idxs), ...]`` chain over a growing value pool [x, w, ...].

    Operand indices are drawn against the *current* pool size, so every program
    is well-formed by construction — hypothesis shrinks toward shorter chains and
    lower indices (i.e. toward reusing x/w), which is exactly the minimal repro.
    """
    n_ops = draw(st.integers(min_value=1, max_value=max_ops))
    prog = []
    pool = 2
    for _ in range(n_ops):
        if draw(st.booleans()):
            op = draw(st.sampled_from(_BINARY))
            idxs = (draw(st.integers(0, pool - 1)), draw(st.integers(0, pool - 1)))
        else:
            op = draw(st.sampled_from(_UNARY))
            idxs = (draw(st.integers(0, pool - 1)),)
        prog.append((op, idxs))
        pool += 1
    return prog


_SEED_ST = st.integers(min_value=0, max_value=2**31 - 1)


# ── runtime-free: every drawn program traces, op-names round-trip ──────────── #
@_SETTINGS
@given(prog=_programs())
def test_drawn_program_traces(prog):
    fn = straightline_fn(prog)
    tb = trace(fn, np.zeros((N, N), np.float32), np.zeros((N, N), np.float32))
    got = [op.op_name for op in tb.body]
    assert got == [_EXPECT[op] for op, _ in prog]


# ── numerical differential — straight-line (oracle vs Metal) ───────────────── #
@gpu
@_SETTINGS
@given(prog=_programs(), seed=_SEED_ST)
def test_straightline_matches_oracle(prog, seed):
    nrng = np.random.default_rng(seed)
    fn = straightline_fn(prog)
    x, w = inputs(nrng)
    oracle = stable(fn, x, w)
    if oracle is None:
        return                              # unstable generated program — skip
    cand = np.asarray(run_traced(fn, x, w), dtype=np.float32)
    np.testing.assert_allclose(
        cand, oracle, rtol=2e-3, atol=2e-3,
        err_msg=f"MISCOMPILE prog={prog} seed={seed}")


# ── numerical differential — fori_loop (fused run_graph_loop path) ─────────── #
@gpu
@_SETTINGS
@given(body=_programs(max_ops=3), trip=st.integers(2, 4), seed=_SEED_ST)
def test_fori_loop_matches_oracle(body, trip, seed):
    nrng = np.random.default_rng(seed)

    def fn(x, w, _bp=body, _t=trip):
        def step(i, c):
            vals = [c, w]
            for op, idxs in _bp:
                vals.append(apply_op(op, vals, idxs))
            return vals[-1]
        return ts.control.fori_loop(0, _t, step, x)

    x, w = inputs(nrng)
    oracle = stable(fn, x, w)
    if oracle is None:
        return
    cand = np.asarray(run_traced(fn, x, w), dtype=np.float32)
    np.testing.assert_allclose(
        cand, oracle, rtol=3e-3, atol=3e-3,
        err_msg=f"LOOP MISCOMPILE body={body} trip={trip} seed={seed}")


# ── numerical differential — cond (fused run_graph_cond path) ──────────────── #
# pred must be a traced input (a Tracer), so `flag` is the first arg (0-d f32).
@gpu
@_SETTINGS
@given(then_p=_programs(max_ops=3), else_p=_programs(max_ops=3),
       flag=st.sampled_from([0, 1]), seed=_SEED_ST)
def test_cond_matches_oracle(then_p, else_p, flag, seed):
    nrng = np.random.default_rng(seed)
    flag_arr = np.array(float(flag), dtype=np.float32)

    def fn(flag, x, w, _tp=then_p, _ep=else_p):
        def branch(prog):
            vals = [x, w]
            for op, idxs in prog:
                vals.append(apply_op(op, vals, idxs))
            return vals[-1]
        return ts.control.cond(flag, lambda: branch(_tp), lambda: branch(_ep))

    x, w = inputs(nrng)
    oracle = stable(lambda x, w: fn(flag_arr, x, w), x, w)
    if oracle is None:
        return
    cand = np.asarray(run_traced(fn, flag_arr, x, w), dtype=np.float32)
    np.testing.assert_allclose(
        cand, oracle, rtol=3e-3, atol=3e-3,
        err_msg=f"COND MISCOMPILE flag={flag} seed={seed}")
