"""Compiler-testing program #2 — differential program generator.

Synthesize random well-typed programs over the **executable lane** (the
`tessera.ops` elementwise/matmul/norm subset that `_gpu_straightline_op` /
`run_graph_*` can actually run, plus `tessera.control.fori_loop` / `cond`) and
diff two evaluators of the *same* function:

  * **oracle**    — eager ``fn(*arrays)``: with no tracer active,
    ``tessera.ops`` runs the numpy reference and ``tessera.control.*`` runs the
    Python loop eagerly.
  * **candidate** — ``run_traced(fn, *arrays, target="apple_gpu")``: the real
    trace → GraphFn / ``execute_traced`` (fused ``run_graph_*``) Metal path.

A mismatch is a **miscompile** in the production lane — exactly the bug class a
hand-written test tail (item #1's ~123 ``needs_direct_test`` ops) misses.
Generation is deterministic (stdlib ``random`` + fixed seeds); square N×N
tensors so every op composes without a shape solver.

The trace-shape / op-name properties are pure and run everywhere; the numerical
differential needs the Apple GPU runtime and skips otherwise.
"""

import random

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.trace import run_traced, trace

from _diff_lane import (  # noqa: E402  (sibling helper module, see _diff_lane.py)
    N, _BINARY, _EXPECT, _UNARY, apply_op, gpu, inputs, ldt_cases,
    numeric_cases, stable, straightline_fn,
)

# Back-compat local aliases (kept so the test bodies read as before).
_apply, _straightline_fn, _inputs, _stable = (
    apply_op, straightline_fn, inputs, stable)

_SEEDS = [0, 1, 2, 3, 7, 13, 42, 123, 999, 31337]


def _gen_prog(rng, n_ops):
    """A list of (op, idxs) instructions over a growing value pool [x, w, ...]."""
    prog = []
    pool = 2                       # x, w
    for _ in range(n_ops):
        if rng.random() < 0.5:
            op = rng.choice(_BINARY)
            idxs = (rng.randrange(pool), rng.randrange(pool))
        else:
            op = rng.choice(_UNARY)
            idxs = (rng.randrange(pool),)
        prog.append((op, idxs))
        pool += 1
    return prog


# ── runtime-free: every generated program traces, op-names round-trip ──────── #
@pytest.mark.parametrize("seed", _SEEDS)
def test_generated_program_traces(seed):
    rng = random.Random(seed)
    for _ in range(12):
        prog = _gen_prog(rng, rng.randint(1, 6))
        fn = _straightline_fn(prog)
        tb = trace(fn, np.zeros((N, N), np.float32), np.zeros((N, N), np.float32))
        got = [op.op_name for op in tb.body]
        assert got == [_EXPECT[op] for op, _ in prog], (
            f"trace op-name mismatch seed={seed}: {got}")


def test_executable_vocab_is_a_subset_of_the_runner():
    """Guard the generator can only emit ops the apple_gpu executor handles —
    a drift in `_gpu_straightline_op` that drops an op surfaces here."""
    from tessera.compiler import trace as T
    src = T._gpu_straightline_op.__code__.co_consts
    flat = " ".join(str(c) for c in src)
    for op in _UNARY + _BINARY:
        assert f"tessera.{op}" in flat or op in flat, (
            f"generator emits {op!r} but the runner doesn't handle it")


# ── numerical differential — straight-line (oracle vs Metal) ───────────────── #
@gpu
@pytest.mark.parametrize("seed", _SEEDS)
def test_straightline_matches_oracle(seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    checked = 0
    for _ in range(20):
        prog = _gen_prog(rng, rng.randint(1, 5))
        fn = _straightline_fn(prog)
        x, w = _inputs(nrng)
        oracle = _stable(fn, x, w)
        if oracle is None:
            continue
        cand = np.asarray(run_traced(fn, x, w), dtype=np.float32)
        np.testing.assert_allclose(
            cand, oracle, rtol=2e-3, atol=2e-3,
            err_msg=f"MISCOMPILE seed={seed} prog={prog}")
        checked += 1
    assert checked > 0, f"seed={seed}: every generated program was unstable"


# ── numerical differential — fori_loop (fused run_graph_loop path) ─────────── #
@gpu
@pytest.mark.parametrize("seed", _SEEDS)
def test_fori_loop_matches_oracle(seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed + 100)
    checked = 0
    for _ in range(12):
        body_prog = _gen_prog(rng, rng.randint(1, 3))
        trip = rng.randint(2, 4)

        def fn(x, w, _bp=body_prog, _t=trip):
            def body(i, c):
                vals = [c, w]
                for op, idxs in _bp:
                    vals.append(_apply(op, vals, idxs))
                return vals[-1]
            return ts.control.fori_loop(0, _t, body, x)

        x, w = _inputs(nrng)
        oracle = _stable(fn, x, w)
        if oracle is None:
            continue
        cand = np.asarray(run_traced(fn, x, w), dtype=np.float32)
        np.testing.assert_allclose(
            cand, oracle, rtol=3e-3, atol=3e-3,
            err_msg=f"LOOP MISCOMPILE seed={seed} body={body_prog} trip={trip}")
        checked += 1
    assert checked > 0, f"seed={seed}: every generated loop was unstable"


# ── numerical differential — cond (fused run_graph_cond path) ──────────────── #
# The pred must be a *traced input* (a Tracer) — a concrete flag can't select a
# fused branch — so `flag` is the first function arg, passed as a 0-d f32 array.
@gpu
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("flag", [0, 1])
def test_cond_matches_oracle(seed, flag):
    rng = random.Random(seed * 7 + flag)
    nrng = np.random.default_rng(seed + 200 + flag)
    flag_arr = np.array(float(flag), dtype=np.float32)
    checked = 0
    for _ in range(8):
        then_p = _gen_prog(rng, rng.randint(1, 3))
        else_p = _gen_prog(rng, rng.randint(1, 3))

        def fn(flag, x, w, _tp=then_p, _ep=else_p):
            def branch(prog):
                vals = [x, w]
                for op, idxs in prog:
                    vals.append(_apply(op, vals, idxs))
                return vals[-1]
            return ts.control.cond(flag, lambda: branch(_tp), lambda: branch(_ep))

        x, w = _inputs(nrng)
        oracle = _stable(lambda x, w: fn(flag_arr, x, w), x, w)
        if oracle is None:
            continue
        cand = np.asarray(run_traced(fn, flag_arr, x, w), dtype=np.float32)
        np.testing.assert_allclose(
            cand, oracle, rtol=3e-3, atol=3e-3,
            err_msg=f"COND MISCOMPILE seed={seed} flag={flag}")
        checked += 1
    assert checked > 0, f"seed={seed} flag={flag}: all branches unstable"


# ── LDT-op differential — @jit(apple_gpu) vs an independent numpy oracle ────── #
# count_nonzero / popcount / asymmetric_bce / masked_categorical aren't
# shape-preserving N×N, so they run as their own cases (see _diff_lane.ldt_cases)
# rather than in the op-chain grammar above. Integer/index ops compare exactly;
# the loss compares at f32 tolerance.
@gpu
@pytest.mark.parametrize("seed", _SEEDS)
def test_ldt_ops_match_oracle(seed):
    nrng = np.random.default_rng(seed + 500)
    for label, fn, args, oracle, exact in ldt_cases(nrng):
        cand = np.asarray(fn(*args))
        if exact:
            np.testing.assert_array_equal(
                cand, oracle, err_msg=f"LDT MISCOMPILE {label} seed={seed}")
        else:
            np.testing.assert_allclose(
                cand, np.asarray(oracle), rtol=2e-3, atol=2e-3,
                err_msg=f"LDT MISCOMPILE {label} seed={seed}")


# ── numeric elementwise / reduction differential — @jit(apple_gpu) vs an
#    independent numpy oracle (item #4 differential-generator extension) ─────── #
@gpu
@pytest.mark.parametrize("seed", _SEEDS)
def test_numeric_ops_match_oracle(seed):
    nrng = np.random.default_rng(seed + 900)
    for label, fn, args, oracle, exact in numeric_cases(nrng):
        cand = np.asarray(fn(*args), dtype=np.float64)
        np.testing.assert_allclose(
            cand, np.asarray(oracle, dtype=np.float64), rtol=1e-4, atol=1e-4,
            err_msg=f"NUMERIC MISCOMPILE {label} seed={seed}")
