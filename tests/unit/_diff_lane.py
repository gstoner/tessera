"""Shared executable-lane grammar + oracle for the differential harnesses.

Not a test module (``python_files = "test_*.py"`` won't collect it). Imported by
both ``test_differential_generator.py`` (stdlib ``random`` + fixed seeds) and
``test_differential_generator_hypothesis.py`` (hypothesis ``@given`` + shrinking)
so the two harnesses share one program grammar and one numpy oracle — no drift.

The "executable lane" is the ``tessera.ops`` subset that ``_gpu_straightline_op``
/ ``run_graph_*`` can actually run, over square N×N f32 tensors so every op
composes without a shape solver.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb

N = 8
# Executable-lane vocab (must be a subset of what `_gpu_straightline_op` runs).
_UNARY = ["silu", "relu", "sigmoid", "tanh", "gelu", "softmax",
          "rmsnorm", "layer_norm"]
_BINARY = ["matmul", "add", "sub", "mul"]   # div excluded (oracle stability)
_EXPECT = {op: f"tessera.{op}" for op in _UNARY + _BINARY}

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(
    not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def apply_op(op, vals, idxs):
    """Apply one op given the value pool + chosen operand indices."""
    if op in _BINARY:
        return getattr(ts.ops, op)(vals[idxs[0]], vals[idxs[1]])
    return getattr(ts.ops, op)(vals[idxs[0]])


def straightline_fn(prog):
    """Compile a ``[(op, idxs), ...]`` instruction list into ``fn(x, w)``."""
    def fn(x, w):
        vals = [x, w]
        for op, idxs in prog:
            vals.append(apply_op(op, vals, idxs))
        return vals[-1]
    return fn


def inputs(nrng):
    x = (nrng.standard_normal((N, N)) / 8).astype(np.float32)
    w = (nrng.standard_normal((N, N)) / 3).astype(np.float32)
    return x, w


def stable(fn, *arrays):
    """Eager oracle output; ``None`` if not finite / too large — so a numerically
    unstable *generated* program can't masquerade as a miscompile."""
    try:
        out = np.asarray(fn(*arrays), dtype=np.float32)
    except Exception:                       # noqa: BLE001
        return None
    if not np.isfinite(out).all() or np.max(np.abs(out)) > 1e4:
        return None
    return out


# ── LDT / lattice-reasoning op differential cases ──────────────────────────── #
# These ops aren't shape-preserving N×N (count_nonzero reduces, popcount needs
# ints, asymmetric_bce → scalar, masked_categorical → indices), so they get
# their own generators. Each case is (label, jitted_fn, args, oracle, exact):
# the candidate is the op run on the **@jit(apple_gpu)** path (numpy-fallback
# chain for non-Metal ops), diffed against an **independent** numpy oracle.

def _agpu(fn):
    import tessera as ts
    return ts.jit(target="apple_gpu")(fn)


def ldt_cases(nrng):
    """Build the four LDT differential cases for one RNG draw."""
    import tessera as ts
    import tessera.losses as L

    cases = []

    # count_nonzero — candidate cardinality along the last axis.
    x = (nrng.standard_normal((N, N)) * (nrng.random((N, N)) > 0.4)).astype(np.float32)
    cases.append((
        "count_nonzero", _agpu(lambda x: ts.ops.count_nonzero(x, axis=-1)),
        (x,), np.count_nonzero(x, axis=-1), True))

    # popcount — independent oracle via Python int.bit_count (not the impl path).
    b = nrng.integers(0, 256, size=(N, N)).astype(np.int64)
    oracle_pc = np.vectorize(lambda v: int(v).bit_count())(b).astype(np.int64)
    cases.append((
        "popcount", _agpu(lambda b: ts.ops.popcount(b)),
        (b,), oracle_pc, True))

    # asymmetric_bce — weighted logits loss; weights are kwargs (non-tensor).
    z = nrng.standard_normal((N, N)).astype(np.float32)
    t = (nrng.random((N, N)) < 0.5).astype(np.float32)
    cases.append((
        "asymmetric_bce",
        _agpu(lambda z, t: ts.ops.asymmetric_bce(z, t, pos_weight=2.0, neg_weight=0.5)),
        (z, t), np.asarray(L.asymmetric_bce(z, t, 2.0, 0.5)), False))

    # masked_categorical — deterministic greedy decision over a candidate mask.
    logits = nrng.standard_normal((N, N)).astype(np.float32)
    mask = (nrng.random((N, N)) < 0.6)
    mask[:, 0] = True                       # guarantee ≥1 candidate per row
    mask = mask.astype(np.int32)
    cases.append((
        "masked_categorical",
        _agpu(lambda lo, m: ts.ops.masked_categorical(lo, m)),
        (logits, mask), np.asarray(ts.ops.masked_categorical(logits, mask)), True))

    return cases


# ── numeric elementwise + reduction differential cases ─────────────────────── #
# Audit 2026-06-10 (item #4) — extends the differential generator across the
# needs-direct-test tail. These run on the **@jit(apple_gpu)** dispatch envelope
# (beyond the 13-op straight-line tracer lane) and diff against an *independent*
# numpy oracle — a true impl-vs-reference check (catches implementation bugs,
# not just trace miscompiles). Each case is (label, jitted_fn, args, oracle,
# exact); all compare at f32 tolerance (exact=False).

def numeric_cases(nrng):
    """Elementwise (unary/binary) + last-axis reduction ops, each @jit on
    apple_gpu vs an independent numpy oracle. log/sqrt/rsqrt use a strictly
    positive input so the reference is well-defined."""
    import tessera as ts

    x = (nrng.standard_normal((N, N)) / 4).astype(np.float32)
    y = (nrng.standard_normal((N, N)) / 4).astype(np.float32)
    xp = (np.abs(nrng.standard_normal((N, N))) + 0.5).astype(np.float32)  # > 0

    return [
        # shape-preserving elementwise unary (defined on all reals)
        ("exp",      _agpu(lambda a: ts.ops.exp(a)),      (x,),  np.exp(x),           False),
        ("abs",      _agpu(lambda a: ts.ops.abs(a)),      (x,),  np.abs(x),           False),
        ("softplus", _agpu(lambda a: ts.ops.softplus(a)), (x,),  np.log1p(np.exp(x)), False),
        # positive-domain unary
        ("log",      _agpu(lambda a: ts.ops.log(a)),      (xp,), np.log(xp),          False),
        ("sqrt",     _agpu(lambda a: ts.ops.sqrt(a)),     (xp,), np.sqrt(xp),         False),
        ("rsqrt",    _agpu(lambda a: ts.ops.rsqrt(a)),    (xp,), 1.0 / np.sqrt(xp),   False),
        # elementwise binary
        ("maximum",  _agpu(lambda a, b: ts.ops.maximum(a, b)), (x, y), np.maximum(x, y), False),
        ("minimum",  _agpu(lambda a, b: ts.ops.minimum(a, b)), (x, y), np.minimum(x, y), False),
        # last-axis reductions (not shape-preserving)
        ("sum",      _agpu(lambda a: ts.ops.sum(a, axis=-1)),  (x,), x.sum(-1),       False),
        ("mean",     _agpu(lambda a: ts.ops.mean(a, axis=-1)), (x,), x.mean(-1),      False),
        ("amax",     _agpu(lambda a: ts.ops.amax(a, axis=-1)), (x,), x.max(-1),       False),
        ("amin",     _agpu(lambda a: ts.ops.amin(a, axis=-1)), (x,), x.min(-1),       False),
    ]
