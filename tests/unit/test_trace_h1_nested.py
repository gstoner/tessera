"""Phase-H H1 — nested control flow via host-orchestration.

`run_graph_*` bodies are flat op-lists, so a control op INSIDE a region body can't
be serialized to a single fused dispatch. H1 makes `execute_traced` recursive: a
region whose body/branches are flat keeps the fused `run_graph_*` path; a region
containing a nested control op is host-orchestrated (the outer construct runs as a
Python loop threading the concrete carry, recursively executing the body — so the
inner construct still fuses). This unblocks loop-in-loop / cond-in-loop /
loop-in-cond, which previously raised.

trace-shape checks are pure; execution cases need the Apple GPU runtime.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler.trace import run_traced, trace
from tessera.compiler.trace import _region_flat

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def _silu(z):
    return z / (1.0 + np.exp(-z))


# --- region-flatness classifier (no runtime) -------------------------------- #
def test_region_flat_classifier():
    def flat(x, w):
        return ts.control.fori_loop(
            0, 2, lambda i, c: ts.ops.silu(ts.ops.matmul(c, w)), x)

    def nested(x, w):
        return ts.control.fori_loop(
            0, 2, lambda i, c: ts.control.fori_loop(
                0, 2, lambda j, cc: ts.ops.matmul(cc, w), c), x)

    fb = trace(flat, np.zeros((1, 8), np.float32), np.zeros((8, 8), np.float32))
    nb = trace(nested, np.zeros((1, 8), np.float32), np.zeros((8, 8), np.float32))
    assert _region_flat(fb.body[0].kwargs["_body"]) is True
    assert _region_flat(nb.body[0].kwargs["_body"]) is False  # inner control_for


# --- nested execution (apple_gpu) ------------------------------------------- #
@gpu
def test_loop_in_loop_matches_numpy():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(x, w):
        return ts.control.fori_loop(
            0, 3, lambda i, c: ts.control.fori_loop(
                0, 2, lambda j, cc: ts.ops.silu(ts.ops.matmul(cc, w)), c), x)

    out = run_traced(f, x, w)
    ref = x.copy()
    for _ in range(3):
        for _ in range(2):
            ref = _silu(ref @ w)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_cond_in_loop_matches_numpy(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 1)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(flag, x, w):
        return ts.control.fori_loop(
            0, 3, lambda i, c: ts.control.cond(
                flag,
                lambda: ts.ops.silu(ts.ops.matmul(c, w)),
                lambda: ts.ops.relu(ts.ops.matmul(c, w))), x)

    out = run_traced(f, np.array([flagv], np.float32), x, w)
    ref = x.copy()
    for _ in range(3):
        z = ref @ w
        ref = _silu(z) if flagv > 0 else np.maximum(z, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_loop_in_cond_matches_numpy(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 7)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(flag, x, w):
        return ts.control.cond(
            flag,
            lambda: ts.control.fori_loop(
                0, 3, lambda i, c: ts.ops.silu(ts.ops.matmul(c, w)), x),
            lambda: ts.ops.relu(x))

    out = run_traced(f, np.array([flagv], np.float32), x, w)
    if flagv > 0:
        ref = x.copy()
        for _ in range(3):
            ref = _silu(ref @ w)
    else:
        ref = np.maximum(x, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_flat_loop_still_fuses_and_matches():
    """A flat loop keeps the single fused dispatch (correctness sentinel)."""
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(x, w):
        return ts.control.fori_loop(
            0, 4, lambda i, c: ts.ops.silu(ts.ops.matmul(c, w)), x)

    out = run_traced(f, x, w)
    ref = x.copy()
    for _ in range(4):
        ref = _silu(ref @ w)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
