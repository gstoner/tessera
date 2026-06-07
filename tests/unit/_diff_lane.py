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
