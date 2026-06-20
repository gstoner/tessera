"""Phase E1c — corroborate conformance ``complete`` cells with the
execution-derived Evaluator (docs/audit/compiler/EVALUATOR_PLAN.md).

The conformance matrix grants ``numerical_check = complete`` when a
**hand-declared** ``execute_compare_fixture`` exists and looks like an
execute-compare (the P0 structural gate). This module adds an **independent,
generative** authority: for each complete cell it builds a program from the
cell's component ops, runs it through ``evaluator.evaluate`` on the real
backend, and requires the run to reach ``HARDWARE_VERIFIED`` (native execution +
oracle match). It is the "derive validates declare" bridge — it does not touch
the drift-gated dashboard or its counts; it cross-checks that the matrix's
``complete`` claims actually reproduce on hardware.

A divergence here is a real finding: a cell the registry calls complete that the
generic Evaluator cannot reproduce natively. Coverage is asserted both ways
(every executable complete cell has a builder; every builder maps to a real
complete cell), so the corroboration set cannot silently shrink.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import tessera as ts

from tessera.compiler.evaluator import BackendVerdict, Rung, evaluate

# Backends that genuinely execute here (Apple Silicon). NVIDIA/ROCm corroborate
# through the same path once a launcher registers (G7) — no change needed here.
EXECUTABLE_TARGETS: frozenset[str] = frozenset({"apple_gpu", "apple_cpu"})


# ── module-level base programs (source-inspectable for @jit) ─────────────────

def _p_matmul(a, b):
    return ts.ops.matmul(a, b)


def _p_softmax(a):
    return ts.ops.softmax(a, axis=-1)


def _p_matmul_softmax(a, b):
    return ts.ops.softmax(ts.ops.matmul(a, b), axis=-1)


def _p_flash_attn(q, k, v):
    return ts.ops.flash_attn(q, k, v)


_BASE_FN: dict[str, Callable[..., Any]] = {
    "matmul": _p_matmul,
    "softmax": _p_softmax,
    "matmul_softmax": _p_matmul_softmax,
    "flash_attn": _p_flash_attn,
}

# (jitted-fn cache keyed by (op, target) so we compile each once.)
# Bounded LRU so it can't grow without limit if used generatively (the seed op
# set is small, so this is a defensive cap, not a hot-path concern).
_JIT_CACHE_MAX = 256
_JIT_CACHE: "OrderedDict[tuple[str, str], Any]" = OrderedDict()


def _jitted(op: str, target: str) -> Any:
    key = (op, target)
    fn = _JIT_CACHE.get(key)
    if fn is None:
        fn = ts.jit(target=target)(_BASE_FN[op])
        _JIT_CACHE[key] = fn
        if len(_JIT_CACHE) > _JIT_CACHE_MAX:
            _JIT_CACHE.popitem(last=False)  # evict oldest
    else:
        _JIT_CACHE.move_to_end(key)  # LRU touch
    return fn


def _np_softmax(x: Any, axis: int = -1) -> Any:
    import numpy as np

    m = x.max(axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis, keepdims=True)


def _build(op: str, rng: Any) -> tuple[tuple[Any, ...], Any, bool, dict[str, float]]:
    """Return ``(args, oracle, exact, tol)`` for one corroboration program."""
    import numpy as np

    if op == "matmul":
        a = rng.standard_normal((16, 16)).astype(np.float32)
        b = rng.standard_normal((16, 16)).astype(np.float32)
        return (a, b), a @ b, False, {"rtol": 2e-3, "atol": 1e-4}
    if op == "softmax":
        a = rng.standard_normal((16, 16)).astype(np.float32)
        return (a,), _np_softmax(a, -1), False, {"rtol": 2e-3, "atol": 1e-4}
    if op == "matmul_softmax":
        a = rng.standard_normal((16, 16)).astype(np.float32)
        b = rng.standard_normal((16, 16)).astype(np.float32)
        return (a, b), _np_softmax(a @ b, -1), False, {"rtol": 3e-3, "atol": 1e-4}
    if op == "flash_attn":
        q = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
        k = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
        v = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
        d = q.shape[-1]
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(d)
        out = np.einsum("bhqk,bhkd->bhqd", _np_softmax(scores, -1), v)
        return (q, k, v), out, False, {"rtol": 5e-3, "atol": 1e-3}
    raise KeyError(f"no corroboration program builder for op {op!r}")


def has_builder(op: str) -> bool:
    return op in _BASE_FN


def corroborate(op: str, target: str, rng: Any) -> BackendVerdict:
    """Run ``op`` on ``target`` through the Evaluator and return its verdict."""
    args, oracle, exact, tol = _build(op, rng)
    return evaluate(target, _jitted(op, target), args, oracle, exact=exact, **tol)


def complete_cells() -> list[tuple[str, str]]:
    """The ``(op, target)`` cells the conformance matrix marks ``complete``."""
    from tessera.compiler import conformance_matrix as C

    return [(c.op, c.target) for c in C.build_matrix() if c.overall == "complete"]


def corroboratable_complete_cells() -> list[tuple[str, str]]:
    """Complete cells on an executable backend that have a program builder."""
    return [
        (op, t) for (op, t) in complete_cells()
        if t in EXECUTABLE_TARGETS and has_builder(op)
    ]


def corroborate_complete(rng: Any) -> list[tuple[str, str, BackendVerdict]]:
    """Corroborate every executable complete cell; returns (op, target, verdict)."""
    out: list[tuple[str, str, BackendVerdict]] = []
    for op, target in corroboratable_complete_cells():
        out.append((op, target, corroborate(op, target, rng)))
    return out


def uncovered_complete_cells() -> list[tuple[str, str]]:
    """Executable complete cells with NO Evaluator builder — the corroboration
    gap. Should stay empty: a complete cell the Evaluator can't reproduce is a
    registry-vs-reality risk."""
    return [
        (op, t) for (op, t) in complete_cells()
        if t in EXECUTABLE_TARGETS and not has_builder(op)
    ]


__all__ = [
    "EXECUTABLE_TARGETS",
    "Rung",
    "corroborate",
    "corroborate_complete",
    "corroboratable_complete_cells",
    "complete_cells",
    "uncovered_complete_cells",
    "has_builder",
]
