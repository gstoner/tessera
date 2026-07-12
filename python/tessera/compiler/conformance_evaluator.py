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

from tessera.compiler.evaluator import (
    BackendVerdict,
    Rung,
    evaluate,
    kv_cache_read_native_equivalence,
    verdict_for,
)

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


def _p_matmul_relu(a, b):
    return ts.ops.relu(ts.ops.matmul(a, b))


def _p_flash_attn(q, k, v):
    return ts.ops.flash_attn(q, k, v)


def _p_conv2d(x, w):
    return ts.ops.conv2d(x, w, stride=1, padding=0, layout="nhwc")


def _p_kv_cache_read(cache, start, end):
    return ts.ops.kv_cache_read(cache, start, end)


_BASE_FN: dict[str, Callable[..., Any]] = {
    "matmul": _p_matmul,
    "softmax": _p_softmax,
    "matmul_softmax": _p_matmul_softmax,
    "matmul_relu": _p_matmul_relu,
    "flash_attn": _p_flash_attn,
    "conv2d": _p_conv2d,
    "kv_cache_read": _p_kv_cache_read,
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
    if op == "matmul_relu":
        a = rng.standard_normal((16, 16)).astype(np.float32)
        b = rng.standard_normal((16, 16)).astype(np.float32)
        return (a, b), np.maximum(a @ b, 0.0), False, {
            "rtol": 3e-3,
            "atol": 1e-4,
        }
    if op == "flash_attn":
        q = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
        k = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
        v = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
        d = q.shape[-1]
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(d)
        out = np.einsum("bhqk,bhkd->bhqd", _np_softmax(scores, -1), v)
        return (q, k, v), out, False, {"rtol": 5e-3, "atol": 1e-3}
    if op == "conv2d":
        # NHWC source, HWIO weights, stride 1 / no pad (matches _p_conv2d).
        x = rng.standard_normal((1, 8, 8, 3)).astype(np.float32)
        w = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)
        oh, ow = x.shape[1] - 2, x.shape[2] - 2
        out = np.zeros((1, oh, ow, w.shape[3]), np.float32)
        for i in range(oh):
            for j in range(ow):
                patch = x[0, i:i + 3, j:j + 3, :]
                for co in range(w.shape[3]):
                    out[0, i, j, co] = np.sum(patch * w[:, :, :, co])
        return (x, w), out, False, {"rtol": 3e-3, "atol": 1e-3}
    raise KeyError(f"no corroboration program builder for op {op!r}")


def has_builder(op: str) -> bool:
    return op in _BASE_FN


def _corroborate_kv_cache_read(target: str, rng: Any) -> BackendVerdict:
    """Corroborate the stateful cache accessor without forcing it through the
    pure-tensor JIT builder registry."""
    import numpy as np

    from tessera.cache import KVCacheHandle

    cache = KVCacheHandle(num_heads=2, head_dim=4, max_seq=16, page_size=4)
    keys = rng.standard_normal((8, 2, 4)).astype(np.float32)
    values = rng.standard_normal((8, 2, 4)).astype(np.float32)
    cache.append(keys, values)

    if target == "apple_gpu":
        cross = kv_cache_read_native_equivalence(cache, 2, 7)
        return verdict_for(
            target,
            "metal_runtime" if cross.relation == "equivalent" else "reference",
            "success",
            True if cross.relation == "equivalent" else None,
        )

    if target == "apple_cpu":
        read_keys, read_values = ts.ops.kv_cache_read(cache, 2, 7)
        matches = bool(
            np.allclose(read_keys, keys[2:7], rtol=1e-5, atol=1e-6)
            and np.allclose(read_values, values[2:7], rtol=1e-5, atol=1e-6)
        )
        return verdict_for(target, "native_cpu", "success", matches)

    raise ValueError(f"kv_cache_read corroboration unsupported on {target!r}")


def corroborate(op: str, target: str, rng: Any) -> BackendVerdict:
    """Run ``op`` on ``target`` through the Evaluator and return its verdict."""
    if op == "kv_cache_read":
        return _corroborate_kv_cache_read(target, rng)
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
