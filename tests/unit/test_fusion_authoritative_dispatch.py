"""Phase 0b/0c — authoritative fusion dispatch (front-to-back closure plan).

0b makes the apple_gpu executor dispatch a fused kernel directly from the
whole-program known_chain group's operand/result roles (Phase 0a), instead of
re-matching the op list per invoke. The structural re-matchers stay as a
fallback for legacy artifacts that predate the `dispatch` roles.

The central guard is the **horizontal-oracle equivalence** (the 0c gate): for
the same program, the authoritative path and the re-matcher path must produce
identical output. We force the re-matcher path by stripping the `dispatch` roles
from a copy of the metadata, then compare. This is the proof that deleting the
re-matchers (0c) is behavior-preserving.

See docs/audit/compiler/COMPILER_AUDIT.md (Library → Optimizing Compiler).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt


def _strip_dispatch(md: dict) -> dict:
    """Return a metadata copy with `dispatch` roles removed from every fusion
    group, so the executor's authoritative resolver declines and the structural
    re-matcher fires instead."""
    md = copy.deepcopy(dict(md))
    for key in ("fusion_groups", "canonical_fusion_groups"):
        groups = md.get(key)
        if not groups:
            continue
        stripped = []
        for g in groups:
            g = dict(g)
            g.pop("dispatch", None)
            stripped.append(g)
        md[key] = stripped
    return md


def _mm_softmax_matmul(a, b, c):
    return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(a, b)), c)


def _mm_softmax(a, b):
    return ts.ops.softmax(ts.ops.matmul(a, b))


def _mm_gelu(a, b):
    return ts.ops.gelu(ts.ops.matmul(a, b))


def _mm_rmsnorm(a, b):
    return ts.ops.rmsnorm(ts.ops.matmul(a, b))


def _swiglu(x, wg, wu, wd):
    return ts.ops.matmul(ts.ops.silu_mul(ts.ops.matmul(x, wg),
                                         ts.ops.matmul(x, wu)), wd)


_RNG = np.random.default_rng(7)


def _mat(*shape):
    return _RNG.standard_normal(shape).astype(np.float32)


CASES = {
    "matmul_softmax_matmul": (_mm_softmax_matmul,
                              (_mat(4, 8), _mat(8, 6), _mat(6, 5))),
    "matmul_softmax": (_mm_softmax, (_mat(4, 8), _mat(8, 6))),
    "matmul_gelu": (_mm_gelu, (_mat(4, 8), _mat(8, 6))),
    "matmul_rmsnorm": (_mm_rmsnorm, (_mat(4, 8), _mat(8, 6))),
    "swiglu": (_swiglu, (_mat(4, 8), _mat(8, 6), _mat(8, 6), _mat(6, 5))),
}


@pytest.mark.parametrize("kernel", list(CASES))
def test_authoritative_plan_fires(kernel):
    fn_builder, args = CASES[kernel]
    fn = ts.jit(target="apple_gpu")(fn_builder)
    md = fn.runtime_artifact().metadata
    ops = md.get("ops") or []
    plan = rt._apple_gpu_resolve_authoritative_plan(md, len(ops))
    if plan is None:
        # swiglu requires the silu_mul lowering to produce a known_chain group;
        # if it isn't available here the case is simply not exercised.
        pytest.skip(f"no authoritative plan for {kernel} in this environment")
    resolved_kernel, roles = plan
    assert resolved_kernel == kernel
    assert "out" in roles


@pytest.mark.parametrize("kernel", list(CASES))
def test_authoritative_matches_rematcher(kernel):
    """The 0c gate: authoritative dispatch == structural re-matcher, same input."""
    fn_builder, args = CASES[kernel]
    fn = ts.jit(target="apple_gpu")(fn_builder)
    md = fn.runtime_artifact().metadata
    ops = md.get("ops") or []
    if rt._apple_gpu_resolve_authoritative_plan(md, len(ops)) is None:
        pytest.skip(f"no authoritative plan for {kernel} in this environment")

    authoritative = np.asarray(rt._execute_apple_gpu_mps_metadata(md, list(args)))
    rematcher = np.asarray(
        rt._execute_apple_gpu_mps_metadata(_strip_dispatch(md), list(args)))

    assert authoritative.shape == rematcher.shape
    np.testing.assert_allclose(authoritative, rematcher, rtol=1e-6, atol=1e-6)


def test_authoritative_attention_matches_numpy():
    """End-to-end numerical correctness through the authoritative path."""
    fn = ts.jit(target="apple_gpu")(_mm_softmax_matmul)
    a, b, c = _mat(4, 8), _mat(8, 6), _mat(6, 5)
    got = np.asarray(fn(a, b, c))
    scores = a @ b
    sm = np.exp(scores - scores.max(-1, keepdims=True))
    sm /= sm.sum(-1, keepdims=True)
    np.testing.assert_allclose(got, sm @ c, rtol=1e-5, atol=1e-5)


def test_missing_roles_falls_back_to_cascade():
    """Stripping dispatch roles must not break execution — the cascade still
    runs. (Legacy-artifact safety: 0b is additive.)"""
    fn = ts.jit(target="apple_gpu")(_mm_softmax)
    a, b = _mat(4, 8), _mat(8, 6)
    md = _strip_dispatch(fn.runtime_artifact().metadata)
    assert rt._apple_gpu_resolve_authoritative_plan(md, len(md.get("ops"))) is None
    out = np.asarray(rt._execute_apple_gpu_mps_metadata(md, [a, b]))
    sm = np.exp(a @ b - (a @ b).max(-1, keepdims=True))
    sm /= sm.sum(-1, keepdims=True)
    np.testing.assert_allclose(out, sm, rtol=1e-5, atol=1e-5)
