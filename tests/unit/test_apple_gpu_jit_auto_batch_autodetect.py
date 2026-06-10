"""P3 (2026-06-09) — auto_batch polish: auto-detection + Graph-IR-emission skip.

Closes the APPLE_AUDIT "Still Open" P3 item:
  (a) `@jit(target="apple_gpu")` auto-detects a recognized decode chain (a body
      of >=2 encode-eligible ops and nothing else) and turns the
      one-command-buffer route on by default (auto_batch default is now None);
  (b) when the route is on, the unused AST Graph IR emission is skipped.

These are decoration-time / introspection tests — no GPU required.
"""

from __future__ import annotations

import textwrap

import pytest

import tessera as ts
from tessera.compiler.jit import (
    TesseraJitError,
    _APPLE_GPU_ENCODE_OP_NAMES,
    _recognized_decode_chain,
    _resolve_auto_batch,
)


# ── Drift gate: detection op-name set mirrors the encode registry ─────

def test_encode_op_names_match_registry():
    from tessera.apple_gpu_chain import ENCODE_OP_REGISTRY

    registry_names = {name for (name, _dtype) in ENCODE_OP_REGISTRY}
    assert _APPLE_GPU_ENCODE_OP_NAMES == registry_names


# ── Detection helper ──────────────────────────────────────────────────

_CHAIN = """
def fn(x, g):
    a = ts.ops.rmsnorm(x, g, rows=4, cols=16, eps=1e-5)
    b = ts.ops.silu(a, n=64)
    c = ts.ops.rmsnorm(b, g, rows=4, cols=16, eps=1e-5)
    return ts.ops.silu(c, n=64)
"""
_NESTED = """
def fn(x, g):
    return ts.ops.silu(ts.ops.rmsnorm(x, g, rows=4, cols=16), n=4)
"""
_SINGLE = """
def fn(x):
    return ts.ops.silu(x, n=4)
"""
_ARITH = """
def fn(x, g):
    a = ts.ops.rmsnorm(x, g, rows=4, cols=16)
    return ts.ops.silu(a, n=4) * 2.0
"""
_CONTROL = """
def fn(x, g):
    a = ts.ops.rmsnorm(x, g, rows=4, cols=16)
    for _ in range(3):
        a = ts.ops.silu(a, n=4)
    return a
"""
_NONOP = """
def fn(x, g):
    a = ts.ops.rmsnorm(x, g, rows=4, cols=16)
    b = helper(a)
    return ts.ops.silu(b, n=4)
"""
_SUBSCRIPT = """
def fn(x, g):
    a = ts.ops.rmsnorm(x, g, rows=4, cols=16)
    return ts.ops.silu(a[0], n=4)
"""


@pytest.mark.parametrize("src,expected", [
    (_CHAIN, True),       # 4-op chain
    (_NESTED, True),      # 2 nested op calls
    (_SINGLE, False),     # only 1 op (chain needs >=2)
    (_ARITH, False),      # arithmetic on an op result
    (_CONTROL, False),    # control flow + range() call
    (_NONOP, False),      # a non-encode helper call
    (_SUBSCRIPT, False),  # subscript on an op result
    ("", False),          # no source
    ("def fn(x):\n    return x\n", False),  # no ops at all
])
def test_recognized_decode_chain(src, expected):
    assert _recognized_decode_chain(textwrap.dedent(src)) is expected


# ── auto_batch resolution ─────────────────────────────────────────────

def test_resolve_auto_batch_explicit_overrides_detection():
    # Explicit True/False win regardless of body shape / target.
    assert _resolve_auto_batch(True, "apple_gpu", _SINGLE) is True
    assert _resolve_auto_batch(False, "apple_gpu", _CHAIN) is False
    # Explicit True even off-apple resolves True (the non-apple guard fires
    # later in the decorator, preserving the misuse error).
    assert _resolve_auto_batch(True, "x86", _CHAIN) is True


def test_resolve_auto_batch_none_autodetects_on_apple_only():
    assert _resolve_auto_batch(None, "apple_gpu", _CHAIN) is True
    assert _resolve_auto_batch(None, "apple_gpu", _SINGLE) is False
    # Auto-detection never flips on for a non-apple target.
    assert _resolve_auto_batch(None, "x86", _CHAIN) is False
    assert _resolve_auto_batch(None, None, _CHAIN) is False


# ── End-to-end at decoration: emission skipped on the auto_batch path ─

def _decorate(src, **kw):
    ns: dict = {"ts": ts}
    exec(compile(textwrap.dedent(src), "<test>", "exec"), ns)  # noqa: S102
    return ts.jit(target="apple_gpu", source=textwrap.dedent(src), **kw)(ns["fn"])


def test_autodetected_chain_skips_graph_ir_emission():
    fn = _decorate(_CHAIN)  # auto_batch=None default -> detected
    # Emission skipped: empty module, no compile bundle, the auto_batch
    # diagnostic present, and the call path falls through to the wrapper
    # (not the surgical tracer).
    assert len(fn.graph_ir.functions) == 0
    assert fn.compile_bundle is None
    assert fn.cpu_plan is None
    assert fn._needs_trace is False
    codes = {d.code for d in fn.lowering_diagnostics}
    assert "JIT_APPLE_GPU_AUTO_BATCH" in codes


def test_explicit_true_skips_emission():
    fn = _decorate(_SINGLE, auto_batch=True)  # single op, but forced on
    assert len(fn.graph_ir.functions) == 0
    assert fn.compile_bundle is None


def test_explicit_false_keeps_emission():
    fn = _decorate(_CHAIN, auto_batch=False)  # recognized chain, forced off
    # Emission runs normally: the module carries the body.
    assert len(fn.graph_ir.functions) == 1
    codes = {d.code for d in fn.lowering_diagnostics}
    assert "JIT_APPLE_GPU_AUTO_BATCH" not in codes


def test_non_chain_keeps_emission():
    fn = _decorate(_ARITH)  # arithmetic body -> not auto-batched
    assert len(fn.graph_ir.functions) == 1


def test_emit_package_keeps_emission_even_when_autobatched():
    # emit_package needs the recognized region, so emission is NOT skipped
    # even though the body is a recognized chain.
    fn = _decorate(_CHAIN, emit_package=False)  # default; sanity
    assert len(fn.graph_ir.functions) == 0
    fn2 = _decorate(_CHAIN, auto_batch=True, emit_package=True)
    assert len(fn2.graph_ir.functions) == 1


# ── Misuse guards still hold under the None default ───────────────────

def test_max_ops_per_cb_without_effective_auto_batch_raises():
    # A non-chain body (no auto-detect) + max_ops_per_cb -> error.
    with pytest.raises(TesseraJitError, match="auto_batch one-command-buffer"):
        _decorate("def fn(x):\n    return x\n", max_ops_per_cb=8)


def test_explicit_auto_batch_on_non_apple_target_raises():
    with pytest.raises(TesseraJitError, match="only supports"):
        @ts.jit(target="x86", auto_batch=True)
        def _bad(x):
            return x
