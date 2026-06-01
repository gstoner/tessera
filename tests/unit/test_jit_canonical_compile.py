"""C.3 — ``@tessera.jit`` carries the canonical CompileResult on JitFn.

After C.1 (canonical wrapper) and C.2 (runtime trusts the canonical
answer), C.3 retrofits the *decorator* so a ``@jit``-produced ``JitFn``
exposes the unified C answer (typed artifacts + seven gates + executable
| reason) on a single attribute: ``jit_fn.compile_result``.

Crucially, the retrofit doesn't run a second compile — ``@jit`` already
calls ``compile_graph_module``, and C.3 uses ``compile_result_from_bundle``
to synthesize the canonical answer from that same bundle (gate evaluation
is cheap; ladder isn't).

Tests pin:

1. Every ``JitFn`` has a non-None ``compile_result`` of the canonical
   typed shape — no caller has to thread through ``compile_bundle`` /
   ``execution_kind`` / gate evaluator separately.
2. The result's ``bundle`` is the same object as ``compile_bundle``
   (the legacy field) — no two-bundle confusion.
3. ``@jit(target="nvidia_sm90")`` returns at decoration time with
   ``compile_result.first_failing_gate.gate == "toolchain"`` — the
   audit-named answer is available before any call.
4. Existing ``cpu_plan`` / ``compile_bundle`` / ``execution_kind`` fields
   are unchanged — the retrofit is additive.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import tessera
from tessera.compiler.canonical_compile import CompileResult


@tessera.jit
def _cpu_eager_demo(a, b):  # noqa: D401 — fixture for tests
    return a @ b


def test_jit_fn_has_compile_result():
    """Every decorated function exposes the canonical answer."""
    assert _cpu_eager_demo.compile_result is not None
    assert isinstance(_cpu_eager_demo.compile_result, CompileResult)


def test_compile_result_bundle_is_compile_bundle():
    """No two-bundle confusion. The canonical result wraps the same bundle
    that lives on the legacy ``compile_bundle`` field."""
    r = _cpu_eager_demo.compile_result
    assert r.bundle is _cpu_eager_demo.compile_bundle


def test_compile_result_target_matches_jit_target():
    """The canonical target matches what @jit recorded."""
    r = _cpu_eager_demo.compile_result
    assert r.target == "cpu"


def test_compile_result_carries_seven_gates():
    """The canonical capability set on JitFn — every gate present."""
    from tessera.compiler import pipeline_gates as pg
    r = _cpu_eager_demo.compile_result
    assert len(r.gate_results) == len(pg.GATE_ORDER)
    assert tuple(g.gate for g in r.gate_results) == pg.GATE_ORDER


def test_jit_target_nvidia_surfaces_toolchain_gate_at_decoration_time(tmp_path):
    """The C.3 payoff: ``@jit(target="nvidia_sm90")`` resolves the
    audit-named gate at *decoration* time — no call needed. The diagnostic
    is available before the user discovers the missing toolchain at
    launch.

    Uses a temp-file module because @jit needs introspectable source.
    """
    src = textwrap.dedent("""
        import tessera
        @tessera.jit(target="nvidia_sm90")
        def f(a, b):
            return a @ b
    """).strip() + "\n"
    mod_dir = tmp_path
    (mod_dir / "c3_mod.py").write_text(src)
    import sys
    sys.path.insert(0, str(mod_dir))
    try:
        import c3_mod
        r = c3_mod.f.compile_result
        assert r is not None
        assert r.target == "nvidia_sm90"
        assert r.executable is False
        assert r.first_failing_gate is not None
        assert r.first_failing_gate.gate == "toolchain"
        assert "nvcc" in r.first_failing_gate.detail
        # The reason mirrors the gate.
        assert r.reason.startswith("first failing gate `toolchain`")
    finally:
        sys.path.remove(str(mod_dir))
        # Don't pollute future imports.
        sys.modules.pop("c3_mod", None)


def test_existing_legacy_fields_still_present():
    """The retrofit is additive — the cpu_plan / compile_bundle /
    execution_kind path callers depend on must keep working."""
    fn = _cpu_eager_demo
    # These existed pre-C.3 and must continue to exist.
    assert hasattr(fn, "compile_bundle")
    assert hasattr(fn, "cpu_plan")
    assert hasattr(fn, "execution_kind")


def test_compile_result_to_dict_round_trips_on_jit_fn():
    """A consumer can serialize the canonical answer directly from JitFn —
    proves the single typed surface is fully consumable."""
    r = _cpu_eager_demo.compile_result
    d = r.to_dict()
    assert "target" in d
    assert "gates" in d
    assert "executable" in d
    assert "first_failing_gate" in d
