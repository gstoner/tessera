"""Regression: ``@tessera.jit(native_required=True)`` surfaces
native-CPU launch failures instead of silently falling back to
the numpy reference plan.

Findings audit (2026-05-19) flagged that ``_native_cpu_fast_call``
caught every exception from ``_execute_native_cpu_metadata`` and
quietly ran ``cpu_plan.execute(...)`` — hiding ABI mismatches
from ordinary @jit users and making ``execution_kind=native_cpu``
misleading.

Fixes locked here:

  1. ``native_required=True`` raises
     :class:`TesseraNativeRequiredError` on a native-CPU launch
     failure instead of falling through.
  2. Without ``native_required=True``, the fallback still happens
     but the reason is recorded on ``self.last_fallback_reason``
     so callers can inspect it.
  3. A successful native run clears ``last_fallback_reason`` to
     ``None``.
"""

from __future__ import annotations

import pytest

from tessera.compiler.fallback import (
    FallbackReason,
    TesseraNativeRequiredError,
)
from tessera.compiler.jit import JitFn


class _StubJit:
    """Minimal stand-in that mirrors ``JitFn`` enough to exercise
    ``_native_cpu_fast_call``'s fallback / native-required paths
    without needing a full @jit pipeline."""

    def __init__(self, native_required: bool, native_raises: Exception | None):
        self.native_required = native_required
        self._native_raises = native_raises
        self.last_fallback_reason = None
        self.arg_names = ("x",)
        self.cpu_plan_called = False
        self._fn_name = "stub_kernel"

    @property
    def _fn(self):
        # Mimic the attribute access ``getattr(self._fn, "__name__", "")``
        # does inside _native_cpu_fast_call.
        class _F:
            __name__ = self._fn_name
        return _F()

    def runtime_artifact(self):
        class _A:
            metadata = {}
        return _A()

    class _CpuPlan:
        def execute(self_inner, args, kwargs, arg_names):
            return ("cpu_plan_result", args, kwargs)

    @property
    def cpu_plan(self):
        return self._CpuPlan()


def _native_call(stub: _StubJit, monkeypatch) -> object:
    """Invoke ``_native_cpu_fast_call`` via ``JitFn`` bound to a
    stub instance.  Patches the runtime entry point to raise when
    ``stub._native_raises`` is set."""

    def fake_execute(metadata, launch_args):
        if stub._native_raises is not None:
            raise stub._native_raises
        return ("native_result", metadata, launch_args)

    monkeypatch.setattr(
        "tessera.runtime._execute_native_cpu_metadata",
        fake_execute,
    )
    # Call the unbound method on the stub.
    return JitFn._native_cpu_fast_call(stub, args=(1,), kwargs={})


def test_native_success_clears_last_fallback_reason(monkeypatch) -> None:
    stub = _StubJit(native_required=False, native_raises=None)
    # Seed a stale value to confirm a clean run resets it.
    stub.last_fallback_reason = FallbackReason.CAPABILITY_NOT_READY
    result = _native_call(stub, monkeypatch)
    assert result[0] == "native_result"
    assert stub.last_fallback_reason is None


def test_native_failure_falls_through_and_records_reason(monkeypatch) -> None:
    stub = _StubJit(
        native_required=False,
        native_raises=RuntimeError("ABI mismatch on f32 GEMM"),
    )
    result = _native_call(stub, monkeypatch)
    # Fell through to the numpy reference plan.
    assert result[0] == "cpu_plan_result"
    # ... and recorded the reason on the JitFn so callers / reports
    # can inspect it.
    assert stub.last_fallback_reason == FallbackReason.CAPABILITY_NOT_READY


def test_native_required_raises_on_failure(monkeypatch) -> None:
    stub = _StubJit(
        native_required=True,
        native_raises=RuntimeError("ABI mismatch on f32 GEMM"),
    )
    with pytest.raises(TesseraNativeRequiredError) as excinfo:
        _native_call(stub, monkeypatch)
    err = excinfo.value
    assert err.reason == FallbackReason.CAPABILITY_NOT_READY
    assert err.target == "cpu"
    assert "ABI mismatch" in err.detail
    # And the JitFn still has the reason on it for inspection.
    assert stub.last_fallback_reason == FallbackReason.CAPABILITY_NOT_READY


def test_jit_decorator_threads_native_required() -> None:
    """The public ``@tessera.jit(native_required=True)`` parameter
    actually reaches the JitFn instance."""
    import inspect
    from tessera.compiler.jit import jit as jit_decorator

    params = inspect.signature(jit_decorator).parameters
    assert "native_required" in params, (
        "@jit(native_required=True) is the documented escape hatch; "
        "its parameter must appear in jit()'s signature"
    )
    assert params["native_required"].default is False
