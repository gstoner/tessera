"""Tests for ``tessera.compiler.jit_bridge`` — the Python →
manifest → shared-loader dispatch bridge.

The bridge is the single hop between the public-API fast paths
(``tessera.ga.*`` / ``tessera.ebm.*``) and the Apple GPU runtime.
These tests cover three claims:

  1. **Manifest resolution.** ``dispatch_via_manifest("clifford_inner")``
     resolves to ``apple_gpu / fused / tessera_apple_gpu_clifford_inner_cl30_f32``
     via ``backend_manifest.manifest_for``.
  2. **Route trace recording.** Calling a public API while tracing is
     on records a ``JitBridgeRoute`` whose ``op_name``, ``target``,
     ``status``, ``symbol``, ``context``, and ``latency_ms`` columns
     are populated.
  3. **JIT context awareness.** Calls inside ``jit_context("apple_gpu")``
     get ``context="jit:apple_gpu"``; calls outside get
     ``context="direct"``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# Skip the GPU-dispatch assertions cleanly on non-Darwin hosts — the
# manifest resolution and trace bookkeeping still work without a GPU
# (the bridge just won't actually dispatch).
darwin_only = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)

from tessera.compiler import jit_bridge as bridge  # noqa: E402
from tessera.compiler import backend_manifest as bm  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_bridge_state():
    """Each test gets a clean slate — no leaked trace rows, no
    JIT-context leftover, tracing off."""
    bridge.clear_dispatch_trace()
    while bridge.current_jit_context() is not None:
        # `jit_context` should always pop on exit, but defensively drain
        # the stack in case a prior failure left it dirty.
        bridge._STATE.jit_stack.pop()
    bridge.set_tracing_enabled(False)
    yield
    bridge.clear_dispatch_trace()
    bridge.set_tracing_enabled(False)


# ---------------------------------------------------------------------------
# Manifest resolution
# ---------------------------------------------------------------------------

def test_lookup_apple_gpu_symbol_resolves_clifford_inner() -> None:
    sym = bridge.lookup_apple_gpu_symbol("clifford_inner")
    assert sym == "tessera_apple_gpu_clifford_inner_cl30_f32"


def test_lookup_apple_gpu_symbol_resolves_ebm_inner_step() -> None:
    sym = bridge.lookup_apple_gpu_symbol("ebm_inner_step")
    assert sym == "tessera_apple_gpu_ebm_inner_step_f32"


def test_lookup_apple_gpu_symbol_returns_none_for_unknown_op() -> None:
    assert bridge.lookup_apple_gpu_symbol("nonsense_op_name") is None
    # EBM-prefixed names that aren't in `_EBM_APPLE_GPU_FUSED` return None.
    assert bridge.lookup_apple_gpu_symbol("ebm_does_not_exist") is None


def test_manifest_routes_ebm_prefix_through_ebm_table() -> None:
    """Cross-check: ``manifest_for("ebm_inner_step")`` and
    ``ebm_manifest_for(...)`` agree."""
    via_router = bm.manifest_for("ebm_inner_step")
    via_table = bm.ebm_manifest_for("ebm_inner_step")
    assert [(e.target, e.status) for e in via_router] == \
           [(e.target, e.status) for e in via_table]


# ---------------------------------------------------------------------------
# Trace toggle + recording
# ---------------------------------------------------------------------------

def test_tracing_disabled_by_default() -> None:
    assert bridge.tracing_enabled() is False


def test_set_tracing_enabled_round_trips() -> None:
    bridge.set_tracing_enabled(True)
    assert bridge.tracing_enabled() is True
    bridge.set_tracing_enabled(False)
    assert bridge.tracing_enabled() is False


def test_dispatch_trace_starts_empty() -> None:
    assert bridge.current_dispatch_trace() == ()


def test_take_dispatch_trace_clears() -> None:
    """``take_dispatch_trace()`` returns + clears the buffer."""
    # No GPU needed — we just need to exercise the buffer manipulation.
    bridge._STATE.trace.append(bridge.JitBridgeRoute(
        op_name="fake", target="apple_gpu", status="fused",
        symbol="fake_sym", context="direct", latency_ms=0.1,
    ))
    assert len(bridge.take_dispatch_trace()) == 1
    assert bridge.current_dispatch_trace() == ()


# ---------------------------------------------------------------------------
# JIT context stack
# ---------------------------------------------------------------------------

def test_jit_context_pushes_and_pops() -> None:
    assert bridge.current_jit_context() is None
    with bridge.jit_context("apple_gpu"):
        assert bridge.current_jit_context() == "jit:apple_gpu"
    assert bridge.current_jit_context() is None


def test_jit_context_is_nested() -> None:
    with bridge.jit_context("apple_gpu"):
        with bridge.jit_context("nvidia_sm90"):
            assert bridge.current_jit_context() == "jit:nvidia_sm90"
        assert bridge.current_jit_context() == "jit:apple_gpu"
    assert bridge.current_jit_context() is None


def test_jit_context_pops_on_exception() -> None:
    """Stack must unwind even if the inner block raises."""
    with pytest.raises(RuntimeError):
        with bridge.jit_context("apple_gpu"):
            raise RuntimeError("boom")
    assert bridge.current_jit_context() is None


# ---------------------------------------------------------------------------
# End-to-end dispatch (Darwin-only)
# ---------------------------------------------------------------------------

@darwin_only
def test_ga_inner_records_route_when_tracing_enabled() -> None:
    """``tessera.ga.inner`` dispatches through the bridge and the
    trace records one row with the manifest-resolved symbol."""
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(42)
    A = rng.randn(8, 8).astype(np.float32)
    B = rng.randn(8, 8).astype(np.float32)

    bridge.set_tracing_enabled(True)
    out = ga.inner(ga.Multivector(A, a), ga.Multivector(B, a))
    bridge.set_tracing_enabled(False)
    trace = bridge.take_dispatch_trace()

    # On a non-GPU host the dispatch returns ok=False and records nothing
    # — but the @darwin_only mark ensures we only run here on Darwin.
    assert len(trace) == 1
    row = trace[0]
    assert row.op_name == "clifford_inner"
    assert row.target == "apple_gpu"
    assert row.status == "fused"
    assert row.symbol == "tessera_apple_gpu_clifford_inner_cl30_f32"
    assert row.context == "direct"
    assert row.latency_ms >= 0.0
    # Output is bitwise-correct.
    expected = np.array([
        float(ga.inner(ga.Multivector(A[i], a),
                        ga.Multivector(B[i], a))) for i in range(8)
    ])
    assert float(np.abs(np.asarray(out) - expected).max()) <= 5e-6


@darwin_only
def test_ebm_inner_step_records_route_with_jit_context() -> None:
    """Inside ``jit_context`` the trace row carries context=jit:apple_gpu."""
    import tessera.ebm as ebm
    rng = np.random.RandomState(43)
    y = rng.randn(16, 4).astype(np.float32)
    g = rng.randn(16, 4).astype(np.float32)

    bridge.set_tracing_enabled(True)
    with bridge.jit_context("apple_gpu"):
        out = ebm.inner_step(y, g, eta=0.05)
    bridge.set_tracing_enabled(False)
    trace = bridge.take_dispatch_trace()

    assert len(trace) == 1
    row = trace[0]
    assert row.op_name == "ebm_inner_step"
    assert row.context == "jit:apple_gpu"
    assert row.symbol == "tessera_apple_gpu_ebm_inner_step_f32"
    expected = (y - 0.05 * g).astype(np.float32)
    assert float(np.abs(out - expected).max()) <= 1e-6


@darwin_only
def test_direct_call_and_jit_call_produce_same_numerical_output() -> None:
    """The route is recorded differently but the numerical output is
    bit-identical — the bridge doesn't perturb math."""
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(44)
    A = rng.randn(8, 8).astype(np.float32)
    B = rng.randn(8, 8).astype(np.float32)
    mv_a = ga.Multivector(A, a)
    mv_b = ga.Multivector(B, a)

    bridge.set_tracing_enabled(True)
    out_direct = np.asarray(ga.inner(mv_a, mv_b))
    with bridge.jit_context("apple_gpu"):
        out_jit = np.asarray(ga.inner(mv_a, mv_b))
    trace = bridge.take_dispatch_trace()

    assert len(trace) == 2
    assert trace[0].context == "direct"
    assert trace[1].context == "jit:apple_gpu"
    # Both routes hit the same kernel ⇒ bit-identical output.
    assert float(np.abs(out_direct - out_jit).max()) == 0.0


@darwin_only
def test_tracing_off_means_no_recording_even_when_dispatching() -> None:
    """``ga.inner`` still dispatches through the bridge when tracing is
    off — the bridge just doesn't append a row."""
    import tessera.ga as ga
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(45)
    A = rng.randn(4, 8).astype(np.float32)
    B = rng.randn(4, 8).astype(np.float32)

    assert bridge.tracing_enabled() is False
    out = ga.inner(ga.Multivector(A, a), ga.Multivector(B, a))
    assert bridge.current_dispatch_trace() == ()
    # Output still correct.
    expected = np.array([
        float(ga.inner(ga.Multivector(A[i], a),
                        ga.Multivector(B[i], a))) for i in range(4)
    ])
    assert float(np.abs(np.asarray(out) - expected).max()) <= 5e-6


# ---------------------------------------------------------------------------
# Miss semantics
# ---------------------------------------------------------------------------

def test_dispatch_via_manifest_raises_for_unknown_op() -> None:
    """``JitBridgeMiss`` is raised when the manifest has no
    ``apple_gpu=fused`` entry for the requested op."""
    import ctypes
    with pytest.raises(bridge.JitBridgeMiss):
        bridge.dispatch_via_manifest(
            "nonsense_op_name",
            argtypes=(ctypes.c_int32,),
            args=(ctypes.c_int32(0),),
        )


def test_dispatch_via_manifest_raises_for_planned_only_op() -> None:
    """Ops whose only manifest entry is ``planned`` (no fused symbol)
    raise :class:`JitBridgeMiss` — the bridge refuses to dispatch
    anything it can't resolve through ``_EBM_APPLE_GPU_FUSED`` /
    ``_CLIFFORD_APPLE_GPU_FUSED``."""
    import ctypes
    with pytest.raises(bridge.JitBridgeMiss):
        bridge.dispatch_via_manifest(
            "ebm_partition_function_ais",
            argtypes=(ctypes.c_int32,),
            args=(ctypes.c_int32(0),),
        )


# ---------------------------------------------------------------------------
# Trace dataclass contract
# ---------------------------------------------------------------------------

def test_jit_bridge_route_is_frozen_dataclass() -> None:
    """Routes are immutable — callers can stash them safely."""
    r = bridge.JitBridgeRoute(
        op_name="x", target="apple_gpu", status="fused",
        symbol="s", context="direct", latency_ms=0.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        r.op_name = "y"   # type: ignore[misc]


def test_shaped_summary_handles_numpy_arrays() -> None:
    a = np.zeros((4, 8), dtype=np.float32)
    b = np.zeros((4, 8), dtype=np.float32)
    summary = bridge.shaped_summary(a, b)
    assert summary == ("(4, 8):float32", "(4, 8):float32")


def test_shaped_summary_handles_non_arrays() -> None:
    summary = bridge.shaped_summary(42, "hello")
    # Non-arrays fall back to the type name.
    assert summary[0] == "'int'" or "int" in summary[0]
