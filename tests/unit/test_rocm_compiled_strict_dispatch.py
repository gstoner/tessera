"""A compiled-ROCm FAILURE must not masquerade as a successful native launch.

Measured 2026-08-04, before this landed: injecting a deliberately broken pass
pipeline into the compiled GEMM lane produced

    ok=True, compiler_path="rocm_compiled", execution_kind="native_gpu"

with numerically correct output — because `tessera-opt` failed (rc=1), every
handler caught `_RocmCompiledUnavailable`, and another lane served the result
under the compiled lane's labels. `TESSERA_STRICT_DISPATCH=1` exists precisely
to stop this (audit 2026-06-10 finding #5, whose docstring says "the fallback
masks exactly the bugs the tests exist to catch") and did not cover this path.

`_RocmCompiledUnavailable`'s own docstring claimed the invariant that failed:
"NOT on a genuine kernel failure (those surface, so a real compiled-lane bug is
never masked)". It did not surface.

**Why this matters beyond tidiness.** It makes hardware numerics untrustworthy
as a gate: a test asserting "the compiled K-loop matches numpy on gfx1151"
passes identically whether the lowering works or silently fell back. W1.1 step
2b is gated on exactly that measurement, so the gate had to be repaired before
the measurement could mean anything.

The distinction the fix preserves is the one the funnel already documents:

  * ENVELOPE LIMIT — no libamdhip64, hipInit failed, tessera-opt not built, a
    dtype/rank/arch outside the lane's documented range. Falling back is
    correct and must NOT trip strict dispatch, or strict CI breaks on every
    CPU-only host.
  * FAILURE — tessera-opt ran and produced no kernel, or a non-ELF blob. The
    compiled lane is broken and must say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt


@pytest.fixture(autouse=True)
def _clean_log():
    rt.reset_dispatch_fallback_log()
    yield
    rt.reset_dispatch_fallback_log()


def test_failure_class_is_recorded_in_the_fallback_log():
    """The failure funnel is reached at all — the property that was missing.

    Calls the helper directly so this runs without an AMD GPU: the point under
    test is the classification, not the kernel.
    """
    with pytest.raises(rt._RocmCompiledUnavailable):
        rt._rocm_compiled_failed("tessera-opt did not serialize the compiled GEMM (test)")
    log = rt.dispatch_fallback_log()
    assert any(op == "rocm_compiled" for op, _ in log), (
        f"failure-class fallback was not recorded: {log}"
    )


def test_strict_dispatch_raises_on_a_failure_class_fallback(monkeypatch):
    """Under strict, a compiled-lane failure raises instead of degrading."""
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    with pytest.raises(rt.TesseraStrictDispatchError, match="rocm_compiled"):
        rt._rocm_compiled_failed("tessera-opt did not serialize the compiled GEMM (test)")


def test_envelope_limits_do_not_trip_strict_dispatch(monkeypatch):
    """The safety property, and the reason this is not simply 'raise on error'.

    An unsupported dtype is a documented envelope limit, not a broken kernel.
    If it tripped strict, every strict run on a CPU-only host — and every
    f32/f64 matmul anywhere — would fail, which is how a gate like this gets
    switched off within a week.
    """
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    artifact = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_compiled", "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "c",
        "ops": [{"op_name": "tessera.matmul", "result": "c",
                 "operands": ["a", "b"], "kwargs": {}}],
    })
    a = np.zeros((8, 8), dtype=np.float64)
    result = rt.launch(artifact, (a, a))
    reason = result.get("reason") or ""
    assert "STRICT_DISPATCH" not in reason, (
        f"an envelope limit tripped strict dispatch: {reason}"
    )


def test_the_two_classes_are_distinguishable():
    """A ratchet on the split itself.

    Collapsing these — making every `_RocmCompiledUnavailable` failure-class, or
    none of them — is the tempting simplification, and either direction breaks
    something: all-failure breaks CPU-only strict CI, none-failure restores the
    masking this file exists to prevent.
    """
    import inspect
    src = inspect.getsource(rt._rocm_compiled_failed)
    assert "_note_dispatch_fallback" in src, (
        "the failure path no longer routes through the shared dispatch funnel"
    )
    # Envelope raise sites must NOT go through the helper.
    runtime_src = inspect.getsource(rt)
    for envelope in ("libamdhip64.so not loadable",
                     "tessera-opt not built"):
        idx = runtime_src.find(envelope)
        assert idx != -1, f"envelope message vanished: {envelope!r}"
        window = runtime_src[max(0, idx - 200):idx]
        assert "_rocm_compiled_failed(" not in window, (
            f"envelope case {envelope!r} was routed through the failure helper"
        )
