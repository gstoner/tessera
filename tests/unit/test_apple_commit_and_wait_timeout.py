"""Apple-sample Pattern 4 — shared-event timeout wrapper for the legacy
MPS / MPSGraph queue.

Pre-Pattern-4 the legacy lane used ``[cb commit]; [cb waitUntilCompleted]``
with no timeout — a hung kernel or driver crash hung the test forever.
Pattern 4 wraps this with ``MTLSharedEvent`` + ``waitUntilSignaledValue:
timeoutMS:`` (mirroring Apple's sample at
``MLMatrixMultiplier.m:241-255``), giving us a named-op timeout that
surfaces "GPU dispatch did not signal" with a precise op name.

The probe ``tessera_apple_gpu_commit_and_wait_timeout_probe(timeout_ms)``
submits a 64-byte fill (the cheapest GPU dispatch we can encode) and
returns:

* ``1`` — completed in time
* ``0`` — timed out (would only happen if the timeout machinery itself
          is broken, since the workload is microseconds)
* ``-1`` — runtime not available (skip)
* ``-2`` — buffer alloc / command buffer creation failed
"""

from __future__ import annotations

import ctypes

import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol


def _bind_timeout_probe():
    runtime = apple_gpu_runtime()
    if runtime is None:
        return None
    return bind_symbol(
        "tessera_apple_gpu_commit_and_wait_timeout_probe",
        (ctypes.c_uint64,),
        restype=ctypes.c_int32,
    )


def _probe(timeout_ms: int) -> int:
    fn = _bind_timeout_probe()
    if fn is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    return fn(timeout_ms)


# ---- The timeout machinery is wired and a real GPU op signals -----------

def test_generous_timeout_completes_for_trivial_dispatch():
    """A 64-byte fill must complete well under 1 second on any working
    Apple-Silicon Metal lane. Generous 30s timeout. Result must be 1
    (completed); 0 (timeout) means the shared-event signal path is
    broken; -1 means the runtime didn't load."""
    rc = _probe(30_000)
    if rc == -1:
        pytest.skip("Apple GPU runtime initialized to ok=False")
    if rc == -2:
        pytest.skip("Buffer / command-buffer alloc failed")
    assert rc == 1, f"expected completion (1); got rc={rc}"


def test_event_value_is_monotonic_across_calls():
    """Pattern 4's signal counter ``legacy_event_val`` is monotonic — every
    call increments it. Verify by submitting 5 trivial dispatches in
    sequence; all five must succeed. (The probe re-enters
    ``commit_and_wait_with_timeout`` which increments the counter.)"""
    for i in range(5):
        rc = _probe(30_000)
        if rc < 0:
            pytest.skip("Apple GPU runtime not available")
        assert rc == 1, f"call {i}: rc={rc}"


# ---- Zero-timeout behavior is well-defined ------------------------------

def test_zero_timeout_returns_a_definite_answer():
    """A 0 ms timeout is a degenerate but well-defined input. The probe
    must return EITHER 1 (the GPU somehow signaled before the wait even
    started — unlikely but legal) OR 0 (timeout fired immediately). Must
    not crash, must not return a value outside the documented enum."""
    rc = _probe(0)
    if rc < 0:
        pytest.skip("Apple GPU runtime not available")
    assert rc in (0, 1), f"unexpected rc={rc}"


# ---- The wrapper is a real helper, not just dead code -------------------

def test_dispatch_mps_gemm_f32_uses_the_timeout_wrapper():
    """The canonical conversion target. Verify by source-reading: the
    matmul dispatch path must call ``commit_and_wait_with_timeout`` with
    a named op string, not raw ``waitUntilCompleted``. Catches a
    regression where someone reverts the conversion."""
    from pathlib import Path
    rt = (Path(__file__).resolve().parents[2]
          / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend"
          / "runtime" / "apple_gpu_runtime.mm")
    src = rt.read_text()
    # The canonical conversion site uses the helper with the op name
    # "mps_gemm_f32".
    assert 'commit_and_wait_with_timeout(ctx, cb,' in src, (
        "expected the matmul dispatch to call "
        "commit_and_wait_with_timeout — was the conversion reverted?")
    assert '"mps_gemm_f32"' in src, (
        "expected the timeout helper to be called with the named op "
        "'mps_gemm_f32' so timeouts have a clear diagnostic")


def test_helper_is_defined_with_apple_sample_pattern_comment():
    """Drift guard — the ``commit_and_wait_with_timeout`` helper's
    docstring names Apple's sample as the source pattern, so a future
    reader knows where the design came from."""
    from pathlib import Path
    rt = (Path(__file__).resolve().parents[2]
          / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend"
          / "runtime" / "apple_gpu_runtime.mm")
    src = rt.read_text()
    assert "Apple-sample Pattern 4" in src
    assert "MLMatrixMultiplier.m:241-255" in src or (
        "MLMatrixMultiplier.m" in src)
