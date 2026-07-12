"""available_backends() / backend_capabilities() must be process-cached.

_runtime_snapshot() constructs + init()s a TesseraRuntime and probes every
device (~30ms). available_backends() is on the launch() hot path (via
backend_capabilities), so re-probing per call added ~32ms to EVERY device_verified_jit-lane
launch — dwarfing the kernels (a kernel is µs–ms). This test locks the cache by
call-count (not timing, which is flaky): the device probe must run at most once
across many launches, and reset must re-enable it.
"""

from __future__ import annotations

from tessera import runtime as rt


def test_available_backends_probes_device_at_most_once(monkeypatch):
    rt._reset_backend_probe_cache()
    calls = {"n": 0}
    real = rt._runtime_snapshot

    def counting_snapshot():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(rt, "_runtime_snapshot", counting_snapshot)

    # Many backend_capabilities calls (the launch() hot path) → one probe.
    for _ in range(50):
        rt.backend_capabilities("rocm")
        rt.backend_capabilities("cpu")
        rt.available_backends()
    assert calls["n"] <= 1, (
        f"device probe ran {calls['n']}x — available_backends() is not cached; "
        "this is the ~32ms/launch regression"
    )


def test_reset_re_enables_the_probe(monkeypatch):
    rt._reset_backend_probe_cache()
    rt.available_backends()                      # populate cache
    assert rt._available_backends_cache is not None

    calls = {"n": 0}
    real = rt._runtime_snapshot

    def counting_snapshot():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(rt, "_runtime_snapshot", counting_snapshot)
    rt._reset_backend_probe_cache()              # tests swapping the runtime
    assert rt._available_backends_cache is None
    rt.available_backends()
    assert calls["n"] == 1
