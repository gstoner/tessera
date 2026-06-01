"""PK audit P2 — packaged ML dispatch concurrency contract.

The audit found that ``tessera_apple_gpu_mlpkg_dispatch`` accessed
``ctx.mtl4_event`` + bumped ``ctx.mtl4_event_val`` WITHOUT holding
``ctx.mtl4_dispatch_mu``. That mutex is what the canonical MTL4 lane
(``mtl4_matmul2d_dispatch`` and friends) uses to serialize encode →
commit → signal → wait. The packaged ML lane is intentionally an
"outlier" — it creates per-call allocator + command buffer so it
doesn't have to serialize against the canonical lane. But sharing
the monotonic counter + the shared-event object across the two
lanes was a real race: an interleaved increment could hand the
packaged lane a signal value the canonical lane was about to
overwrite, or vice versa.

The fix: give packaged ML its own ``mlpkg_event`` + ``mlpkg_event_val``
+ ``mlpkg_event_mu``. The queue stays shared (the queue itself
serializes submission, so we don't lose ordering), but each lane
has its own scoreboard.

These tests pin:

* **Concurrent packaged dispatches succeed** — 4 threads each running
  N dispatches on shared+per-thread pipelines. Each dispatch must
  complete cleanly without timing out. A regression where the mutex
  is missing would either hang (signal value never reaches a waiter)
  or return ``False`` for some dispatches under load.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from tessera.apple_mlpkg import (
    compile_mlpackage,
    last_error_kind,
    packaged_ml_available,
    packaged_ml_skip_reason,
)


_FIXTURES_DIR = (Path(__file__).resolve().parent.parent
                 / "fixtures" / "apple_gpu")


def _find_mtlpackage() -> Path | None:
    if not _FIXTURES_DIR.is_dir():
        return None
    for entry in _FIXTURES_DIR.iterdir():
        if entry.suffix == ".mtlpackage" and entry.is_dir():
            return entry
    return None


def test_concurrent_packaged_dispatches_do_not_race_the_event_counter():
    """Run 4 threads × 8 dispatches each over independent pipelines
    (so the per-pipeline ``argumentTable`` / intermediates heap are
    NOT contended). The shared resources under contention are
    ``ctx.mlpkg_event`` + ``ctx.mlpkg_event_val`` + ``ctx.mlpkg_event_mu``
    + ``ctx.mtl4_queue``. If the P2 fix is in place, every dispatch
    must report success; if the mutex is missing, the monotonic
    counter races, signal values collide, and at least one
    ``waitUntilSignaledValue`` either hangs (caught by the test
    timeout) or returns False (caught by the assert)."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")

    N_THREADS = 4
    DISPATCHES_PER_THREAD = 8
    M = K = N = 4

    # One pipeline per thread — concurrent dispatches against the same
    # pipeline+argument-table would race the per-pipeline state, which
    # is NOT what this test is gating. We're gating the shared event /
    # counter contention.
    pipes = []
    for _ in range(N_THREADS):
        pipe = compile_mlpackage(pkg, function_name="main",
                                 input_dimensions={0: (K, M), 1: (N, K)})
        if pipe is None:
            for p in pipes:
                p.destroy()
            pytest.fail(f"compile_mlpackage failed; "
                        f"last_error_kind={last_error_kind()}")
        if not pipe.prepare_tensors():
            for p in pipes:
                p.destroy()
            pipe.destroy()
            pytest.skip("prepare_tensors failed on this host")
        pipes.append(pipe)

    try:
        rng = np.random.default_rng(0xBEEF)
        A_np = rng.standard_normal((M, K), dtype=np.float32)
        B_np = rng.standard_normal((K, N), dtype=np.float32)
        # Pre-fill each pipeline once — the dispatch itself is the
        # contended call, not the host-side fill.
        for p in pipes:
            assert p.fill_input("inputA", A_np.tobytes())
            assert p.fill_input("inputB", B_np.tobytes())

        failures: list[str] = []
        barrier = threading.Barrier(N_THREADS)

        def worker(idx: int) -> None:
            pipe = pipes[idx]
            # Sync all threads to the same starting line so the
            # dispatches actually race rather than sequence themselves.
            barrier.wait()
            for j in range(DISPATCHES_PER_THREAD):
                ok = pipe.dispatch(timeout_ms=30_000)
                if not ok:
                    failures.append(
                        f"thread {idx} dispatch {j} returned False "
                        f"(last_error_kind={last_error_kind()})")
                    return

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120.0)
            if t.is_alive():
                failures.append(f"thread {t.name} did not finish — "
                                f"likely a signal-counter race hang")

        assert not failures, "\n".join(failures)
    finally:
        for p in pipes:
            p.destroy()
