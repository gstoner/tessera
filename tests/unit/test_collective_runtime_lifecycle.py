"""Lifecycle regression for the collective runtime C ABI.

Drives the ``tessera-collective-runtime-smoke`` binary (built from
``src/collectives/tools/tessera-collective-runtime-smoke/``) and
asserts that the four lifecycle / thread-safety scenarios it exercises
all pass:

  1. Concurrent submitters across 8 threads do not corrupt state.
  2. ``tessera_shutdown_runtime()`` while a submit is in flight
     does not crash (``shared_ptr``-based slot keeps the runtime
     alive past the concurrent shutdown).
  3. Init-after-shutdown re-creates the runtime on demand.
  4. ``PerfettoTraceWriter`` is internally synchronized and does not
     lose events under direct concurrent mutation.

Findings audit (2026-05-19) flagged the architectural concern that
the global mutex was held through ``ExecRuntime::submit()``.  The
runtime now drops that global lock before submit by returning a
``shared_ptr`` strong reference from the runtime slot.  The prerequisite
fix is that per-runtime components own their own synchronization:
``TokenLimiter`` already did, and ``PerfettoTraceWriter`` now locks
mutating methods and snapshots before flushing.  This test locks that
shipping contract end to end.

Skips cleanly when the binary isn't built (the smoke binary lives
behind ``cmake --build build --target tessera-collective-runtime-smoke``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_BINARY = (
    REPO_ROOT / "build" / "src" / "collectives" / "tools"
    / "tessera-collective-runtime-smoke"
    / "tessera-collective-runtime-smoke"
)


def _have_binary() -> bool:
    return SMOKE_BINARY.is_file() and os.access(SMOKE_BINARY, os.X_OK)


_REQUIRES_BINARY = pytest.mark.skipif(
    not _have_binary(),
    reason=(
        "tessera-collective-runtime-smoke not built; run "
        "`cmake --build build --target tessera-collective-runtime-smoke`"
    ),
)


@_REQUIRES_BINARY
def test_collective_runtime_lifecycle_smoke_succeeds() -> None:
    proc = subprocess.run(
        [str(SMOKE_BINARY)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"smoke binary failed (rc={proc.returncode}):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    # Spot-check each scenario reported success.
    for marker in (
        "[OK] 8 threads",
        "[OK] shutdown-while-submitting survived",
        "[OK] init-after-shutdown cycle",
        # PerfettoTraceWriter must be internally thread-safe.  This
        # is the prerequisite that made dropping the global mutex
        # from the submit hot path safe; a future refactor that
        # silently removes the writer's internal lock would fail
        # this scenario before destabilizing the wider runtime.
        "[OK] PerfettoTraceWriter survived",
        "[ALL OK]",
    ):
        assert marker in proc.stdout, (
            f"missing expected marker {marker!r} in smoke output:\n{proc.stdout}"
        )
