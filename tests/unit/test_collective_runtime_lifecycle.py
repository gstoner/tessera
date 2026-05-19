"""Lifecycle regression for the collective runtime C ABI.

Drives the ``tessera-collective-runtime-smoke`` binary (built from
``src/collectives/tools/tessera-collective-runtime-smoke/``) and
asserts that the three lifecycle scenarios it exercises all pass:

  1. Concurrent submitters across 8 threads do not corrupt state.
  2. ``tessera_shutdown_runtime()`` while a submit is in flight
     does not crash (``shared_ptr``-based slot keeps the runtime
     alive past the concurrent shutdown).
  3. Init-after-shutdown re-creates the runtime on demand.

Findings audit (2026-05-19) flagged the architectural concern that
the global mutex was held through ``ExecRuntime::submit()``.  We
investigated dropping the lock under submit but found that
``ExecRuntime`` is not yet internally thread-safe
(``PerfettoTraceWriter`` shares state without an internal mutex),
so the current shipping state holds the mutex by design and the
followup is to add an internal per-runtime lock.  This test
locks the contract that survives today: safe under concurrent
submits and across shutdown/reinit cycles.

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
        "[ALL OK]",
    ):
        assert marker in proc.stdout, (
            f"missing expected marker {marker!r} in smoke output:\n{proc.stdout}"
        )
