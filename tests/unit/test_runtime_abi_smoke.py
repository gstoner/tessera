"""Runtime ABI smoke regression — drives the unsanitized
``tessera-runtime-abi-smoke`` binary so the C ABI contract is
checked in every Python sweep (not just sanitizer lanes).

This is the unsanitized companion to the parametrized
``test_sanitizer_smoke.py`` tests.  Findings audit (2026-05-19)
flagged that the previous sanitizer driver only exercised the
*collective* runtime — the *device runtime* (``tsr*``) C ABI was
silently uncovered.  This test ensures the runtime lifecycle
contract is exercised in the base ctest / pytest sweep too;
sanitizer lanes add red-zone / race / UB instrumentation on top.

Skips cleanly when the smoke binary isn't built (the
``cmake --build build --target tessera-runtime-abi-smoke``
side-effect of the normal monorepo build).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_BINARY = (
    REPO_ROOT / "build" / "src" / "runtime" / "tessera-runtime-abi-smoke"
)


def _have_binary() -> bool:
    return SMOKE_BINARY.is_file() and os.access(SMOKE_BINARY, os.X_OK)


@pytest.mark.skipif(
    not _have_binary(),
    reason=(
        "tessera-runtime-abi-smoke not built; run "
        "`cmake --build build --target tessera-runtime-abi-smoke`"
    ),
)
def test_runtime_abi_smoke_lifecycle() -> None:
    """The seven scenarios the smoke binary exercises must all
    pass — they cover the runtime lifecycle contract end to end."""
    proc = subprocess.run(
        [str(SMOKE_BINARY)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"runtime ABI smoke failed (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )
    expected = (
        "[OK] init idempotent",
        "[OK] malloc/memset/map/free round-trip",
        "[OK] memcpy intra-device round-trip",
        "[OK] 16x stream+event create/record/sync/destroy cycle",
        "[OK] tsrShutdown refuses live handles",
        "[OK] init",   # init→handles→shutdown→init cycle
        "[OK] multi-device",   # may be skipped on CPU-only builds
        "[ALL OK]",
    )
    for marker in expected:
        assert marker in proc.stdout, (
            f"missing expected marker {marker!r} in smoke output:\n{proc.stdout}"
        )
