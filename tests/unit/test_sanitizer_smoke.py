"""Sanitizer smoke wrapper — drives the smoke binary under each
sanitizer build, when those build directories exist.

The actual sanitizer builds are gated behind
``scripts/run_sanitizers.sh`` because they cost ~2 minutes each.
This test only runs against build directories that the developer
or CI has already populated; otherwise it skips cleanly.

Catches the regression shape where someone re-introduces a data
race or use-after-free in the runtime — TSAN flagged the
``PerfettoTraceWriter`` race the first time, and ASAN would catch
the next variant.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_REL = (
    "src/collectives/tools/tessera-collective-runtime-smoke/"
    "tessera-collective-runtime-smoke"
)


def _sanitizer_binary(sanitizer: str) -> Path | None:
    binary = REPO_ROOT / f"build-{sanitizer}" / SMOKE_REL
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    return None


def _options_for(sanitizer: str) -> dict[str, str]:
    """Sanitizer runtime env vars — must abort on any finding so the
    test surfaces regressions instead of just logging them."""
    env = dict(os.environ)
    common = "halt_on_error=1:print_stacktrace=1"
    if sanitizer == "asan":
        env["ASAN_OPTIONS"] = common + ":abort_on_error=1"
    elif sanitizer == "tsan":
        env["TSAN_OPTIONS"] = common + ":second_deadlock_stack=1"
    elif sanitizer == "ubsan":
        env["UBSAN_OPTIONS"] = common
    return env


@pytest.mark.parametrize("sanitizer", ["asan", "tsan", "ubsan"])
def test_sanitizer_smoke(sanitizer: str) -> None:
    binary = _sanitizer_binary(sanitizer)
    if binary is None:
        pytest.skip(
            f"build-{sanitizer}/ smoke binary missing; build it with "
            f"`scripts/run_sanitizers.sh {sanitizer}` to enable this test"
        )
    proc = subprocess.run(
        [str(binary)],
        capture_output=True, text=True, timeout=90,
        env=_options_for(sanitizer),
    )
    assert proc.returncode == 0, (
        f"{sanitizer} smoke binary failed (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )
    for marker in (
        "[OK] 8 threads",
        "[OK] shutdown-while-submitting survived",
        "[OK] init-after-shutdown cycle",
        "[OK] PerfettoTraceWriter survived",
        "[ALL OK]",
    ):
        assert marker in proc.stdout, (
            f"{sanitizer}: missing marker {marker!r} in stdout:\n{proc.stdout}"
        )
