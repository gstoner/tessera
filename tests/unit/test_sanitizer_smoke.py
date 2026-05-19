"""Sanitizer smoke wrapper — drives both smoke binaries under each
sanitizer build, when those build directories exist.

The actual sanitizer builds are gated behind
``scripts/run_sanitizers.sh`` because they cost ~2 minutes each.
This test only runs against build directories that the developer
or CI has already populated; otherwise it skips cleanly.

Two distinct C-side surfaces are covered:

  * ``tessera-collective-runtime-smoke`` — collective ``tessera_*``
    entry points (qos_limit_set, submit_chunk_async, shutdown,
    trace_write).  Catches PerfettoTraceWriter races under TSAN
    and lifetime issues in the shared_ptr-based runtime slot.

  * ``tessera-runtime-abi-smoke`` — device runtime ``tsr*`` C ABI
    (tsrInit / tsrMalloc / tsrCreateStream / tsrFree / tsrShutdown).
    Catches handle-lifetime corruption under ASAN (the
    use-after-free shape the live-handle ratchet prevents),
    counter-accounting races under TSAN, arithmetic/aliasing UB
    under UBSAN.

A regression in either surface — TSAN flagged the
PerfettoTraceWriter race the first time, ASAN would catch any
re-introduction of the freed-device-pointer shape — fails the
relevant parametrization here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``(target_name, build-relative path)`` for every smoke we exercise.
#: Keep in sync with ``SMOKES`` in ``scripts/run_sanitizers.sh``.
SMOKE_BINARIES: tuple[tuple[str, str], ...] = (
    (
        "tessera-collective-runtime-smoke",
        "src/collectives/tools/tessera-collective-runtime-smoke/"
        "tessera-collective-runtime-smoke",
    ),
    (
        "tessera-runtime-abi-smoke",
        "src/runtime/tessera-runtime-abi-smoke",
    ),
)

#: Per-smoke success markers each binary prints on a clean run.
SMOKE_MARKERS: dict[str, tuple[str, ...]] = {
    "tessera-collective-runtime-smoke": (
        "[OK] 8 threads",
        "[OK] shutdown-while-submitting survived",
        "[OK] init-after-shutdown cycle",
        "[OK] PerfettoTraceWriter survived",
        "[ALL OK]",
    ),
    "tessera-runtime-abi-smoke": (
        "[OK] init idempotent",
        "[OK] malloc/memset/map/free round-trip",
        "[OK] memcpy intra-device round-trip",
        "[OK] 16x stream+event create/record/sync/destroy cycle",
        "[OK] tsrShutdown refuses live handles",
        "[OK] init",   # init→handles→shutdown→init line
        "[ALL OK]",
    ),
}


def _binary_path(sanitizer: str, rel: str) -> Path:
    return REPO_ROOT / f"build-{sanitizer}" / rel


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


@pytest.mark.parametrize(
    "sanitizer,smoke,rel",
    [
        (s, name, rel)
        for s in ("asan", "tsan", "ubsan")
        for name, rel in SMOKE_BINARIES
    ],
    ids=lambda v: v,
)
def test_sanitizer_smoke(sanitizer: str, smoke: str, rel: str) -> None:
    binary = _binary_path(sanitizer, rel)
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        pytest.skip(
            f"build-{sanitizer}/{rel} not present; build via "
            f"`scripts/run_sanitizers.sh {sanitizer}` to enable this test"
        )
    proc = subprocess.run(
        [str(binary)],
        capture_output=True, text=True, timeout=120,
        env=_options_for(sanitizer),
    )
    assert proc.returncode == 0, (
        f"{sanitizer}/{smoke} failed (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )
    for marker in SMOKE_MARKERS[smoke]:
        assert marker in proc.stdout, (
            f"{sanitizer}/{smoke}: missing marker {marker!r}\n"
            f"stdout:\n{proc.stdout}"
        )
