"""Capability-aware `tessera-opt` invocation for codegen tests.

`tessera-opt` is a *variable* binary. Which passes it registers depends on how
it was configured (`TESSERA_BUILD_{ROCM,NVIDIA,APPLE}_BACKEND`, whether a real
CUDA/HIP toolchain was present, ...), and a working tree commonly holds several
build directories at once.

Tests that drive a backend pass have historically resolved
`build/tools/tessera-opt/tessera-opt` themselves and skipped only when the file
is *missing*. That leaves the far more common case unhandled: the binary exists
but was built without the backend under test. The test then fails with

    tessera-opt: Unknown command line argument '--generate-rocm-where-kernel'

which reads as a broken test rather than what it is — a build-selection
problem. On a Mac configured `TESSERA_BUILD_ROCM_BACKEND=OFF` that accounts for
hundreds of failures that say nothing about the code under test.

`run_tessera_opt` skips instead, naming the missing pass and the binary's build
profile. A test that genuinely wants to assert a pass *is* registered should
assert on `registered_passes()` directly rather than relying on a crash.
"""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

#: Search order for the driver. `TESSERA_OPT` wins so a developer with several
#: build directories can select one without editing tests.
_DEFAULT_CANDIDATES = (
    REPO_ROOT / "build/tools/tessera-opt/tessera-opt",
    REPO_ROOT / "build-apple/tools/tessera-opt/tessera-opt",
)


def tessera_opt_path() -> Path | None:
    """Resolve the driver, honouring `TESSERA_OPT`, or return None."""
    configured = os.environ.get("TESSERA_OPT")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    return next((path for path in _DEFAULT_CANDIDATES if path.is_file()), None)


@lru_cache(maxsize=8)
def registered_passes(tool: Path) -> frozenset[str]:
    """Pass/pipeline names this binary registers. Read once per tool path."""
    try:
        help_text = subprocess.run(
            [str(tool), "--help"], capture_output=True, text=True,
            check=False, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(re.findall(r"--([A-Za-z0-9][\w-]*)", help_text))


@lru_cache(maxsize=8)
def build_profile(tool: Path) -> str:
    """`--tessera-build-info` summary, or a note that the binary predates it."""
    try:
        result = subprocess.run(
            [str(tool), "--tessera-build-info"], capture_output=True,
            text=True, check=False, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0 or not result.stdout.strip():
        return "unknown (binary predates --tessera-build-info)"
    return " ".join(result.stdout.split())


def _pass_name(argument: str) -> str | None:
    """The registered name behind a CLI argument, or None if it isn't one.

    Handles `--pass`, `--pass=value`, and `--pass{opt=1}` spellings alike.
    """
    if not argument.startswith("--"):
        return None
    return re.split(r"[=\{]", argument[2:], maxsplit=1)[0]


def require_tessera_opt(*passes: str) -> Path:
    """Return the driver, skipping unless it registers every named pass."""
    tool = tessera_opt_path()
    if tool is None:
        pytest.skip(
            "tessera-opt not built (ninja -C build tessera-opt) and TESSERA_OPT unset"
        )
    available = registered_passes(tool)
    missing = [name for name in (_pass_name(p) or p for p in passes)
               if name and name not in available]
    if missing:
        pytest.skip(
            f"{tool} does not register {', '.join(sorted(set(missing)))} "
            f"(build profile: {build_profile(tool)}). Configure a build with "
            "the owning backend, or point TESSERA_OPT at one."
        )
    return tool


def run_tessera_opt(directive: str, *passes: str) -> subprocess.CompletedProcess:
    """Run `tessera-opt - <passes>` on `directive`, skipping if unsupported.

    Only `--` arguments are treated as pass names; positional/option values are
    passed through untouched.
    """
    tool = require_tessera_opt(*passes)
    return subprocess.run(
        [str(tool), "-", *passes], input=directive,
        capture_output=True, text=True,
    )
