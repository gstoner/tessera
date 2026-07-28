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

A `--pass-pipeline=` invocation has the same problem in a second spelling —

    error: 'convert-scf-to-cf' does not refer to a registered pass or pass
    pipeline

— so the passes named *inside* a pipeline are checked too, not just the
`--pass-pipeline` flag itself.

`run_tessera_opt` skips instead, naming the missing pass and the binary's build
profile. A test that genuinely wants to assert a pass *is* registered should
assert on `registered_passes()` directly rather than relying on a crash.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

#: Environment selectors, most canonical first. `TESSERA_OPT_BIN` is the older
#: spelling `python/tessera/_jit_boundary.py` still honours; keeping it here
#: means a developer's exported variable selects the same binary for the
#: product and for the tests.
_ENV_SELECTORS = ("TESSERA_OPT", "TESSERA_OPT_BIN")

#: Search order for the driver when nothing is exported. A selector wins so a
#: developer with several build directories can choose one without editing tests.
_DEFAULT_CANDIDATES = (
    REPO_ROOT / "build/tools/tessera-opt/tessera-opt",
    REPO_ROOT / "build-apple/tools/tessera-opt/tessera-opt",
    REPO_ROOT / "build/bin/tessera-opt",
)


def tessera_opt_path() -> Path | None:
    """Resolve the driver, honouring the env selectors, or return None.

    An exported selector is final: if it names something that is not a file we
    return None rather than quietly falling back, because silently running a
    different binary than the one the developer asked for is how a "passing"
    run ends up proving nothing.
    """
    for selector in _ENV_SELECTORS:
        configured = os.environ.get(selector)
        if configured:
            path = Path(configured).expanduser()
            return path if path.is_file() else None
    in_repo = next((path for path in _DEFAULT_CANDIDATES if path.is_file()), None)
    if in_repo is not None:
        return in_repo
    on_path = shutil.which("tessera-opt")
    return Path(on_path) if on_path else None


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


#: The flag whose *value* names further passes rather than being one itself.
_PIPELINE_FLAG = "pass-pipeline"


def _pass_name(argument: str) -> str | None:
    """The registered name behind a CLI argument, or None if it isn't one.

    Handles `--pass`, `-pass`, `--pass=value`, and `--pass{opt=1}` alike; MLIR
    accepts one or two leading dashes and this suite uses both. A bare `-` (the
    stdin marker) names nothing.
    """
    stripped = argument.lstrip("-")
    return re.split(r"[=\{]", stripped, maxsplit=1)[0] if stripped else None


def _pipeline_pass_names(argument: str) -> tuple[str, ...]:
    """Pass names nested inside a `--pass-pipeline=...` value.

    A pipeline names its passes without the leading dashes, so the argument
    itself only proves `--pass-pipeline` exists — the passes inside are what a
    partial build actually lacks. Anchors (`builtin.module(...)`,
    `gpu.module(...)`) open a nesting level rather than naming a pass, and
    `{...}` carries a pass's options; neither is a registry name.
    """
    _, separator, pipeline = argument.partition("=")
    if not separator:
        return ()
    without_options = re.sub(r"\{[^{}]*\}", "", pipeline)
    return tuple(
        name for name, anchor in
        re.findall(r"([A-Za-z0-9][\w.-]*)\s*(\(?)", without_options)
        if not anchor
    )


def _requested_names(arguments: Iterable[str]) -> list[str]:
    """Registry names an invocation depends on, from any accepted spelling.

    Callers reach this two ways: `require_tessera_opt("tessera-apple-...")`
    naming a pass bare, and the `run_*` helpers forwarding real CLI arguments.
    """
    names: list[str] = []
    for argument in arguments:
        name = _pass_name(argument) if argument.startswith("-") else argument
        if not name:
            continue
        names.append(name)
        if name == _PIPELINE_FLAG:
            names.extend(_pipeline_pass_names(argument))
    return names


def missing_passes(tool: Path, *passes: str) -> list[str]:
    """Requested names, sorted, that `tool`'s pass registry does not know."""
    available = registered_passes(tool)
    return sorted({name for name in _requested_names(passes)
                   if name not in available})


def capability_skip_reason(tool: Path, *passes: str) -> str | None:
    """Why `tool` cannot run `passes`, or None when it can.

    The single capability check for the tree — `require_tessera_opt` here and
    `CompilerToolchain.require_tessera_opt` both route through it, so two call
    sites can never disagree about what a binary can do.
    """
    missing = missing_passes(tool, *passes)
    if not missing:
        return None
    return (
        f"{tool} does not register {', '.join(missing)} "
        f"(build profile: {build_profile(tool)}). Configure a build with "
        "the owning backend, or point TESSERA_OPT at one."
    )


def require_tessera_opt(*passes: str) -> Path:
    """Return the driver, skipping unless it registers every named pass."""
    tool = tessera_opt_path()
    if tool is None:
        pytest.skip(
            "tessera-opt not built (ninja -C build tessera-opt) and TESSERA_OPT unset"
        )
    reason = capability_skip_reason(tool, *passes)
    if reason is not None:
        pytest.skip(reason)
    return tool


def _pass_arguments(arguments: Iterable[str]) -> tuple[str, ...]:
    """The dash-prefixed subset — option values stay out of the capability check."""
    return tuple(argument for argument in arguments if argument.startswith("-"))


def run_tessera_opt(directive: str, *passes: str) -> subprocess.CompletedProcess:
    """Run `tessera-opt - <passes>` on `directive`, skipping if unsupported.

    Only dash-prefixed arguments are treated as pass names; positional/option
    values are passed through untouched.
    """
    tool = require_tessera_opt(*_pass_arguments(passes))
    return subprocess.run(
        [str(tool), "-", *passes], input=directive,
        capture_output=True, text=True,
    )


def run_tessera_opt_file(
    source: Path | str, *passes: str, timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run `tessera-opt <passes> <source>` on a file, skipping if unsupported.

    The stdin form above covers directives a test builds in memory; fixtures
    that already live on disk (lit inputs) come through here so neither shape
    needs a resolver of its own.
    """
    tool = require_tessera_opt(*_pass_arguments(passes))
    return subprocess.run(
        [str(tool), *passes, str(source)],
        capture_output=True, text=True, timeout=timeout,
    )
