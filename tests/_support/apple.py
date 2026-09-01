"""Shared Apple exact-device capability and provenance assertions.

The portable lane may validate the same numerical oracle through the explicit
reference fallback.  These helpers deliberately make that result insufficient
for a test that claims native Apple execution.
"""
from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import subprocess
import sys
from typing import Any, Mapping

import pytest


def apple_metal_available() -> bool:
    """Whether this process can make a real Apple Metal placement claim."""
    if sys.platform != "darwin":
        return False
    try:
        from tessera.runtime import DeviceTensor
        return bool(DeviceTensor.is_metal())
    except Exception:
        return False


def require_apple_metal() -> None:
    """Skip an exact-device test with one stable capability-specific reason."""
    if sys.platform != "darwin":
        pytest.skip("hardware_apple_gpu requires a Darwin host")
    if not apple_metal_available():
        pytest.skip(
            "hardware_apple_gpu requires an available Metal device "
            "(run the exact-device lane outside the sandbox with a fresh runtime)"
        )


def apple_gpu_jit_runtime_available() -> bool:
    """Whether the Apple runtime and JIT bridge are both loadable."""
    try:
        from tessera import _apple_gpu_backend as apple_gpu_backend
        from tessera import _jit_boundary as jit_boundary

        return bool(apple_gpu_backend.is_available() and jit_boundary.is_available())
    except Exception:
        return False


def require_apple_gpu_jit_runtime() -> None:
    """Require the runtime/JIT ABI after the shared Metal device boundary."""
    if not apple_gpu_jit_runtime_available():
        pytest.skip(
            "integration requires the Apple GPU runtime and libtessera_jit ABI"
        )


def require_apple_package_fixture(path: Any) -> None:
    """Require an authored Metal package fixture for package integration tests."""
    if path is None:
        pytest.skip("integration requires a checked-in .mtlpackage fixture")


def require_apple_metal4() -> None:
    """Require the Metal 4 runtime surface after the exact-device boundary."""
    require_apple_metal()
    try:
        from tessera import runtime

        available = bool(runtime.apple_gpu_metal4_caps().get("available"))
    except Exception:
        available = False
    if not available:
        pytest.skip("metal4 requires an available Apple Metal 4 runtime")


def metal_compiler_available() -> bool:
    """Whether the offline Apple ``metal`` compiler can actually *run* here.

    Resolving the binary is not enough, and that distinction bit us: Xcode ships
    a ``metal`` driver that `xcrun -f metal` finds happily and that then exits
    with "cannot execute tool 'metal' due to missing Metal Toolchain" until
    ``xcodebuild -downloadComponent MetalToolchain`` has been run. A presence
    check therefore reports available on a host where every compile fails. Ask
    for the version instead — it is the cheapest call that proves execution.
    """
    try:
        return subprocess.run(
            ["xcrun", "metal", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def require_metal_compiler() -> None:
    """Skip an offline MSL compiler test with one capability-specific reason."""
    if not metal_compiler_available():
        pytest.skip(
            "compiler_tool requires a runnable Apple `metal` compiler — point "
            "xcode-select at Xcode and run "
            "`xcodebuild -downloadComponent MetalToolchain`; verify with "
            "`xcrun metal --version`"
        )


def apple_gpu_memory_abi_available() -> bool:
    """Whether the loaded Apple runtime exports the memory-budget ABI."""
    try:
        from tessera import runtime

        return runtime._apple_gpu_memory_api() is not None
    except Exception:
        return False


def require_apple_gpu_memory_abi() -> None:
    """Skip a memory-budget ABI test with one capability-specific reason."""
    if not apple_gpu_memory_abi_available():
        pytest.skip(
            "integration requires the Apple GPU memory-budget runtime ABI"
        )


def require_apple_accelerate() -> None:
    """Skip an Apple CPU integration test without conflating it with Metal."""
    if sys.platform != "darwin":
        pytest.skip("integration requires Darwin's Accelerate framework")


def require_darwin_host() -> None:
    """Skip a host-API integration test with a stable Darwin capability reason."""
    if sys.platform != "darwin":
        pytest.skip("integration requires a Darwin host")


def require_apple_chip_identity() -> None:
    """Require the unmasked Darwin sysctl capability used for chip calibration."""
    require_darwin_host()
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        brand = ""
    if not brand.lower().startswith("apple m"):
        pytest.skip(
            "integration requires unmasked Darwin Apple-chip identity via sysctl"
        )


def assert_native_apple_gpu(
    result: Mapping[str, Any], *, compiler_path: str | None = None,
) -> None:
    """Require successful native Metal provenance, not a semantic fallback."""
    assert result.get("ok") is True, result.get("reason")
    assert result.get("execution_kind") == "native_gpu", result.get("reason")
    assert result.get("execution_mode") == "metal_runtime", result.get("reason")
    if compiler_path is not None:
        assert result.get("compiler_path") == compiler_path, result.get("reason")


def assert_reference_cpu(result: Mapping[str, Any]) -> None:
    """Lock an unsupported or unavailable path to the explicit CPU fallback."""
    assert result.get("ok") is True, result.get("reason")
    assert result.get("execution_kind") == "reference_cpu", result.get("reason")


def assert_native_apple_jit(compiled: Any) -> None:
    """Require a JIT callable to have executed on Metal, not a fallback lane."""

    assert getattr(compiled, "execution_kind", None) == "native_gpu"
    metadata = compiled.runtime_artifact().metadata
    assert metadata.get("execution_mode") == "metal_runtime"


# --- Apple backend compiled into tessera-opt -------------------------------
#
# `tessera-opt` only registers the Apple pipelines when it was configured with
# -DTESSERA_BUILD_APPLE_BACKEND=ON (compile-time `TESSERA_HAVE_APPLE_BACKEND`).
# The ROCm / x86 fleet boxes build it OFF, so on those hosts the Apple fixtures
# used to FAIL on `Unknown command line argument
# '-tessera-lower-to-apple_cpu-full'` -- 57 red in a full sweep that say nothing
# about Apple and drown out real signal. A capability the host's build lacks
# must SKIP, not fail (the same rule CLAUDE.md applies to a missing device).
#
# This is the existing `tessera-opt not built` guard's missing other half: that
# one asks whether the binary exists, this one asks whether the binary has the
# Apple backend in it.
#
# EVERY entry point takes the tool path explicitly. The fixtures do not agree on
# how to find tessera-opt -- some honour `TESSERA_OPT`, others `TESSERA_OPT_PATH`,
# with different default build dirs -- so a helper that resolved the binary
# independently could inspect a DIFFERENT build than the one that produced the
# failure. On a host with several build trees that silently inverts the guard:
# it could skip a real registration regression in an Apple-enabled binary
# because some other preferred binary lacks the backend (#620 review). Probe the
# executable that actually ran, and cache by its path.

#: A pipeline registered only under `TESSERA_HAVE_APPLE_BACKEND`. Matched by
#: exact flag name -- a bare "apple" substring would also match LLVM's own
#: `=apple` NEON asm flavour in --help and report a false positive.
_APPLE_PROBE_PIPELINE = "tessera-lower-to-apple_cpu"


@functools.cache
def tessera_opt_registers(flag: str, opt_path: str) -> bool:
    """True iff the ``tessera-opt`` at ``opt_path`` registers ``flag``.

    Reads ``--help`` rather than running the pipeline, so the probe cannot be
    confused by an unrelated pipeline failure. Returns False (never raises) when
    the binary is missing or unrunnable. Cached per (flag, binary)."""
    if not opt_path or not Path(opt_path).is_file():
        return False
    try:
        out = subprocess.run(
            [str(opt_path), "--help"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in (out.stdout + out.stderr)


def apple_backend_in_tessera_opt(opt_path: str) -> bool:
    """True iff the ``tessera-opt`` at ``opt_path`` was built with the Apple backend."""
    return tessera_opt_registers(_APPLE_PROBE_PIPELINE, str(opt_path))


def apple_backend_configured_in_build(opt_path: str) -> bool:
    """True iff the CMake cache of the tree that produced ``opt_path`` says the
    Apple backend is ON.

    Deliberately independent of :func:`apple_backend_in_tessera_opt` -- it reads
    the cache, not ``--help`` -- so the two can be cross-checked. A guard that
    silently skips a suite it should have run is worse than the loud failure it
    replaced, and this is what makes that detectable."""
    if not opt_path:
        return False
    for parent in Path(opt_path).resolve().parents:
        cache = parent / "CMakeCache.txt"
        if cache.is_file():
            try:
                return "TESSERA_BUILD_APPLE_BACKEND:BOOL=ON" in cache.read_text(
                    errors="ignore"
                )
            except OSError:
                return False
    return False


def skip_if_apple_pipeline_unregistered(proc: Any, opt_path: str) -> None:
    """Skip when ``opt_path`` rejected an Apple pipeline it was never built with.

    Call right after invoking the pipeline, before asserting on the result;
    ``opt_path`` must be the binary that produced ``proc``.

    Deliberately narrow in three ways, so it cannot hide a real defect:
      * only fires on the two "this pipeline is not registered" signatures, one
        per invocation form (``-flag`` and ``--pass-pipeline=``);
      * only when that binary genuinely lacks the Apple backend -- if it HAS it,
        an unregistered pipeline is a registration regression and stays loud;
      * never touches a nonzero exit for any other reason, so the negative
        fixtures that expect a rejection still assert their own diagnostics.
    """
    import pytest

    if getattr(proc, "returncode", 0) == 0:
        return
    stderr = getattr(proc, "stderr", "") or ""
    unregistered = (
        "Unknown command line argument" in stderr
        or "does not refer to a registered pass or pass pipeline" in stderr
    )
    if not unregistered:
        return
    if apple_backend_in_tessera_opt(opt_path):
        return
    pytest.skip(
        "tessera-opt was built without the Apple backend "
        "(configure -DTESSERA_BUILD_APPLE_BACKEND=ON to run this)"
    )


#: The promotion thresholds `aggregate_stable_route_reports` seals into every
#: Apple strict route ledger, and that `load_strict_route_ledger` now holds a
#: promotion to. Kept here rather than in one test module because two test
#: files build ledger fixtures and a third checks the committed ones; three
#: copies of a threshold set is how they drift apart.
STRICT_PROMOTION_RULES = {
    "minimum_speedup_fraction_each_run": 0.05,
    "maximum_cross_run_drift_fraction": 0.15,
    "absolute_time_drift_is_diagnostic_only": True,
    "minimum_paired_win_fraction_each_run": 0.75,
    "maximum_cross_run_speedup_spread": 0.05,
    "requires_native_dispatch": True,
    "requires_numerical_validation": True,
    "requires_repeated_measurement": True,
    "requires_interleaved_paired_trials": True,
    "requires_resource_evidence": True,
}


def strict_promotion_evidence(**overrides: object) -> dict[str, object]:
    """Evidence a real promotion carries, clearing `STRICT_PROMOTION_RULES`.

    Ledger fixtures used to assert `status: "promote_candidate"` with **no**
    `route_evidence` and no rules block -- a forged promotion, which the loader
    accepted because it validated provenance exhaustively and the promotion
    criteria not at all. Building fixtures the way `seal_strict_route_ledger`
    actually builds them is what lets a negative case mean something: the
    rejection has to come from the field under test, not from evidence that
    was never there.

    Pass an override to forge a specific violation.
    """
    return {
        "present_in_all_runs": True,
        "placement_and_numerical_proof": True,
        "repeated_measurement": True,
        "paired_measurement": True,
        "resource_evidence_retained": True,
        "paired_median_speedups": [0.31, 0.29],
        "paired_win_fractions": [1.0, 1.0],
        "cross_run_speedup_spread": 0.02,
        **overrides,
    }
