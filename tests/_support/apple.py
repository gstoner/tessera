"""Shared Apple exact-device capability and provenance assertions.

The portable lane may validate the same numerical oracle through the explicit
reference fallback.  These helpers deliberately make that result insufficient
for a test that claims native Apple execution.
"""
from __future__ import annotations

import shutil
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
    """Whether the offline Apple ``metal`` compiler is available on this host."""
    if shutil.which("metal") is not None:
        return True
    try:
        return bool(
            subprocess.run(
                ["xcrun", "-f", "metal"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return False


def require_metal_compiler() -> None:
    """Skip an offline MSL compiler test with one capability-specific reason."""
    if not metal_compiler_available():
        pytest.skip(
            "compiler_tool requires the Apple `metal` compiler "
            "(install Xcode or Command Line Tools)"
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
