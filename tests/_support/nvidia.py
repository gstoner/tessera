"""Shared NVIDIA exact-device test probes and provenance assertions."""
from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from typing import Any

import pytest
from tests._support.environment import (
    ensure_cuda_bin_on_path,
    ensure_nvidia_driver_on_path,
    nvidia_gpu_is_plausibly_present,
)


def nvidia_cuda_toolchain_available() -> bool:
    """Whether the host exposes an NVIDIA CUDA compiler."""
    ensure_cuda_bin_on_path()
    return bool(shutil.which("nvcc") or os.path.isfile("/usr/local/cuda/bin/nvcc"))


def nvidia_mma_runtime_available() -> bool:
    """Whether the shipped NVIDIA MMA runtime can execute on this host."""
    if not nvidia_cuda_toolchain_available():
        return False
    try:
        from tessera import runtime as rt
        return rt._nvidia_mma_runtime_available()
    except Exception:
        return False


def nvidia_mma_ptx_launch_available() -> bool:
    """Whether the MMA runtime and the shipped PTX launch bridge are usable."""
    if not nvidia_mma_runtime_available():
        return False
    try:
        from tessera import runtime as rt
        return rt._load_nvidia_ptx_launch() is not None
    except Exception:
        return False


def require_nvidia_mma_runtime() -> Any:
    """Return the runtime or skip with a stable, capability-specific reason."""
    if not nvidia_cuda_toolchain_available():
        pytest.skip("nvcc not installed")
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available():
        pytest.skip("no usable NVIDIA CUDA device")
    return rt


def nvidia_cuda_host_ready() -> bool:
    """Whether CUDA tooling and the NVIDIA driver are reachable from this host.

    Repairs PATH first, then WARNS if a GPU is evidently present but still
    unreachable. A silent skip is the correct answer for a host with no NVIDIA
    hardware; it is the WRONG answer for a host that has one and cannot see it,
    because the run then reports success having executed nothing. That is not
    hypothetical -- it hid 80 real failures on The-Super-Bear until 2026-08-30,
    two of which were compiler defects.
    """
    ensure_nvidia_driver_on_path()
    ready = nvidia_cuda_toolchain_available() and shutil.which("nvidia-smi") is not None
    if ready:
        try:
            ready = subprocess.run(
                ["nvidia-smi"], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=5, check=False,
            ).returncode == 0
        except OSError:
            ready = False
    if not ready and nvidia_gpu_is_plausibly_present():
        warnings.warn(
            "NVIDIA device lanes are being SKIPPED on a host that appears to "
            "have an NVIDIA GPU. This is an environment problem, not an absent "
            "device, and the skips will look like a clean run. Source "
            "scripts/_nvidia_env.sh (or put the driver shim -- /usr/lib/wsl/lib "
            "under WSL2 -- and the CUDA toolkit on PATH) before treating any "
            "result from this session as device evidence.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ready


def assert_native_gpu(result: dict[str, Any]) -> None:
    """Require a successful result with actual NVIDIA device provenance."""
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_gpu", result.get("reason")
