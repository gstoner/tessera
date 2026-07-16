"""Shared NVIDIA exact-device test probes and provenance assertions."""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

import pytest


def nvidia_cuda_toolchain_available() -> bool:
    """Whether the host exposes an NVIDIA CUDA compiler."""
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
    """Whether CUDA tooling and the NVIDIA driver are reachable from this host."""
    if not nvidia_cuda_toolchain_available() or shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5, check=False,
        ).returncode == 0
    except OSError:
        return False


def assert_native_gpu(result: dict[str, Any]) -> None:
    """Require a successful result with actual NVIDIA device provenance."""
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_gpu", result.get("reason")
