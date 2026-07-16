"""Compatibility imports for NVIDIA helpers retained by unit tests.

New exact-device tests import from :mod:`tests._support.nvidia`, so their
location does not make helper availability depend on pytest's test-directory
import path.
"""

from tests._support.nvidia import (
    assert_native_gpu,
    nvidia_cuda_host_ready,
    nvidia_cuda_toolchain_available,
    nvidia_mma_runtime_available,
    require_nvidia_mma_runtime,
)

__all__ = [
    "assert_native_gpu",
    "nvidia_cuda_host_ready",
    "nvidia_cuda_toolchain_available",
    "nvidia_mma_runtime_available",
    "require_nvidia_mma_runtime",
]
